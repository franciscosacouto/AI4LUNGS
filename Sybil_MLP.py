import torch 
import pytorch_lightning as L
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.loss import weibull
from torchmetrics.classification import BinaryF1Score, Accuracy, Recall, Precision
from lifelines.utils import concordance_index

class SybilEncoderSurvival(L.LightningModule):
    def __init__(self, sybil_wrapper, survival_head, learning_rate, pos_weight, freeze_encoder=True):
        super().__init__()
        self.sybil = sybil_wrapper
        self.survival_head = survival_head
        self.learning_rate = learning_rate
        self.freeze_encoder = freeze_encoder

     # Metrics
        self.cindex_metric = ConcordanceIndex()
        self.auroc_metric = Auc() 
        self.f1score = BinaryF1Score()
        self.cindex_metric = ConcordanceIndex()

        self.test_preds = []
        self.test_events = []
        self.test_time= []
        self.val_preds = []
        self.val_events = []
        self.val_time = []
        self.train_preds = [] 
        self.train_events = []
        self.train_time = []
        self.train_log_hz_t = []
        self.val_log_hz_t = []
        self.test_log_hz_t = []
        self.test_auroc = None
        self.test_f1_score = None
        self.test_balanced_accuracy = None
        self.freeze_encoder = freeze_encoder
        # Set the freeze state
        if self.freeze_encoder:
            for param in self._encoder_wrapper.parameters():
                param.requires_grad = False
        else:
            self._encoder_wrapper.train()
            for param in self._encoder_wrapper.parameters():
                param.requires_grad = True
        
        # We store the ensemble models from the wrapper
        self.models = self.sybil.models

    def forward(self, dicom_paths):
        """
        Sybil is unique: It takes paths, not tensors.
        We average embeddings across the ensemble before the survival head.
        """
        all_embeddings = []
        
        # Extract embeddings from the ensemble
        # Based on Sybil's architecture, we capture the output of the relu layer
        for model in self.models:
            # We use a context manager or hook if needed, 
            # but usually Sybil wrappers provide an 'embed' mode or similar.
            # Here we assume a forward pass that returns latent features.
            with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                # This is a conceptual call; Sybil typically returns a dict or tensor
                # We expect a (Batch, 512) embedding
                emb = model.embed(dicom_paths) 
                all_embeddings.append(emb)
        
        # Average ensemble embeddings: (Batch, 512)
        avg_emb = torch.stack(all_embeddings).mean(dim=0)
        
        # Pass through Weibull Survival Head: (Batch, 2)
        log_params = self.survival_head(avg_emb)
        return log_params

    def training_step(self, batch, batch_idx):
        # paths is a list of lists (batch of DICOM path lists)
        paths, event, time = batch
        event = event.squeeze().bool()
        time = time.squeeze().float()

        log_params = self(paths).squeeze()
        loss = weibull.neg_log_likelihood(log_params, event, time)
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        paths, event, time = batch
        event = event.squeeze().bool()
        time = time.squeeze().float()

        log_params = self(paths).squeeze()
        loss = weibull.neg_log_likelihood(log_params, event, time)
        
        # Get hazard at t=5 years (scaled)
        time_5y = torch.tensor(1825.0 / 2786.0).to(self.device)
        log_hz_t = weibull.log_hazard(log_params, time_5y)
        
        output = {
            "loss": loss,
            "preds": log_params.detach().cpu(),
            "events": event.detach().cpu(),
            "time": time.detach().cpu(),
            "log_hz_t": log_hz_t.detach().cpu()
        }
        self.validation_step_outputs.append(output)
        self.log("val_loss", loss, prog_bar=True)
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        preds = torch.cat([x["preds"] for x in outputs])
        events = torch.cat([x["events"] for x in outputs])
        time = torch.cat([x["time"] for x in outputs])
        log_hz_t = torch.cat([x["log_hz_t"] for x in outputs])

        self._calculate_balanced_metrics(preds, events, time, "val", log_hz_t)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        paths, event, time = batch
        log_params = self(paths).squeeze()
        
        time_5y = torch.tensor(1825.0 / 2786.0).to(self.device)
        log_hz_t = weibull.log_hazard(log_params, time_5y)

        output = {
            "preds": log_params.detach().cpu(),
            "events": event.detach().cpu(),
            "time": time.detach().cpu(),
            "log_hz_t": log_hz_t.detach().cpu()
        }
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        preds = torch.cat([x["preds"] for x in outputs])
        events = torch.cat([x["events"] for x in outputs])
        time = torch.cat([x["time"] for x in outputs])
        log_hz_t = torch.cat([x["log_hz_t"] for x in outputs])

        metrics = self._calculate_balanced_metrics(preds, events, time, "test", log_hz_t, return_metrics=True)
        
        # Save to CSV (similar to your previous model)
        time_5y = torch.tensor(1825.0 / 2786.0).cpu()
        surv_probs = weibull.survival_function(preds, time_5y).squeeze().cpu().numpy()
        
        df = pd.DataFrame({
            'Actual_Time': time.cpu().numpy(),
            'Actual_Event': events.cpu().numpy(),
            'Surv_Prob_5y': surv_probs
        })
        df.to_csv(f"sybil_test_results.csv", index=False)
        
        self.test_step_outputs.clear()

    def _calculate_balanced_metrics(self, preds, events, time, prefix, log_hz_t, return_metrics=False):
        # Standard survival metrics using Weibull logic
        time_5y = torch.tensor(1825.0 / 2786.0).cpu()
        log_hz = weibull.log_hazard(preds, time)
        surv_prob_t = weibull.survival_function(preds, time_5y)
        
        # Binary classification proxy (Probability of surviving 5 years < 0.6 = High Risk)
        hard_preds = (surv_prob_t < 0.6).int()

        # Integrated Brier Score
        surv = weibull.survival_function(preds, time)
        ibs_val = BrierScore()(surv, events, time).integral()
        
        # AUC and C-Index
        auroc_val = Auc()(log_hz_t, events, time, new_time=time_5y)
        cindex_val = ConcordanceIndex()(log_hz, events, time)

        self.log_dict({
            f'{prefix}_auroc': auroc_val,
            f'{prefix}_cindex': cindex_val,
            f'{prefix}_ibs': ibs_val,
            f'{prefix}_f1': self.f1score(hard_preds, events)
        }, on_epoch=True)

        if return_metrics:
            return {'auroc': auroc_val, 'cindex': cindex_val, 'ibs': ibs_val}

    def configure_optimizers(self):
        if not self.freeze_encoder:
            param_groups = [
                {'params': self.survival_head.parameters(), 'lr': self.learning_rate},
                {'params': self.sybil.parameters(), 'lr': self.learning_rate / 10}
            ]
        else:
            param_groups = self.survival_head.parameters()

        return torch.optim.Adam(param_groups, lr=self.learning_rate, weight_decay=1e-5)
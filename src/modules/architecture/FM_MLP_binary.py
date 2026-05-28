import os
import torch 
import pytorch_lightning as L
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from lifelines.utils import concordance_index

# Consolidated TorchMetrics imports
from torchmetrics.classification import (
    BinaryAUROC, 
    BinaryF1Score, 
    BinaryAccuracy, 
    BinaryRecall, 
    BinaryPrecision
)

class encoder_decoder(L.LightningModule):
    def __init__(self, encoder, survival_head, learning_rate, pos_weight, config, freeze_encoder=True, fold_id='unknown'):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'survival_head'])
        
        self.survival_head = survival_head
        self.learning_rate = learning_rate
        self.config = config
        self.freeze_encoder = freeze_encoder
        self.fold_id = fold_id
        
        # 1. Extract the vision model from the dict cleanly
        full_model = encoder.get('vision')      
        if hasattr(full_model, 'model'): 
            self.vision_encoder = full_model.model.image_encoder
        elif hasattr(full_model, 'image_encoder'): 
            self.vision_encoder = full_model.image_encoder
        else:
            self.vision_encoder = full_model  
        
        # 2. Handle encoder freezing systematically
        self._set_encoder_freeze_state()

        # 3. Define Stable Loss Function
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # 4. Correctly Instantiate Metric Collections (attached to the module lifecycle)
        # Note: Using explicit Binary versions to match your 0.5 threshold workflow safely
        self.auroc_metric = BinaryAUROC()
        self.f1_metric = BinaryF1Score()
        self.acc_metric = BinaryAccuracy(threshold=0.5)
        self.recall_metric = BinaryRecall(threshold=0.5)
        self.precision_metric = BinaryPrecision(threshold=0.5)

        # 5. Modular storage lists for tracking raw outputs per epoch
        self._reset_epoch_containers('train')
        self._reset_epoch_containers('val')
        self._reset_epoch_containers('test')

    def _set_encoder_freeze_state(self):
        """Helper to handle freezing logic uniformly."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = not self.freeze_encoder
        if not self.freeze_encoder:
            self.vision_encoder.train()

    def _reset_epoch_containers(self, stage: str):
        """Resets the clean memory lists per lifecycle stage."""
        if stage == 'train':
            self.train_outputs = {'preds': [], 'events': [], 'times': []}
        elif stage == 'val':
            self.val_outputs = {'preds': [], 'events': [], 'times': []}
        elif stage == 'test':
            self.test_outputs = {'pids': [], 'preds': [], 'events': [], 'times': []}

    def forward(self, inputs):
        # Extract inputs safely
        x = inputs.get('image') if isinstance(inputs, dict) else inputs.get('pixel_values', None)
        if x is None:
            x = inputs  # Fallback if raw tensor passed
            
        outputs = self.vision_encoder(x)
        
        # Extract embedding vector cleanly
        if hasattr(outputs, "pooler_output"):
            img_emb = outputs.pooler_output
        else:
            img_emb = outputs[0][:, 0, :] if isinstance(outputs, (list, tuple)) else outputs

        img_emb = img_emb.view(img_emb.size(0), -1)
        return self.survival_head(img_emb)

    def training_step(self, batch, batch_idx):
        x, event, time = batch
        logits = self(x)
        loss = self.loss_fn(logits, event)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Safe collection tracking (detached to eliminate memory leaks)
        self.train_outputs['preds'].append(logits.detach().cpu())
        self.train_outputs['events'].append(event.detach().cpu())
        self.train_outputs['times'].append(time.detach().cpu())
        return loss

    def on_training_epoch_end(self):
        self._evaluate_and_log_stage('train')
        self._reset_epoch_containers('train')

    def validation_step(self, batch, batch_idx):
        x, event, time = batch
        logits = self(x)
        loss = self.loss_fn(logits, event)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_outputs['preds'].append(logits.detach().cpu())
        self.val_outputs['events'].append(event.detach().cpu())
        self.val_outputs['times'].append(time.detach().cpu())

    def on_validation_epoch_end(self):
        self._evaluate_and_log_stage('val')
        self._reset_epoch_containers('val')

    def test_step(self, batch, batch_idx):
        inputs, event, time = batch
        
        if isinstance(inputs, dict) and 'pid' in inputs:
            pids = inputs['pid']
        else:
            raise KeyError("Could not find 'pid' inside the inputs dictionary.")

        logits = self(inputs).squeeze()

        # Unify tracking formats to strings or tensors
        if isinstance(pids, torch.Tensor):
            self.test_outputs['pids'].extend(pids.detach().cpu().view(-1).tolist())
        elif isinstance(pids, (list, np.ndarray)):
            self.test_outputs['pids'].extend(list(pids))
        else:
            self.test_outputs['pids'].append(pids)

        self.test_outputs['preds'].append(logits.detach().cpu().view(-1))
        self.test_outputs['events'].append(event.detach().cpu().view(-1))
        self.test_outputs['times'].append(time.detach().cpu().view(-1))

    def on_test_epoch_end(self):
        if not self.test_outputs['preds']:
            print("❌ ERROR: No predictions were captured during test_step.")
            return

        # Unpack & Align arrays
        pids = np.array(self.test_outputs['pids']).ravel()
        logits = torch.cat(self.test_outputs['preds']).numpy().ravel()
        events = torch.cat(self.test_outputs['events']).bool().numpy().ravel()
        times = torch.cat(self.test_outputs['times']).numpy().ravel()

        print(f"\n📊 EXTRACTION SIZE CHECK:\n -> PIDs: {len(pids)} | Logits: {len(logits)} | Events: {len(events)} | Times: {len(times)}")

        # Handle size mismatches safely
        min_len = min(len(pids), len(logits), len(events), len(times))
        if min_len != max(len(pids), len(logits), len(events), len(times)):
            print(f"⚠️ LENGTH MISMATCH DETECTED! Truncating to {min_len}...")
            pids, logits, events, times = pids[:min_len], logits[:min_len], events[:min_len], times[:min_len]

        # Convert probabilities
        prob_event = 1.0 / (1.0 + np.exp(-logits))

        # Save to File System Safely
        base_dir = "/nas-ctm01/homes/fmferreira/AI4LUNGS/results/Imaging/binary"
        os.makedirs(base_dir, exist_ok=True) # Ensure path exists to avoid crash
        
        

        df_probs = pd.DataFrame({'pid': pids, 'imaging_prob': prob_event, 'true_label': events.astype(int)})
        df_probs.to_csv(f"{base_dir}/best_imaging_probabilities_fold_{self.fold_id}.csv", index=False)

        # Final evaluation run
        self._evaluate_and_log_stage('test', external_data=(torch.tensor(logits), torch.tensor(events), torch.tensor(times)))
        self._reset_epoch_containers('test')

    def _evaluate_and_log_stage(self, stage: str, external_data=None):
        """Unified method for calculating and logging metric values securely."""
        if external_data:
            logits, events, times = external_data
        else:
            container = getattr(self, f"{stage}_outputs")
            logits = torch.cat(container['preds'])
            events = torch.cat(container['events'])
            times = torch.cat(container['times'])

        events_bool = events.bool()
        probs = torch.sigmoid(logits)

        # Print debug balance metrics
        self.print_imbalance(probs, events_bool, stage_name=stage.upper())

        # Compute Survival Specific Concordance Index
        try:
            cindex_val = concordance_index(times.numpy(), logits.numpy(), events_bool.numpy())
        except Exception as e:
            cindex_val = 0.5
            print(f"Warning computing c-index for {stage}: {e}")

        # Direct Metric evaluations (passing tensors directly on identical device structures)
        metrics = {
            f'{stage}_auroc': self.auroc_metric(probs, events_bool.int()),
            f'{stage}_cindex': torch.tensor(cindex_val),
            f'{stage}_f1_score': self.f1_metric(probs, events_bool.int()),
            f'{stage}_balanced_acc': self.acc_metric(probs, events_bool.int()),
            f'{stage}_recall': self.recall_metric(probs, events_bool.int()),
            f'{stage}_precision': self.precision_metric(probs, events_bool.int())
        }

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def print_imbalance(self, probs: torch.Tensor, labels: torch.Tensor, stage_name=""):
        hard_preds = (probs > 0.5).int()
        num_pred_0 = (hard_preds == 0).sum().item()
        num_pred_1 = (hard_preds == 1).sum().item()
        num_true_0 = (labels == 0).sum().item()
        num_true_1 = (labels == 1).sum().item()

        print(f"\nStage {stage_name} | Pred: [0s: {num_pred_0}, 1s: {num_pred_1}] | True: [0s: {num_true_0}, 1s: {num_true_1}]")

    def configure_optimizers(self):
        if not self.freeze_encoder:
            # Separate backbones with fine-tuning Differential Learning Rates
            param_groups = [
                {'params': self.survival_head.parameters(), 'lr': self.learning_rate},
                {'params': self.vision_encoder.parameters(), 'lr': self.learning_rate / 10}
            ]
        else:
            # Gather parameters dynamically filtered by your initialization definitions
            param_groups = [{'params': filter(lambda p: p.requires_grad, self.parameters())}]

        return torch.optim.Adam(param_groups, lr=self.learning_rate, weight_decay=1e-5)
import torch 
from torchsurv.metrics.cindex import ConcordanceIndex
import torch
import pytorch_lightning as L
import numpy as np
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryStatScores, BinaryAccuracy, Accuracy
from torchsurv.loss import cox, weibull
from torchsurv.loss.cox import neg_partial_log_likelihood 
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.stats.kaplan_meier import KaplanMeierEstimator
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import pandas as pd

class encoder_decoder(L.LightningModule):
    def __init__(self, encoder, survival_head, learning_rate, pos_weight):
        super().__init__()
        self.survival_head = survival_head
        self._encoder_wrapper = encoder        
        self.vision_encoder = encoder.model.image_encoder
        self.learning_rate = learning_rate
        self.vision_encoder.eval() 
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
     # Metrics
        self.cindex_metric = ConcordanceIndex()
        self.auroc_metric = Auc() 
        self.f1score = BinaryF1Score()
        self.cindex_metric = ConcordanceIndex()


        self.stats_metric = BinaryStatScores(threshold=0.5, average='none')
        # Define the binary classification loss function
        # BCEWithLogitsLoss is numerically stable for logits (unbounded outputs)
        # self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)

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

    def encode_batch(self, base64_list):
        embeddings = []

        # MedImageInsight expects: encode(images=[base64_str, ...])
        out = self._encoder_wrapper.encode(images=base64_list)
        img_emb = out["image_embeddings"]  # numpy array or tensor

        # convert each embedding to tensor
        if isinstance(img_emb, np.ndarray):
            img_emb = torch.tensor(img_emb)
            
        img_emb = img_emb.to(self.device)
        return img_emb.float()

    def forward(self, x):
        embeddings = self.encode_batch(x)
        log_hz = self.survival_head(embeddings)
        return log_hz

        
    
    def training_step(self, batch, batch_idx):
        x, event, time = batch
        event = event.bool()
        event = event.squeeze() # Ensure 1D
        time = time.squeeze()          # Ensure 1D
        log_params = self(x).squeeze()
        if batch_idx == 0 and self.current_epoch == 0:
            print(f"\n[EPOCH {self.current_epoch}] First Training Batch Info:")
            print(f" - Image List Len: {len(x)}")
            print(f" - Event Tensor Shape: {event.shape}")
            print(f" - Time Tensor Shape: {time.shape}")
            print(f'event tensor:{event}')
            print(f'time tensor:{time}')

        print(log_params.max())
        print(log_params.min())

        loss = weibull.neg_log_likelihood(log_params, event, time)
        log_hz =weibull.log_hazard(log_params, time)
        new_time = torch.tensor(1825.0 / 2786.0).to(self.device)
        log_hz_t= weibull.log_hazard(log_params, new_time)        
        self.log("train_loss", loss)
        self.train_preds.append(log_params.detach().cpu())
        self.train_events.append(event.detach().cpu())
        self.train_time.append(time.detach().cpu())
        self.train_log_hz_t.append(log_hz_t.detach().cpu())

        # wandb.log({"train_loss": loss})
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, event, time = batch
        event = event.bool()
        x, event, time = batch
        event = event.squeeze() # Ensure 1D
        time = time.squeeze()          # Ensure 1D
        
        log_params = self(x).squeeze()
        print(log_params.max())
        print(log_params.min())
        loss = weibull.neg_log_likelihood(log_params, event, time, reduction='mean')
        log_hz =weibull.log_hazard(log_params, time)
        new_time = torch.tensor(1825.0 / 2786.0).to(self.device)
        log_hz_t= weibull.log_hazard(log_params,  new_time)
        self.log("val_loss", loss, prog_bar=True)
        self.val_preds.append(log_params.detach().cpu())
        self.val_events.append(event.detach().cpu())
        self.val_time.append(time.detach().cpu())
        self.val_log_hz_t.append(log_hz_t.detach().cpu())

    

    def print_inbalance(self, predicted_activated_labels, labels, stage_name=""):
        # Check how many predictions are 0 and 1
        num_pred_0 = (predicted_activated_labels == 0).sum().item()
        num_pred_1 = (predicted_activated_labels == 1).sum().item()

        # Check how many actual labels are 0 and 1
        num_true_0 = (labels == 0).sum().item()
        num_true_1 = (labels == 1).sum().item()

        print("\n" + "="*50)
        print(f"Stage: {stage_name}")

        print(f"Predicted class distribution: 0s = {num_pred_0}, 1s = {num_pred_1}")
        print(f"Actual label distribution:    0s = {num_true_0}, 1s = {num_true_1}")
        print(f"Actual Imbalance Ratio (0:1): {num_true_0 / (num_true_1 + 1e-8):.2f}:1")
        print("="*50)

        if num_pred_1 == 0 and num_pred_0 > 0:
            print("⚠️ Model is predicting only class 0 (majority class). It is ignoring the minority class!")
        elif num_pred_0 == 0 and num_pred_1 > 0:
            print("⚠️ Model is predicting only class 1 (minority class). It is ignoring the majority class!")
        else:
            print("✅ Model is predicting both classes.")

        return
    
    def _calculate_balanced_metrics(self, preds: torch.Tensor, events: torch.Tensor, time: torch.Tensor, prefix: str,  log_hz_t: torch.Tensor, return_metrics=False):
        # Calculate True Positives (TP), False Negatives (FN), etc.
        # 1. Move everything to CPU
        preds = preds.cpu()
        events = events.cpu()
        time = time.cpu()
        log_hz_t = log_hz_t.cpu()
        self.stats_metric = self.stats_metric.cpu()
        new_time = torch.tensor(1825.0 / 2786.0).cpu()
        surv_prob = weibull.survival_function(preds, new_time)
        hard_preds = (surv_prob < 0.65).int()
        events_bool = events.squeeze().bool()
        self.print_inbalance(hard_preds, events_bool, stage_name=prefix.upper())
        log_hz = weibull.log_hazard(preds, time) 
        eval_times = torch.linspace(time.min(), time.max(), steps=10).cpu().unsqueeze(0)    
        # Weibull survival function expects (params, time)
        # Shape of surv should be (num_samples, num_time_points)
        surv = weibull.survival_function(preds, time) 

        # 3. Use the BrierScore Class
        # Note: BrierScore() handles the IPCW (censoring weights) internally
        bs_metric = BrierScore()
        
        # surv: (N, 10), events_bool: (N,), time: (N,), eval_times: (10,)
        scores = bs_metric(surv, events_bool, time)
        
        # Calculate the Integrated Brier Score (IBS)
        ibs_val = bs_metric.integral()
        weibull_auc = Auc() # Torchsurv objects also need to be on same device
        auroc_val = weibull_auc(log_hz_t, events_bool, time, new_time=new_time)
        f1 = BinaryF1Score()
        f1_score = f1(hard_preds,events_bool)
        balanced_acc = Accuracy(task='multiclass', num_classes=2, average='macro')        
        acc = balanced_acc(hard_preds,events_bool)

        weibull_cindex = ConcordanceIndex()
        cindex= weibull_cindex(log_hz, events_bool, time)
        
        self.log_dict({
            f'{prefix}_auroc': auroc_val,
            f'{prefix}_cindex': cindex,
            f'{prefix}_ibs': ibs_val,
            f'{prefix}_f1_score': f1_score,
            f'{prefix}_balanced_acc': acc,
         }, on_step=False, on_epoch=True)
        if return_metrics: 
            return {
            'auroc': auroc_val.item(), 
            'cindex': cindex.item(),
            'ibs': ibs_val.item(),
            'f1_score': f1_score.item(),
            'balanced_acc': acc.item(), 
        }

    def plot_risk_distribution(self, preds, events, epoch):
        plt.figure(figsize=(10, 6))
        
        # Weibull: Higher log_alpha (params[:, 0]) usually means higher survival
        # We use -log_alpha as a proxy for "Risk"
        risk_scores = -preds[:, 0].numpy() 
        events = events.numpy()

        sns.kdeplot(risk_scores[events == 1], fill=True, label="Actual Events", color="red")
        sns.kdeplot(risk_scores[events == 0], fill=True, label="Censored", color="blue")
        
        plt.title(f"Epoch {epoch}: Risk Score Distribution")
        plt.xlabel("Predicted Risk Score (-log_alpha)")
        plt.ylabel("Density")
        plt.legend()
        
        # Log to WandB via the trainer's logger
        if self.logger:
            self.logger.experiment.log({"risk_distribution": wandb.Image(plt)})
        
        plt.close()


    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        events = torch.cat(self.val_events)
        time = torch.cat(self.val_time)
        log_hz_t = torch.cat(self.val_log_hz_t)
        self._calculate_balanced_metrics(preds, events, time, 'val', log_hz_t)
        if self.current_epoch % 5 == 0:
            self.plot_risk_distribution(preds, events, self.current_epoch)
        # Clear lists for the next epoch
        self.val_preds.clear()
        self.val_events.clear()
        self.val_time.clear()
        self.val_log_hz_t.clear()

    def test_step(self, batch, batch_idx):
        x, event, time = batch
        event=event.squeeze()
        time= time.squeeze()
        preds = self(x).squeeze()

        new_time = torch.tensor(1825.0 / 2786.0).to(self.device)

        log_hz_t =weibull.log_hazard(preds, new_time)
        log_hz =weibull.log_hazard(preds, time)

        # store for epoch_end
        self.test_preds.append(preds.detach().cpu())
        self.test_events.append(event.detach().cpu())
        self.test_time.append(time.detach().cpu())
        self.test_log_hz_t.append(log_hz_t.detach().cpu())

    
    def on_training_epoch_end(self):
        preds = torch.cat(self.train_preds)
        events = torch.cat(self.train_events)
        time = torch.cat(self.train_time)
        log_hz_t = torch.cat(self.train_log_hz_t)
        self._calculate_balanced_metrics(preds, events, time, 'train', log_hz_t)
        self.train_preds.clear()
        self.train_events.clear()
        self.train_time.clear()
        self.train_log_hz_t.clear()

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds)
        events = torch.cat(self.test_events)
        time = torch.cat(self.test_time)
        log_hz_t= torch.cat(self.test_log_hz_t)

        eval_time = torch.tensor([1825.0 / 2786.0]).to(preds.device)
        
        # 3. Calculate Survival Probability S(t | x)
        # Shape will be (num_samples, 1)
        with torch.no_grad():
            surv_probs = weibull.survival_function(preds, eval_time.unsqueeze(0))
        
        # 4. Prepare data for Excel
        # Convert to CPU/Numpy for Pandas compatibility
        preds_np = preds.cpu().numpy()
        events_np = events.cpu().numpy()
        time_np = time.cpu().numpy()
        surv_probs_np = surv_probs.squeeze().cpu().numpy()

        # 5. Create DataFrame
        df = pd.DataFrame({
            'Actual_Time': time_np,
            'Actual_Event': events_np,
            'Surv_Prob_5y': surv_probs_np
        })
        
        # 6. Save to Excel
        # We use a unique name to avoid overwriting files in Cross-Validation
        filename = f"test_results_fold_{getattr(self, 'fold_id', 'final')}.xlsx"
        df.to_excel(filename, index=False)
        print(f"Saved test predictions and probabilities to {filename}")        # Calculate metrics for the test set
        metrics=  self._calculate_balanced_metrics(preds, events, time, 'test', log_hz_t, return_metrics=True)
        self.test_auroc = metrics['auroc']
        self.test_cindex = metrics['cindex']
        self.test_ibs = metrics['ibs']
        self.test_f1_score = metrics['f1_score']
        self.test_balanced_acc = metrics['balanced_acc']
        self.plot_risk_distribution(preds, events, self.current_epoch)
        # Clear lists
        self.test_preds.clear()
        self.test_events.clear()
        self.test_log_hz_t.clear()

    def configure_optimizers(self):
        # Unfreeze the encoder parameters for fine-tuning
        for param in self.vision_encoder.parameters():
            param.requires_grad = True

        encoder_lr = self.learning_rate / 10.0 
        
        param_groups = [
            {'params': self.survival_head.parameters(), 'lr': self.learning_rate},
            {'params': self.vision_encoder.parameters(), 'lr': encoder_lr},
        ]
        
        optimizer = torch.optim.Adam(
            param_groups, 
            weight_decay=1e-5 
        )
        return optimizer
   

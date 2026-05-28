import torch 
from torchsurv.metrics.cindex import ConcordanceIndex
import torch
import pytorch_lightning as L
import numpy as np
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryStatScores, BinaryAccuracy, Accuracy, Recall, Precision
from torchsurv.loss import cox, weibull
from torchsurv.loss.cox import neg_partial_log_likelihood 
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.stats.kaplan_meier import KaplanMeierEstimator
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import pandas as pd
from lifelines.utils import concordance_index



class encoder_decoder(L.LightningModule):
    def __init__(self, encoder, survival_head, learning_rate, pos_weight, freeze_encoder=True):
        super().__init__()
        self.survival_head = survival_head
        self._encoder_wrapper = encoder        
        self.learning_rate = learning_rate
        
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
        self.freeze_encoder = freeze_encoder
        # Set the freeze state
        if self.freeze_encoder:
            for param in self._encoder_wrapper.parameters():
                param.requires_grad = False
        else:
            self._encoder_wrapper.train()
            for param in self._encoder_wrapper.parameters():
                param.requires_grad = True



    def encode_batch(self, pixel_values):
        img_emb = self._encoder_wrapper(pixel_values) 
        return img_emb

    def forward(self, x):
        
        # This creates the mathematical link for backpropagation
        outputs = self.encode_batch(x)
        logits = self.survival_head(outputs)
        return logits

        

    
    def training_step(self, batch, batch_idx):
    # x is a Tensor from your new _get_data return
        x, event, time = batch
        
        # Ensure survival labels are 1D and correct type
        event = event.squeeze().bool()
        time = time.squeeze().float()

        # The magic happens here: gradients flow through log_params back to vision_encoder
        log_params = self(x).squeeze() 

        # Calculate loss
        loss = weibull.neg_log_likelihood(log_params, event, time)
        
        # Log metrics
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        
        # Verify gradients are working (only once)
        if batch_idx == 0 and self.current_epoch == 0:
            for name, param in self._encoder_wrapper.named_parameters():
                if param.requires_grad:
                    print(f"🔥 Successfully training: {name}")
                    break # Just confirm one to be sure

        return loss

    # def _check_gradient_flow(self):
    #     """Helper to verify the 360M parameters are actually 'awake'"""
    #     for name, param in self.named_parameters():
    #         if "vision_encoder" in name and param.requires_grad:
    #             print(f"✅ Gradient Path Verified: {name} is set to learn.")
    #             return
    #     print("❌ WARNING: Vision Encoder is still frozen or disconnected!")
    
    def validation_step(self, batch, batch_idx):
        x, event, time = batch
        print(f'shape x{x.shape}')
        event = event.bool()
        x, event, time = batch
        event = event.squeeze() # Ensure 1D
        time = time.squeeze()          # Ensure 1D
        
        log_params = self(x).squeeze()
        print(f'shape x{log_params.shape}')

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
        # Ensure preds is [N, 2] for Weibull
        preds, events, time, log_hz_t= preds.cpu(), events.cpu(), time.cpu(), log_hz_t.cpu()

        print(f'shape PREDS: {preds.shape}')
        print(f'shape time: {time.shape}')
        # Now log_hz will have the same length as time (N)
        log_hz = weibull.log_hazard(preds, time)
        preds, events, time, log_hz_t= preds.cpu(), events.cpu(), time.cpu(), log_hz_t.cpu()
        self.stats_metric = self.stats_metric.cpu()
        time_5y = torch.tensor(1825.0 / 2786.0).cpu()
        print(f'shape log_hz: {log_hz.shape}')
        print(f'shape log_hz_t: {log_hz_t.shape}')

        surv_prob_t = weibull.survival_function(preds, time_5y)
        surv = weibull.survival_function(preds, time) 

        hard_preds = (surv_prob_t < 0.6).int()

        events = events.squeeze().bool()
        self.print_inbalance(hard_preds, events, stage_name=prefix.upper())
        
        # Calculate the Integrated Brier Score (IBS)
        bs_metric = BrierScore()
        scores = bs_metric(surv, events, time)
        ibs_val = bs_metric.integral()
        weibull_auc = Auc() # Torchsurv objects also need to be on same device
        auroc_val = weibull_auc(log_hz_t, events, time, new_time=time_5y)
        f1 = BinaryF1Score()
        f1_score = f1(hard_preds,events)
        
        balanced_acc = Accuracy(task='multiclass', num_classes=2, average='macro')        
        acc = balanced_acc(hard_preds,events)

        rec= Recall(task='multiclass', num_classes=2, average='macro')
        recall= rec(hard_preds,events)

        prec= Precision(task='multiclass', num_classes=2, average='macro')
        precision = prec(hard_preds,events)

        weibull_cindex = ConcordanceIndex()
        cindex= weibull_cindex(log_hz, events, time)
        # 1. Convert to numpy and immediately flatten to 1D
        log_hz= torch.diagonal(log_hz)
        # We detach() and cpu() first to ensure we are off the graph and on the right device
        time_np = time.detach().cpu().numpy().ravel()
        log_hz_np = log_hz_t.detach().cpu().numpy().ravel()
        events_np = events.detach().cpu().numpy().ravel()

        # 2. Check shapes for peace of mind (optional debugging)
        print(f"DEBUG SHAPES: time={time_np.shape}, log_hz={log_hz_np.shape}, events={events_np.shape}")

        # 3. Calculate lifelines c-index
        # REMEMBER: Use -log_hz_np because lifelines expects: higher score = longer life
        cindex_life = concordance_index(time_np, -log_hz_np, events_np)
        print(f'Cindex TorchSurv: {cindex}')
        print(f'Cindex Lifelines: {cindex_life}')

        self.log_dict({
            f'{prefix}_auroc': auroc_val,
            f'{prefix}_cindex': cindex,
            f'{prefix}_ibs': ibs_val,
            f'{prefix}_f1_score': f1_score,
            f'{prefix}_balanced_acc': acc,
            f'{prefix}_recall': recall,
            f'{prefix}_precision': precision,

         }, on_step=False, on_epoch=True)
        if return_metrics: 
            return {
            'auroc': auroc_val.item(), 
            'cindex': cindex.item(),
            'ibs': ibs_val.item(),
            'f1_score': f1_score.item(),
            'balanced_acc': acc.item(), 
            'recall': recall.item(),
            'precision': precision.item(),
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
        # Use dim=0 to stack [32, 2] + [32, 2] into [64, 2]
        preds = torch.cat(self.val_preds) 
        
        # Ensure time and events are flat 1D vectors
        events = torch.cat(self.val_events)
        time = torch.cat(self.val_time)
        
        # This must also be [N, 1] or [N] depending on your log_hazard call
        log_hz_t = torch.cat(self.val_log_hz_t)
        self._calculate_balanced_metrics(preds, events, time, 'val', log_hz_t)
        if self.current_epoch % 5 == 0:
            self.plot_risk_distribution(preds, events, self.current_epoch)
        
        eval_time = torch.tensor(1825.0 / 2786.0).cpu()

        with torch.no_grad():
            surv_probs = weibull.survival_function(preds, eval_time)
        
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
        filename = f"validation_results_fold_{getattr(self, 'fold_id', 'final')}.csv"
        df.to_csv(filename, index=False, mode= 'a')
        print(f"Saved test predictions and probabilities to {filename}")        # Calculate metrics for the test set
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

        eval_time = torch.tensor(1825.0 / 2786.0).cpu()

        
        # 3. Calculate Survival Probability S(t | x)
        # Shape will be (num_samples, 1)
        with torch.no_grad():
            surv_probs = weibull.survival_function(preds, time = eval_time)
        
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


    # def on_after_backward(self):
    # # Check the exact parameter you just found
    #     for name, param in self.named_parameters():
    #         if "vision_encoder.blocks.2.1" in name:
    #             if param.grad is not None:
    #                 grad_norm = param.grad.norm().item()
    #                 if grad_norm > 1e-9: # If it's not zero, it's learning!
    #                     print(f"SUCCESS: {name} is Updating! Grad Norm: {grad_norm:.6f}")
    #             else:
    #                 print(f"WARNING: {name} has NO gradient.")
    #             break

    def configure_optimizers(self):
            # 1. Collect only parameters that have requires_grad = True
            trainable_params = list(filter(lambda p: p.requires_grad, self.parameters()))
            
            if self.freeze_encoder== True:
                for param in self._encoder_wrapper.parameters():
                    param.requires_grad = False
            else:
                for param in self._encoder_wrapper.parameters():
                    param.requires_grad = True

            
            # 2. If we are unfreezing, we might want a lower LR for the backbone (Fine-tuning)
            if not self.freeze_encoder:
                # Separate the head and the encoder for different learning rates
                param_groups = [
                    {'params': self.survival_head.parameters(), 'lr': self.learning_rate},
                    {'params': self._encoder_wrapper.parameters(), 'lr': self.learning_rate / 10}
                ]
            else:
                # Only the head is training
                param_groups = list(filter(lambda p: p.requires_grad, self.parameters()))


            optimizer = torch.optim.Adam(
                param_groups, 
                lr=self.learning_rate,
                weight_decay=1e-5 
            )
            return optimizer

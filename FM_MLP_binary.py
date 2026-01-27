import torch 
import torch
import pytorch_lightning as L
import numpy as np
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryStatScores, BinaryAccuracy, Accuracy, Recall, Precision
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import pandas as pd
import lifelines
from lifelines.utils import concordance_index

class encoder_decoder(L.LightningModule):
    def __init__(self, encoder, survival_head, learning_rate, pos_weight, freeze_encoder=True):
        super().__init__()
        self.survival_head = survival_head
        self._encoder_wrapper = encoder        
        self.vision_encoder = encoder.model.image_encoder
        self.learning_rate = learning_rate
        
     # Metrics
        # self.cindex_metric = ConcordanceIndex()
        self.auroc_metric = BinaryAUROC() 
        self.f1score = BinaryF1Score()

        self.stats_metric = BinaryStatScores(threshold=0.5, average='none')
        # Define the binary classification loss function
        # BCEWithLogitsLoss is numerically stable for logits (unbounded outputs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)


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
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        else:
            self.vision_encoder.train()
            for param in self.vision_encoder.parameters():
                param.requires_grad = True



    # def encode_batch(self, base64_list):
    #     embeddings = []

    #     # MedImageInsight expects: encode(images=[base64_str, ...])
    #     out = self._encoder_wrapper.encode(images=base64_list)
    #     img_emb = out["image_embeddings"]  # numpy array or tensor

    #     # convert each embedding to tensor
    #     if isinstance(img_emb, np.ndarray):
    #         img_emb = torch.tensor(img_emb)
            
    #     img_emb = img_emb.to(self.device)
    #     return img_emb.float()
    def encode_batch(self, pixel_values):
        img_emb = self.vision_encoder(pixel_values) 
        return img_emb

    def forward(self, x):
        
        # This creates the mathematical link for backpropagation
        outputs = self.vision_encoder(x)
        
        # 2. Extract the embedding vector
        # ViT models return an object; we want the pooled output (e.g., shape [Batch, 1024])
        if hasattr(outputs, "pooler_output"):
            img_emb = outputs.pooler_output
        else:
            # Fallback for different model versions
            img_emb = outputs[0][:, 0, :] if isinstance(outputs, (list, tuple)) else outputs

        # 3. Pass to your 150k parameter survival head
        log_params = self.survival_head(img_emb)
        
        return log_params


    
    def training_step(self, batch, batch_idx):
    # x is a Tensor from your new _get_data return
        x, event,time = batch
        
        logits = self(x)
        loss = self.loss_fn(logits, event) # event needs to be (B, 1) for BCEWithLogitsLoss if logits is (B, 1)
        self.log("train_loss", loss)
        self.train_preds.append(logits.detach().cpu())
        self.train_events.append(event.detach().cpu())
        self.train_time.append(time.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        x, event,time = batch
        logits = self(x)
        loss = self.loss_fn(logits, event)
        self.log("val_loss", loss, prog_bar=True)
        self.val_preds.append(logits.detach().cpu())
        self.val_events.append(event.detach().cpu())
        self.val_time.append(time.detach().cpu())
       
    

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
    
    def _calculate_balanced_metrics(self, preds: torch.Tensor, events: torch.Tensor, time, prefix: str, return_metrics=False):
        # Calculate True Positives (TP), False Negatives (FN), etc.
        # 1. Move everything to CPU
        preds = preds.cpu()
        hard_preds = (preds > 0).int()
        events_bool=events.bool()
        self.print_inbalance(preds, events_bool, stage_name=prefix.upper())
        time= time.cpu()
        cindex=concordance_index(time, events_bool, preds)

        auroc = BinaryAUROC()
        auroc_val = auroc(preds,events_bool)
        f1 = BinaryF1Score()
        f1_score = f1(preds,events_bool)
        balanced_acc = Accuracy(task='multiclass', num_classes=2, average='macro')        
        acc = balanced_acc(hard_preds,events_bool)
        rec= Recall(task='multiclass', num_classes=2, average='macro')
        recall= rec(hard_preds,events_bool)
        prec= Precision(task='multiclass', num_classes=2, average='macro')
        precision = prec(hard_preds,events_bool)
        
        self.log_dict({
            f'{prefix}_auroc': auroc_val,
            f'{prefix}_cindex': cindex,
            # f'{prefix}_ibs': ibs_val,
            f'{prefix}_f1_score': f1_score,
            f'{prefix}_balanced_acc': acc,
            f'{prefix}_recall': recall,
            f'{prefix}_precision': precision,

         }, on_step=False, on_epoch=True)
        if return_metrics: 
            return {
            'auroc': auroc_val.item(), 
            'cindex': cindex.item(),
            # 'ibs': ibs_val.item(),
            'f1_score': f1_score.item(),
            'balanced_acc': acc.item(), 
            'recall': recall.item(),
            'precision': precision.item(),
        }

    # def plot_risk_distribution(self, preds, events, epoch):
    #     plt.figure(figsize=(10, 6))
        
    #     # Weibull: Higher log_alpha (params[:, 0]) usually means higher survival
    #     # We use -log_alpha as a proxy for "Risk"
    #     risk_scores = -preds[:, 0].numpy() 
    #     events = events.numpy()

    #     sns.kdeplot(risk_scores[events == 1], fill=True, label="Actual Events", color="red")
    #     sns.kdeplot(risk_scores[events == 0], fill=True, label="Censored", color="blue")
        
    #     plt.title(f"Epoch {epoch}: Risk Score Distribution")
    #     plt.xlabel("Predicted Risk Score (-log_alpha)")
    #     plt.ylabel("Density")
    #     plt.legend()
        
    #     # Log to WandB via the trainer's logger
    #     if self.logger:
    #         self.logger.experiment.log({"risk_distribution": wandb.Image(plt)})
        
    #     plt.close()


    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        events = torch.cat(self.val_events)
        time = torch.cat(self.val_time)
        self._calculate_balanced_metrics(preds, events, time, 'val')
        # if self.current_epoch % 5 == 0:
        #     self.plot_risk_distribution(preds, events, self.current_epoch)
      
        self.val_preds.clear()
        self.val_events.clear()
        self.val_time.clear()

    def test_step(self, batch, batch_idx):
        x, event,time = batch
        preds = self(x)
        # store for epoch_end
        self.test_preds.append(preds.detach().cpu())
        self.test_events.append(event.detach().cpu())
        self.test_time.append(time.detach().cpu())
    
    def on_training_epoch_end(self):
        preds = torch.cat(self.train_preds)
        events = torch.cat(self.train_events)
        time = torch.cat(self.train_time)
        self._calculate_balanced_metrics(preds, events, time, 'train')
        self.train_preds.clear()
        self.train_events.clear()
        self.train_time.clear()

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds)
        events = torch.cat(self.test_events)
        time = torch.cat(self.test_time)
        metrics=  self._calculate_balanced_metrics(preds, events, time, 'test', return_metrics=True)
        self.test_auroc = metrics['auroc']
        self.test_cindex = metrics['cindex']
        # self.test_ibs = metrics['ibs']
        self.test_f1_score = metrics['f1_score']
        self.test_balanced_acc = metrics['balanced_acc']
        # self.plot_risk_distribution(preds, events, self.current_epoch)
        # Clear lists
        self.test_preds.clear()
        self.test_events.clear()
        self.test_time.clear()


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
                for param in self.vision_encoder.parameters():
                    param.requires_grad = False
            else:
                for param in self.vision_encoder.parameters():
                    param.requires_grad = True

            
            # 2. If we are unfreezing, we might want a lower LR for the backbone (Fine-tuning)
            if not self.freeze_encoder:
                # Separate the head and the encoder for different learning rates
                param_groups = [
                    {'params': self.survival_head.parameters(), 'lr': self.learning_rate},
                    {'params': self.vision_encoder.parameters(), 'lr': self.learning_rate / 10}
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

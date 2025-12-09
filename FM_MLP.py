import torch 
from torchsurv.metrics.cindex import ConcordanceIndex
import torch
import pytorch_lightning as L
import numpy as np
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryStatScores


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
        self.auroc_metric = BinaryAUROC()
        self.f1score = BinaryF1Score()

        self.stats_metric = BinaryStatScores(threshold=0.5, average='none')
        # Define the binary classification loss function
        # BCEWithLogitsLoss is numerically stable for logits (unbounded outputs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)

        self.test_preds = []
        self.test_events = []
        self.val_preds = []
        self.val_events = []
        self.train_preds = [] 
        self.train_events = []
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
        logits = self.survival_head(embeddings)
        return logits

        
    
    def training_step(self, batch, batch_idx):
        x, event = batch
        logits = self(x)
        loss = self.loss_fn(logits, event.unsqueeze(1)) # event needs to be (B, 1) for BCEWithLogitsLoss if logits is (B, 1)
        self.log("train_loss", loss)
        self.train_preds.append(logits.detach().cpu())
        self.train_events.append(event.detach().cpu())

        # wandb.log({"train_loss": loss})
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, event = batch
        logits = self(x)
        loss = self.loss_fn(logits, event.unsqueeze(1))
        self.log("val_loss", loss, prog_bar=True)
        self.val_preds.append(logits.detach().cpu())
        self.val_events.append(event.detach().cpu())

    

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
        print(f"Actual label distribution:    0s = {num_true_0}, 1s = {num_true_1}")
        print(f"Actual Imbalance Ratio (0:1): {num_true_0 / (num_true_1 + 1e-8):.2f}:1")
        print("="*50)

        if num_pred_1 == 0 and num_pred_0 > 0:
            print("⚠️ Model is predicting only class 0 (majority class). It is ignoring the minority class!")
        elif num_pred_0 == 0 and num_pred_1 > 0:
            print("⚠️ Model is predicting only class 1 (minority class). It is ignoring the majority class!")
        else:
            print("✅ Model is predicting both classes.")

        return
    
    def _calculate_balanced_metrics(self, preds: torch.Tensor, events: torch.Tensor, prefix: str,  return_metrics=False):
        # Calculate True Positives (TP), False Negatives (FN), etc.
        # stats is a tensor of shape (5,) [TP, FP, TN, FN, SUPS]
        preds = preds.squeeze(-1)
        hard_preds = (preds > 0).int()
        events_int = events.int() # True labels must be int for comparisons
        self.print_inbalance(hard_preds, events_int, stage_name=prefix.upper())
        stats = self.stats_metric(preds, events)
        
        TP, FP, TN, FN, _ = stats.unbind() 
        
        # Calculate Sensitivity (Recall): TP / (TP + FN)
        sensitivity = TP / (TP + FN + 1e-8) 
        
        # Calculate Specificity: TN / (TN + FP)
        specificity = TN / (TN + FP + 1e-8)
        
        # Calculate Balanced Accuracy: (Sensitivity + Specificity) / 2
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Calculate F1 Score and AUROC (which you still want to keep)
        auroc_val = self.auroc_metric(preds, events)
        f1_val = self.f1score(preds, events)
        
        self.log_dict({
            f'{prefix}_auroc': auroc_val,
            f'{prefix}_f1_score': f1_val,
            f'{prefix}_balanced_accuracy': balanced_accuracy, # The desired metric
        }, on_step=False, on_epoch=True)
        if return_metrics: 
            return {
            'auroc': auroc_val.item(), 
            'f1': f1_val.item(), 
            'balanced_accuracy': balanced_accuracy.item()
        }

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        events = torch.cat(self.val_events)

        self._calculate_balanced_metrics(preds, events, 'val')

        # Clear lists for the next epoch
        self.val_preds.clear()
        self.val_events.clear()

    def test_step(self, batch, batch_idx):
        x, event = batch
        logits = self(x)

        # store for epoch_end
        self.test_preds.append(logits.detach().cpu())
        self.test_events.append(event.detach().cpu())
        
    def on_training_epoch_end(self):
        preds = torch.cat(self.train_preds)
        events = torch.cat(self.train_events)
        self._calculate_balanced_metrics(preds, events, 'train')
        self.train_preds.clear()
        self.train_events.clear()

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds)
        events = torch.cat(self.test_events)

        # Calculate metrics for the test set
        metrics=  self._calculate_balanced_metrics(preds, events, 'test', return_metrics=True)
        self.test_auroc = metrics['auroc']
        self.test_f1_score = metrics['f1']
        self.test_balanced_accuracy = metrics['balanced_accuracy']
        # Clear lists
        self.test_preds.clear()
        self.test_events.clear()

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
   

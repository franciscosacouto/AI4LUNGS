import os
import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from torchsurv.loss import weibull
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.auc import Auc

class EncoderDecoderSystem(L.LightningModule):
    def __init__(self, fusion_module, survival_head, learning_rate, tokenizer, config, fold_id):
        super().__init__()
        # 1. Store structural models and hyperparameters
        self.fusion_module = fusion_module
        self.survival_head = survival_head
        self.learning_rate = learning_rate
        self.tokenizer = tokenizer
        self.config = config
        self.fold_id = fold_id

        # 2. Allocate validation monitoring buckets
        self.val_preds = []
        self.val_events = []
        self.val_time = []
        self.val_log_hz_t = []
        self.test_preds = []
        self.test_events = []
        self.test_time = []
        self.test_log_hz_t = []
        self.test_pids = []
        self.test_pids = []
        self.torch_cindex = ConcordanceIndex()
        self.brier_score = BrierScore()
        self.auc = Auc()

        should_freeze = getattr(self.config, 'Freeze_weights', True)

        if should_freeze:
            print("🔒 [SYSTEM INIT] Freezing Encoders (requires_grad = False)")
            # Freeze the entire multi-modal fusion framework
            for param in self.fusion_module.parameters():
                param.requires_grad = False
            self.fusion_module.eval() # Puts vision/text norms & dropouts into static mode
        else:
            print("🔓 [SYSTEM INIT] Unfreezing Encoders for Active Fine-Tuning!")
            
            # Unfreeze everything inside the fusion layer first
            for param in self.fusion_module.parameters():
                param.requires_grad = True
                
            # Now safely apply modality specific rules based on what branches are actually enabled
            if not getattr(self.config, 'image', False):
                # If image is disabled, freeze the vision backbone parameters specifically
                for param in self.fusion_module.vision_net.parameters():
                    param.requires_grad = False
                    
            if not getattr(self.config, 'text', False) and self.fusion_module.text_net is not None:
                # If text is disabled, freeze the language transformer parameters specifically
                for param in self.fusion_module.text_net.parameters():
                    param.requires_grad = False
                    
            # Explicitly force the fusion module to remain in active train mode
            self.fusion_module.train()

        # Ensure your linear head layers are ALWAYS tracking gradients
        for param in self.survival_head.parameters():
            param.requires_grad = True

        self.grad_clip_val = getattr(config, 'grad_clip_val', 1.0)
            
    def forward(self, inputs):
        """Pass inputs through the modular pipeline."""
        combined = self.fusion_module(inputs, self.tokenizer)

        log_params = self.survival_head(combined)
        return torch.clamp(log_params, min=-10.0, max=10.0)

    def training_step(self, batch, _batch_idx):
        inputs, event, time = batch
        event, time = event.view(-1).bool(), time.view(-1).float()

        log_params = self(inputs)
        loss = weibull.neg_log_likelihood_weibull(log_params, event, time)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, event, time = batch
        event, time = event.view(-1).bool(), time.view(-1).float()
        
        # Forward pass through model
        log_params = self(inputs)
        
        # 1. Log loss immediately (Lightning handles the batch-averaging automatically)
        loss = weibull.neg_log_likelihood_weibull(log_params, event, time, reduction='mean')
        self.log("val_loss", loss, prog_bar=True)
        
        # 2. Collect raw tensors for the global epoch calculations
        self.val_preds.append(log_params.detach().cpu())
        self.val_events.append(event.detach().cpu())
        self.val_time.append(time.detach().cpu())
        return loss

    def test_step(self, batch, batch_idx):
        inputs, event, time = batch
        event, time = event.view(-1).bool(), time.view(-1).float()

        log_params = self(inputs)

        loss = weibull.neg_log_likelihood_weibull(log_params, event, time, reduction='mean')
        self.log("test_loss", loss, prog_bar=True)

        self.test_preds.append(log_params.detach().cpu())
        self.test_events.append(event.detach().cpu())
        self.test_time.append(time.detach().cpu())

        pids = inputs.get('pid', [])
        if isinstance(pids, torch.Tensor):
            self.test_pids.extend(pids.cpu().view(-1).tolist())
        else:
            self.test_pids.extend(list(pids))


    def on_validation_epoch_end(self):

        preds = torch.cat(self.val_preds, dim=0)
        events = torch.cat(self.val_events, dim=0).bool()
        time = torch.cat(self.val_time, dim=0)

        # scalar new_time → log_hazard returns (n,); 1-D vector → (n, n)
        eval_time_scalar = torch.tensor(1825.0 / 2786.0)

        # C-Index: (n, n) log-hazard matrix (all patients × all times)
        log_hz = weibull.log_hazard(preds, time)
        ts_cindex = self.torch_cindex(log_hz, events, time)

        # Incident AUC at 5y: scalar new_time → (n,) risk scores
        log_hz_t = weibull.log_hazard(preds, new_time=eval_time_scalar)
        final_auc = self.auc(log_hz_t, events, time, new_time=eval_time_scalar).item()

        # IBS: (n, n) survival matrix (all patients × all times)
        surv_all = weibull.survival_function_weibull(preds, time)
        self.brier_score(surv_all, events, time)
        final_brier = self.brier_score.integral().item()

        print(f"[VAL] C-Index: {ts_cindex.item():.4f} | AUC@5y: {final_auc:.4f} | IBS: {final_brier:.4f}")

        self.log_dict({
            'val_cindex': ts_cindex.item(),
            'val_auc': final_auc,
            'val_brier_score': final_brier
        }, prog_bar=True, on_epoch=True)

        # Reset cache buckets to free memory before next epoch begins
        self.val_preds.clear()
        self.val_events.clear()
        self.val_time.clear()
       
    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds, dim=0)
        events = torch.cat(self.test_events, dim=0).bool()
        time = torch.cat(self.test_time, dim=0)

        eval_time_scalar = torch.tensor(1825.0 / 2786.0)

        # C-Index: (n, n) log-hazard matrix
        log_hz = weibull.log_hazard(preds, time)
        ts_cindex = self.torch_cindex(log_hz, events, time)

        # Incident AUC at 5y: scalar new_time → (n,) risk scores
        log_hz_t = weibull.log_hazard(preds, new_time=eval_time_scalar)
        final_auc = self.auc(log_hz_t, events, time, new_time=eval_time_scalar).item()

        # IBS: (n, n) survival matrix
        surv_all = weibull.survival_function_weibull(preds, time)
        self.brier_score(surv_all, events, time)
        final_brier = self.brier_score.integral().item()

        print(f"[TEST] C-Index: {ts_cindex.item():.4f} | AUC@5y: {final_auc:.4f} | IBS: {final_brier:.4f}")

        self.log_dict({
            'test_cindex': ts_cindex.item(),
            'test_auc': final_auc,
            'test_brier_score': final_brier
        }, prog_bar=False, on_epoch=True)

        # Save predictions for late fusion
        # test_pids is already a flat list of individual values from test_step
        surv_5y = weibull.survival_function_weibull(preds, new_time=eval_time_scalar)  # (n,)
        prob_death_5y = (1.0 - surv_5y).numpy()

        df_out = pd.DataFrame({
            'pid':             np.array(self.test_pids).ravel(),
            'true_event':      events.numpy().astype(int),
            'fup_days':        (time * 2786.0).numpy(),   # de-normalize back to days
            'weibull_param_1': preds[:, 0].numpy(),
            'weibull_param_2': preds[:, 1].numpy(),
            'imaging_prob_5y': prob_death_5y,
        })

        save_dir = "/nas-ctm01/homes/fmferreira/AI4LUNGS/results/Imaging"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"test_weibull_parameters_fold_{self.fold_id}.csv")
        df_out.to_csv(save_path, index=False)
        print(f"[SAVED] {len(df_out)} predictions → {save_path}")

        self.test_pids.clear()
        self.test_preds.clear()
        self.test_events.clear()
        self.test_time.clear()

    def configure_optimizers(self):
        """Define your parameter tuning parameters right here."""
        param_groups = []
        
        # A. Always track and optimize the primary Survival Head weights
        for param in self.survival_head.parameters():
            param.requires_grad = True
        param_groups.append({'params': self.survival_head.parameters(), 'lr': self.learning_rate})

        # B. Grab parameters from your text net prompt learner inside the fusion layer
        if hasattr(self.fusion_module.text_net, 'prompt_learner') and self.fusion_module.text_net.prompt_learner is not None:
            param_groups.append({'params': self.fusion_module.text_net.prompt_learner.parameters(), 'lr': self.learning_rate})

        # C. Handle Fine-tuning or freezing logic for the backbones safely
        if getattr(self.config, 'image', False) and hasattr(self.fusion_module.vision_net, 'vision_encoder'):
            is_frozen = getattr(self.config, 'freeze_encoder', True)
            for param in self.fusion_module.vision_net.vision_encoder.parameters():
                param.requires_grad = not is_frozen
            
            if not is_frozen:
                param_groups.append({
                    'params': self.fusion_module.vision_net.vision_encoder.parameters(),
                    'lr': self.learning_rate / 10  # Fine-tune backbone slower
                })

        # Filter out parameters that don't require gradients (frozen weights)
        trainable_param_groups = []
        for group in param_groups:
            trainable_params = [p for p in group['params'] if p.requires_grad]
            if trainable_params:
                trainable_param_groups.append({'params': trainable_params, 'lr': group['lr']})

        optimizer = torch.optim.Adam(trainable_param_groups, lr=self.learning_rate, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_cindex",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        self.clip_gradients(optimizer, gradient_clip_val=self.grad_clip_val, gradient_clip_algorithm="norm")
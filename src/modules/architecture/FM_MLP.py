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
import torch.nn as nn
from transformers import BertTokenizer, AutoTokenizer
import  sys
import os
sys.path.insert(0, '/nas-ctm01/homes/fmferreira/CT-CLIP/CT_CLIP')

from ct_clip.tokenizer import SimpleTokenizer

# 2. Initialize it 
# It usually expects the 'data' folder to be present (which is in your image!)
class TextContextPrompter(nn.Module):
    def __init__(self, n_ctx=16, embedding_dim=512, tokenizer=None):
        super().__init__()
        # n_ctx is the number of learnable "tokens"
        self.n_ctx = n_ctx
        
        # Initialize with random normal or actual word embeddings
        ctx_vectors = torch.empty(n_ctx, embedding_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors) # These are the learnable parameters

    def forward(self, prefix_embed, suffix_embed, content_embed):
        # Concatenate: [SOT] + [Learnable Context] + [Content] + [EOT] + [PADs]
        # Shapes must match: (Batch, 1, Dim) + (Batch, n_ctx, Dim) + (Batch, Seq, Dim) ...
        return torch.cat([prefix_embed, self.ctx.unsqueeze(0).expand(prefix_embed.shape[0], -1, -1), content_embed, suffix_embed], dim=1)

class encoder_decoder(L.LightningModule):
    def __init__(self, encoder, survival_head, learning_rate, pos_weight, config, freeze_encoder=True):
        super().__init__()
        self.survival_head = survival_head
        self._encoder_wrapper = encoder        
        full_model = encoder.get('vision') # This is the MedImageInsight object
        
        if hasattr(full_model, 'model'): 
            self.vision_encoder = full_model.model.image_encoder
        elif hasattr(full_model, 'image_encoder'): # CT-CLIP structure
            self.vision_encoder = full_model.image_encoder
        else:
            # Fallback for RadioDino/Timm models
            self.vision_encoder = full_model
        provided_lang_encoder = encoder.get('language')

        if provided_lang_encoder is not None:
            self.text_encoder = provided_lang_encoder
        elif hasattr(full_model, 'model'):
            self.text_encoder = full_model.model.lang_encoder
        elif hasattr(full_model, 'lang_encoder'):
            self.text_encoder = full_model.lang_encoder
        else:
            self.text_encoder = None
        self.learning_rate = learning_rate
        self.config = config 
        self.n_ctx = 16
        self.prompt_learner = None
        if getattr(self.config, 'text', False) and self.text_encoder is not None:
            # Check for width attribute (common in CLIP-style models) 
            # or fallback to a standard dimension if 'width' isn't the attribute name
            emb_dim = getattr(self.text_encoder, 'width', 512) 
            
            self.prompt_learner = TextContextPrompter(
                n_ctx=self.n_ctx, 
                embedding_dim=emb_dim
            )
            for param in self.prompt_learner.parameters():
                param.requires_grad = True
     # Metrics
        self.cindex_metric = ConcordanceIndex()
        self.auroc_metric = Auc() 
        self.f1score = BinaryF1Score()


        # self.stats_metric = BinaryStatScores(threshold=0.5, average='none')
        self.test_preds = []
        self.test_events = []
        self.test_time= []
        self.test_pids = []
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
        if self.freeze_encoder and  getattr(self.config, 'image', False):
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        elif  getattr(self.config, 'image', False):
            self.vision_encoder.train()
            for param in self.vision_encoder.parameters():
                param.requires_grad = True

        if self.freeze_encoder and getattr(self.config, 'text', False):

            for param in self.text_encoder.parameters():
                param.requires_grad = False
        elif getattr(self.config, 'text', False):
            for param in self.text_encoder.parameters():
                param.requires_grad = True

    def encode_batch(self, inputs):
            features = []
            
            # 1. Image Branch (MedImageInsight)
            if 'image' in inputs and getattr(self.config, 'image', False):
                v_out = self.vision_encoder(inputs['image'])
                
                # Use the pooled output
                img_emb = v_out.pooler_output if hasattr(v_out, "pooler_output") else v_out
                
                # Handle list/tuple outputs from standard ViTs
                if isinstance(img_emb, (list, tuple)): 
                    img_emb = img_emb[0][:, 0, :]
                                    
                # CRITICAL: Flatten to [Batch, Dim]
                # This ensures 1024 or 2048 is a clean vector for the MLP
                img_emb = img_emb.view(img_emb.size(0), -1)
                features.append(img_emb)

            # 2. Text Branch (CT-CLIP)
            if 'text' in inputs and getattr(self.config, 'text', False):
                if getattr(self.config, 'text_encoder', 'MedImageInsights') == 'MedImageInsights':

                    sentences = inputs['text']
                    max_len = 77
                    # Room for SOT (1) + EOT (1) + Learnable Prompts (n_ctx)
                    content_per_chunk = max_len - 2 - self.prompt_learner.n_ctx 
                    
                    # 1. Get Tokenizer and Embeddings from your Transformer class
                    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True, trust_remote_code=True)

                    # tokenizer = getattr(self._encoder_wrapper, 'tokenizer', None) or getattr(self._encoder_wrapper.model, 'tokenizer', None)
                    token_embedding_layer = self.text_encoder.token_embedding
                    pos_embedding = self.text_encoder.positional_embedding # [77, width]

                    tokenized_output = tokenizer(sentences, truncation=False, add_special_tokens=False)
                    full_encodings = tokenized_output['input_ids']
                    
                    sot = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 49406
                    eot = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 49407
                    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

                    batch_embeddings = []

                    for ids in full_encodings:
                        # Split into chunks
                        chunks = [ids[i : i + content_per_chunk] for i in range(0, len(ids), content_per_chunk)]
                        if not chunks or len(chunks[0]) == 0: 
                            chunks = [[]] # Handle empty strings
                        
                        chunk_outputs = []
                        for c in chunks:
                            # A. Convert IDs to Embeddings
                            sot_idx = torch.tensor([sot], device=self.device)
                            eot_idx = torch.tensor([eot], device=self.device)
                            content_idx = torch.tensor([c], device=self.device) if len(c) > 0 else torch.tensor([[]], device=self.device, dtype=torch.long)
                            
                            sot_emb = token_embedding_layer(sot_idx).unsqueeze(0)    # [1, 1, width]
                            eot_emb = token_embedding_layer(eot_idx).unsqueeze(0)    # [1, 1, width]
                            
                            # Handle empty content embedding if text is empty
                            if len(c) > 0:
                                content_emb = token_embedding_layer(content_idx)    # [1, len, width]
                            else:
                                content_emb = torch.empty((1, 0, self.text_encoder.width), device=self.device)

                            # B. CoOp Injection: [SOT] + [Prompt] + [Content] + [EOT]
                            # self.prompt_learner returns (1, 1 + n_ctx + len(c) + 1, width)
                            full_embeddings = self.prompt_learner(sot_emb, eot_emb, content_emb)

                            # C. Manual Padding and Masking
                            curr_len = full_embeddings.shape[1]
                            pad_amt = max_len - curr_len
                            
                            if pad_amt > 0:
                                pad_idx = torch.tensor([pad] * pad_amt, device=self.device)
                                pad_emb = token_embedding_layer(pad_idx).unsqueeze(0)
                                full_embeddings = torch.cat([full_embeddings, pad_emb], dim=1)
                            
                            # Mask: 1 for tokens/prompts, 0 for pads (used for MultiHeadAttention)
                            # key_padding_mask in your Transformer is (attention_mask == 0)
                            # So we create an attention_mask where 1 is real and 0 is pad
                            attn_mask = torch.zeros(max_len, device=self.device)
                            attn_mask[:curr_len] = 1
                            key_padding_mask = (attn_mask == 0).unsqueeze(0) # [1, 77]

                            # D. Add Positional Encodings
                            full_embeddings = full_embeddings + pos_embedding.unsqueeze(0)

                            # E. Manual Transformer Forward (Since we have embeds, not IDs)
                            x = full_embeddings.permute(1, 0, 2)  # [L, N, D]
                            for block in self.text_encoder.resblocks:
                                x = block(x, key_padding_mask=key_padding_mask)
                            
                            x = x.permute(1, 0, 2)  # [N, L, D]
                            x = self.text_encoder.ln_final(x)
                            chunk_outputs.append(x)
                        
                        # Mean pool across all chunks for this specific sentence
                        combined_sentence = torch.cat(chunk_outputs, dim=1)
                        batch_embeddings.append(combined_sentence.mean(dim=1))

                    # 2. Safety check before cat
                    if len(batch_embeddings) == 0:
                        # Fallback to zero tensor if batch is somehow empty
                        print("just zeros")
                        text_emb = torch.zeros((len(sentences), self.text_encoder.width), device=self.device)
                    else:
                        text_emb = torch.cat(batch_embeddings, dim=0)
                    
                    features.append(text_emb)
                else:

                    
                    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
                    # Tokenize and ensure tensors are on CPU
                    tokens = tokenizer(
                    inputs['text'], 
                    padding="max_length", 
                    truncation=True, 
                    max_length=256, 
                    return_tensors="pt"
                    )
                    print("After diagnosis tokens:")
                    print(tokens['input_ids'].size(1))             
                    actual_count = tokens['attention_mask'][0].sum().item()
                    print(f"Actual tokens used: {actual_count}")

                    # print(f"Text Encoder Class: {self.text_encoder.__class__.__name__}")
                    # print(f"DEBUG: All attributes of text_encoder: {dir(self.text_encoder)}")
                    self.text_encoder.text_transformer.resize_token_embeddings(len(tokenizer))

                    input_ids = tokens['input_ids'].to(self.device)

                    # 2. Now you can slice it because it is a Tensor
                    # input_ids = input_ids[:, :254]
                    # Fix the vocab mismatch without resizing
                    # vocab_limit = self.text_encoder.text_transformer.token_emb.num_embeddings
                    # input_ids[input_ids >= vocab_limit] = tokenizer.unk_token_id

                    # Pass to transformer (Correct keyword)
                    model_output = self.text_encoder.text_transformer(input_ids=input_ids)

                    # Extract Hidden State then CLS
                    enc_text = model_output.last_hidden_state
                    text_embeds = enc_text[:, 0, :]

                    # Project
                    text_latents = self.text_encoder.to_text_latent(text_embeds)
                    features.append(text_latents)
                    

            # 3. Tabular Branch
            if 'tabular' in inputs and getattr(self.config, 'tabular', False):
                features.append(inputs['tabular'])

            # 4. Final Fusion
            if len(features) > 1:
                return torch.cat(features, dim=1)
            elif len(features) == 1:
                return features[0]
            else:
                raise ValueError("No modalities enabled in config!")

    def forward(self, inputs):
        combined = self.encode_batch(inputs)
        log_params = self.survival_head(combined)
        log_params = torch.clamp(log_params, min=-10.0, max = 10.0)
        return log_params

    
    def training_step(self, batch, batch_idx):
    # x is a Tensor from your new _get_data return
        inputs, event, time = batch
        
        # Ensure survival labels are 1D and correct type
        event = event.squeeze().bool()
        time = time.squeeze().float()

        # The magic happens here: gradients flow through log_params back to vision_encoder
        log_params = self(inputs).squeeze() 
        print("Printing log parameters shape:")
        print(log_params.shape)

        # Calculate loss
        loss = weibull.neg_log_likelihood_weibull(log_params, event, time)
        
        # Log metrics
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        
   

        return loss

    
    

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
    
    def _calculate_balanced_metrics(self, preds: torch.Tensor, events: torch.Tensor, time: torch.Tensor, prefix: str, log_hz_t: torch.Tensor, return_metrics=False):
        # 1. Relocate to CPU and enforce strict shapes
        preds, events, time, log_hz_t = preds.cpu(), events.cpu(), time.cpu(), log_hz_t.cpu()
        events_bool = events.squeeze().bool()

        # 2. Extract continuous hazard risks at each patient's actual event time
        log_hz_matrix = weibull.log_hazard(preds, time)
        log_hz = torch.diagonal(log_hz_matrix) # Grabs the 1D diagonal elements

        # 3. LIFELINES C-INDEX IMPLEMENTATION
        time_np = time.numpy().ravel()
        events_np = events_bool.numpy().ravel()
        log_hz_np = log_hz.numpy().ravel()

        # CRITICAL: Invert hazard with a negative sign. 
        # Lifelines expects: higher value = longer survival. 
        # Our model evaluates: higher log_hz = higher risk/earlier event.
        cindex_life = concordance_index(time_np, -log_hz_np, events_np)
        print(f"📊 [{prefix.upper()}] Lifelines Concordance Index: {cindex_life:.4f}")

        # 4. Continuous Time-Dependent AUC (td_auc) via torchsurv at 5-Year mark
        time_5y = torch.tensor(1825.0 / 2786.0).cpu()
        weibull_auc = Auc() 
        td_auc_val = weibull_auc(log_hz, events_bool, time)
        td_auc_val = torch.mean(td_auc_val).item()
        print(f"📈 [{prefix.upper()}] Time-Dependent AUC (5-Year): {td_auc_val:.4f}")

        # 5. Integrated Brier Score (IBS) for continuous curve calibration
        bs_metric = BrierScore()
        surv_curves = weibull.survival_function_weibull(preds, time)
        # Handle cases where batch sizes are small or boundary edges match
        try:
            bs_metric(surv_curves, events_bool, time)
            ibs_val = bs_metric.integral()
        except Exception:
            ibs_val = torch.tensor(0.0) # Fallback for edge runtime anomalies

        # 6. Optional: Diagnostic printing for patient risk visualization
        # We define an event split threshold exclusively for distribution checking logs
        surv_prob_t = weibull.survival_function_weibull(preds, time_5y)
        hard_predictions_check = (surv_prob_t < 0.5).int() 
        self.print_inbalance(hard_predictions_check, events_bool, stage_name=prefix.upper())

        # Log pure continuous evaluation metrics out to Lightning Loggers/WandB
        self.log_dict({
            f'{prefix}_cindex': cindex_life,
            f'{prefix}_td_auc': td_auc_val,
            f'{prefix}_ibs': ibs_val,
        }, on_step=False, on_epoch=True)

        if return_metrics: 
            return {
                'cindex': cindex_life,
                'td_auc': td_auc_val.item(), 
                'ibs': ibs_val.item()
            }



    def on_validation_epoch_end(self):
        # Use dim=0 to stack [32, 2] + [32, 2] into [64, 2]
        preds = torch.cat(self.val_preds) 
        
        # Ensure time and events are flat 1D vectors
        events = torch.cat(self.val_events)
        time = torch.cat(self.val_time)
        
        # This must also be [N, 1] or [N] depending on your log_hazard call
        log_hz_t = torch.cat(self.val_log_hz_t)
        self._calculate_balanced_metrics(preds, events, time, 'val', log_hz_t)

        
        eval_time = torch.tensor(1825.0 / 2786.0).cpu()

        with torch.no_grad():
            surv_probs = weibull.survival_function_weibull(preds, eval_time)
        
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


    def validation_step(self, batch, batch_idx):
        inputs, event, time = batch
        event = event.squeeze() # Ensure 1D
        time = time.squeeze()          # Ensure 1D
        event= event.bool()
        log_params = self(inputs).squeeze()
       
        loss = weibull.neg_log_likelihood_weibull(log_params, event, time, reduction='mean')
        log_hz =weibull.log_hazard(log_params, time)
        new_time = torch.tensor(1825.0 / 2786.0).to(self.device)
        log_hz_t= weibull.log_hazard(log_params,  new_time)
        self.log("val_loss", loss, prog_bar=True)
        self.val_preds.append(log_params.detach().cpu())
        self.val_events.append(event.detach().cpu())
        self.val_time.append(time.detach().cpu())
        self.val_log_hz_t.append(log_hz_t.detach().cpu())

    def test_step(self, batch, batch_idx):
        inputs, event, time = batch
        
        pids = inputs['pid']
            
        # 2. Forward pass returns the 2 Weibull parameters. 
        # DO NOT use global .squeeze() here so it remains [Batch, 2]
        log_params = self(inputs) 
        if log_params.ndim == 1:
            log_params = log_params.unsqueeze(0) # Safeguard for batch size = 1

        # 3. FORCE time to be a flat 1D vector to satisfy torchsurv expectations
        time_1d = time.flatten().to(self.device)
        event_1d = event.flatten().to(self.device)

        # 4. Compute continuous risk metrics using the safely flattened timelines
        new_time_scalar = torch.tensor(1825.0 / 2786.0).to(self.device)
        
        # Pass a 1D vector for the specific validation time evaluation
        # We broadcast the scalar to a 1D vector matching your current batch size
        # new_time_vector = new_time_scalar.expand(preds.size(0))

        log_hz = weibull.log_hazard(log_params, time_1d)
        log_hz_t = weibull.log_hazard(log_params, new_time_scalar)

        # 5. Store items safely into individual containers
        if isinstance(pids, torch.Tensor):
            self.test_pids.append(pids.detach().cpu().view(-1))
        elif isinstance(pids, (list, np.ndarray)):
            self.test_pids.append(list(pids))
        else:
            self.test_pids.append([pids])

        # Save detached variables explicitly 
        self.test_preds.append(log_params.detach().cpu())
        self.test_events.append(event_1d.detach().cpu())
        self.test_time.append(time_1d.detach().cpu())
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
        
        unpacked_pids = []
        for item in self.test_pids:
            if isinstance(item, torch.Tensor):
                unpacked_pids.extend(item.cpu().view(-1).tolist())
            elif isinstance(item, (list, np.ndarray)):
                unpacked_pids.extend(list(item))
            else:
                unpacked_pids.append(item)
        pids = np.array(unpacked_pids).ravel()

        # 2. Standardize multi-dimensional regression outputs and survival labels
        preds = torch.cat(self.test_preds, dim=0).cpu() # Maintained as [N, 2] matrix
        events = torch.cat([t.view(-1) for t in self.test_events]).bool().cpu()
        times = torch.cat([t.view(-1) for t in self.test_time]).cpu()
        log_hz_t = torch.cat([t.view(-1) for t in self.test_log_hz_t]).cpu()

        # Extract continuous parametric outputs: Column 0 = log_scale, Column 1 = log_shape
        weibull_param_1 = preds[:, 0].numpy().ravel()
        weibull_param_2 = preds[:, 1].numpy().ravel()

        # 3. Compute continuous 5-Year Probability of Death via the cumulative density function
        # 1825 days / 2786 normalization factor (if applicable to your time-scale)
        time_5y = torch.tensor(1825.0 / 2786.0) 
        surv_prob_5y = weibull.survival_function_weibull(preds, time_5y).numpy().ravel()
        prob_death_5y = 1.0 - surv_prob_5y  # F(t) = 1.0 - S(t)

        print(f"\n📊 [REGRESSION] EXTRACTION SIZE CHECK:\n -> PIDs: {len(pids)} | Params: {len(weibull_param_1)} | Events: {len(events)} | Times: {len(times)}")

        

        # 5. Build and save the exact target survival parameter file
        df_weibull = pd.DataFrame({
            'pid': pids,
            'true_event': events.numpy().astype(int),
            'fup_days': times.numpy(),
            'weibull_param_1': weibull_param_1,
            'weibull_param_2': weibull_param_2,
            'imaging_prob_5y': prob_death_5y  # True Probability of Death/Event
        })

        base_dir = "/nas-ctm01/homes/fmferreira/AI4LUNGS/results/Imaging"
        os.makedirs(base_dir, exist_ok=True) 
        df_weibull.to_csv(f"{base_dir}/test_weibull_parameters_fold_{self.fold_id}.csv", index=False)
        print(f"✅ Successfully saved regression parameters to {base_dir}")

        # 6. Execute metrics loop using the lifelines C-index logic
        self._calculate_balanced_metrics(preds, events, times, 'test', log_hz_t)

        # 7. Reset containers for subsequent cross-validation folds
        self.test_pids.clear()
        self.test_preds.clear()
        self.test_events.clear()
        self.test_time.clear()
        self.test_log_hz_t.clear()

    def on_after_backward(self):
    # Check the exact parameter you just found
        for name, param in self.named_parameters():
            if "vision_encoder.blocks.2.1" in name:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1e-9: # If it's not zero, it's learning!
                        print(f"SUCCESS: {name} is Updating! Grad Norm: {grad_norm:.6f}")
                else:
                    print(f"WARNING: {name} has NO gradient.")
                break

    def configure_optimizers(self):
        # Determine modality from config
        param_groups = []
        use_image = getattr(self.config, 'image', False)
        use_text = getattr(self.config, 'text', False)
        # 1. ALWAYS include the Survival Head (Primary LR)
        for param in self.survival_head.parameters():
            param.requires_grad = True
        param_groups.append({'params': self.survival_head.parameters(), 'lr': self.learning_rate})

        if self.prompt_learner is not None:
            param_groups.append({'params': self.prompt_learner.parameters(), 'lr': self.learning_rate})

        # 2. Vision Encoder Logic
        if self.vision_encoder is not None and use_image is not None:
            if self.freeze_encoder:
                for param in self.vision_encoder.parameters():
                    param.requires_grad = False
            else:
                for param in self.vision_encoder.parameters():
                    param.requires_grad = True
                # Use a lower LR for the backbone (Fine-tuning)
                param_groups.append({
                    'params': self.vision_encoder.parameters(), 
                    'lr': self.learning_rate / 10
                })

        # 3. Text Encoder Logic
        if self.text_encoder is not None and use_text is not None:
            # Usually, text encoders are kept frozen unless you have a lot of data
            # We'll follow the same logic as the vision encoder
            if self.freeze_encoder:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
            else:
                for param in self.text_encoder.parameters():
                    param.requires_grad = True
                param_groups.append({
                    'params': self.text_encoder.parameters(), 
                    'lr': self.learning_rate / 10
                })

        # 4. Final Safety Filter
        # In case freeze_encoder was True, we filter out all frozen params 
        # to ensure the optimizer doesn't try to track them.
        trainable_param_groups = []
        for group in param_groups:
            trainable_params = [p for p in group['params'] if p.requires_grad]
            if trainable_params:
                trainable_param_groups.append({
                    'params': trainable_params, 
                    'lr': group['lr']
                })

        optimizer = torch.optim.AdamW(
            trainable_param_groups, 
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        return optimizer
# =================================================================
# 1. IMPORTS
# =================================================================
import torch 
import sys
import pandas as pd
import os
import random
import numpy as np
import hydra
import wandb
import pytorch_lightning as L
import matplotlib.pyplot as plt
import seaborn as sns # <-- NEW IMPORT for KDE plotting
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryStatScores

# =================================================================
# 2. DATASET CLASS (SurvivalDataset)
# =================================================================
# (SurvivalDataset class remains the same)
class SurvivalDataset(Dataset):
    def __init__(self, df):
        self.x = torch.tensor(df.drop(columns=['5y']).values, dtype=torch.float32) 
        self.event = torch.tensor(df['5y'].values, dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.event[idx]
    
pos_weight=torch.tensor([4.0]) 

# =================================================================
# 3. MODEL CLASS (MLP_decoder)
# =================================================================
class MLP_decoder(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.auroc_metric = BinaryAUROC()
        self.f1score = BinaryF1Score()
        self.stats_metric = BinaryStatScores(threshold=0.5, average='none')
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight= pos_weight)
        
        # Lists to store ALL predictions (logits) and true events
        self.test_preds = []
        self.test_events = []
        self.val_preds = []
        self.val_events = []
        
        # Store final test results for plotting/saving
        self.final_test_logits = None # <-- NEW ATTRIBUTE
        self.final_test_events = None # <-- NEW ATTRIBUTE

    def forward(self, x):
        x = self.model(x).squeeze(-1)
        return x
    
    # (training_step, validation_step, print_inbalance, 
    #  _calculate_balanced_metrics, on_validation_epoch_end, test_step remain the same) 
    
    def training_step(self, batch, batch_idx):
        x, event = batch
        logits = self(x)
        loss = self.loss_fn(logits, event) 
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, event = batch
        logits = self(x)
        loss = self.loss_fn(logits, event)
        self.log("val_loss", loss, prog_bar=True)
        self.val_preds.append(logits.detach().cpu())
        self.val_events.append(event.detach().cpu())

    def print_inbalance(self, predicted_activated_labels, labels, stage_name=""):
        num_pred_0 = (predicted_activated_labels == 0).sum().item()
        num_pred_1 = (predicted_activated_labels == 1).sum().item()
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
        
    def _calculate_balanced_metrics(self, preds: torch.Tensor, events: torch.Tensor, prefix: str):
        probs = torch.sigmoid(preds) 
        hard_preds = (probs > 0.5).int()
        events_int = events.int()
        self.print_inbalance(hard_preds, events_int, stage_name=prefix.upper())
        stats = self.stats_metric(probs, events) 
        
        TP, FP, TN, FN, _ = stats.unbind() 
        
        sensitivity = TP / (TP + FN + 1e-8) 
        specificity = TN / (TN + FP + 1e-8)
        balanced_accuracy = (sensitivity + specificity) / 2
        
        auroc_val = self.auroc_metric(probs, events)
        f1_val = self.f1score(probs, events)
        
        self.log_dict({
            f'{prefix}_auroc': auroc_val,
            f'{prefix}_f1_score': f1_val,
            f'{prefix}_balanced_accuracy': balanced_accuracy,
        }, on_step=False, on_epoch=True)
        

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        events = torch.cat(self.val_events)
        self._calculate_balanced_metrics(preds, events, 'val')
        self.val_preds.clear()
        self.val_events.clear()

    def test_step(self, batch, batch_idx):
        x, event = batch
        logits = self(x)
        self.test_preds.append(logits.detach().cpu())
        self.test_events.append(event.detach().cpu())
        
    def on_test_epoch_end(self, outs=None): # outs=None is for PL compatibility
        preds = torch.cat(self.test_preds)
        events = torch.cat(self.test_events)

        # Calculate metrics (used for logging and potentially saving to Excel)
        self._calculate_balanced_metrics(preds, events, 'test')
        
        # --- 🔑 CRITICAL: Store final test logits and events ---
        self.final_test_logits = preds.numpy()
        self.final_test_events = events.numpy()
        
        # Clear lists
        self.test_preds.clear()
        self.test_events.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-5 
        )
        return optimizer

# =================================================================
# 4. RESULTS UTILITIES (New Plotting Function)
# =================================================================

def plot_logit_distribution(logits, events, stage_name):
    """
    Generates a KDE plot showing the distribution of the final logit scores
    for the two classes (Survived vs. Event).
    """
    if logits is None or events is None:
        print("⚠️ Cannot plot logit distribution: Test logits or events not available.")
        return

    print(f"\n--- Generating Logit Distribution Plot for {stage_name} ---")

    # 1. Create a DataFrame for Seaborn
    plot_df = pd.DataFrame({
        'Logit_Score': logits.flatten(),
        'Event': events.flatten()
    })
    # Create descriptive labels for the legend
    plot_df['Event_Label'] = plot_df['Event'].map({0: '5Y Survived (0)', 1: '5Y Event (1)'})

    # 2. Generate the KDE Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=plot_df, 
        x='Logit_Score', 
        hue='Event_Label', 
        fill=True, 
        common_norm=False, # Plot independent distributions
        palette={'5Y Survived (0)': 'blue', '5Y Event (1)': 'red'}
    )
    # Add a vertical line at the decision boundary (Logit=0, which equals Prob=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1.5, label='Decision Boundary (Logit=0)')
    
    plt.title(f'Logit Score Distribution by Survival Outcome')
    plt.xlabel('Logit Score (Risk)')
    plt.ylabel('Density')
    plt.legend()
    
    plot_filename = f'logit_distribution_{stage_name}.png'
    plt.savefig(plot_filename)
    print(f"✅ Logit distribution plot saved as {plot_filename}")
    plt.close()

# =================================================================
# 5. DATA LOADING UTILITY
# =================================================================
def load_data(embeds_path,cancer_path):
    # (load_data function remains the same)
    data = torch.load(embeds_path)
    data_df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    cancer_data = pd.read_csv(cancer_path,usecols=['pid', '5y'])
    cancer_data_df = pd.DataFrame(cancer_data)
    
    data_df.rename(columns={'index': 'pid'}, inplace=True)
    data_df['pid'] = data_df['pid'].astype(str)
    embed_cols = [c for c in data_df.columns if c != "pid"]
    data_df[embed_cols] = data_df[embed_cols].astype(float) 
    cancer_data_df['pid'] = cancer_data_df['pid'].astype(str)

    merged_data_df = pd.merge(cancer_data_df, data_df, on="pid", how="inner")
    merged_data_df.set_index('pid', inplace=True)

    return merged_data_df

# =================================================================
# 6. MAIN TRAINING FUNCTION (Adapted for t-SNE)
# =================================================================

@hydra.main(version_base=None, config_path="Configs/", config_name="test")
def main(config):
    # --- 6.1 Setup ---
    # ... (Setup code remains the same) ...
    if any([torch.cuda.is_available(), torch.backends.mps.is_available()]):
        print("CUDA-enabled GPU/TPU is available.")
        BATCH_SIZE = config.BATCH_SIZE_GPU 
    else:
        print("No CUDA-enabled GPU found, using CPU.")
        BATCH_SIZE = config.BATCH_SIZE_CPU 

    EPOCHS = config.EPOCHS
    LEARNING_RATE = config.LEARNING_RATE
    embeds_path = config.embeds_path
    cancer_path = config.cancer_path
    SEED = config.SEED
    test_size = config.test_size
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # --- 6.2 Data Loading ---
    merged_data_df = load_data(embeds_path,cancer_path)
    # Replicate your splitting logic
    df_train, df_test_val = train_test_split(merged_data_df, test_size=2*test_size, random_state=SEED) 
    df_train, df_val = train_test_split(df_train, test_size=test_size, random_state=SEED) 
    df_val, df_test = train_test_split(df_test_val, test_size=0.5, random_state=SEED) 
    
    print(f"(Sample size) Training:{len(df_train)} | Validation:{len(df_val)} |Testing:{len(df_test)}")

    dataloader_train = DataLoader(SurvivalDataset(df_train), batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(SurvivalDataset(df_val), batch_size=BATCH_SIZE, shuffle=False)
    dataloader_test = DataLoader(SurvivalDataset(df_test), batch_size=BATCH_SIZE, shuffle=False)

    x, event = next(iter(dataloader_train))
    num_features = x.size(1)
    print(f"num_features = {num_features}")

    # --- 6.3 Model Initialization ---
    cox_model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(num_features), 
        torch.nn.Linear(num_features,128),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.7),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.7),
        torch.nn.Linear(128, 1),
    )

    wandb_logger = WandbLogger(
        project="survival_analysis",
        name="MLP_Binary_Classifier_5yr",
        config={
            "learning_rate": LEARNING_RATE, "epochs": EPOCHS, "batch_size": BATCH_SIZE, 
            "model_type": "MLP_Binary",
        },
    )

    lightning_model = MLP_decoder(cox_model, LEARNING_RATE)
    # NOTE: Set log_every_n_steps to 1 or a low number if your step count per epoch is small
    trainer = L.Trainer(max_epochs=EPOCHS, accelerator="auto", devices=1, logger = wandb_logger, log_every_n_steps=1) 
    
    # --- 6.4 Training and Testing ---
    trainer.fit(lightning_model, dataloader_train, dataloader_val)
    lightning_model.eval()

    trainer.test(lightning_model, dataloaders=dataloader_test)

    # --- 6.5 t-SNE Visualization ---
    
    # 1. Collect all features and events from the test set for t-SNE
    all_features = []
    all_events_tsne = []
    
    print("\n--- Collecting input features for t-SNE ---")
    with torch.no_grad():
        for batch in dataloader_test:
            x, event = batch 
            all_features.append(x.cpu().numpy())
            all_events_tsne.append(event.cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    events_tsne = np.concatenate(all_events_tsne, axis=0)
    
    # 2. Perform and save the t-SNE plot of INPUT FEATURES
    
    # 3. Plot the Logit Score Distribution (the final score given by the model)
    plot_logit_distribution(
        lightning_model.final_test_logits,
        lightning_model.final_test_events,
        stage_name='Test_Logit_Distribution'
    )
    
    # --- 6.6 Finalization ---
    torch.save(lightning_model.state_dict(), "mlp_binary_model.pth")
    wandb.finish()


if __name__ == "__main__":
    main()
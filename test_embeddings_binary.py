import torch 
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
# Removed unused survival loss and metric imports
import torch
from torch.utils.data import Dataset
import pytorch_lightning as L
import torchmetrics
# Import the required binary metrics
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy 
import random
import numpy as np
import hydra
import wandb
from lightning.pytorch.loggers import WandbLogger   


class SurvivalDataset(Dataset):
    def __init__(self, df):
        # Features are all columns *except* the '5y' event column
        self.x = torch.tensor(df.drop(columns=['5y']).values, dtype=torch.float32)   
        # Target must be float for BCE loss
        self.event = torch.tensor(df['5y'].values, dtype=torch.float32)    

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.event[idx]
    
    

class MLP_decoder(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        # Initialize Binary Classification Metrics
        self.auroc_metric = BinaryAUROC()
        self.accuracy_metric = BinaryAccuracy()

        # Define the binary classification loss function
        # BCEWithLogitsLoss is numerically stable for logits (unbounded outputs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.test_preds = []
        self.test_events = []

    def forward(self, x):
        x = self.model(x).squeeze(-1) # Ensure output is [Batch_Size]
        return x
    
    def training_step(self, batch, batch_idx):
        x, event = batch
        logits = self(x)
        # Use BCEWithLogitsLoss with float targets
        loss = self.loss_fn(logits, event) 
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, event = batch
        logits = self(x)
        loss = self.loss_fn(logits, event)
        self.log("val_loss", loss, prog_bar=True)
        # Log validation AUROC for sweep metric monitoring
        self.log("val_auroc", self.auroc_metric(logits, event), on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        x, event = batch
        logits = self(x)

        # store for epoch_end
        self.test_preds.append(logits.detach().cpu())
        self.test_events.append(event.detach().cpu())
        
    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds)
        events = torch.cat(self.test_events)

        # 1. Calculate AUROC (using logits, as it handles sigmoid internally)
        auroc_val = self.auroc_metric(preds, events)
        
        # 2. Calculate Accuracy (needs probabilities or hard predictions)
        # We use threshold=0 (equivalent to sigmoid > 0.5) for hard prediction
        accuracy_val = self.accuracy_metric(preds, events) 

        print(f"\nTest AUROC = {auroc_val:.4f}")
        print(f"Test Accuracy = {accuracy_val:.4f}")

        self.log("test_auroc", auroc_val, prog_bar=True)
        self.log("test_accuracy", accuracy_val, prog_bar=True)

        # Clear lists
        self.test_preds.clear()
        self.test_events.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def load_data(embeds_path,cancer_path):
    data = torch.load(embeds_path)
    data_df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    cancer_data = pd.read_csv(cancer_path,usecols=['pid', '5y'])
    cancer_data_df = pd.DataFrame(cancer_data)
    
    # Clean up column names and types
    data_df.rename(columns={'index': 'pid'}, inplace=True)
    data_df['pid'] = data_df['pid'].astype(str) # PID as string for merging

    # Convert embedding columns to float
    embed_cols = [c for c in data_df.columns if c != "pid"]
    # FIX: Use standard pandas method for type conversion
    data_df[embed_cols] = data_df[embed_cols].astype(float) 

    # Ensure cancer data PID is string
    cancer_data_df['pid'] = cancer_data_df['pid'].astype(str)

    # Merge
    merged_data_df = pd.merge(cancer_data_df, data_df, on="pid", how="inner")
    
    # PID remains as a column or index for splitting. We keep it as a column for now.
    # We drop the PID column before passing to the Dataset, as the Dataset expects only features and target.
    merged_data_df.set_index('pid', inplace=True)  

    return merged_data_df


@hydra.main(version_base=None, config_path=".", config_name="test")
def main(config):
    # ... (unchanged setup code) ...
    if any([torch.cuda.is_available(), torch.backends.mps.is_available()]):
        print("CUDA-enabled GPU/TPU is available.")
        BATCH_SIZE = config.BATCH_SIZE_GPU # batch size for training
    else:
        print("No CUDA-enabled GPU found, using CPU.")
        BATCH_SIZE = config.BATCH_SIZE_CPU  # batch size for training

    EPOCHS = config.EPOCHS
    LEARNING_RATE = config.LEARNING_RATE
    embeds_path = config.embeds_path
    cancer_path = config.cancer_path
    SEED = config.SEED
    test_size = config.test_size
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED) # Removed, use torch.manual_seed for CUDA too


    merged_data_df = load_data(embeds_path,cancer_path)
    df_train, df_test = train_test_split(merged_data_df, test_size=test_size, random_state=SEED)
    df_train, df_val = train_test_split(df_train, test_size=test_size, random_state=SEED)
    print(f"(Sample size) Training:{len(df_train)} | Validation:{len(df_val)} |Testing:{len(df_test)}")

    dataloader_train = DataLoader(SurvivalDataset(df_train), batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(SurvivalDataset(df_val), batch_size=len(df_val), shuffle=False)
    dataloader_test = DataLoader(SurvivalDataset(df_test), batch_size=len(df_test), shuffle=False)

    x, event = next(iter(dataloader_train))

    num_features = x.size(1)

    print(f"x (shape)    = {x.shape}")
    print(f"num_features = {num_features}")
    print(f"event        = {event.shape}")

    # Initiate MLP model (Architecture remains the same, but function changes)
    cox_model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(num_features),  # Batch normalization
        torch.nn.Linear(num_features, 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(64, 1),  # Outputs one logit for BCE Loss
    )

    torch.manual_seed(SEED)
    wandb_logger = WandbLogger(
        project="survival_analysis",
        # Renamed run for clarity
        name="MLP_Binary_Classifier_5yr",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "model_type": "MLP_Binary",
        },
    )

    lightning_model = MLP_decoder(cox_model, LEARNING_RATE)
    trainer = L.Trainer(max_epochs=EPOCHS, accelerator="auto", devices=1, deterministic=True,logger = wandb_logger)
    trainer.fit(lightning_model, dataloader_train, dataloader_val)
    lightning_model.eval()

    #test the model

    # plot loss curve
    import matplotlib.pyplot as plt     
    # Get training and validation losses


    trainer.test(lightning_model, dataloaders=dataloader_test)

    # Save the trained model
    torch.save(lightning_model.state_dict(), "mlp_binary_model.pth")

    wandb.finish()


if __name__ == "__main__":
    main()
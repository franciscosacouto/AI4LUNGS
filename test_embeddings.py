import torch 
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchsurv.loss import cox, weibull
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.kaplan_meier import KaplanMeierEstimator
import torch
from torch.utils.data import Dataset
import sympy
import pytorch_lightning as L
import torchmetrics
from torchmetrics.functional import accuracy, auroc, precision, recall
import random
import numpy as np
import hydra
import wandb
from lightning.pytorch.loggers import WandbLogger   


class SurvivalDataset(Dataset):
    def __init__(self, df):
        self.x = torch.tensor(df.iloc[:, 2:].values, dtype=torch.float32)   # embeddings
        self.event = torch.tensor(df['5y'].values, dtype=torch.bool)    # event
        self.time = torch.tensor(df['fup_days'].values, dtype=torch.float32)  # time

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], (self.event[idx], self.time[idx])
    

class MLP_decoder(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

     # Metrics
        self.cindex_metric = ConcordanceIndex()
        self.auc_metric = Auc()   # time-dependent AUC

        self.test_preds = []
        self.test_events = []
        self.test_times = []

    def forward(self, x):
        x= self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, (event, time) = batch
        log_hz = self(x)
        loss = cox.neg_partial_log_likelihood(log_hz, event, time, reduction="mean")
        self.log("train_loss", loss)
        # wandb.log({"train_loss": loss})
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, (event, time) = batch
        log_hz = self(x)
        loss = cox.neg_partial_log_likelihood(log_hz, event, time, reduction="mean")
        self.log("val_loss", loss, prog_bar=True)
        # wandb.log({"val_loss": loss})

    def test_step(self, batch, batch_idx):
        x, (event, time) = batch
        preds = self(x).squeeze()

        # store for epoch_end
        self.test_preds.append(preds.detach().cpu())
        self.test_events.append(event.detach().cpu())
        self.test_times.append(time.detach().cpu())
        
    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds)
        events = torch.cat(self.test_events)
        times = torch.cat(self.test_times)

        # C-index
        cindex_val = self.cindex_metric(preds, events, times)
        ci_low, ci_high = self.cindex_metric.confidence_interval()

        print(f"\nTest C-index = {cindex_val:.4f}  (95% CI: {ci_low:.4f}, {ci_high:.4f})")

        # Time-dependent AUC
        auc_values = self.auc_metric(preds, events, times)   # 1D tensor
        auc_time = self.auc_metric.time                   

        print("\nTime-dependent AUC(t):")
        print("Times:", auc_time)
        print("AUC:", auc_values)

        self.log("test_cindex", cindex_val, prog_bar=True)
        self.log("test_auc_mean", auc_values.mean(), prog_bar=True)
        # wandb.log({"test_cindex": cindex_val, "test_auc_mean": auc_values.mean()})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def load_data(embeds_path,cancer_path):
    data = torch.load(embeds_path)
    data_df= pd.DataFrame.from_dict(data, orient='index').reset_index()
    cancer_data= pd.read_csv(cancer_path,usecols=['pid', '5y' ,'fup_days'])
    cancer_data_df = pd.DataFrame(cancer_data)
    
    embed_cols = [c for c in data_df.columns if c != "pid"]
    data_df[embed_cols] = data_df[embed_cols].applymap(lambda x: float(x))

    data_df.rename(columns={'index': 'pid'}, inplace=True)
    data_df['pid'] = data_df['pid'].astype(int)
    merged_data_df = pd.merge(cancer_data_df, data_df, on="pid", how="inner")
    merged_data_df.set_index('pid', inplace=True)  

    return merged_data_df





@hydra.main(version_base=None, config_path=".", config_name="test")
def main(config):

    if any([torch.cuda.is_available(), torch.backends.mps.is_available()]):
        print("CUDA-enabled GPU/TPU is available.")
        BATCH_SIZE = config.BATCH_SIZE_GPU # batch size for training
    else:
        print("No CUDA-enabled GPU found, using CPU.")
        BATCH_SIZE = config.BATCH_SIZE_CPU  # batch size for training

    EPOCHS = config.EPOCHS
    LEARNING_RATE = config.LEARNING_RATE
    embeds_path = config.embeds_path
    cancer_path = config.cancer_path
    SEED = config.SEED
    test_size = config.test_size
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)





    merged_data_df = load_data(embeds_path,cancer_path)
    df_train, df_test = train_test_split(merged_data_df, test_size=test_size, random_state=SEED)
    df_train, df_val = train_test_split(df_train, test_size=test_size, random_state=SEED)
    print(f"(Sample size) Training:{len(df_train)} | Validation:{len(df_val)} |Testing:{len(df_test)}")

    dataloader_train = DataLoader(SurvivalDataset(df_train), batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(SurvivalDataset(df_val), batch_size=len(df_val), shuffle=False)
    dataloader_test = DataLoader(SurvivalDataset(df_test), batch_size=len(df_test), shuffle=False)

    x, (event, time) = next(iter(dataloader_train))

    num_features = x.size(1)

    print(f"x (shape)    = {x.shape}")
    print(f"num_features = {num_features}")
    print(f"event        = {event.shape}")
    print(f"time         = {time.shape}")

    # Initiate Weibull model
    cox_model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(num_features),  # Batch normalization
        torch.nn.Linear(num_features, 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(64, 1),  # Estimating log hazards for Cox models
    )

    torch.manual_seed(SEED)
    wandb_logger = WandbLogger(
        project="survival_analysis",
        name="MLP_Cox_model",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "model_type": "MLP_Cox",
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
    torch.save(lightning_model.state_dict(), "mlp_cox_model.pth")

    wandb.finish()


if __name__ == "__main__":
    main()

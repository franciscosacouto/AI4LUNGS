import base64
import torch 
import sys
import pandas as pd
import os
import io
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchsurv.loss import cox
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.cindex import ConcordanceIndex
import torch
from torch.utils.data import Dataset
import pytorch_lightning as L
import random
import numpy as np
import hydra
import wandb
from lightning.pytorch.loggers import WandbLogger   

sys.path.insert(1, '/nas-ctm01/homes/fmferreira/MedImageInsights')
from medimageinsightmodel import MedImageInsight
classifier = MedImageInsight(
    model_dir="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27",

    vision_model_name="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27/vision_model/medimageinsigt-v1.0.0.pt",
    language_model_name="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27/language_model/language_model.pth"
)

class SurvivalDataset(Dataset):
    def __init__(self, df):
        """
        df must contain:
            - 'file_path' : path to .npy image
            - '5y' : event indicator
            - 'fup_days' : follow-up time
        """

        self.df = df.reset_index()   # keep pid but use sequential indices

        # Extract outcomes
        self.event = torch.tensor(self.df['5y'].values, dtype=torch.bool)
        self.time = torch.tensor(self.df['fup_days'].values, dtype=torch.float32)

        # Optional: detect tabular columns besides file_path, 5y, fup_days
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        npy_path = self.df.loc[idx, "file_path"]

        npy_img = np.load(npy_path)

        if not isinstance(npy_img, np.ndarray):
            raise ValueError(f"Loaded object is not a numpy array: {npy_path}")

        print("DEBUG:", idx, npy_path, npy_img.shape, npy_img.dtype)

        img_b64 = pil_to_base64(npy_path)

        return img_b64, (self.event[idx], self.time[idx])

class encoder_decoder(L.LightningModule):
    def __init__(self, encoder, survival_head, learning_rate):
        super().__init__()
        self.survival_head = survival_head
        self.encoder = encoder
        self.learning_rate = learning_rate

     # Metrics
        self.cindex_metric = ConcordanceIndex()
        self.auc_metric = Auc()   # time-dependent AUC

        self.test_preds = []
        self.test_events = []
        self.test_times = []
    
    def encode_batch(self, base64_list):

        embeddings = []

        for b64 in base64_list:
            emb = self.encoder.encode_image(b64)
            if isinstance(emb, np.ndarray):
                emb = torch.tensor(emb)
                embeddings.append(emb.float())
            
        return torch.stack(embeddings, dim=0)

    def forward(self, x):
        embeddings = self.encode_batch(x)
        logits = self.survival_head(embeddings)
        return logits

        
    
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


def load_data(image_path,cancer_path):
    data_df = pd.read_csv(image_path)
    data_df['pid'] = data_df['pid'].astype(int)
    # Load cancer metadata
    cancer_df = pd.read_csv(cancer_path, usecols=['pid', '5y', 'fup_days'])
    cancer_df['pid'] = cancer_df['pid'].astype(int)

    # Merge
    merged_df = pd.merge(cancer_df, data_df, on="pid", how="inner")

    # PID as index
    merged_df.set_index('pid', inplace=True)

    return merged_df


def search_files(rootdir, df):
    for dirpath, _, filenames in os.walk(rootdir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            pid = filename.split('_')[0]   # extract PID
            new_row = {"pid": pid, "file_path": full_path}
            df.loc[len(df)] = new_row
            print(f"Found file: {full_path}")
    return df


def pil_to_base64(npy_img_path):
    npy_img = np.load(npy_img_path)
    pil_img = Image.fromarray((npy_img * 255).astype(np.uint8)) 
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):

    if any([torch.cuda.is_available(), torch.backends.mps.is_available()]):
        print("CUDA-enabled GPU/TPU is available.")
        BATCH_SIZE = config.BATCH_SIZE_GPU # batch size for training
    else:
        print("No CUDA-enabled GPU found, using CPU.")
        BATCH_SIZE = config.BATCH_SIZE_CPU  # batch size for training

    EPOCHS = config.EPOCHS
    LEARNING_RATE = config.LEARNING_RATE

    SEED = config.SEED
    test_size = config.test_size
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    classifier.load_model()
    
    df= pd.DataFrame( columns=[config.data.columns.pid, 
                               config.data.columns.file_path])

    rootdir_lung = config.directories.rootdir_lung
    rootdir_ws = config.directories.rootdir_ws
    rootdir_masked = config.directories.rootdir_masked
    print("Searching lung files...")
    df = search_files( rootdir_lung, df)  
    save_df_as_csv = df.to_csv('lung_file_paths.csv', index=False)
    
    merged_data_df = load_data(config.directories.image_df_path,config.directories.cancer_path)
    df_train, df_test = train_test_split(merged_data_df, test_size=test_size, random_state=SEED)
    df_train, df_val = train_test_split(df_train, test_size=test_size, random_state=SEED)
    print(f"(Sample size) Training:{len(df_train)} | Validation:{len(df_val)} |Testing:{len(df_test)}")

    dataloader_train = DataLoader(SurvivalDataset(df_train), batch_size=BATCH_SIZE, shuffle=True,collate_fn=lambda x: list(zip(*x)))
    dataloader_val = DataLoader(SurvivalDataset(df_val), batch_size=len(df_val), shuffle=False)
    dataloader_test = DataLoader(SurvivalDataset(df_test), batch_size=len(df_test), shuffle=False)

    x, (event, time)= next(iter(dataloader_train))
    print("X[0] type:", type(x[0]))
    print("X[0] example:", x[0])
    exit()
    sample_emb = classifier.encode(x[0])
    
    num_features = sample_emb.shape[0]
    print("Embedding dimension =", num_features)

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
        name="MLP_Cox_model_300",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "model_type": "MLP_Cox",
        },
    )

    lightning_model = encoder_decoder(classifier,cox_model, LEARNING_RATE)
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

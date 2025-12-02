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
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryStatScores


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
        """

        self.df = df.reset_index()   # keep pid but use sequential indices

        # Extract outcomes
        self.event = torch.tensor(self.df['5y'].values, dtype=torch.bool)

        # Optional: detect tabular columns besides file_path, 5y, fup_days
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        npy_path = self.df.loc[idx, "file_path"]

        npy_img = np.load(npy_path)
        

        if not isinstance(npy_img, np.ndarray):
            raise ValueError(f"Loaded object is not a numpy array: {npy_path}")

        img_b64 = array_to_base64(npy_img)

        return img_b64, self.event[idx]

class encoder_decoder(L.LightningModule):
    def __init__(self, encoder, survival_head, learning_rate):
        super().__init__()
        self.survival_head = survival_head
        self.encoder = encoder
        self.learning_rate = learning_rate

     # Metrics
        self.cindex_metric = ConcordanceIndex()
        self.auroc_metric = BinaryAUROC()
        self.f1score = BinaryF1Score()

        self.stats_metric = BinaryStatScores(threshold=0.5, average='none')
        # Define the binary classification loss function
        # BCEWithLogitsLoss is numerically stable for logits (unbounded outputs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.test_preds = []
        self.test_events = []
        self.val_preds = []
        self.val_events = []

    
    def encode_batch(self, base64_list):
        embeddings = []

        # MedImageInsight expects: encode(images=[base64_str, ...])
        out = self.encoder.encode(images=base64_list) 
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
        x, (event, time) = batch
        logits = self(x)
        loss = self.loss_fn(logits, event) 

        self.log("train_loss", loss)
        # wandb.log({"train_loss": loss})
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, (event, time) = batch
        logits = self(x)
        loss = self.loss_fn(logits, event)
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
            print("⚠️  Model is predicting only class 0 (majority class). It is ignoring the minority class!")
        elif num_pred_0 == 0 and num_pred_1 > 0:
            print("⚠️  Model is predicting only class 1 (minority class). It is ignoring the majority class!")
        else:
            print("✅ Model is predicting both classes.")

        return
    
    def _calculate_balanced_metrics(self, preds: torch.Tensor, events: torch.Tensor, prefix: str):
        # Calculate True Positives (TP), False Negatives (FN), etc.
        # stats is a tensor of shape (5,) [TP, FP, TN, FN, SUPS]
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
        

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds)
        events = torch.cat(self.test_events)

        # Calculate metrics for the test set
        self._calculate_balanced_metrics(preds, events, 'test')
        
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


def load_data(cancer_path, rootdir):
    """
    Loads cancer metadata and merges with dynamically found image file paths.
    
    Parameters:
        image_path (str): Path to the CSV with image metadata (optional columns like 'pid')
        cancer_path (str): Path to the CSV with cancer outcomes ('pid', '5y', 'fup_days')
        rootdir (str): Root directory to search for image .npy files
    
    Returns:
        pd.DataFrame: Merged dataframe with 'file_path' and outcomes, indexed by 'pid'
    """

    # Load cancer metadata
    cancer_df = pd.read_csv(cancer_path, usecols=['pid', '5y'])
    cancer_df['pid'] = cancer_df['pid'].astype(str)

    # Optionally load image metadata CSV if needed
    
    # Dynamically search for file paths
    df_paths = search_files(rootdir, pd.DataFrame())  # returns DataFrame with 'pid' index and 'file_path'
    
    # Merge dynamically found file paths
    merged_df = cancer_df.merge(df_paths, left_on='pid', right_index=True, how='inner')

    # Set PID as index
    merged_df.set_index('pid', inplace=True)

    return merged_df



def search_files(rootdir, df):
    records = []
    for dirpath, _, filenames in os.walk(rootdir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            pid = filename.split('_')[0]   # extract PID

            records.append({"pid": str(pid), "file_path": full_path})

       # FIX: Return a new DataFrame with PID as index
    return pd.DataFrame(records).set_index('pid')


def array_to_base64(npy_img):

    pil_img = Image.fromarray((npy_img * 255).astype(np.uint8)) 
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def collate_survival(batch):
    # batch is a list of: (img_b64, (event, time))
    imgs = [item[0] for item in batch]
    events = torch.stack([item[1][0] for item in batch])
    times = torch.stack([item[1][1] for item in batch])
    return imgs, (events, times)


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
    torch.use_deterministic_algorithms(True)

    classifier.load_model()
    
    df= pd.DataFrame( columns=[config.data.columns.pid, 
                               config.data.columns.file_path])

    rootdir_lung = config.directories.rootdir_lung
    rootdir_ws = config.directories.rootdir_ws
    rootdir_masked = config.directories.rootdir_masked
    # df_paths = search_files(rootdir_lung, pd.DataFrame())
    print("Loading survival outcomes and merging paths...")
    merged_data_df = load_data( config.directories.cancer_path, rootdir_lung)
    # merged_data_df = df_outcomes.merge(df_paths, on="pid", how="inner")
    print(merged_data_df.columns)
    df_train, df_test_val = train_test_split(merged_data_df, test_size=2*test_size, random_state=SEED)
    df_val, df_test = train_test_split(df_test_val, test_size=0.5, random_state=SEED)
    print(f"(Sample size) Training:{len(df_train)} | Validation:{len(df_val)} |Testing:{len(df_test)}")

    dataloader_train = DataLoader(SurvivalDataset(df_train), batch_size=BATCH_SIZE, shuffle=True,num_workers=8,
    pin_memory=True,collate_fn=collate_survival)
    dataloader_val = DataLoader(SurvivalDataset(df_val), batch_size=len(df_val), shuffle=False,num_workers=8,
    pin_memory=True,collate_fn=collate_survival)
    dataloader_test = DataLoader(SurvivalDataset(df_test), batch_size=len(df_test), shuffle=False,num_workers=8,
    pin_memory=True,collate_fn=collate_survival)

    x, (event, time)= next(iter(dataloader_train))
    
    sample_emb = classifier.encode(images=[x[0]])
    print(sample_emb.keys())
    image_emb = sample_emb['image_embeddings']  # extract the actual embeddings
    num_features = image_emb.shape[-1] 
    print("Embedding dimension =", num_features)

  
    # Initiate Weibull model
    cox_model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(num_features), # Batch normalization
        torch.nn.Linear(num_features,128),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.7),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.7),
        torch.nn.Linear(128, 1),# Outputs one logit for BCE Loss
    )

    torch.manual_seed(SEED)
    wandb_logger = WandbLogger(
        project="decoder_encoder",
        name="mlp_cox_model_32",
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

# import base64
import torch 
import sys
import pandas as pd
import os
# import io
# from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
# from torchsurv.loss import cox
# from torchsurv.metrics.auc import Auc
# from torchsurv.metrics.cindex import ConcordanceIndex
import torch
# from torch.utils.data import Dataset
import pytorch_lightning as L
import random
import numpy as np
import hydra
import wandb
from lightning.pytorch.loggers import WandbLogger   
# from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryStatScores
from FM_MLP import encoder_decoder
from Dataset import SurvivalDataset


sys.path.insert(1, '/nas-ctm01/homes/fmferreira/MedImageInsights')
from medimageinsightmodel import MedImageInsight


classifier = MedImageInsight(
    model_dir="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27",

    vision_model_name="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27/vision_model/medimageinsigt-v1.0.0.pt",
    language_model_name="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27/language_model/language_model.pth"
)


def load_data(cancer_path, rootdir):
    # Load cancer metadata
    cancer_df = pd.read_csv(cancer_path, usecols=['pid', '5y'])
    cancer_df['pid'] = cancer_df['pid'].astype(str)
    
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


def collate_survival(batch):
    imgs = [item[0] for item in batch]
    events = torch.stack([item[1] for item in batch])
    return imgs, events

def save_results_to_excel(file_path, new_row):
   
    
    # Convert dict to DataFrame
    new_df = pd.DataFrame([new_row])
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Load existing, concatenate, and save back
        try:
            existing_df = pd.read_excel(file_path)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_excel(file_path, index=False)
            print(f"✅ Results successfully appended to {file_path}")
        except Exception as e:
            # Handle potential file access errors or corruption
            print(f"⚠️ Error reading or writing to existing Excel file: {e}")
            new_df.to_excel(file_path, index=False)
            print("Attempted to create a new file with the current run data.")
    else:
        # Create a new file
        new_df.to_excel(file_path, index=False)
        print(f"✅ Created new results file: {file_path}")

@hydra.main(version_base=None, config_path="Configs/", config_name="config")
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

    rootdir = config.directories.rootdir
    
    # df_paths = search_files(rootdir_lung, pd.DataFrame())
    print("Loading survival outcomes and merging paths...")
    merged_data_df = load_data( config.directories.cancer_path, rootdir)
    # merged_data_df = df_outcomes.merge(df_paths, on="pid", how="inner")
    print(merged_data_df.columns)
    df_train, df_test_val = train_test_split(merged_data_df, test_size=2*test_size, random_state=SEED)
    df_val, df_test = train_test_split(df_test_val, test_size=0.5, random_state=SEED)
    print(f"(Sample size) Training:{len(df_train)} | Validation:{len(df_val)} |Testing:{len(df_test)}")

    n_pos = df_train['5y'].sum()
    n_neg = len(df_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    dataloader_train = DataLoader(SurvivalDataset(df_train), batch_size=BATCH_SIZE, shuffle=True,num_workers=8,
    pin_memory=True,collate_fn=collate_survival)
    dataloader_val = DataLoader(SurvivalDataset(df_val), batch_size=len(df_val), shuffle=False,num_workers=8,
    pin_memory=True,collate_fn=collate_survival)
    dataloader_test = DataLoader(SurvivalDataset(df_test), batch_size=len(df_test), shuffle=False,num_workers=8,
    pin_memory=True,collate_fn=collate_survival)

    x, event= next(iter(dataloader_train))
    
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
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(128, 1),# Outputs one logit for BCE Loss
    )

    torch.manual_seed(SEED)
    wandb_logger = WandbLogger(
        project=config.project,
        name=config.model_name,
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "model_type": "MLP_Cox", 
        },
    )

    early_stop_callback = L.callbacks.EarlyStopping(
        monitor=config.early_stopping.monitor, 
        patience=config.early_stopping.patience,   
        verbose=config.early_stopping.verbose,
        mode=config.early_stopping.mode 
    )

    trainer_callbacks = [early_stop_callback]

    lightning_model = encoder_decoder(classifier,cox_model, LEARNING_RATE, pos_weight)
    trainer = L.Trainer(max_epochs=EPOCHS, accelerator="auto", devices=1, deterministic=True,logger = wandb_logger,enable_checkpointing=False, callbacks=trainer_callbacks, log_every_n_steps=1)
    trainer.fit(lightning_model, dataloader_train, dataloader_val)
    lightning_model.eval()

    # plot loss curve
    import matplotlib.pyplot as plt     
    # Get training and validation losses


    trainer.test(lightning_model, dataloaders=dataloader_test)
    test_results = {
        # Metadata and Hyperparameters (from config)
        "Model_Name": config.model_name,
        "Image_Root_Dir": config.directories.rootdir,
        "Learning_Rate": config.LEARNING_RATE,
        "Epochs_Trained": trainer.current_epoch,
        "Batch_Size_GPU": config.BATCH_SIZE_GPU,
        "Test_Size": config.test_size,
        "Seed": config.SEED,
        
        # Test Metrics (from the updated model instance)
        "Test_AUROC": lightning_model.test_auroc,
        "Test_F1_Score": lightning_model.test_f1_score,
        "Test_Balanced_Accuracy": lightning_model.test_balanced_accuracy,
    }
    
    # 2. Save the results to the central Excel file
    results_file_path = "Experiments_Summary.xlsx"
    save_results_to_excel(results_file_path, test_results)

    # Save the trained model
    torch.save(lightning_model.state_dict(), "Models/mlp_cox_model.pth")

    wandb.finish()


if __name__ == "__main__":
    main()

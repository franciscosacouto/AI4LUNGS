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
from collections import defaultdict 
from lightning.pytorch.loggers import WandbLogger   
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryStatScores
from FM_MLP import encoder_decoder as encoder_decoder
from NLSTPreprocessedKFoldDataLoader import NLSTPreprocessedKFoldDataLoader
from NLSTPreprocessedKFoldDataLoader import NLSTPreprocessedDataLoader
from FM_MLP_binary import encoder_decoder as encoder_decoder_binary
from RadioDino_MLP import encoder_decoder as radiodino_decoder

import timm


sys.path.insert(1, '/nas-ctm01/homes/fmferreira/MedImageInsights')
from medimageinsightmodel import MedImageInsight
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

classifier = MedImageInsight(
    model_dir="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27",

    vision_model_name="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27/vision_model/medimageinsigt-v1.0.0.pt",
    language_model_name="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27/language_model/language_model.pth"
)

radioDino = timm.create_model("hf_hub:Snarcy/RadioDino-s16", pretrained=True)

def load_data(cancer_path, rootdir):
    # Load cancer metadata
    cancer_df = pd.read_csv(cancer_path, usecols=[
        'pid', 
        '5y', 
        # ADD THESE COLUMNS FROM YOUR CSV
        'study_yr', 
        'reversed', 
        'sct_slice_final_mapped',
        'fup_days'
        # Ensure these are the correct names in your CSV
    ])
    print(cancer_df.columns)
    max_time = cancer_df['fup_days'].max()
    cancer_df['fup_days'] = cancer_df['fup_days'] / max_time
    

    # Dynamically search for file paths
    df_paths = search_files(rootdir, pd.DataFrame())  # returns DataFrame with 'pid' index and 'file_path'
    
    # Merge dynamically found file paths
    merged_df = cancer_df.merge(df_paths, left_on='pid', right_index=True, how='inner')
    print(merged_df.columns)
    # Set PID as index

    return merged_df



def search_files(rootdir, df):
    records = []
    for dirpath, _, filenames in os.walk(rootdir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            pid = filename.split('_')[0]   # extract PID

            records.append({"pid": int(pid), "file_path": full_path})

       # FIX: Return a new DataFrame with PID as index
    return pd.DataFrame(records).set_index('pid')


def collate_survival(batch):
    # item[0] is now a Tensor [3, H, W] from _get_data
    imgs = torch.stack([item[0] for item in batch]) 
    events = torch.stack([item[1] for item in batch])
    time_to_event = torch.stack([item[2] for item in batch])

    return imgs, events, time_to_event

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



@hydra.main(version_base=None, config_path="Configs/", config_name="config_ws")
def main(config):

    EPOCHS = config.EPOCHS
    LEARNING_RATE = config.LEARNING_RATE

    SEED = config.SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)

    ENCODER_MAP = {
        'MedImageInsights': classifier,
        'RadioDino': radioDino
    }

    # Mapping logic for which LightningModule wrapper to use
    # Key: (is_binary, encoder_name)
    MODULE_MAP = {
        (True, 'MedImageInsights'): encoder_decoder_binary,
        (False, 'MedImageInsights'): encoder_decoder,
        (True, 'RadioDino'): radiodino_decoder,
        (False, 'RadioDino'): radiodino_decoder, # Use same for both if logic allows
    }
    if any([torch.cuda.is_available(), torch.backends.mps.is_available()]):
        print("CUDA-enabled GPU/TPU is available.")
        BATCH_SIZE = config.BATCH_SIZE_GPU # batch size for training
    else:
        print("No CUDA-enabled GPU found, using CPU.")
        BATCH_SIZE = config.BATCH_SIZE_CPU  # batch size for training

    encoder_obj = ENCODER_MAP[config.Encoder]
    if config.Encoder == 'MedImageInsights':
        encoder_obj.load_model()


    


    rootdir = config.directories.rootdir

    lung_metadataframe = load_data(config.directories.cancer_path, rootdir)
    print(lung_metadataframe.columns) # <-- Run this line to check column names!
    lung_metadataframe = lung_metadataframe.rename(columns={'file_path': 'path', '5y': 'label', 'sct_slice_final_mapped': 'sct_slice_num'})
    lung_metadataframe['label'] = lung_metadataframe['label'].astype(int)

    pd.set_option('display.max_columns', None)

    # 2. Show the full string content (don't truncate long file paths)
    pd.set_option('display.max_colwidth', None)

    # 3. Increase the width of the display so it doesn't wrap lines too much
    pd.set_option('display.width', 1000)




    # 3. Instantiate the NLSTPreprocessedKFoldDataLoader
    data_loader_manager = NLSTPreprocessedKFoldDataLoader(
        config=config,
        lung_metadataframe=lung_metadataframe
    )

  
    dataloaders = data_loader_manager.get_dataloaders()
    all_fold_results = []
    num_folds = len(dataloaders['train'])
    train_labels = dataloaders['train'][0].dataset.labels
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    x_sample, event_sample, time_sample = next(iter(dataloaders['train'][0])) 
    
    encoder_obj = ENCODER_MAP.get(config.Encoder)
    module_class = MODULE_MAP.get((config.Binary_model, config.Encoder))

    if not encoder_obj or not module_class:
        raise ValueError(f"Unsupported combination: {config.Encoder} & Binary={config.Binary_model}")

    # Initialize temp model
    temp_model = module_class(
        encoder_obj, 
        torch.nn.Identity(), 
        LEARNING_RATE, 
        pos_weight, 
        freeze_encoder=True
    )

    
    # Move model to GPU (this is what caused the mismatch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_model.to(device)
    
    with torch.no_grad():
        temp_model.eval()
        # MOVE INPUT TO DEVICE HERE:
        x_sample = x_sample.to(device) 
        
        sample_output = temp_model(x_sample) 
        num_features = sample_output.shape[-1]

    print(f"Embedding dimension determined: {num_features}")

    for fold_id in range(num_folds):
            print(f"\n====================== STARTING FOLD {fold_id + 1}/{num_folds} ======================")

            # 1. Select DataLoaders for the Current Fold
            dataloader_train = dataloaders['train'][fold_id]
            dataloader_val = dataloaders['validation'][fold_id]
            dataloader_test = dataloaders['test'][fold_id]
            
            # 1. Determine the output head architecture
            out_features = 1 if config.Binary_model else 2
            head_architecture = torch.nn.Sequential(
                torch.nn.BatchNorm1d(num_features),
                torch.nn.Linear(num_features, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(128, out_features),
            )


            early_stop_callback = L.callbacks.EarlyStopping(
            monitor=config.early_stopping.monitor, 
            patience=config.early_stopping.patience,   
            verbose=config.early_stopping.verbose,
            mode=config.early_stopping.mode,
            check_on_train_epoch_end=False
            )
            model_str = f"best{config.model_name}"

            checkpoint_callback = L.callbacks.ModelCheckpoint(
            monitor='val_loss',     # Or whatever your monitor is
            dirpath=f'checkpoints/fold_{fold_id:02d}',
            filename=model_str +'-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,           # Only keep the 1 best model
            mode='min',             # 'min' for loss, 'max' for C-index
        )
            trainer_callbacks = [early_stop_callback]

                        # 2. Instantiate the Lightning model using the registry lookup from earlier
            lightning_model = module_class(
                encoder_obj, 
                head_architecture, 
                LEARNING_RATE, 
                pos_weight, 
                freeze_encoder=config.Freeze_weights
            )


            trainable = sum(p.numel() for p in lightning_model.parameters() if p.requires_grad)
            print(f"The model has {trainable:,} trainable parameters.")
     
            # 4. Initialize Trainer and Logger for the Current Fold
            current_wandb_logger = WandbLogger(
                project=config.project,
                name=f"{config.model_name}_Fold_{fold_id+1}",
                config={"learning_rate": LEARNING_RATE, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "model_type": "MLP_Cox"},
            )
            current_trainer = L.Trainer(
                max_epochs=EPOCHS, 
                accelerator="auto", 
                devices=1, 
                deterministic=True, 
                logger=current_wandb_logger, 
                callbacks=[early_stop_callback, checkpoint_callback],
                log_every_n_steps=10
            )
            
            # 5. Train, Validate, and Test
            current_trainer.fit(lightning_model, dataloader_train, dataloader_val)
            
            # Run test and capture results
            test_results_list = current_trainer.test(ckpt_path='best', dataloaders=dataloader_test)
            test_results_fold = test_results_list[0] # Test returns a list of dictionaries

            # 6. Store Results and Finalize Run
            all_fold_results.append(test_results_fold)
            wandb.finish() 

        # --- END CV LOOP ---

    
    # Calculate average metrics across all folds
    avg_auroc = np.mean([r.get('test_auroc', 0) for r in all_fold_results])
    std_auroc = np.std([r.get('test_auroc', 0) for r in all_fold_results])
    avg_f1 = np.mean([r.get('test_f1_score', 0) for r in all_fold_results])
    std_f1 = np.std([r.get('test_f1_score', 0) for r in all_fold_results])
    avg_bal_acc = np.mean([r.get('test_balanced_acc', 0) for r in all_fold_results])
    std_bal_acc = np.std([r.get('test_balanced_acc', 0) for r in all_fold_results])
    avg_cindex = np.mean([r.get('test_cindex',0) for r in all_fold_results])
    std_cindex = np.std([r.get('test_cindex', 0) for r in all_fold_results])
    avg_ibs = np.mean([r.get('test_ibs',0) for r in all_fold_results])
    std_ibs = np.std([r.get('test_ibs', 0) for r in all_fold_results])
    avg_recall = np.mean([r.get('test_recall', 0) for r in all_fold_results])
    std_recall = np.std([r.get('test_recall', 0) for r in all_fold_results])
    avg_precision = np.mean([r.get('test_precision', 0) for r in all_fold_results])
    std_precision = np.std([r.get('test_precision', 0) for r in all_fold_results])
    
    print(f"\n====================== CV COMPLETE ({num_folds} Folds) ======================")
    print(f"Average Test AUROC: {avg_auroc:.4f}")
    print(f"Average Test F1 Score: {avg_f1:.4f}")
    
    # 7. Prepare and save final summary (using average metrics)
    final_summary = {
        "Model_Name": config.model_name,
        "Learning_Rate": LEARNING_RATE,
        "Total_Folds": num_folds,
        "Avg_Test_AUROC": avg_auroc,
        "Std_Test_AUROC": std_auroc,
        "Avg_Test_CIndex": avg_cindex,
        "Std_Test_CIndex": std_cindex,
        "Avg_Test_IBrier_Score": avg_ibs,
        "Std_Test_IBS": std_ibs,
        "Avg_Test_balanced_acc": avg_bal_acc,
        "Std_Test_bal_acc": std_bal_acc,
        "Avg_Test_f1_score":avg_f1,
        "Std_Test_f1_score": std_f1,
        "Avg_Test_recall":avg_recall,
        "Std_Test_recall": std_recall,
        "Avg_Test_precision":avg_precision,
        "Std_Test_precision": std_precision,
        "Seed": config.SEED,
    }
    
    results_file_path = "Experiments_Summary.xlsx"
    save_results_to_excel(results_file_path, final_summary)

if __name__ == "__main__":
    main()

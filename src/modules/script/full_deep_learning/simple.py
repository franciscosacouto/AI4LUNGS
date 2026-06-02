import torch 
import sys
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as L
import random
import numpy as np
import hydra
import wandb
from collections import defaultdict 
from lightning.pytorch.loggers import WandbLogger   

project_root = "/nas-ctm01/homes/fmferreira/AI4LUNGS" 
if project_root not in sys.path:
    sys.path.append(project_root)

# 1. IMPORT YOUR BRAND NEW CLEAN MODULAR FILES
from src.modules.architecture.MedImage import MedicalVisionEncoder
from src.modules.architecture.CTClip import ClinicalTextEncoder
from src.modules.architecture.fusion import MultiModalFusionModule
from src.modules.architecture.system import EncoderDecoderSystem

from NLSTPreprocessedKFoldDataLoader import NLSTPreprocessedKFoldDataLoader
from model_setup import get_encoders
from transformers import AutoTokenizer 

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def load_data(cancer_path, rootdir, text_path):
    cancer_df = pd.read_csv(cancer_path, usecols=['pid', '5y', 'study_yr', 'reversed', 'sct_slice_final_mapped', 'fup_days'])
    max_time = cancer_df['fup_days'].max()
    cancer_df['fup_days'] = cancer_df['fup_days'] / max_time
    
    df_paths = search_files(rootdir, pd.DataFrame())
    text_df = pd.read_csv(text_path)

    merged_df = cancer_df.merge(df_paths, on='pid', how='inner')
    merged_df = merged_df.merge(text_df, how='inner', on='pid')    
    return merged_df

def search_files(rootdir, df):
    records = []
    for dirpath, _, filenames in os.walk(rootdir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            pid = filename.split('_')[0]
            records.append({"pid": int(pid), "file_path": full_path})
    return pd.DataFrame(records).set_index('pid')

def save_results_to_excel(file_path, new_row):
    new_df = pd.DataFrame([new_row])
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_excel(file_path)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_excel(file_path, index=False)
            print(f"✅ Results successfully appended to {file_path}")
        except Exception as e:
            print(f"⚠️ Error appending to Excel file: {e}")
            new_df.to_excel(file_path, index=False)
    else:
        new_df.to_excel(file_path, index=False)
        print(f"✅ Created new results file: {file_path}")


@hydra.main(version_base=None, config_path="/nas-ctm01/homes/fmferreira/AI4LUNGS/Configs/", config_name="config_ws")
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCHS = config.EPOCHS
    LEARNING_RATE = config.LEARNING_RATE
    SEED = config.SEED
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        torch.use_deterministic_algorithms(False)

    # 2. LOAD BACKBONES FROM DISK ONCE
    raw_encoders = get_encoders(config.get('image_encoder'), config.get('text_encoder'), device)  

    # 3. INITIALIZE GLOBAL TOKENIZER
    global_tokenizer = None
    if getattr(config, 'text', False):
        global_tokenizer = AutoTokenizer.from_pretrained(
            'microsoft/BiomedVLP-CXR-BERT-specialized',
            do_lower_case=True, 
            trust_remote_code=True
        )

    # 4. INSTANTIATE YOUR ISOLATED FEATURE ENCODERS NATIVELY
    # No more complex MODULE_MAP lookup dictionary needed!
    vision_net = MedicalVisionEncoder(raw_encoders.get('vision'))
    text_net = ClinicalTextEncoder(raw_encoders.get('language')) if raw_encoders.get('language') is not None else None

    # Assemble your unified multi-modal fusion wrapper
    fusion_module = MultiModalFusionModule(vision_net=vision_net, text_net=text_net, config=config)

    # Setup batch limits
    BATCH_SIZE = config.BATCH_SIZE_GPU if torch.cuda.is_available() else config.BATCH_SIZE_CPU

    # Load Dataframes and process tracking rows
    lung_metadataframe = load_data(config.directories.cancer_path, config.directories.rootdir, config.directories.rootdir_text)
    lung_metadataframe = lung_metadataframe.rename(columns={'file_path': 'path', '5y': 'label', 'sct_slice_final_mapped': 'sct_slice_num'})
    lung_metadataframe['label'] = lung_metadataframe['label'].astype(int)

    data_loader_manager = NLSTPreprocessedKFoldDataLoader(config=config, lung_metadataframe=lung_metadataframe)
    dataloaders = data_loader_manager.get_dataloaders()
    all_fold_results = []
    num_folds = len(dataloaders['train'])
    
    # Calculate pos_weights for survival head configurations
    train_labels = dataloaders['train'][0].dataset.labels
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)], dtype=torch.float32)

    # Calculate dynamic input sizing for the fully connected survival head layers
    batch_sample = next(iter(dataloaders['train'][0])) 
    inputs_sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_sample[0].items()}
   
    with torch.no_grad():
        fusion_module.eval()
        sample_output = fusion_module(inputs_sample, global_tokenizer)
        total_input_features = sample_output.shape[-1]

    print(f"✅ Dynamic Multimodal Combined Feature Input Dimension: {total_input_features}")
    
    # --- CROSS VALIDATION LOOP ---
    for fold_id in range(num_folds):
        print(f"\n====================== STARTING FOLD {fold_id + 1}/{num_folds} ======================")

        dataloader_train = dataloaders['train'][fold_id]
        dataloader_val = dataloaders['validation'][fold_id]
        dataloader_test = dataloaders['test'][fold_id]
        
        # Continuous models output 2 parameters (Weibull scale + shape), binary outputs 1
        out_features = 1 if config.Binary_model else 2
        
        # Build your survival head layers natively
        head_architecture = torch.nn.Sequential(
            torch.nn.BatchNorm1d(total_input_features),
            torch.nn.Linear(total_input_features, 128),
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
        
        model_str = f"best_{config.model_name}"
        checkpoint_callback = L.callbacks.ModelCheckpoint(
            monitor='val_loss',     
            dirpath=f'checkpoints/fold_{fold_id:02d}',
            filename=model_str + '-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,           
            mode='min',             
        )

        # 5. INITIALIZE YOUR RESTURCTURED PYTORCH LIGHTNING SYSTEM FILE
        lightning_model = EncoderDecoderSystem(
            fusion_module=fusion_module,
            survival_head=head_architecture,
            learning_rate=LEARNING_RATE,
            tokenizer=global_tokenizer,
            config=config,
            fold_id=fold_id + 1
        )
        
        trainable = sum(p.numel() for p in lightning_model.parameters() if p.requires_grad)
        print(f"The model has {trainable:,} trainable parameters.")
 
        current_wandb_logger = WandbLogger(
            project=config.project,
            name=f"{config.model_name}_Fold_{fold_id+1}",
            config={"learning_rate": LEARNING_RATE, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "model_type": "Modular_Weibull_Survival"},
        )
        
        current_trainer = L.Trainer(
            max_epochs=EPOCHS, 
            accelerator="auto", 
            devices=1, 
            deterministic=False, 
            logger=current_wandb_logger, 
            callbacks=[early_stop_callback, checkpoint_callback],
            log_every_n_steps=10
        )
        
        # Train and validate model setup parameters via system configurations
        current_trainer.fit(lightning_model, dataloader_train, dataloader_val)
        
        # Run test set using the top checkpoint tracked across training
        test_results_list = current_trainer.test(lightning_model, dataloaders=dataloader_test, ckpt_path='best', verbose=True)
        test_results_fold = test_results_list[0]

        all_fold_results.append(test_results_fold)
        wandb.finish() 

    # --- END CV LOOP ---
    
    # Gather global results across cross-validation blocks using torchsurv metrics
    avg_loss = np.mean([r.get('test_loss', 0) for r in all_fold_results])
    std_loss = np.std([r.get('test_loss', 0) for r in all_fold_results])
    avg_cindex = np.mean([r.get('test_cindex', 0) for r in all_fold_results])
    std_cindex = np.std([r.get('test_cindex', 0) for r in all_fold_results])
    avg_auc = np.mean([r.get('test_auc', 0) for r in all_fold_results])
    std_auc = np.std([r.get('test_auc', 0) for r in all_fold_results])
    avg_brier = np.mean([r.get('test_brier_score', 0) for r in all_fold_results])
    std_brier = np.std([r.get('test_brier_score', 0) for r in all_fold_results])
    
    print(f"\n====================== CV COMPLETE ({num_folds} Folds) ======================")
    print(f"Average Test C-Index     : {avg_cindex:.4f} ± {std_cindex:.4f}")
    print(f"Average Test 5Y AUC      : {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"Average Test Brier Score : {avg_brier:.4f} ± {std_brier:.4f}")
    
    final_summary = {
        "Model_Name": config.model_name,
        "Learning_Rate": LEARNING_RATE,
        "Total_Folds": num_folds,
        "Avg_Test_Loss": avg_loss,
        "Std_Test_Loss": std_loss,
        "Avg_Test_CIndex": avg_cindex,
        "Std_Test_CIndex": std_cindex,
        "Avg_Test_AUC_5Y": avg_auc,
        "Std_Test_AUC_5Y": std_auc,
        "Avg_Test_Brier_Score": avg_brier,
        "Std_Test_Brier_Score": std_brier,
        "Seed": config.SEED,
    }
    
    save_results_to_excel("Experiments_Summary_v2.xlsx", final_summary)

if __name__ == "__main__":
    main()
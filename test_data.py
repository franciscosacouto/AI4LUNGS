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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.loggers import WandbLogger
from sklearn.manifold import TSNE
from FM_MLP import encoder_decoder
from AI4LUNGS.NLSTPreprocessedKFoldDataLoader import SurvivalDataset
# NOTE: Ensure 'FM_MLP' (encoder_decoder) and 'Dataset' (SurvivalDataset)
# are in your path, or replace with actual imports.

# --- PLACEHOLDER 1: Assuming the MedImageInsight Model is loaded like this ---
sys.path.insert(1, '/nas-ctm01/homes/fmferreira/MedImageInsights')
from medimageinsightmodel import MedImageInsight # Assuming this is your Foundation Model class

classifier = MedImageInsight(
    model_dir="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27",
    vision_model_name="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27/vision_model/medimageinsigt-v1.0.0.pt",
    language_model_name="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27/language_model/language_model.pth"
)

# --- PLACEHOLDER 2: Assumed external classes (Imported from FM_MLP and Dataset) ---
# NOTE: You MUST ensure these two classes are correctly imported/defined.
# from FM_MLP import encoder_decoder
# from Dataset import SurvivalDataset 


# =================================================================
# 2. DATA UTILITIES
# =================================================================

def load_data(cancer_path, rootdir):
    """Loads cancer metadata and merges it with image file paths."""
    cancer_df = pd.read_csv(cancer_path, usecols=['pid', '5y'])
    cancer_df['pid'] = cancer_df['pid'].astype(str)
    
    df_paths = search_files(rootdir) 
    
    merged_df = cancer_df.merge(df_paths, left_on='pid', right_index=True, how='inner')
    merged_df.set_index('pid', inplace=True)

    return merged_df

def search_files(rootdir):
    """Dynamically searches the root directory for image files and extracts PIDs."""
    records = []
    for dirpath, _, filenames in os.walk(rootdir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            pid = filename.split('_')[0] 
            records.append({"pid": str(pid), "file_path": full_path})

    return pd.DataFrame(records).set_index('pid')

def collate_survival(batch):
    """Collates a list of (Base64 image, event) tuples into batch tensors/lists."""
    imgs = [item[0] for item in batch]
    events = torch.stack([item[1] for item in batch])
    return imgs, events

# =================================================================
# 3. RESULTS UTILITIES
# =================================================================

def save_results_to_excel(file_path, new_row):
    """Appends a new row of results to an Excel file, creating it if necessary."""
    new_df = pd.DataFrame([new_row])
    
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_excel(file_path)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_excel(file_path, index=False)
            print(f"✅ Results successfully appended to {file_path}")
        except Exception as e:
            print(f"⚠️ Error reading or writing to existing Excel file: {e}")
            new_df.to_excel(file_path, index=False)
            print("Attempted to create a new file with the current run data.")
    else:
        new_df.to_excel(file_path, index=False)
        print(f"✅ Created new results file: {file_path}")


# =================================================================
# 3. RESULTS UTILITIES (Modified visualize_embeddings)
# =================================================================

def visualize_embeddings(lightning_model, dataloader, stage_name):
    """
    Generates a 2D t-SNE visualization of the model's image embeddings,
    colored by the survival outcome, and saves the raw data.
    """
    print(f"\n--- Generating {stage_name} Embeddings for t-SNE ---")
    all_embeddings = []
    all_events = []

    lightning_model.eval()
    
    # We rely on lightning_model.encode_batch from the encoder_decoder class
    with torch.no_grad():
        for batch in dataloader:
            x, event = batch
            # Embeddings are extracted using the model's current state
            embeddings = lightning_model.encode_batch(x)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_events.append(event.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    events = np.concatenate(all_events, axis=0)

    # --- 🔑 NEW CODE: Save the raw data for future analysis ---
    save_filename = f'embeddings_data_{stage_name}.npz'
    np.savez_compressed(
        save_filename, 
        embeddings=embeddings, 
        events=events
    )
    print(f"✅ Raw embeddings and events saved to {save_filename} for future t-SNE runs.")
    # -----------------------------------------------------------

    # Apply t-SNE
    print("Running t-SNE...")
    # ... (rest of the t-SNE and plotting code remains the same) ...
    tsne = TSNE(n_components=2, random_state=42, perplexity=10, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plotting 
    plt.figure(figsize=(10, 8))
    # ... (Plotting code) ...
    plt.close()

# =================================================================
# 4. MAIN TRAINING FUNCTION
# =================================================================

@hydra.main(version_base=None, config_path="Configs/", config_name="config")
def main(config):
    # --- 4.1 Setup ---
    if any([torch.cuda.is_available(), torch.backends.mps.is_available()]):
        print("CUDA-enabled GPU/TPU is available.")
        BATCH_SIZE = config.BATCH_SIZE_GPU 
    else:
        print("No CUDA-enabled GPU found, using CPU.")
        BATCH_SIZE = config.BATCH_SIZE_CPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = config.EPOCHS
    LEARNING_RATE = config.LEARNING_RATE # NOTE: This should be the head's LR

    SEED = config.SEED
    test_size = config.test_size
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)

    classifier.load_model()
    try:
            # We access the PyTorch model stored inside the MedImageInsight wrapper
            internal_model = classifier.model
            if internal_model is not None:
                internal_model.to(device)
                print(f"✅ MedImageInsight's internal PyTorch model successfully moved to {device}")
            else:
                print("⚠️ Warning: classifier.model is None. Check if classifier.load_model() completed.")
    except AttributeError:
            # This will catch the error if 'model' is not the correct attribute name
        print("❌ CRITICAL ERROR: 'classifier' object does not have a 'model' attribute to move. Check MedImageInsight source code.")
        sys.exit(1) # Exit if we can't move the model    # --- 4.2 Data Loading ---
    print("Loading survival outcomes and merging paths...")
    merged_data_df = load_data( config.directories.cancer_path, config.directories.rootdir)
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

    # Determine embedding size
    x, event= next(iter(dataloader_train))
    sample_emb = classifier.encode(images=[x[0]])
    num_features = sample_emb['image_embeddings'].shape[-1] 
    print("Embedding dimension =", num_features)

    # --- 4.3 Model Initialization ---
    cox_model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(num_features), 
        torch.nn.Linear(num_features,128),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(128, 1),
    )

    wandb_logger = WandbLogger(
        project=config.project, name=config.model_name,
        config={
            "learning_rate_head": LEARNING_RATE, "epochs": EPOCHS, "batch_size": BATCH_SIZE, 
            "model_type": "MLP_Cox", "pos_weight": pos_weight.item(),
        },
    )

    lightning_model = encoder_decoder(classifier,cox_model,  LEARNING_RATE, pos_weight)
    
    # NOTE: log_every_n_steps=1 to see per-epoch logs due to small batch count
    trainer = L.Trainer(max_epochs=EPOCHS, accelerator="auto", devices=1, deterministic=True,
                        logger = wandb_logger, enable_checkpointing=False, log_every_n_steps=10)

    # --- 4.4 Training and Testing ---
    trainer.fit(lightning_model, dataloader_train, dataloader_val)
    lightning_model.eval()

    trainer.test(lightning_model, dataloaders=dataloader_test)
    # --- CRITICAL FINAL DEVICE PLACEMENT ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Final check: Ensuring all models are on {device} for visualization.")

# 1. Move the Lightning Module (and its head)
    lightning_model.to(device)

    # 2. Re-verify and explicitly move the Foundation Model's internal component
    try:
        if hasattr(lightning_model._encoder_wrapper, 'model') and lightning_model._encoder_wrapper.model is not None:
            # Move the internal PyTorch model that you confirmed was on CUDA earlier
            lightning_model._encoder_wrapper.model.to(device)
            print("✅ Re-confirmed Foundation Model's internal PyTorch weights moved to GPU.")
        
        # 3. CRITICAL: If the image encoder is stored separately, move that too.
        # We must assume the internal structure of MedImageInsight.
        # If the encoder is part of the model, this is redundant. If not, this is the fix.
        if hasattr(lightning_model._encoder_wrapper.model, 'image_encoder') and lightning_model._encoder_wrapper.model.image_encoder is not None:
            lightning_model._encoder_wrapper.model.image_encoder.to(device)
            print("✅ Re-confirmed image encoder moved to GPU.")

    except AttributeError:
        # Catching if the attribute structure is different
        print("⚠️ Warning: Could not find all internal model attributes to move. The main model should suffice.")

    # --- End of Final Device Placement ---

    # --- 4.5 Visualization and Results Saving ---
    visualize_embeddings(lightning_model, dataloader_test, stage_name='test')

    test_results = {
        # Metadata and Hyperparameters
        "Model_Name": config.model_name,
        "Image_Root_Dir": config.directories.rootdir,
        "Learning_Rate_Head": config.LEARNING_RATE,
        "Epochs_Trained": trainer.current_epoch,
        "Batch_Size_GPU": config.BATCH_SIZE_GPU,
        "Test_Size": config.test_size,
        "Seed": config.SEED,
        
        # Test Metrics (Retrieved from encoder_decoder attributes)
        "Test_AUROC": lightning_model.test_auroc,
        "Test_F1_Score": lightning_model.test_f1_score,
        "Test_Balanced_Accuracy": lightning_model.test_balanced_accuracy,
    }
    
    results_file_path = "Experiments_Summary.xlsx"
    save_results_to_excel(results_file_path, test_results)

    # Save the trained model
    torch.save(lightning_model.state_dict(), "Models/mlp_cox_model.pth")

    wandb.finish()


if __name__ == "__main__":
    main()
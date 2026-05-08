import torch
import pandas as pd
import os
import numpy as np
import hydra
from tqdm import tqdm
import sys

# Ensure the project root is in the path
sys.path.insert(1, '/nas-ctm01/homes/fmferreira/MedImageInsights')

from medimageinsightmodel import MedImageInsight
from NLSTPreprocessedKFoldDataLoader import NLSTPreprocessedKFoldDataLoader

def save_combined_fold(df_list, fold_id, output_dir):
    """Saves the combined splits for a single fold into one CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine train, val, and test dataframes
    final_df = pd.concat(df_list, ignore_index=True)
    
    output_path = os.path.join(output_dir, f"fold_{fold_id}_embeddings.csv")
    final_df.to_csv(output_path, index=False)
    print(f"✅ Saved combined Fold {fold_id} (Train/Val/Test) to {output_path}")

@hydra.main(version_base=None, config_path="Configs/", config_name="config_ws")
def main(config):
    # 1. Initialize MedImageInsight
    med_model = MedImageInsight(
        model_dir="/nas-ctm01/homes/fmferreira/MedImageInsights/2024.09.27",
        vision_model_name="medimageinsigt-v1.0.0.pt",
        language_model_name="language_model.pth"
    )
    med_model.load_model()
    
    device = med_model.device
    model = med_model.model
    model.eval()

    # 2. Data Loading Logic
    from AI4LUNGS.src.modules.script.full_deep_learning.encoder_survivalhead import load_data 
    lung_metadataframe = load_data(
        config.directories.cancer_path, 
        config.directories.rootdir, 
        config.directories.rootdir_text
    )
    
    lung_metadataframe = lung_metadataframe.rename(
        columns={'file_path': 'path', '5y': 'label', 'sct_slice_final_mapped': 'sct_slice_num'}
    )

    # Path to PID lookup dictionary
    path_to_pid = dict(zip(lung_metadataframe['path'], lung_metadataframe['pid']))

    # Initialize manager (load_data_name=False to prevent printer crash)
    data_loader_manager = NLSTPreprocessedKFoldDataLoader(
        config=config,
        lung_metadataframe=lung_metadataframe,
        load_data_name=False 
    )
    
    dataloaders = data_loader_manager.get_dataloaders()
    output_dir = "extracted_embeddings_results"
    
    num_folds = config.number_of_k_folds if config.number_of_k_folds > 0 else 1

    # 3. Combined Extraction Loop
    with torch.no_grad():
        for fold_idx in range(num_folds):
            print(f"\n🚀 Processing Fold {fold_idx}...")
            fold_dataframes = []

            for split_name in ['train', 'validation', 'test']:
                print(f"--- Extracting {split_name} split ---")
                
                # Get the loader and the corresponding file paths for this fold/split
                loader = dataloaders[split_name][fold_idx]
                fold_paths = data_loader_manager.data_splits[split_name]['file_names'][fold_idx]
                fold_pids = [str(path_to_pid[p]) for p in fold_paths]
                
                # Force no shuffle to maintain order
                loader.dataset.shuffle = False 
                
                all_embeds = []
                for batch in tqdm(loader):
                    inputs, _, _ = batch
                    images = inputs['image'].to(device)
                    feats = model.encode_image(images)
                    all_embeds.append(feats.cpu().numpy())

                if all_embeds:
                    embeddings = np.concatenate(all_embeds, axis=0)
                    
                    # Truncate PIDs if they don't match (e.g., due to batching drop_last)
                    current_pids = fold_pids[:len(embeddings)]
                    
                    # Create split-specific dataframe
                    temp_df = pd.DataFrame(
                        embeddings, 
                        columns=[f'enc_{i}' for i in range(embeddings.shape[1])]
                    )
                    temp_df.insert(0, 'pid', current_pids)
                    temp_df.insert(1, 'split', split_name) # Identify if it's train, val, or test
                    
                    fold_dataframes.append(temp_df)

            # Save all splits for this fold into one file
            if fold_dataframes:
                save_combined_fold(fold_dataframes, fold_idx, output_dir)

if __name__ == "__main__":
    main()
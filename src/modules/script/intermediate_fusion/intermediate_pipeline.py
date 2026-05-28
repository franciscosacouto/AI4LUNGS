import os 
import sys
import numpy as np
import pandas as pd
import lightgbm
import sklearn
import torch




def extract_fine_tuned_features(module_class, ckpt_path, dataloader, encoders, head_architecture, config, device):
    """
    Loads a specific fold checkpoint and extracts image embeddings.
    """
    # 1. Load the model from the best checkpoint
    model = module_class.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        encoders=encoders,
        head_architecture=head_architecture,
        config=config
    )
    model.to(device)
    model.eval()

    extracted_data = []

    print(f"🧠 Extracting features from checkpoint: {os.path.basename(ckpt_path)}")
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0]
            pids = inputs['pid']
            
            # Move only tensors to GPU
            gpu_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in inputs.items()}
            
            # 2. Extract features from the fine-tuned image encoder
            # Accessing the specific sub-module within your lightning wrapper
            img_embeddings = model.encoders.image_encoder(gpu_inputs['image'])
            
            # Flatten if it's a spatial tensor (B, C, H, W) -> (B, C*H*W)
            if img_embeddings.dim() > 2:
                img_embeddings = img_embeddings.view(img_embeddings.size(0), -1)
            
            feat_np = img_embeddings.cpu().numpy()
            
            for i, pid in enumerate(pids):
                extracted_data.append([pid] + feat_np[i].tolist())

    # Create DataFrame
    cols = ['pid'] + [f'img_feat_{i}' for i in range(feat_np.shape[1])]
    return pd.DataFrame(extracted_data, columns=cols)

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

def run_intermediate_fusion(df_img_features, clinical_csv_path):
    """
    Merges image embeddings with clinical data and trains XGBoost.
    """
    # 1. Load Clinical Data
    df_clinical = pd.read_csv(clinical_csv_path)
    
    # 2. Merge on PID (Intermediate Fusion)
    df_fused = pd.merge(df_clinical, df_img_features, on='pid', how='inner')
    
    # 3. Define target and drop metadata
    # Ensure you drop columns that are NOT features
    DROP_FOR_ML = ['pid', 'label', 'fup_days', 'finaldeathlc', 'path', 'split_fold_1', 'split_fold_2', 'split_fold_3']
    
    X = df_fused.drop(columns=[c for c in DROP_FOR_ML if c in df_fused.columns], errors='ignore')
    y = df_fused['label']
    
    # 4. Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. Train XGBoost
    print(f"🌲 Training Fusion Model on {X.shape[1]} combined features...")
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    clf.fit(X_scaled, y)
    
    # 6. Evaluate
    probs = clf.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, probs)
    
    return clf, auc, df_fused

# ==========================================
# EXAMPLE USAGE (Inside your fold loop)
# ==========================================
path_to_best_ckpt = f'checkpoints/fold_01/best_model.ckpt'
df_img = extract_fine_tuned_features(module_class, path_to_best_ckpt, dataloader_test, ...)
clinical_stage_path = 'data/After_Diagnosis_Features.csv'

model, fused_auc, final_df = run_intermediate_fusion(df_img, clinical_stage_path)
print(f"🔥 Final Intermediate Fusion AUC: {fused_auc:.4f}")
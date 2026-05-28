import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    balanced_accuracy_score, 
    recall_score, 
    precision_score
)
from lifelines.utils import concordance_index

# 1. Load your data
# Requirements: Columns 'pid', 'imaging_prob', 'true_label' and 'clinical_prob'
df_img = pd.read_csv("/nas-ctm01/homes/fmferreira/AI4LUNGS/results/Imaging/final_aggregated_results.csv")
df_clin = pd.read_csv("/nas-ctm01/homes/fmferreira/AI4LUNGS/results/clinical_models/preclinical/regression/final_clinical_results.csv")

# 2. Merge on PID
# We use suffixes to make it clear which is which, 
# though the actual labels should be identical.
df = pd.merge(df_img, df_clin, on='pid', suffixes=('', '_drop'))

# 3. Clean up the merged dataframe
# Drop the redundant label column from the clinical file
if 'true_label_drop' in df.columns:
    df = df.drop(columns=['true_label_drop'])

# Ensure PID types match just in case
df['pid'] = df['pid'].astype(str)


# 2. Grid Search for Weights
results = []

# Iterate from 0.0 to 1.0 in steps of 0.1
for i in range(11):
    w_img = round(i * 0.1, 1)
    w_clin = round(1.0 - w_img, 1)
    
    # Calculate fused probability
    fused_prob = (df['imaging_prob'] * w_img) + (df['clinical_prob'] * w_clin)
    
    # Convert probabilities to hard predictions using standard 0.5 threshold
    fused_preds = (fused_prob >= 0.5).astype(int)
    
    # Calculate Continuous Metrics
    auc = roc_auc_score(df['true_label'], fused_prob)
    c_index = concordance_index(df['true_label'], fused_prob)
    
    # Calculate Threshold-Based Metrics (0.5)
    f1 = f1_score(df['true_label'], fused_preds)
    bal_acc = balanced_accuracy_score(df['true_label'], fused_preds)
    recall = recall_score(df['true_label'], fused_preds)
    precision = precision_score(df['true_label'], fused_preds)
    
    results.append({
        'Weight_Img': w_img,
        'Weight_Clin': w_clin,
        'AUC': auc,
        'C-Index': c_index,
        'Balanced_Acc': bal_acc,
        'F1': f1,
        'Recall': recall,
        'Precision': precision
    })

# 3. Create Results DataFrame
results_df = pd.DataFrame(results)

# 4. Find the Best Weight based on AUC
best_idx = results_df['AUC'].idxmax()
best_row = results_df.loc[best_idx]

print("📊 --- Weight Optimization Results ---")
print(results_df.to_string(index=False))

print("\n🏆 --- BEST COMBINATION ---")
print(f"Best Image Weight:     {best_row['Weight_Img']}")
print(f"Best Clinical Weight:  {best_row['Weight_Clin']}")
print(f"Best AUC:              {best_row['AUC']:.4f}")
print(f"Associated Bal. Acc:   {best_row['Balanced_Acc']:.4f}")

# 5. Save results
results_df.to_csv("/nas-ctm01/homes/fmferreira/AI4LUNGS/results/late_fusion/weight_optimization_results_preclinical.csv", index=False)
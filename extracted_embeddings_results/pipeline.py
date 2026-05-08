import numpy as np
import pandas as pd
import warnings
import os
import pickle

# Feature selection & stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, log_loss,
    balanced_accuracy_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# ================== Utility Constants ==================
DROP_COLS = [
    "pid", "dss_5y", "split_fold_1", "split_fold_2", "split_fold_3",
    "split", "survival_time", "event", "lesionsize"
]

# ================== MODEL FUNCTIONS ==================

def fit_rf(X_train, y_train, custom_folds=None):
    param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 10, 20, 50],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    rf = RandomForestClassifier(random_state=42)
    scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'roc_auc': 'roc_auc', 'neg_log_loss': 'neg_log_loss'}
    grid_search = GridSearchCV(rf, param_grid, scoring=scoring, refit='roc_auc', cv=custom_folds, verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

def fit_svm(X_train, y_train, custom_folds=None):
    pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=42))])
    param_grid = {'svc__C': [0.1, 1, 10, 100], 'svc__kernel': ['linear', 'rbf'], 'svc__gamma': ['scale', 'auto']}
    scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'roc_auc': 'roc_auc', 'neg_log_loss': 'neg_log_loss'}
    grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, refit='roc_auc', cv=custom_folds, verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

def fit_lasso_logreg(X_train, y_train, custom_folds=None):
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42))])
    param_grid = {'clf__C': [0.01, 0.1, 1, 10, 100]}
    scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'roc_auc': 'roc_auc', 'neg_log_loss': 'neg_log_loss'}
    grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, refit='roc_auc', cv=custom_folds, verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

def fit_svm_l1(X_train, y_train, custom_folds=None):
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', CalibratedClassifierCV(estimator=LinearSVC(penalty='l1', dual=False, max_iter=10000), cv=3))])
    param_grid = {'clf__estimator__C': [0.01, 0.1, 1, 10, 100]}
    scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'roc_auc': 'roc_auc', 'neg_log_loss': 'neg_log_loss'}
    grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, refit='roc_auc', cv=custom_folds, verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

def fit_xgboost(X_train, y_train, custom_folds=None):
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
    param_grid = {'clf__n_estimators': [100, 200], 'clf__max_depth': [3, 6, 9], 'clf__learning_rate': [0.01, 0.1]}
    scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'roc_auc': 'roc_auc', 'neg_log_loss': 'neg_log_loss'}
    grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, refit='roc_auc', cv=custom_folds, verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

MODELS = {
    "rf": fit_rf,
    "svm": fit_svm,
    "lasso_logreg": fit_lasso_logreg,
    "svm_l1": fit_svm_l1,
    "xgboost": fit_xgboost
}

# ================== Utility Functions ==================

def drop_unwanted_columns(df):
    rndgroup_cols = [col for col in df.columns if col.startswith('rndgroup')]
    loc_cols = [col for col in df.columns if col.startswith('loc')]
    cols_to_drop = set(DROP_COLS) | set(rndgroup_cols) | set(loc_cols)
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

def save_val_metrics(grid_search, path):
    best = grid_search.best_index_
    res = pd.DataFrame(grid_search.cv_results_).loc[best, ["mean_test_accuracy", "mean_test_precision", "mean_test_recall", "mean_test_roc_auc", "mean_test_neg_log_loss"]]
    metrics = res.rename({"mean_test_accuracy": "val_acc", "mean_test_roc_auc": "val_auroc"}).to_dict()
    pd.DataFrame([metrics | grid_search.best_params_]).to_csv(path, index=False)

def test_model_save_metrics(grid_search, X_test, y_test, path):
    model = grid_search.best_estimator_
    y_pred, y_proba = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }
    pd.DataFrame([metrics]).to_csv(path, index=False)
    print(f"   Test AUROC: {metrics['roc_auc']:.4f}")

# ================== Main Pipeline ==================

def run_intermediate_fusion(clinical_dfs, imaging_dfs, d1_metadata, custom_folds, folder):
    os.makedirs(folder, exist_ok=True)
    
    # Split Dataset 1 metadata into Train/Val and Test groups
    train_meta = d1_metadata[d1_metadata["split_fold_1"] != "test"].reset_index(drop=True)
    test_meta = d1_metadata[d1_metadata["split_fold_1"] == "test"].reset_index(drop=True)

    for clin_name, clin_df in clinical_dfs.items():
        for imag_name, imag_df in imaging_dfs.items():
            combo_name = f"{clin_name}_x_{imag_name}"
            print(f"\n🚀 Training Combo: {combo_name}")

            # Merge features on PID
            fused_all = pd.merge(clin_df, imag_df, on='pid', how='inner')
            
            # Create training and test feature sets aligned with metadata PIDs
            X_train = train_meta[['pid']].merge(fused_all, on='pid', how='left')
            y_train = train_meta['dss_5y']
            
            X_test = test_meta[['pid']].merge(fused_all, on='pid', how='left')
            y_test = test_meta['dss_5y']

            # Final cleaning (drop PIDs and metadata strings)
            X_train_final = drop_unwanted_columns(X_train).fillna(0)
            X_test_final = drop_unwanted_columns(X_test).fillna(0)

            for model_name, fit_func in MODELS.items():
                print(f"--- Model: {model_name} ---")
                res = fit_func(X_train_final, y_train, custom_folds=custom_folds)
                
                # Save results
                save_val_metrics(res, f"{folder}/{combo_name}_{model_name}_val.csv")
                test_model_save_metrics(res, X_test_final, y_test, f"{folder}/{combo_name}_{model_name}_test.csv")

def main():
    # Load Data
    path_target = r'/nas-ctm01/homes/fmferreira/AI4LUNGS/NLST_clinical_final_processed_outcomes.csv'
    metadata_path = r'/nas-ctm01/homes/fmferreira/AI4LUNGS/lung_metadata_with_splits.csv'
    output_folder = r'/nas-ctm01/homes/fmferreira/AI4LUNGS/multimodal/intermediatefusion'

    d1_meta = pd.read_csv(metadata_path)
    df_target = pd.read_csv(path_target)
    
    # Ensure dss_5y target is present in metadata
    # (Assuming calculate_dss_5y logic or merge here)
    d1_meta = d1_meta.merge(df_target[['pid', 'death_days', 'finaldeathlc']], on='pid', how='left')
    d1_meta['dss_5y'] = ((d1_meta['death_days'] <= 1825) & (d1_meta['finaldeathlc'] == 1)).astype(int)

    clinical_feature_sets = {
        "pc": pd.read_csv(r'/nas-ctm01/homes/fmferreira/AI4LUNGS/NLST_preclinical_final_processed_normalized.csv'),
        "ad": pd.read_csv(r'/nas-ctm01/homes/fmferreira/AI4LUNGS/NLST_after_diagnosis_final_processed_normalized.csv'),
        "fc": pd.read_csv(r'/nas-ctm01/homes/fmferreira/AI4LUNGS/NLST_clinical_final_processed_normalized.csv')
    }

    imaging_feature_sets = {
        "fold1": pd.read_csv(r'/nas-ctm01/homes/fmferreira/AI4LUNGS/extracted_embeddings_results/fold_0_embeddings.csv'),
        "fold2": pd.read_csv(r'/nas-ctm01/homes/fmferreira/AI4LUNGS/extracted_embeddings_results/fold_1_embeddings.csv'),
        "fold3": pd.read_csv(r'/nas-ctm01/homes/fmferreira/AI4LUNGS/extracted_embeddings_results/fold_2_embeddings.csv')
    }

    # Prepare CV folds using indices from the training subset of d1
    train_only = d1_meta[d1_meta["split_fold_1"] != "test"].reset_index(drop=True)
    custom_folds = [
        (train_only[train_only[col] == "train"].index.tolist(),
         train_only[train_only[col] == "val"].index.tolist())
        for col in ["split_fold_1", "split_fold_2", "split_fold_3"]
    ]

    run_intermediate_fusion(clinical_feature_sets, imaging_feature_sets, d1_meta, custom_folds, output_folder)

if __name__ == "__main__":
    main()
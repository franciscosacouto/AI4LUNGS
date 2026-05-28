import numpy as np
import pandas as pd
import warnings
import os
import pickle

# Feature selection & stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    balanced_accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Survival Analysis
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# Suppress warnings
warnings.filterwarnings('ignore')

# ================== Utility Constants ==================
DROP_COLS = [
    "pid", "path", "key", "label", "dss_5y", 
    "split_fold_1", "split_fold_2", "split_fold_3",
    "survival_time", "event", "study_yr"
]

SCORING = {
    'accuracy': 'accuracy',
    'roc_auc': 'roc_auc'
}
TARGET_COL = "label"

# ================== UTILITIES ==================

def drop_unwanted_columns(df):
    """Ensure we only keep numeric features and ignore metadata/targets."""
    df_clean = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    return df_clean.select_dtypes(include=[np.number])

def calculate_vif(df, threshold=5.0):
    X = drop_unwanted_columns(df.copy())
    X = X.fillna(X.select_dtypes(include=[np.number]).median())
    X = X.loc[:, X.nunique() > 1]
    
    if X.empty: return []
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vals = X.values.astype(float)
    vif_data["VIF"] = [variance_inflation_factor(vals, i) for i in range(vals.shape[1])]
    return vif_data[vif_data["VIF"] <= threshold]["feature"].tolist()

def get_lasso_features(df, y):
    X = drop_unwanted_columns(df.copy())
    X = X.fillna(X.select_dtypes(include=[np.number]).median())
    if X.empty: return []
    
    X_scaled = StandardScaler().fit_transform(X)
    logreg_lasso = LogisticRegressionCV(Cs=50, cv=5, penalty="l1", solver="liblinear", 
                                        random_state=42, max_iter=10000).fit(X_scaled, y)
    coef = pd.Series(logreg_lasso.coef_[0], index=X.columns)
    return coef[coef != 0].index.tolist()

def cox_univariate_feature_selection(X_train, df_target):
    # Align survival data
    X = X_train.copy()
    X = X.merge(df_target[['pid', 'fup_days', 'finaldeathlc']], on='pid', how='left')
    
    features = drop_unwanted_columns(X).columns.tolist()
    # Ensure survival columns are not in the feature list
    features = [f for f in features if f not in ['fup_days', 'finaldeathlc']]
    
    X = X.fillna(X.select_dtypes(include=[np.number]).median())
    uni_features = []
    for feature in features:
        try:
            cph = CoxPHFitter()
            cph.fit(X[[feature, 'fup_days', 'finaldeathlc']], 
                    duration_col='fup_days', event_col='finaldeathlc')
            if cph.summary.loc[feature, "p"] < 0.05:
                uni_features.append(feature)
        except: continue
    return uni_features

# ================== MODEL FITTERS ==================

def fit_rf(X, y, custom_folds):
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    gs = GridSearchCV(rf, {'n_estimators': [100, 200], 'max_depth': [3, 5, 10]}, 
                      scoring=SCORING, refit='roc_auc', cv=custom_folds, n_jobs=-1)
    return gs.fit(X, y)

def fit_svm(X, y, custom_folds):
    pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, class_weight='balanced', random_state=42))])
    gs = GridSearchCV(pipe, {'svc__C': [0.1, 1, 10], 'svc__kernel': ['linear', 'rbf']}, 
                      scoring=SCORING, refit='roc_auc', cv=custom_folds, n_jobs=-1)
    return gs.fit(X, y)

def fit_lasso_logreg(X, y, custom_folds):
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42))])
    gs = GridSearchCV(pipe, {'clf__C': [0.01, 0.1, 1, 10]}, scoring=SCORING, refit='roc_auc', cv=custom_folds, n_jobs=-1)
    return gs.fit(X, y)

def fit_xgboost(X, y, custom_folds):
    pos_w = (y == 0).sum() / (y == 1).sum()
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', XGBClassifier(scale_pos_weight=pos_w, eval_metric='logloss', random_state=42))])
    gs = GridSearchCV(pipe, {'clf__n_estimators': [100], 'clf__max_depth': [3, 5]}, scoring=SCORING, refit='roc_auc', cv=custom_folds, n_jobs=-1)
    return gs.fit(X, y)

MODELS = {"rf": fit_rf, "svm": fit_svm, "lasso_logreg": fit_lasso_logreg, "xgboost": fit_xgboost}

# ================== EVALUATION ==================

def save_val_metrics(gs, path):
    res = pd.DataFrame(gs.cv_results_).loc[gs.best_index_]
    out = {k: v for k, v in res.items() if 'mean_test' in k}
    out.update(gs.best_params_)
    pd.DataFrame([out]).to_csv(path, index=False)

def test_model_save_metrics(gs, X_test, y_test, test_pids, df_target, path, prob_path):
    model = gs.best_estimator_
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # --- LATE FUSION PREP: Save per-patient probabilities ---
    probs_df = pd.DataFrame({
        'pid': test_pids,
        'clinical_prob': y_prob,
        'true_label': y_test.values
    })
    probs_df.to_csv(prob_path, index=False)
    # -------------------------------------------------------
    
    # Survival mapping for C-Index
    eval_df = pd.DataFrame({'pid': test_pids, 'risk': y_prob})
    eval_df = eval_df.merge(df_target[['pid', 'fup_days', 'finaldeathlc']], on='pid', how='left').dropna()
    
    c_index = concordance_index(eval_df['fup_days'], -eval_df['risk'], eval_df['finaldeathlc'])
    
    metrics = {
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "balanced_acc": balanced_accuracy_score(y_test, model.predict(X_test))
    }
    pd.DataFrame([metrics]).to_csv(path, index=False)
    return metrics

# ================== MAIN ==================

def main(folder=r'/nas-ctm01/homes/fmferreira/AI4LUNGS/results/clinical_models/after_diagnosis/binary'):
    # Paths
    path_features = r'/nas-ctm01/homes/fmferreira/AI4LUNGS/data/NLST_after_diagnosis_final_processed_normalized.csv'
    path_target = r'/nas-ctm01/homes/fmferreira/AI4LUNGS/data/NLST_clinical_final_processed_outcomes.csv'
    splits_path = r'/nas-ctm01/homes/fmferreira/AI4LUNGS/data/lung_metadata_with_splits.csv'
    os.makedirs(folder, exist_ok=True)

    # 1. Load Data
    features_df = pd.read_csv(path_features)
    meta_df = pd.read_csv(splits_path)
    df_target = pd.read_csv(path_target).dropna(subset=['fup_days', 'finaldeathlc'])
    
    d1 = features_df.merge(meta_df[['pid', 'label', 'split_fold_1', 'split_fold_2', 'split_fold_3']], on="pid")
    d1_train = d1[d1["split_fold_1"] != "test"].reset_index(drop=True)
    d1_test = d1[d1["split_fold_1"] == "test"].reset_index(drop=True)
    
    y_train, y_test = d1_train[TARGET_COL], d1_test[TARGET_COL]
    test_pids = d1_test["pid"].values
    # 1. Identify columns that are NOT numeric types
    non_numeric_cols = d1.select_dtypes(exclude=[np.number]).columns.tolist()

    print("--- Columns currently NOT detected as numbers ---")
    print(non_numeric_cols)

    # 2. Check for "Hidden" non-numerics in columns that SHOULD be numbers
    # This checks if any value in the column fails a conversion to numeric
    def find_offenders(df):
        offending_columns = {}
        for col in df.columns:
            # Try converting to numeric; identify where it fails (returns NaN)
            converted = pd.to_numeric(df[col], errors='coerce')
            mask = converted.isna() & df[col].notna()
            if mask.any():
                # Get the unique non-numeric values found in this column
                offending_columns[col] = df.loc[mask, col].unique().tolist()
        return offending_columns

    print("\n--- Columns with non-numeric values found inside ---")
    offenders = find_offenders(d1)
    for col, values in offenders.items():
        print(f"Column '{col}' contains non-numeric values: {values}")
        # 2. CV Folds
    custom_folds = [(d1_train[d1_train[col] == "train"].index.tolist(),
                     d1_train[d1_train[col] == "val"].index.tolist())
                    for col in ["split_fold_1", "split_fold_2", "split_fold_3"]]

    # 3. Feature Selection
    fs_path = os.path.join(folder, "feature_selection_lists.pkl")
    if os.path.exists(fs_path):
        with open(fs_path, "rb") as f: fs_lists = pickle.load(f)
    else:
        print("Running Feature Selection...")
        fs_lists = {
            "vif": calculate_vif(d1_train),
            "lasso": get_lasso_features(d1_train, y_train),
            "cox_uni": cox_univariate_feature_selection(d1_train, df_target)
        }
        fs_lists["union"] = list(set(fs_lists["vif"]) | set(fs_lists["lasso"]) | set(fs_lists["cox_uni"]))
        with open(fs_path, "wb") as f: pickle.dump(fs_lists, f)

    all_results = []
    best_auc = -1
    best_gs = None
    best_info = {}

    # 4. Training Loop
    for method, features in fs_lists.items():
        valid_features = [f for f in features if f in d1_train.columns]
        if not valid_features: continue
        
        for model_name, fit_func in MODELS.items():
            print(f"🚀 Testing {model_name} + {method}...")
            gs = fit_func(d1_train[valid_features], y_train, custom_folds)
            
            # Temporary evaluation to check performance
            y_prob = gs.best_estimator_.predict_proba(d1_test[valid_features])[:, 1]
            current_auc = roc_auc_score(y_test, y_prob)
            
            # Tracking for Global Ranking
            test_res = {"method": method, "model": model_name, "roc_auc": current_auc}
            all_results.append(test_res)

            # 🏆 Check if this is our new winner
            if current_auc > best_auc:
                best_auc = current_auc
                best_gs = gs
                best_info = {
                    "method": method, 
                    "model": model_name, 
                    "features": valid_features
                }

    # 5. Final Saving (Only the best)
    if best_gs:
        print(f"\n🥇 Best Model Found: {best_info['model']} via {best_info['method']} (AUC: {best_auc:.4f})")
        
        # Save Global Ranking for reference
        summary = pd.DataFrame(all_results).sort_values(by="roc_auc", ascending=False)
        summary.to_csv(os.path.join(folder, "GLOBAL_RANKING.csv"), index=False)

        # Save ONLY the best model's probabilities for Late Fusion
        test_metrics = test_model_save_metrics(
            best_gs, 
            d1_test[best_info['features']], 
            y_test, 
            test_pids, 
            df_target, 
            os.path.join(folder, "best_clinical_model_metrics.csv"),
            os.path.join(folder, "best_clinical_probabilities.csv") # <--- Use this for fusion
        )
        
        # Optional: Save the actual model object
        with open(os.path.join(folder, "best_clinical_model.pkl"), "wb") as f:
            pickle.dump(best_gs.best_estimator_, f)
            
    print("\n✅ DONE. Best model artifacts saved to folder.")

if __name__ == "__main__":
    main()
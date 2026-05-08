# ================== Imports ==================
from pyexpat import model
import numpy as np
import pandas as pd
import warnings
import os
import pickle
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
from scipy.stats import rankdata

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LogisticRegression, LogisticRegressionCV, LassoCV
)
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
# ================== Boosting Models ==================
from xgboost import XGBClassifier
import lightgbm as lgb

# ================== Survival Analysis ==================
from lifelines import CoxPHFitter

# ================== MODEL FUNCTIONS ==================
def fit_rf(X_train, y_train, custom_folds=None):
    """
    Fit a Random Forest model and return the fitted model.
    """
    param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 10, 20, 50],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    # Initialize model
    rf = RandomForestClassifier(random_state=42)

    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'roc_auc': 'roc_auc',
        'neg_log_loss': 'neg_log_loss'
    }
    
    if custom_folds is None:
        grid_search = GridSearchCV(
        rf,
        param_grid,
        scoring=scoring,
        refit='roc_auc',  # Use AUROC to select best model
        cv=3,
        verbose=1,
        n_jobs=-1,
        return_train_score=True
        )
    else:
        grid_search = GridSearchCV(
        rf,
        param_grid,
        scoring=scoring,
        refit='roc_auc',
        cv=custom_folds,  # Use your predefined splits
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )
        
    # Fit the model
    grid_search.fit(X_train, y_train)
        

    return grid_search

def fit_svm(X_train, y_train, custom_folds=None):
    """
    Fit a SVM model and return the fitted model.
    """
    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # SVM benefits from feature scaling
        ('svc', SVC(probability=True))  # Enable probabilities for AUROC, log loss
    ])

    # Define hyperparameter grid
    param_grid = {
        'svc__C': [0.1, 0.5, 1, 5, 10, 100],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto']
    }

    # GridSearchCV
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'roc_auc': 'roc_auc',
        'neg_log_loss': 'neg_log_loss'
    }
    
    if custom_folds is None:
        grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='roc_auc',  # Use AUROC to select best model
        cv=3,
        verbose=1,
        n_jobs=-1,
        return_train_score=True
        )
    else:
        grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='roc_auc',
        cv=custom_folds,  # Use your predefined splits
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )
        
    # Fit the model
    grid_search.fit(X_train, y_train)
        

    return grid_search

def fit_lasso_logreg(X_train, y_train, custom_folds=None):
    """
    Fit a SVM model and return the fitted model.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000))
    ])

    # Hyperparameter grid
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10, 100]  # Smaller C = stronger regularization
    }

    # GridSearchCV
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'roc_auc': 'roc_auc',
        'neg_log_loss': 'neg_log_loss'
    }
    
    if custom_folds is None:
        grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='roc_auc',  # Use AUROC to select best model
        cv=3,
        verbose=1,
        n_jobs=-1,
        return_train_score=True
        )
    else:
        grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='roc_auc',
        cv=custom_folds,  # Use your predefined splits
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )
        
    # Fit the model
    grid_search.fit(X_train, y_train)
        

    return grid_search

def fit_svm_l1(X_train, y_train, custom_folds=None):
    """
    Fit a SVM model and return the fitted model.
    """
    # Define pipeline
    pipeline= Pipeline([
        ('scaler', StandardScaler()),
        ('clf', CalibratedClassifierCV(
            estimator=LinearSVC(penalty='l1', dual=False, max_iter=10000),
            cv=3
        ))
    ])

    # Hyperparameter grid for C in LinearSVC
    param_grid = {
        'clf__estimator__C': [0.01, 0.1, 1, 10, 100]
    }
    
    # GridSearchCV
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'roc_auc': 'roc_auc',
        'neg_log_loss': 'neg_log_loss'
    }
    
    if custom_folds is None:
        grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='roc_auc',  # Use AUROC to select best model
        cv=3,
        verbose=1,
        n_jobs=-1,
        return_train_score=True
        )
    else:
        grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='roc_auc',
        cv=custom_folds,  # Use your predefined splits
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )
        
    # Fit the model
    grid_search.fit(X_train, y_train)
        

    return grid_search

def fit_xgboost(X_train, y_train, custom_folds=None):
    """
    Fit a XGBoost model and return the fitted model.
    """
    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=22))
    ])

    # Grid
    param_grid = {
        'clf__n_estimators': [50, 100, 200, 500],
        'clf__max_depth': [3, 5, 7, 9],
        'clf__learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5],
        'clf__subsample': [0.5, 0.8, 1.0]
    }

    # GridSearchCV
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'roc_auc': 'roc_auc',
        'neg_log_loss': 'neg_log_loss'
    }
    
    if custom_folds is None:
        grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='roc_auc',  # Use AUROC to select best model
        cv=3,
        verbose=0,
        n_jobs=-1,
        return_train_score=True
        )
    else:
        grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='roc_auc',
        cv=custom_folds,  # Use your predefined splits
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )
        
    # Fit the model
    grid_search.fit(X_train, y_train)
        

    return grid_search

# ================== Modeling Functions ==================
def fit_rf(X_train, y_train, custom_folds=None):
    """Random Forest + GridSearch."""
    param_grid = {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [None, 10, 20, 50],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }
    rf = RandomForestClassifier(random_state=22)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
        "neg_log_loss": "neg_log_loss"
    }
    cv = custom_folds or 3
    grid_search = GridSearchCV(
        rf, param_grid, scoring=scoring, refit="roc_auc",
        cv=cv, verbose=1, n_jobs=-1, return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    return grid_search


# ================== Utility Constants ==================
DROP_COLS = [
    "pid", "dss_5y", "split_fold_1", "split_fold_2", "split_fold_3",
    "split", "survival_time", "event"
]

MODELS = {
    "rf": fit_rf,
    "svm": fit_svm,
    "lasso_logreg": fit_lasso_logreg,
    "svm_l1": fit_svm_l1,
    "xgboost": fit_xgboost
}

# ================== Utility Functions ==================
def drop_unwanted_columns(df, exclude=None):
    """Drop columns that interfere with feature selection or modeling."""
    if exclude is None:
        exclude = []
    cols_to_drop = set(DROP_COLS) - set(exclude)
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")


def safe_print_features(title, features):
    print(f"{title} ({len(features)} features):")
    print(features)


# ================== Core Functions ==================
def calculate_dss_5y_apply(row_pid, df_target, cutoff):
    """Return DSS-5y for a given pid."""
    target_row = df_target[df_target["pid"] == row_pid]
    if target_row.empty:
        return 0
    death_date = target_row["death_days"].values[0]
    final_death_lc = target_row["finaldeathlc"].values[0]
    return int((death_date <= cutoff) and (final_death_lc == 1))


def get_homogeneous_features(df):
    """Return features with no variance."""
    df = drop_unwanted_columns(df)
    return [col for col in df.columns if df[col].nunique() <= 1]



def save_val_metrics(grid_search, path="validation_metrics.csv"):
    """Save validation metrics from GridSearchCV."""
    best = grid_search.best_index_
    results = pd.DataFrame(grid_search.cv_results_).loc[best, [
        "mean_test_accuracy", "mean_test_precision", "mean_test_recall",
        "mean_test_roc_auc", "mean_test_neg_log_loss"
    ]].rename({
        "mean_test_accuracy": "val_accuracy",
        "mean_test_precision": "val_precision",
        "mean_test_recall": "val_recall",
        "mean_test_roc_auc": "val_auroc",
        "mean_test_neg_log_loss": "val_log_loss"
    })
    df = pd.DataFrame([results.to_dict() | grid_search.best_params_])
    df.to_csv(path, index=False)
    # Print the validation metrics as well
    print("Validation Metrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    print("Validation metrics saved to", path)
    return df


# ================== Main Pipeline ==================

def load_datasets(path_features, path_target, dataset1_splits_path, cutoff_days=5*365):
    """Load features, targets, and create DSS-5y column."""
    df = pd.read_csv(path_features)
    df_target = pd.read_csv(path_target)

    # Compute DSS-5y
    df["dss_5y"] = df["pid"].apply(lambda x: calculate_dss_5y_apply(x, df_target, cutoff_days))

    dataset1_splits = pd.read_csv(dataset1_splits_path)

    # Dataset 1
    d1 = df[df["pid"].isin(dataset1_splits["pid"])].reset_index(drop=True)
    d1_target = df_target[df_target["pid"].isin(dataset1_splits["pid"])].reset_index(drop=True)
    d1 = d1.merge(dataset1_splits[["pid", "split_fold_1", "split_fold_2", "split_fold_3"]],
                  on="pid", how="left")

    # Dataset 2 (same patients but possibly larger set)
    d2 = df.merge(dataset1_splits[["pid", "split_fold_1", "split_fold_2", "split_fold_3"]],
                  on="pid", how="left")

    return d1, d1_target, d2, df_target

def make_splits(d1, d1_target, d2, df_target, test_fraction=0.2, seed=22):
    """Create consistent train/test splits for d1 and d2."""
    # D1 train/test
    d1_train = d1[d1["split_fold_1"] != "test"].reset_index(drop=True)
    d1_train_target = d1_target[d1_target["pid"].isin(d1_train["pid"])].reset_index(drop=True)
    d1_test = d1[d1["split_fold_1"] == "test"].reset_index(drop=True)
    d1_test_target = d1_target[d1_target["pid"].isin(d1_test["pid"])].reset_index(drop=True)

    # Ensure consistency in D2 test split
    d1_train_pids = d1_train["pid"].unique().tolist()
    d1_test_pids = d1_test["pid"].unique().tolist()
    total_rows_d2 = len(d2)
    test_size_d2 = int(total_rows_d2 * test_fraction)

    # Start with d1 test pids
    test_pids = d1_test_pids[:]

    remaining_d2_pids = d2[~d2["pid"].isin(d1_test_pids + d1_train_pids)]["pid"].unique().tolist()
    remaining_d2_df = d2[d2["pid"].isin(remaining_d2_pids)].reset_index(drop=True)

    test_rows_needed = max(0, test_size_d2 - len(test_pids))
    if test_rows_needed > 0 and len(remaining_d2_df) > 0:
        _, additional_test_pids = train_test_split(
            remaining_d2_df["pid"].unique(),
            test_size=test_rows_needed,
            stratify=remaining_d2_df["dss_5y"],
            random_state=seed
        )
        test_pids.extend(additional_test_pids.tolist())

    d2["split"] = "train_val"
    d2.loc[d2["pid"].isin(test_pids), "split"] = "test"

    d2_train = d2[d2["split"] != "test"].reset_index(drop=True)
    d2_train_target = df_target[df_target["pid"].isin(d2_train["pid"])].reset_index(drop=True)
    d2_test = d2[d2["split"] == "test"].reset_index(drop=True)
    d2_test_target = df_target[df_target["pid"].isin(d2_test["pid"])].reset_index(drop=True)

    # Useful extra splits
    d2_test_not_in_d1 = d2_test[~d2_test["pid"].isin(d1["pid"])].reset_index(drop=True)
    d2_test_not_in_d1_target = d2_test_target[~d2_test_target["pid"].isin(d1_target["pid"])].reset_index(drop=True)
    d2_not_in_d1 = d2[~d2["pid"].isin(d1["pid"])].reset_index(drop=True)
    d2_not_in_d1_target = df_target[~df_target["pid"].isin(d1["pid"])].reset_index(drop=True)

    return (d1_train, d1_train_target, d1_test, d1_test_target,
            d2_train, d2_train_target, d2_test, d2_test_target,
            d2_test_not_in_d1, d2_test_not_in_d1_target,
            d2_not_in_d1, d2_not_in_d1_target)


def get_final_estimator(model):
    """Unwrap GridSearchCV and Pipeline to get the actual model."""
    # If GridSearchCV, take best_estimator_
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
    # If Pipeline, take the last step
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]  # get the final estimator
    return model

def explain_model(model, X_train, y_train, X_test, y_test, save_dir="explain_outputs", model_name="model", key='id'):
    """
    Explain a trained model (XGBoost, Lasso Logistic Regression, or SVM).
    Produces plots + feature importance CSV at dataset level.

    Args:
        model: trained sklearn/xgboost model (or pipeline)
        X_train, y_train, X_test, y_test: data
        save_dir: folder to save outputs
        model_name: label to use in file names
    """
    os.makedirs(save_dir, exist_ok=True)
    feature_names = X_train.columns
    model = get_final_estimator(model)

    print(f"Explaining model: {model_name}")
    print(str(type(model)))

    # --- Case 1: Lasso Logistic Regression ---
    if "logreg" in model_name.lower():
        print("Explaining Lasso Logistic Regression...")
        try:
            explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_train)

            # SHAP summary bar plot
            shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
            plt.title(f"{key}_{model_name} - LASSO LOGREG SHAP (Bar)")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{key}_{model_name}_lasso_logreg_shap_bar.png"))
            plt.close()

            # SHAP beeswarm
            shap.summary_plot(shap_values, X_train, show=False)
            plt.title(f"{key}_{model_name} - LASSO LOGREG SHAP (Beeswarm)")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{key}_{model_name}_lasso_logreg_shap_beeswarm.png"))
            plt.close()

            # Global importance (mean |SHAP|)
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": np.abs(shap_values).mean(0)
            }).sort_values("importance", ascending=False)

            importance_df.to_csv(os.path.join(save_dir, f"{key}_{model_name}_lasso_logreg_importance.csv"), index=False)
        except Exception as e:
            print(f"Error creating explainer: {e}")


        coefs = model.coef_.flatten()
        coef_df = pd.DataFrame({"feature": feature_names, "importance": coefs})
        coef_df["abs_importance"] = np.abs(coef_df["importance"])
        coef_df = coef_df.sort_values("abs_importance", ascending=False)

        # Plot
        plt.figure(figsize=(8, 6))
        coef_df.head(20).plot(x="feature", y="importance", kind="barh", legend=False)
        plt.title(f"{key}_{model_name} - Lasso Coefficients")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{key}_{model_name}_lasso_coeffs.png"))
        plt.close()

        # Save CSV
        coef_df.to_csv(os.path.join(save_dir, f"{key}_{model_name}_lasso_coeffs.csv"), index=False)
        return coef_df

    # --- Case 2: XGBoost ---
    elif "xgboost" in model_name.lower():
        print("Explaining XGBoost...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # SHAP summary bar plot
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        plt.title(f"{key}_{model_name} - XGB SHAP (Bar)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{key}_{model_name}_xgb_shap_bar.png"))
        plt.close()

        # SHAP beeswarm
        shap.summary_plot(shap_values, X_train, show=False)
        plt.title(f"{key}_{model_name} - XGB SHAP (Beeswarm)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{key}_{model_name}_xgb_shap_beeswarm.png"))
        plt.close()

        # Global importance (mean |SHAP|)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(shap_values).mean(0)
        }).sort_values("importance", ascending=False)

        importance_df.to_csv(os.path.join(save_dir, f"{key}_{model_name}_xgb_importance.csv"), index=False)
        return importance_df

    # --- Case 3: SVM ---
    elif "svm" in model_name.lower():
        print("Explaining SVM with permutation importance...")

        try:
            #explainer = shap.KernelExplainer(model.predict, X_train)
            explainer = shap.KernelExplainer(model.predict_proba, X_train)

            shap_values = explainer.shap_values(X_train)

            # SHAP summary bar plot
            shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
            plt.title(f"{key}_{model_name} - SVM SHAP (Bar)")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{key}_{model_name}_svm_shap_bar.png"))
            plt.close()

            # SHAP beeswarm
            shap.summary_plot(shap_values, X_train, show=False)
            plt.title(f"{key}_{model_name} - SVM SHAP (Beeswarm)")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{key}_{model_name}_svm_shap_beeswarm.png"))
            plt.close()

            # Global importance (mean |SHAP|)
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": np.abs(shap_values).mean(0)
            }).sort_values("importance", ascending=False)

            importance_df.to_csv(os.path.join(save_dir, f"{key}_{model_name}_svm_importance.csv"), index=False)
        except Exception as e:
            print(f"Error creating explainer: {e}")

        r = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42)
        perm_df = pd.DataFrame({
            "feature": feature_names,
            "importance": r.importances_mean
        }).sort_values("importance", ascending=False)

        # Plot
        plt.figure(figsize=(8, 6))
        perm_df.head(20).plot(x="feature", y="importance", kind="barh", legend=False)
        plt.title(f"{key}_{model_name} - SVM Permutation Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{key}_{model_name}_svm_perm.png"))
        plt.close()

        # Save CSV
        perm_df.to_csv(os.path.join(save_dir, f"{key}_{model_name}_svm_perm.csv"), index=False)
        return perm_df

    else:
        raise ValueError("Model type not recognized! Supported: Lasso Logistic Regression, XGBoost, SVM.")


# --- Utility: Save/load models ---
def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def train_best_model(best_models, path_features_map, path_target, dataset1_splits_path, custom_folds=None):
    """
    Train and return the best model(s) defined in best_models dict.

    Args:
        best_models (dict): Dictionary like
            { 'pd_d1_train': { 'time_set': 'preclinical', 
                               'model': 'xgboost',
                               'feature_selection': 'union',
                               'd_train': 'd1' } }
        path_features_map (dict): Maps time_set → path_features
            e.g. { 'preclinical': '/path/preclinical.csv',
                   'after_diagnosis': '/path/after_diag.csv',
                   'final_clinical': '/path/final.csv' }
        path_target (str): Path to target CSV.
        dataset1_splits_path (str): Path to dataset1 splits CSV.
        custom_folds: Optional predefined CV folds.

    Returns:
        dict: { name: fitted_model }
    """
    trained_models = {}



    for name, best_cfg in best_models.items():
        print(f"\n=== Training best model: {name} ===")
        # 1. Load correct feature file based on time_set
        path_features = path_features_map[best_cfg['time_set']]
        time_set = best_cfg["time_set"]
        model_name = best_cfg["model"]
        d_train_choice = best_cfg["d_train"]

        d1, d1_target, d2, df_target = load_datasets(path_features, path_target, dataset1_splits_path)
        (d1_train, d1_train_target, d1_test, d1_test_target,
         d2_train, d2_train_target, d2_test, d2_test_target,
         d2_test_not_in_d1, d2_test_not_in_d1_target,
         d2_not_in_d1, d2_not_in_d1_target) = make_splits(d1, d1_target, d2, df_target)
        
        # Custom CV folds for consistency
        custom_folds = [(d1_train[d1_train[col] == "train"].index,
                        d1_train[d1_train[col] == "val"].index)
                        for col in ["split_fold_1", "split_fold_2", "split_fold_3"]]

        
        if d_train_choice == "d1":
            df_train = d1_train
        elif d_train_choice == "d2":
            df_train = d2_train
        else:
            raise ValueError(f"Unknown dataset: {d_train_choice}")

        clin_fold_name = time_set

        imaging_feature_sets = {
        "fold1": pd.read_csv(r'/nas-ctm01/homes/fmferreira/AI4LUNGS/extracted_embeddings_results/fold_0_embeddings.csv'),
        "fold2": pd.read_csv(r'/nas-ctm01/homes/fmferreira/AI4LUNGS/extracted_embeddings_results/fold_1_embeddings.csv'),
        "fold3": pd.read_csv(r'/nas-ctm01/homes/fmferreira/AI4LUNGS/extracted_embeddings_results/fold_2_embeddings.csv')
        }

        test_probs = pd.DataFrame({"pid": d1_test["pid"]})

        for img_fold_name,  img_df in imaging_feature_sets.items():
            combo_name = f"{clin_fold_name}_x_{img_fold_name}"
            print(f"\n=== Processing {combo_name} ===")

            # ------------------ Merge clinical + imaging ------------------
            X_train = pd.merge(d1_train[['pid']], d1_train, on='pid')
            X_train = pd.merge(X_train, img_df, on='pid')
            y_train = d1_train['dss_5y']


            X_test_d1 = pd.merge(d1_test[['pid']], d1_test, on='pid', how='left')
            X_test_d1 = pd.merge(X_test_d1, img_df, on='pid', how='left')

            # print the number of rows in each set
            print(f"Training set size: {X_train.shape[0]} samples")
            print(f"Test set size: {X_test_d1.shape[0]} samples")

            # Drop pid column
            for df in [X_train, X_test_d1]:
                df.drop(columns=['pid'], inplace=True)

            X_all = drop_unwanted_columns(X_train)
            y_all = df_train["dss_5y"]

            X_test_d1 = drop_unwanted_columns(X_test_d1)
            y_test_d1 = d1_test["dss_5y"]

            model_func = MODELS[model_name]
            best_model = model_func(X_all, y_all, custom_folds=custom_folds)
            save_model(best_model, f"{name}_best_model.pkl")

            # Save the probabilities of the test set df with
            test_probs[f'prob_{combo_name}'] = best_model.predict_proba(X_test_d1)[:, 1]
            print(f"Saved test probabilities for {combo_name}")


            # Explain it
            results_df = explain_model(best_model, X_all, y_all, X_test_d1, y_test_d1,
                                    save_dir=r"/nas-ctm01/homes/fmferreira/AI4LUNGS/multimodal/intermediatefusion/interpretability", model_name=best_cfg['model'], key=f"{name}_{img_fold_name}")

        # Save the test probabilities to CSV
        test_probs.to_csv(f"{name}_test_probabilities.csv", index=False)        

        trained_models[name] = {
            "fitted_model": best_model,
            "train_set": best_cfg['d_train'],
            "time_set": best_cfg['time_set']
        }

        print(f"✅ Trained {best_cfg['model']} on {best_cfg['d_train']} ({best_cfg['time_set']})")

    return trained_models



# ================== Example Usage ==================
if __name__ == "__main__":
    best_models = {
        # 'pd_d1_train': {
        #     'time_set': 'preclinical',
        #     'model': 'xgboost',
        #     'feature_selection': 'union',
        #     'd_train': 'd1'
        # },
        # 'pd_d2_train': {
        #     'time_set': 'preclinical',
        #     'model': 'lasso_logreg',
        #     'feature_selection': 'cox_uni',
        #     'd_train': 'd2'
        # },
        # 'ad_d1_train': {
        #     'time_set': 'after_diagnosis',
        #     'model': 'xgboost',
        #     'feature_selection': 'vif',
        #     'd_train': 'd1'
        # },
        # 'ad_d2_train': {
        #     'time_set': 'after_diagnosis',
        #     'model': 'lasso_logreg',
        #     'feature_selection': 'vif',
        #     'd_train': 'd2'
        # },
        # 'ad_svm_l1': {
        #     'time_set': 'after_diagnosis',
        #     'model': 'svm_l1',
        #     'd_train': 'd1'
        # },
        # 'fc_lasso_logreg': {
        #     'time_set': 'final_clinical',
        #     'model': 'lasso_logreg',
        #     'd_train': 'd1'
        # },
        'pc_svm_l1': {
            'time_set': 'preclinical',
            'model': 'xgboost',
            'd_train': 'd1'
        },
    }

    path_features_map = {
        'preclinical': r'/nas-ctm01/homes/fmferreira/AI4LUNGS/NLST_preclinical_final_processed_normalized.csv',
        'after_diagnosis': r'/nas-ctm01/homes/fmferreira/AI4LUNGS/NLST_after_diagnosis_final_processed_normalized.csv',
        'final_clinical': r'/nas-ctm01/homes/fmferreira/AI4LUNGS/NLST_clinical_final_processed_normalized.csv'
    }
    path_target = r'/nas-ctm01/homes/fmferreira/AI4LUNGS/NLST_clinical_final_processed_outcomes.csv'
    dataset1_splits_path = r'/nas-ctm01/homes/fmferreira/AI4LUNGS/lung_metadata_with_splits.csv'

    trained = train_best_model(best_models, path_features_map, path_target, dataset1_splits_path)

    # # Example: Access fitted model
    # my_model = trained['pd_d1_train']['fitted_model']
    # my_features = trained['pd_d1_train']['features']
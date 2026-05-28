import os
import numpy as np
import pandas as pd
import warnings
import pickle

# Survival Modeling Frameworks (All Curve-Compatible)
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Validation Metrics
from sksurv.metrics import cumulative_dynamic_auc, brier_score, concordance_index_censored
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Suppress warnings
warnings.filterwarnings('ignore')

# ================== DATA PREPARATION UTILITY ==================

def prepare_survival_target(df_main, df_target):
    """
    Formats the target properties into a structured array 
    explicitly required by scikit-survival objects.
    """
    merged = df_main[['pid', 'label']].merge(df_target[['pid', 'fup_days']], on='pid', how='inner')
    merged['label'] = merged['label'].astype(bool)
    merged['fup_days'] = merged['fup_days'].clip(lower=1)  # Survival times must be > 0
    
    y_surv = np.empty(len(merged), dtype=[('event', 'bool'), ('time', 'float')])
    y_surv['event'] = merged['label'].values
    y_surv['time'] = merged['fup_days'].values
    return y_surv

# ================== SURVIVAL MODEL FITTERS ==================

def fit_rsf(X, y_surv, custom_folds):
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomSurvivalForest(random_state=42, n_jobs=-1))
    ])
    param_grid = {
        'model__n_estimators': [100, 200], 
        'model__max_depth': [3, 5, 7]
    }
    gs = GridSearchCV(pipe, param_grid, cv=custom_folds, n_jobs=-1)
    return gs.fit(X, y_surv)

def fit_gradient_boosting_survival(X, y_surv, custom_folds):
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', GradientBoostingSurvivalAnalysis(loss='coxph', random_state=42))
    ])
    param_grid = {
        'model__n_estimators': [100, 200], 
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5]
    }
    gs = GridSearchCV(pipe, param_grid, cv=custom_folds, n_jobs=-1)
    return gs.fit(X, y_surv)

def fit_cox_ph(X, y_surv, custom_folds):
    # Standard scaling is critical for stable linear partial-likelihood convergence
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', CoxnetSurvivalAnalysis(l1_ratio=0.5, fit_baseline_model=True))  # Unpenalized baseline Cox
    ])
    param_grid = {
        'model__alphas': [[0.1], [0.05], [0.01], [0.005], [0.001]]
    }
    gs = GridSearchCV(pipe, param_grid, cv=custom_folds, n_jobs=-1)
    return gs.fit(X, y_surv)

def fit_cox_net(X, y_surv, custom_folds):
    # fit_baseline_model=True is REQUIRED to extract survival probability curves later
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', CoxnetSurvivalAnalysis(l1_ratio=0.5, fit_baseline_model=True))
    ])
    param_grid = {
        'model__n_alphas': [50, 100]
    }
    gs = GridSearchCV(pipe, param_grid, cv=custom_folds, n_jobs=-1)
    return gs.fit(X, y_surv)

SURVIVAL_MODELS = {
    "random_survival_forest": fit_rsf,
    "gradient_boosting_survival": fit_gradient_boosting_survival,
    "cox_ph": fit_cox_ph,
    "cox_net": fit_cox_net
}

# ================== LONGITUDINAL CURVE EXPORTER ==================

def save_survival_curves(model, X_test, test_pids, output_path):
    """
    Extracts and exports full step curves over longitudinal time intervals.
    """
    try:
        survival_funcs = model.predict_survival_function(X_test)
        time_points = survival_funcs[0].x
        curves_data = {"time": time_points}
        
        for i, pid in enumerate(test_pids):
            curves_data[f"pid_{pid}"] = survival_funcs[i].y
            
        curves_df = pd.DataFrame(curves_data)
        curves_df.to_csv(output_path, index=False)
        print(f"📈 Full longitudinal curves saved to: {output_path}")
    except AttributeError:
        print("⚠️ This model type does not natively generate step survival curve functions.")

# ================== MAIN PROCESSING PIPELINE ==================

def main(folder=r'/nas-ctm01/homes/fmferreira/AI4LUNGS/results/clinical_models/preclinical/regression'):
    path_features = r'/nas-ctm01/homes/fmferreira/AI4LUNGS/data/NLST_preclinical_final_processed_normalized.csv'
    path_target = r'/nas-ctm01/homes/fmferreira/AI4LUNGS/data/NLST_clinical_final_processed_outcomes.csv'
    splits_path = r'/nas-ctm01/homes/fmferreira/AI4LUNGS/data/lung_metadata_with_splits.csv'
    os.makedirs(folder, exist_ok=True)

    # Load Core Source Dataframes
    features_df = pd.read_csv(path_features)
    meta_df = pd.read_csv(splits_path, usecols=['pid', 'label', 'split_fold_1', 'split_fold_2', 'split_fold_3'])
    df_target = pd.read_csv(path_target)

    # Merge splits data with features, dropping row items containing structural NaNs
    d1 = features_df.merge(meta_df, on="pid").dropna()
    
    # Establish feature lists by excluding target and tracking columns
    drop_meta = ["pid", "split_fold_1", "split_fold_2", "split_fold_3", "label"]
    features = [c for c in d1.columns if c not in drop_meta]

    # Milestone target configuration for time-dependent evaluations (5 years = 1825 days)
    target_time_5y = np.array([1825.0])
    target_years = np.linspace(365.25,1825, 5)


    # OUTER LOOP: Train and isolate folds independently to enforce complete leakage containment
    for fold_col in ["split_fold_1", "split_fold_2", "split_fold_3"]:
        print(f"\n=================== EXECUTING OUTER LOOP: {fold_col.upper()} ===================")
        
        # Segment arrays based strictly on the current fold tracking context
        train_df = d1[d1[fold_col] == "train"].reset_index(drop=True)
        val_df = d1[d1[fold_col] == "val"].reset_index(drop=True)
        test_df = d1[d1[fold_col] == "test"].reset_index(drop=True)
        
        # Combine train/validation blocks into a single cohort for internal cross-validation mapping
        d1_train = pd.concat([train_df, val_df]).reset_index(drop=True)
        
        # Extract direct explicit indices pointing to the internal splits
        train_idx = d1_train[d1_train[fold_col] == "train"].index.tolist()
        val_idx = d1_train[d1_train[fold_col] == "val"].index.tolist()
        custom_folds = [(train_idx, val_idx)]

        # Generate target structures matching split indexes
        y_surv_train = prepare_survival_target(d1_train, df_target)
        y_surv_test = prepare_survival_target(test_df, df_target)
        test_pids = test_df["pid"].values

        all_results = []
        best_c_index = -1
        best_model_obj = None
        best_model_name = ""

        # Sub-loop traversing different model architectures
        for model_name, fit_func in SURVIVAL_MODELS.items():
            print(f"⏱️ Training {model_name}...")
            try:
                gs = fit_func(d1_train[features], y_surv_train, custom_folds)
                X_test_eval = test_df[features]

                # Predict standardized continuous risk scores (Higher score = Higher Risk)
                risk_scores = gs.best_estimator_.predict(X_test_eval)
                
                # 1. Global Concordance Index via native sksurv function
                c_index = concordance_index_censored(y_surv_test['event'], y_surv_test['time'], risk_scores)[0]
                
                # 2. Time-Dependent AUC (Evaluated at precisely Day 1825)
                va_auc, _ = cumulative_dynamic_auc(y_surv_train, y_surv_test, risk_scores, target_years)
                td_auc_5y = np.mean(va_auc)

                # 3. Time-Dependent Brier Score & True Calibrated Probabilities
                surv_funcs = gs.best_estimator_.predict_survival_function(X_test_eval)
                prob_survival_5y = np.array([fn(1825.0) for fn in surv_funcs])
                
                _, b_score = brier_score(y_surv_train, y_surv_test, prob_survival_5y, target_time_5y)
                brier_score_5y = b_score[0]

                # ------------------------------------------------------------------
                # FIXED-POINT 5-YEAR EVALUATION BLOCK (Accuracy, Precision, Recall)
                # ------------------------------------------------------------------
                true_event_before_5y = y_surv_test['event'] & (y_surv_test['time'] <= 1825.0)
                true_survived_past_5y = y_surv_test['time'] > 1825.0
                evaluable_mask = true_event_before_5y | true_survived_past_5y
                
                acc_5y, prec_5y, rec_5y, static_auc_5y = np.nan, np.nan, np.nan, np.nan
                
                if evaluable_mask.sum() > 0:
                    y_true_5y = true_event_before_5y[evaluable_mask].astype(int)
                    y_prob_5y = 1.0 - prob_survival_5y[evaluable_mask] # Probability of event occurring before 5y
                    y_pred_5y = (y_prob_5y >= 0.5).astype(int)
                    
                    acc_5y = accuracy_score(y_true_5y, y_pred_5y)
                    prec_5y = precision_score(y_true_5y, y_pred_5y, zero_division=0)
                    rec_5y = recall_score(y_true_5y, y_pred_5y, zero_division=0)
                    try:
                        static_auc_5y = roc_auc_score(y_true_5y, y_prob_5y)
                    except ValueError:
                        static_auc_5y = 0.5

                # Pack metrics for summary tables
                result_entry = {
                    "model": model_name,
                    "c_index": c_index,
                    "td_auc_5y": td_auc_5y,
                    "brier_score_5y": brier_score_5y,
                    "accuracy_5y": acc_5y,
                    "precision_5y": prec_5y,
                    "recall_5y": rec_5y,
                    "static_auc_5y": static_auc_5y,
                    "roc_auc":static_auc_5y,
                    "best_params": gs.best_params_
                }
                all_results.append(result_entry)

                # Keep track of the winner for the late fusion export phase
                if c_index > best_c_index:
                    best_c_index = c_index
                    best_model_obj = gs.best_estimator_
                    best_model_name = model_name

            except Exception as e:
                print(f"❌ Failed processing for model {model_name}: {e}")

        # Save Metrics Summary Ranking Sheet for the fold block
        summary = pd.DataFrame(all_results).sort_values(by="c_index", ascending=False)
        summary.to_csv(os.path.join(folder, f"TRUE_SURVIVAL_RANKING_{fold_col}.csv"), index=False)
        print(f"\n--- Results Summary for {fold_col.upper()} ---")
        columns_to_show = ["model", "c_index", "td_auc_5y", "accuracy_5y", "precision_5y", "recall_5y", "brier_score_5y", "roc_auc"]
        print(summary[columns_to_show].to_string(index=False))

        # ================== EXPORT ADVANCED LATE FUSION FIELD MATRICES ==================
        if best_model_obj is not None:
            print(f"🥇 Winner Identified for {fold_col}: {best_model_name} (C-Index: {best_c_index:.4f})")
            X_test_final = test_df[features]

            # Standardized final risks and true calibrated probabilities
            final_risk = best_model_obj.predict(X_test_final)
            surv_funcs_final = best_model_obj.predict_survival_function(X_test_final)
            prob_alive_5y = np.array([fn(1825.0) for fn in surv_funcs_final])
            
            # Save full step curves file for downstream continuous curve blending
            curves_path = os.path.join(folder, f"best_survival_curves_{fold_col}.csv")
            save_survival_curves(best_model_obj, X_test_final, test_pids, curves_path)

            # Re-compute mask to flag evaluable patients for late fusion steps
            true_event_before_5y = y_surv_test['event'] & (y_surv_test['time'] <= 1825.0)
            true_survived_past_5y = y_surv_test['time'] > 1825.0
            evaluable_mask = true_event_before_5y | true_survived_past_5y

            # --- NEW: EXTRACT SEMI/NON-PARAMETRIC UNDERLYING CURVE COEFFICIENTS ---
            # This replaces the Weibull (shape, scale) parameters for these architectures
            underlying_model = best_model_obj.named_steps['model']
            
            # Base data container for coefficients
            fusion_export_df = pd.DataFrame({
                'pid': test_pids,
                'clinical_risk': final_risk,               # Acts like the linear predictor/hazard scale
                'clinical_surv_prob_5y': prob_alive_5y,    # Aligned 5-year mark probability
                'time': y_surv_test['time'],
                'event': y_surv_test['event'].astype(int),
                'evaluable_at_5y': evaluable_mask.astype(int)
            })

            # Append model-specific mathematical curve parameters
            if best_model_name in ['cox_ph', 'cox_net']:
                # For Cox models, survival profile = baseline_survival(t) ^ exp(risk)
                # We save the individual baseline hazard multiplier for absolute alignment
                baseline_hazard_ratio = np.exp(final_risk)
                fusion_export_df['baseline_hazard_multiplier'] = baseline_hazard_ratio
                
                # Export the unique baseline survival curve steps to a secondary file for reconstruction
                if hasattr(underlying_model, "baseline_survival_"):
                    base_surv = underlying_model.baseline_survival_
                    base_df = pd.DataFrame({'time': base_surv.x, 'baseline_survival': base_surv.y})
                    base_df.to_csv(os.path.join(folder, f"baseline_survival_steps_{fold_col}.csv"), index=False)

            elif best_model_name in ['random_survival_forest', 'gradient_boosting_survival']:
                # Tree-based models map directly to non-linear risk pools.
                # We export the predicted cumulative hazard at the 5-year mark as an alternative parameter
                cum_haz_funcs = best_model_obj.predict_cumulative_hazard_function(X_test_final)
                cum_haz_5y = np.array([fn(1825.0) for fn in cum_haz_funcs])
                fusion_export_df['cum_hazard_param_5y'] = cum_haz_5y

            # Export aligned parameters required for downstream decision-level Late Fusion scripts
            export_path = os.path.join(folder, f"best_clinical_fusion_parameters_{fold_col}.csv")
            fusion_export_df.to_csv(export_path, index=False)
            print(f"📦 Advanced Fusion Parametric Matrices exported to: {export_path}")
            
            # Export optimal trained serialized object (Crucial: contains full tree geometries/baselines)
            with open(os.path.join(folder, f"best_survival_model_{fold_col}.pkl"), "wb") as f:
                pickle.dump(best_model_obj, f)
        else:
            print(f"❌ Structural training failure: No winning model recorded for fold {fold_col}.")
if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

# Switch metric library import
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
warnings.filterwarnings('ignore')

# ================== LATE FUSION ENGINE ==================

def execute_late_fusion(clinical_df, imaging_df, time_milestone_days=1825.0, time_normalization_factor=2786.0, weight_imaging=0.5):
    """
    Merges clinical and imaging metrics to perform late decision-level fusion.
    weight_imaging = 1.0 reproduces the standalone imaging ensemble c-index exactly.
    """
    # Align data matrices on unique patient identifiers
    fusion_df = pd.merge(clinical_df, imaging_df, on='pid', suffixes=('_clin', '_img'))
    if fusion_df.empty:
        return None, None

    weight_clinical = 1.0 - weight_imaging

    # --- 1. BLEND RISK SCORES FOR RANKING METRICS (C-INDEX) ---
    # imaging_prob_5y is the true per-fold-averaged death probability — use it directly
    # as the imaging risk score. Weibull hazard from averaged params degrades ranking
    # because averaging params ≠ averaging hazards (Jensen's inequality).
    fusion_df['clin_risk_norm'] = (fusion_df['clinical_risk'] - fusion_df['clinical_risk'].mean()) / (fusion_df['clinical_risk'].std() + 1e-8)
    fusion_df['img_risk_norm'] = (fusion_df['imaging_prob_5y'] - fusion_df['imaging_prob_5y'].mean()) / (fusion_df['imaging_prob_5y'].std() + 1e-8)

    # Higher score = higher risk = earlier death
    fusion_df['fused_risk_score'] = (weight_clinical * fusion_df['clin_risk_norm']) + \
                                    (weight_imaging * fusion_df['img_risk_norm'])

    # --- 2. BLEND PROBABILITIES OF DEATH ---
    clinical_death_prob_5y = 1.0 - fusion_df['clinical_surv_prob_5y']
    imaging_death_prob_5y = fusion_df['imaging_prob_5y'].values

    fusion_df['fused_death_prob_5y'] = (weight_clinical * clinical_death_prob_5y) + \
                                       (weight_imaging * imaging_death_prob_5y)
    
    y_events = fusion_df['event'].astype(int)
    y_times_days = fusion_df['time'].astype(float)

    # lifelines c-index expects: higher score = longer life; flip sign since our score is risk
    c_index_fused = concordance_index(
        event_times=y_times_days,
        predicted_scores=-fusion_df['fused_risk_score'].values,
        event_observed=y_events
    )

    # sksurv structured array required by cumulative_dynamic_auc
    y_surv = np.array(
        list(zip(y_events.astype(bool), y_times_days.astype(float))),
        dtype=[('event', bool), ('time', float)]
    )
    target_time_arr = np.array([time_milestone_days])
    try:
        va_auc, _ = cumulative_dynamic_auc(
            y_surv,
            y_surv,
            fusion_df['fused_risk_score'].values,
            target_time_arr
        )
        td_auc_5y = float(va_auc[0])
    except Exception as e:
        td_auc_5y = np.nan
        print(f"Warning calculating td-AUC: {e}")

    # Evaluable cohort: event before 5y OR survived past 5y (exclude early censored)
    true_event_before_5y = (y_events == 1) & (y_times_days <= time_milestone_days)
    true_survived_past_5y = (y_times_days > time_milestone_days)
    evaluable_mask = true_event_before_5y | true_survived_past_5y

    y_true_eval = true_event_before_5y[evaluable_mask].astype(int)
    prob_eval = fusion_df['fused_death_prob_5y'].values[evaluable_mask]
    pred_eval = (prob_eval >= 0.50).astype(int)

    metrics = {
        "cohort_size": len(fusion_df),
        "evaluable_size": int(evaluable_mask.sum()),
        "c_index_fused": c_index_fused,
        "auc_fused": td_auc_5y,
        "brier_fused": brier_score_loss(y_true_eval, prob_eval),
        "accuracy_fused": accuracy_score(y_true_eval, pred_eval)
    }
    return fusion_df, metrics

# ================== MAIN SWEEP PROCESSOR ==================

def main():
    clinical_results_dir = r'/nas-ctm01/homes/fmferreira/AI4LUNGS/results/clinical_models/full_information/regression'
    imaging_path = r'/nas-ctm01/homes/fmferreira/AI4LUNGS/results/Imaging/final_Imaging_results.csv'
    output_fusion_dir = os.path.join(clinical_results_dir, "late_fusion_outputs")
    os.makedirs(output_fusion_dir, exist_ok=True)
    
    clinical_avg_path = os.path.join(clinical_results_dir, "final_clinical_results.csv")
    if not os.path.exists(clinical_avg_path) or not os.path.exists(imaging_path):
        print("❌ Error: Missing files. Run your aggregation script first.")
        return
        
    clinical_df = pd.read_csv(clinical_avg_path)
    imaging_df = pd.read_csv(imaging_path)
    
    clinical_df['pid'] = clinical_df['pid'].astype(int)
    imaging_df['pid'] = imaging_df['pid'].astype(int)
    
    TARGET_DAYS = 1825.0 
    IMAGING_TIME_NORMALIZATION_FACTOR = 2786 
    
    imaging_weights = np.linspace(0.0, 1.0, 11)
    all_sweep_records = []
    
    print(f"🚀 Sweeping weights from 0 to 1 at 0.1 increments via Lifelines...")
    for w_img in imaging_weights:
        fused_df, metrics = execute_late_fusion(
            clinical_df=clinical_df,
            imaging_df=imaging_df,
            time_milestone_days=TARGET_DAYS,
            time_normalization_factor=IMAGING_TIME_NORMALIZATION_FACTOR,
            weight_imaging=w_img
        )
        
        if metrics:
            metrics["weight_imaging"] = round(w_img, 2)
            metrics["weight_clinical"] = round(1.0 - w_img, 2)
            all_sweep_records.append(metrics)

    if all_sweep_records:
        ranking_df = pd.DataFrame(all_sweep_records)
        ranking_df = ranking_df.sort_values(by="c_index_fused", ascending=False)
        
        ranking_df.to_csv(os.path.join(output_fusion_dir, "LATE_FUSION_WEIGHT_OPTIMIZATION_RANKING.csv"), index=False)
        
        print("\n=================== 📊 GLOBAL WEIGHT OPTIMIZATION RANKING (LIFELINES) ===================")
        print("Columns: [Weight Img | Weight Clin] -> Fused Target Metrics Output")
        columns_to_print = ['weight_imaging', 'weight_clinical', 'c_index_fused', 'auc_fused', 'brier_fused', 'accuracy_fused']
        print(ranking_df[columns_to_print].to_string(index=False))
        
        best_w = ranking_df.iloc[0]
        print("\n=============================================================================")
        print(f"🥇 OPTIMAL CONFIGURATION: Imaging Weight = {best_w['weight_imaging']} | Clinical Weight = {best_w['weight_clinical']}")
        print(f"🏆 Best Performance: C-Index = {best_w['c_index_fused']:.4f} | AUC = {best_w['auc_fused']:.4f}")
        print("=============================================================================")

if __name__ == "__main__":
    main()
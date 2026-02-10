import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.metrics import roc_curve
import pandas as pd

df =np.read_csv('test_results_fold_final.csv')

def calculate_optimal_threshold(y_true, y_prob):
    """Optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    J = tpr - fpr
    return thresholds[np.argmax(J)]


threshold = calculate_optimal_threshold(df["Actual_Event"], df["Surv_Prob_5y"])
print(f"Optimal threshold (Youden's J): {threshold:.4f}")

df["risk_group"] = df["Surv_Prob_5y"].apply(
    lambda x: "High Risk" if x >= threshold else "Low Risk"
)
print("\n--- Risk Group Distribution ---")
print(df["risk_group"].value_counts())

# Kaplan–Meier curves
kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 6))

for group, color in zip(["Low Risk", "High Risk"], ["#009999", "#CC0066"]):
    mask = df["risk_group"] == group
    kmf.fit(df.loc[mask, "fup_days"], 
            event_observed=df.loc[mask, "event"], 
            label=group)
    kmf.plot_survival_function(ci_show=False, color=color, linewidth=2)

plt.title("Kaplan-Meier Survival Curves by Model-Predicted Risk")
plt.xlabel("Time (days)")
plt.ylabel("Survival probability")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
# if save_path:
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

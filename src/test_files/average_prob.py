import pandas as pd
import glob
import os

def aggregate_clinical_files(file_list, output_path="/nas-ctm01/homes/fmferreira/AI4LUNGS/results/clinical_models/full_information/regression/final_clinical_results.csv"):
    all_dfs = []

    for file in file_list:
        if os.path.exists(file):
            print(f"Reading {file}...")
            all_dfs.append(pd.read_csv(file))
        else:
            print(f"⚠️ Warning: {file} not found.")

    if not all_dfs:
        print("No data found to aggregate.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)

    final_df = combined_df.groupby('pid').agg({
        'clinical_surv_prob_5y': 'mean',
        'time': 'first',
        'event': 'first',
        'clinical_risk': 'mean'
    }).reset_index()

    final_df.to_csv(output_path, index=False)
    print(f"\n✅ Successfully aggregated {len(file_list)} clinical files.")
    print(f"Total unique patients: {len(final_df)}")

    return final_df


def aggregate_imaging_files(file_list, output_path="/nas-ctm01/homes/fmferreira/AI4LUNGS/results/Imaging/final_Imaging_results.csv"):
    all_dfs = []

    for file in file_list:
        if os.path.exists(file):
            print(f"Reading {file}...")
            all_dfs.append(pd.read_csv(file))
        else:
            print(f"⚠️ Warning: {file} not found.")

    if not all_dfs:
        print("No data found to aggregate.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)

    final_df = combined_df.groupby('pid').agg({
        'imaging_prob_5y': 'mean',
        'fup_days': 'first',
        'true_event': 'first',
        'weibull_param_1': 'mean',
        'weibull_param_2': 'mean'
    }).reset_index()

    final_df.to_csv(output_path, index=False)
    print(f"\n✅ Successfully aggregated {len(file_list)} imaging files.")
    print(f"Total unique patients: {len(final_df)}")

    return final_df


# --- EXECUTION ---
clinical_files = [
    "/nas-ctm01/homes/fmferreira/AI4LUNGS/results/clinical_models/full_information/regression/best_clinical_fusion_parameters_split_fold_1.csv",
    "/nas-ctm01/homes/fmferreira/AI4LUNGS/results/clinical_models/full_information/regression/best_clinical_fusion_parameters_split_fold_2.csv",
    "/nas-ctm01/homes/fmferreira/AI4LUNGS/results/clinical_models/full_information/regression/best_clinical_fusion_parameters_split_fold_3.csv"
]

imaging_files = [
    "/nas-ctm01/homes/fmferreira/AI4LUNGS/results/Imaging/test_weibull_parameters_fold_1.csv",
    "/nas-ctm01/homes/fmferreira/AI4LUNGS/results/Imaging/test_weibull_parameters_fold_2.csv",
    "/nas-ctm01/homes/fmferreira/AI4LUNGS/results/Imaging/test_weibull_parameters_fold_3.csv"
]

aggregate_clinical_files(clinical_files)
aggregate_imaging_files(imaging_files)
import pandas as pd
import glob
import os

def aggregate_multiple_files(file_list, output_path="/nas-ctm01/homes/fmferreira/AI4LUNGS/results/clinical_models/full_information/regression/final_clinical_results.csv"):
    all_dfs = []
    
    # 1. Load and collect all files
    for file in file_list:
        if os.path.exists(file):
            print(f"Reading {file}...")
            all_dfs.append(pd.read_csv(file))
        else:
            print(f"⚠️ Warning: {file} not found.")

    if not all_dfs:
        print("No data found to aggregate.")
        return None

    # 2. Combine all files into one master table
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 3. Aggregate: Average probability per PID, keep the label
    # This works whether a PID appears in one file or all three
    final_df = combined_df.groupby('pid').agg({
        'clinical_surv_prob_5y': 'mean',
        'time': 'first',
        'event': 'first',
        'clinical_risk': 'mean'
    }).reset_index()

    # 4. Save and return
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ Successfully aggregated {len(file_list)} files.")
    print(f"Total unique patients: {len(final_df)}")
    
    return final_df

# --- EXECUTION ---
# List your three files here
files_to_combine = [                      
    "/nas-ctm01/homes/fmferreira/AI4LUNGS/results/clinical_models/full_information/regression/best_clinical_fusion_parameters_split_fold_1.csv",
    "/nas-ctm01/homes/fmferreira/AI4LUNGS/results/clinical_models/full_information/regression/best_clinical_fusion_parameters_split_fold_1.csv",
    "/nas-ctm01/homes/fmferreira/AI4LUNGS/results/clinical_models/full_information/regression/best_clinical_fusion_parameters_split_fold_1.csv"
]

# If your files follow a pattern, you could also use:
# files_to_combine = glob.glob("imaging_results_fold_*.csv")

result_df = aggregate_multiple_files(files_to_combine)
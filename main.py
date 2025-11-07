import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read data from CSV file in desktop csv folder csv

person_data = pd.read_csv('C:/Users/franc/Desktop/csv/nlst_prsn_idc.csv')
cancer_data = pd.read_csv('C:/Users/franc/Desktop/csv/nlst_canc_idc.csv')
ct_data = pd.read_csv('C:/Users/franc/Desktop/csv/nlst_screen_idc.csv')

#combine 3 csv files into one dataframe based on common column 'person_id'
merged_data = pd.merge(person_data, cancer_data, on='pid', how='inner')
merged_data = pd.merge(merged_data, ct_data, on='pid', how='inner')   
print(merged_data.head())
#save merged data to new csv file
merged_data.to_csv('C:/Users/franc/Desktop/csv/merged_nlst_data.csv', index=False)
print("Merged data saved to 'merged_nlst_data.csv'")

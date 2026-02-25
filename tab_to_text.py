import pandas as pd
import csv
dict_preclinical = {
    'age': 'Patient is {} years old. ',
    'educat': 'Patient has a {} diploma. ',
    'height': ' Height of {} cm. ',
    'weight': 'Patient weights {} kg. ',
    'cigar': 'Patient smokes cigar. ',
    'cigsmok': 'Patient smokes ciggaretes. ',
    'pipe': 'Patient smokes pipe. ',
    'pkyr': 'Patient has a history of {} pack years ',
    'smokeday': 'Patient smokes {} cigarettes a day. ',
    'smokelive': 'Patient lives with a smoker. ',
    'smokework': 'Patient works with exposure to smokers. ',
    'smokeyr': 'Patient smoked for a total of {} years. ',
    'lesionsize': 'Patient has a lesion size of {} mm. ',
    'loccar': 'Patient has the main cancer located in the Carina. ',
    'loclhil': 'Patient has the main cancer located in the left hilum. ',
    'loclin': 'Patient has the main cancer located in the Lingula. ',
    'locllow': 'Patient has the main cancer located in the left lower lobe. ',
    'loclmsb': 'Patient has the main cancer located in the left main stem bronchus. ',
    'loclup': 'Patient has the main cancer located in the left upper lobe. ',
    'locmed': 'Patient has the main cancer located in the Mediastinum. ',
    'locoth': 'Patient has the main cancer located in other location. ',
    'locrhil': 'Patient has the main cancer located in the right Hilum. ',
    'locrlow': 'Patient has the main cancer located in the right lower lobe. ',
    'locrmid': 'Patient has the main cancer located in the right middle lobe. ',
    'locrmsb': 'Patient has the main cancer located in the right main stem bronchus. ',
    'locrup': 'Patient has the main cancer located in the right upper lobe. ',
    'locunk': 'Patient has the main cancer located in unknown location ',
    'wrkasbe': 'Patient worked in asbestos environment. ',
    'wrkbaki': 'Patient worked in baking environment. ',
    'wrkbutc': 'Patient worked in butchering environment. ',
    'wrkchem': 'Patient worked in chemical environment. ',
    'wrkcoal': 'Patient worked in coal environment. ',
    'wrkcott': 'Patient worked in cotton environment. ',
    'wrkfarm': 'Patient worked in farm environment. ',
    'wrkfire': 'Patient worked in fire fighting environment. ',
    'wrkflou': 'Patient worked in flour environment. ',
    'wrkfoun': 'Patient worked in steel milling environment. ',
    'wrkhard': 'Patient worked in hard rock mining environment. ',
    'wrkpain': 'Patient worked in painting environment. ',
    'wrksand': 'Patient worked in sandblasting environment. ',
    'wrkweld': 'Patient worked in welding environment. ',
    'diagadas': 'Patient has adult asthma. ',
    'diagasbe': 'Patient has asbestosis. ',
    'diagbron': 'Patient has bronchiectasis. ',
    'diagchas': 'Patient has childhood asthma. ',
    'diagchro': 'Patient has chronic bronchitis. ',
    'diagcopd': 'Patient has COPD. ',
    'diagdiab': 'Patient has diabetes. ',
    'diagemph': 'Patient has emphysema. ',
    'diagfibr': 'Patient has fibrosis of the lung. ',
    'diaghear': 'Patient has heart disease. ',
    'diaghype': 'Patient has Hypertension. ',
    'diagpneu': 'Patient has pneumonia. ',
    'diagsarc': 'Patient has sarcoidosis. ',
    'diagsili': 'Patient has silicosis. ',
    'diagstro': 'Patient has Stroke history. ',
    'diagtube': 'Patient has tuberculosis. ',
    'cancblad': 'Patient diagnosed with bladder cancer. ',
    'cancbrea': 'Patient diagnosed with breast cancer. ',
    'canccerv': 'Patient diagnosed with cervical cancer. ',
    'canccolo': 'Patient diagnosed with colorectal cancer. ',
    'cancesop': 'Patient diagnosed with esophageal cancer. ',
    'canckidn': 'Patient diagnosed with kidney cancer. ',
    'canclary': 'Patient diagnosed with larynx cancer. ',
    'canclung': 'Patient diagnosed with lung cancer. ',
    'cancnasa': 'Patient diagnosed with nasal cancer. ',
    'cancoral': 'Patient diagnosed with oral cancer. ',
    'cancstom': 'Patient diagnosed with stomach cancer. ',
    'cancthyr': 'Patient diagnosed with thyroid cancer. ',
    'canctran': 'Patient diagnosed with transitional cell cancer. ',
    'fambrother': 'brother ',
    'famchild': 'child ',
    'famfather': 'father ',
    'fammother': 'mother ',
    'famsister': 'sister ',
    'previous_canc': 'History of previous cancer. ',
    'previous_diseases': 'History of previous disease. ',
    'family_history': 'Family history of cancer. ',
    'bmi': 'Patient has BMI of {}. ',
    'work_related': 'Patient has work related cancer. ',
    'gender_1': 'Male. ',
    'gender_2': 'Female. ',
    'race_1': 'White. ',
    'race_2': 'African-American. ',
    'race_3': 'Asian. ',
    'race_4': 'American Indian. ',
    'race_5': 'Native Hawaiian. ',
    'race_6': 'More than one race. ',
    'race_99': 'Unknown. ',
}

dict_after_diagnosis = {
    'de_grade' : 'Lung cancer grade of {}. ',
    'de_stag_7thed' : 'Lung cancer stage of {}. ',
    'num_confirmed' : '{} num of confirmed cancers. ',
    'clinical_m_7thed': 'AJCC staging clinical M component of {}. ',
    'clinical_n_7thed': 'AJCC staging clinical N component of {}. ',
    'clinical_t_7thed': 'AJCC staging clinical T component of {}. ',
    'lc_behav': 'ICD-O-3 behavior of lung cancer of {}. ',
    'path_m_7thed': 'AJCC 7th edition staging path M component of {}. ',
    'path_n_7thed': 'AJCC 7th edition staging path N component of {}. ',
    'path_t_7thed': 'AJCC 7th edition staging path T component of {}. ',
    'valcsg': 'VALGCSG stage of {}. ',
    'de_type_8000.0': '8000.0 ',
    'de_type_8010.0': '8010.0 ',
    'de_type_8012.0': '8012.0 ',
    'de_type_8013.0': '8013.0 ',
    'de_type_8041.0': '8041.0 ',
    'de_type_8046.0': '8046.0 ',
    'de_type_8070.0': '8070.0 ',
    'de_type_8071.0': '8071.0 ',
    'de_type_8140.0': '8140.0 ',
    'de_type_8246.0': '8246.0 ',
    'de_type_8250.0': '8250.0 ',
    'de_type_8252.0': '8252.0 ',
    'de_type_8260.0': '8260.0 ',
    'de_type_8550.0': '8550.0 ',
    'de_type_8560.0': '8560.0 ',
    'de_type_Missing': 'Missing ',
    'de_type_Other': 'Other ',
    'lc_morph_8000': '8000 ',
    'lc_morph_8010': '8010 ',
    'lc_morph_8012': '8012 ',
    'lc_morph_8013': '8013 ',
    'lc_morph_8041': '8041 ',
    'lc_morph_8042': '8042 ',
    'lc_morph_8046': '8046 ',
    'lc_morph_8070': '8070 ',
    'lc_morph_8071': '8071 ',
    'lc_morph_8140': '8140 ',
    'lc_morph_8246': '8246 ',
    'lc_morph_8250': '8250 ',
    'lc_morph_8252': '8252 ',
    'lc_morph_8550': '8550 ',
    'lc_morph_8560': '8560 ',
    'lc_morph_Other': 'Other ',
    'lc_topog_C34.0': 'C34.0 ',
    'lc_topog_C34.1': 'C34.1 ',
    'lc_topog_C34.2': 'C34.2 ',
    'lc_topog_C34.3': 'C34.3 ',
    'lc_topog_C34.8': 'C34.8 ',
    'lc_topog_C34.9': 'C34.9 ',
    'lc_topog_C38.3': 'C38.3 ',


    
}


dict_remaining = {
    'invaslc': 'Had invasive lung cancer precedure. ',
    'medcomplc': 'Had medical complications of lung cancer. ',
    'progressed_ever': 'Had lung cancer progression. ',
    'progsite_adrenal_ever': 'Progression to adrenal glands. ',
    'progsite_bone_ever': 'Progression to bone. ',
    'progsite_brain_ever': 'Progression to brain. ',
    'progsite_liver_ever': 'Progression to liver. ',
    'progsite_lymph_n1_ever': 'Progression to N1 lymph nodes. ',
    'progsite_lymph_n2_ever': 'Progression to N2 lymph nodes. ',
    'progsite_lymph_n3_ever': 'Progression to N3 lymph nodes. ',
    'progsite_mediastinum_ever': 'Progression to mediastinum. ',
    'progsite_orig_lung_ever': 'Progression in original lung. ',
    'progsite_other_ever': 'Progression to other sites. ',
    'progsite_other_lung_ever': 'Progression to other lung. ',
    'progsite_pleura_ever': 'Progression to pleura. ',
    'progsite_skin_ever': 'Progression to skin. ',
    'progsite_unk_ever': 'Progression to unknown site. ',
    'can_link': 'Record can be linked to lung cancer. ',
    'compcat': 'Complication category: {}. ',
    'comp_code_1': 'Complication code 1. ',
    'comp_code_2': 'Complication code 2. ',
    'comp_code_3': 'Complication code 3. ',
    'comp_code_5': 'Complication code 5. ',
    'comp_code_6': 'Complication code 6. ',
    'comp_code_7': 'Complication code 7. ',
    'comp_code_8': 'Complication code 8. ',
    'comp_code_9': 'Complication code 9. ',
    'comp_code_10': 'Complication code 10. ',
    'comp_code_11': 'Complication code 11. ',
    'comp_code_14': 'Complication code 14. ',
    'comp_code_16': 'Complication code 16. ',
    'comp_code_17': 'Complication code 17. ',
    'comp_code_21': 'Complication code 21. ',
    'comp_code_22': 'Complication code 22. ',
    'comp_code_23': 'Complication code 23. ',
    'comp_code_25': 'Complication code 25. ',
    'comp_code_26': 'Complication code 26. ',
    'comp_code_28': 'Complication code 28. ',
    'comp_code_29': 'Complication code 29. ',
    'comp_code_30': 'Complication code 30. ',
    'comp_code_31': 'Complication code 31. ',
    'comp_code_32': 'Complication code 32. ',
    'comp_code_33': 'Complication code 33. ',
    'comp_code_34': 'Complication code 34. ',
    'comp_code_35': 'Complication code 35. ',
    'comp_code_36': 'Complication code 36. ',
    'comp_code_37': 'Complication code 37. ',
    'comp_code_40': 'Complication code 40. ',
    'comp_code_41': 'Complication code 41. ',
    'comp_code_42': 'Complication code 42. ',
    'comp_code_45': 'Complication code 45. ',
    'comp_code_47': 'Complication code 47. ',
    'comp_code_48': 'Complication code 48. ',
    'comp_code_50': 'Complication code 50. ',
    'comp_code_52': 'Complication code 52. ',
    'comp_code_54': 'Complication code 54. ',
    'comp_code_55': 'Complication code 55. ',
    'comp_code_56': 'Complication code 56. ',
    'treatnum_101': 'Treatment code 101. ',
    'treatnum_102': 'Treatment code 102. ',
    'treatnum_103': 'Treatment code 103. ',
    'treatnum_104': 'Treatment code 104. ',
    'treatnum_188': 'Treatment code 188. ',
    'treatnum_199': 'Treatment code 199. ',
    'treatnum_201': 'Treatment code 201. ',
    'treatnum_202': 'Treatment code 202. ',
    'treatnum_203': 'Treatment code 203. ',
    'treatnum_204': 'Treatment code 204. ',
    'treatnum_205': 'Treatment code 205. ',
    'treatnum_206': 'Treatment code 206. ',
    'treatnum_207': 'Treatment code 207. ',
    'treatnum_208': 'Treatment code 208. ',
    'treatnum_209': 'Treatment code 209. ',
    'treatnum_210': 'Treatment code 210. ',
    'treatnum_211': 'Treatment code 211. ',
    'treatnum_212': 'Treatment code 212. ',
    'treatnum_213': 'Treatment code 213. ',
    'treatnum_214': 'Treatment code 214. ',
    'treatnum_215': 'Treatment code 215. ',
    'treatnum_216': 'Treatment code 216. ',
    'treatnum_288': 'Treatment code 288. ',
    'treatnum_299': 'Treatment code 299. ',
    'treatnum_401': 'Treatment code 401. ',
    'treatnum_402': 'Treatment code 402. ',
    'treatnum_406': 'Treatment code 406. ',
    'treatnum_488': 'Treatment code 488. ',
    'treat_1': 'Treatment type 1. ',
    'treat_2': 'Treatment type 2. ',
    'treat_3': 'Treatment type 3. ',
    'treat_4': 'Treatment type 4. '
}

type_keys=['de_type_8000.0',	'de_type_8010.0',	'de_type_8012.0',	'de_type_8013.0',	'de_type_8041.0',	'de_type_8046.0',	'de_type_8070.0',	'de_type_8071.0',	'de_type_8140.0',	'de_type_8246.0',	'de_type_8250.0',	'de_type_8252.0',	'de_type_8260.0',	'de_type_8550.0',	'de_type_8560.0',	'de_type_Missing',	'de_type_Other']

morphology_keys=['lc_morph_8000',	'lc_morph_8010',	'lc_morph_8012',	'lc_morph_8013',	'lc_morph_8041',	'lc_morph_8042',	'lc_morph_8046',	'lc_morph_8070',	'lc_morph_8071',	'lc_morph_8140',	'lc_morph_8246',	'lc_morph_8250',	'lc_morph_8252',	'lc_morph_8550',	'lc_morph_8560',	'lc_morph_Other']

topology_keys=['lc_topog_C34.0',	'lc_topog_C34.1',	'lc_topog_C34.2',	'lc_topog_C34.3',	'lc_topog_C34.8',	'lc_topog_C34.9',	'lc_topog_C38.3']

binary_keys_preclinical=['cigar','cigsmok','pipe','smokelive','smokework','work_related','loccar','loclhil','loclin','locllow','loclmsb','loclup',
             'locmed','locoth','locrhil','locrlow','locrmid','locrmsb','locrup','locunk','wrkasbe','wrkbaki','wrkbutc','wrkchem','wrkcoal','wrkcott','wrkfarm','wrkfire','wrkflou',
             'wrkfoun','wrkhard','wrkpain','wrksand','wrkweld','wrkcoal','diagadas','diagasbe','diagbron','diagchas','diagchro','diagcopd',
             'diagdiab','diagemph','diagfibr','diaghear','diaghype','diagpneu','diagsarc','diagsili','diagstro','diagtube','cancblad',
             'cancbrea','canccerv','canccolo','cancesop','canckidn','canclary','canclung','cancnasa','cancoral','cancpanc','cancphar',
             'cancstom','cancthyr','canctran','previous_canc','previous_disease',
             'family_history','gender_1','gender_2','race_1','race_2','race_3','race_4',
             'race_5','race_6','race_99']


family_keys_preclinical=['fambrother', 'famchild','famfather', 'fammother','famsister']




final_binary_keys= list(dict_remaining.keys())
print(type(final_binary_keys))
final_binary_keys.remove('compcat')




preclinical=r'/nas-ctm01/homes/mipaiva/clinical_models/NLST_preclinical_final_processed_normalized.csv'
results_preclinical=[]
with open(preclinical, mode='r') as f:
    reader = csv.DictReader(f) 
    i=0
    for row in reader:

        pid= row.get('pid')
        string_family=''
        string_binary=''
        string_values=''
        global_string=''
        
        for key, value in row.items():
            value = str(value).strip() 

            if key not in dict_preclinical:
                continue
            try:
                numeric_val = float(value)
            except ValueError:
                numeric_val = None
            if 'fam' in key.lower():
                print(f"DEBUG: Found key {key} with value '{value}'")
            if key in family_keys_preclinical and numeric_val == 1.0:
                if string_family:
                    string_family += 'and '
                string_family+= dict_preclinical[key]

            if key in binary_keys_preclinical and numeric_val==1.0:
                string_binary+= dict_preclinical[key]

            if key in dict_preclinical.keys() and not(key in binary_keys_preclinical) and not(key in family_keys_preclinical):
                    string_values+= dict_preclinical[key].format(value)
        
        if string_family:
            string_family += 'have lung cancer. '
        global_string = string_values + string_binary + string_family
        list=[pid,global_string]
        results_preclinical.append({'pid': pid, 'preclinical': global_string})
        
        # Optional: only print every 100 rows to save console power
        print(f"ID {pid}: {global_string}")

# Create DataFrame once at the end
df = pd.DataFrame(results_preclinical)

afterdiagnosis= '/nas-ctm01/homes/mipaiva/clinical_models/NLST_after_diagnosis_final_processed_normalized.csv'
results_ad =[]
with open(afterdiagnosis, mode='r') as f:
    reader = csv.DictReader(f) 
    i=0
    for row in reader:

        pid= row.get('pid')
        string_type=''
        string_morphology=''
        string_topology=''
        global_string_ad=''
        string_values=''
        
        for key, value in row.items():
            value = str(value).strip() 

            if key not in dict_after_diagnosis:
                continue
            try:
                numeric_val = float(value)
            except ValueError:
                numeric_val = None

            if key in type_keys and numeric_val == 1.0:
                if string_type:
                    string_type += 'and '
                string_type+= dict_after_diagnosis[key]
            if key in morphology_keys and numeric_val == 1.0:
                if string_morphology:
                    string_morphology += 'and '
                string_morphology+= dict_after_diagnosis[key]
            if key in topology_keys and numeric_val == 1.0:
                if string_topology:
                    string_topology += 'and '
                string_topology+= dict_after_diagnosis[key]


            if key in dict_after_diagnosis.keys() and not(key in type_keys) and not(key in morphology_keys) and not(key in topology_keys):
                    string_values+= dict_after_diagnosis[key].format(value)

        
        if string_type:
            string_type += 'type of cancer. '
        if string_morphology:
            string_morphology += 'type of morphology. '
        if string_topology:
            string_topology += 'type of topology. '
        global_string_ad = string_values + string_type + string_morphology+ string_topology
        list=[pid,global_string_ad]
        results_ad.append({'pid': pid, 'afterdiagnosis': global_string_ad})
        
        # Optional: only print every 100 rows to save console power
        print(f"ID {pid}: {global_string_ad}")
# Create DataFrame once at the end
df_after_diagnosis = pd.DataFrame(results_ad)

# 2. Merge on Patient ID
# This ensures data matches even if the CSV rows are in different orders
df_ad = pd.merge(df, df_after_diagnosis, on='pid', how='inner')

# 3. Combine strings into the 'afterdiagnosis' column
# We add a space " " in between so the sentences don't touch
df_ad['afterdiagnosis'] = df_ad['preclinical'] + " " + df_ad['afterdiagnosis']


# len(dict_after_diagnosis) +
print(  len(dict_after_diagnosis) +len(dict_preclinical) + len(dict_remaining))

final= '/nas-ctm01/homes/mipaiva/clinical_models/NLST_clinical_final_processed_normalized.csv'
results_final =[]
with open(final, mode='r') as f:
    reader = csv.DictReader(f) 
    i=0
    for row in reader:

        pid= row.get('pid')
        string_values=''
        string_binary=''
        
        for key, value in row.items():
            value = str(value).strip() 

            if key not in dict_remaining:
                continue
            try:
                numeric_val = float(value)
            except ValueError:
                numeric_val = None

            if key in final_binary_keys and numeric_val==1.0:
                string_binary+= dict_remaining[key]

            if key in dict_remaining.keys() and not(key in final_binary_keys):
                    string_values+= dict_remaining[key].format(value)

        global_string_final = string_values + string_binary
        list=[pid,global_string_final]
        results_final.append({'pid': pid, 'full_information': global_string_final})
        
        # Optional: only print every 100 rows to save console power
        print(f"ID {pid}: {global_string_final}")
# Create DataFrame once at the end
df_final = pd.DataFrame(results_final)

# 2. Merge on Patient ID
# This ensures data matches even if the CSV rows are in different orders
df_real = pd.merge(df_ad, df_final, on='pid', how='inner')

# 3. Combine strings into the 'afterdiagnosis' column
# We add a space " " in between so the sentences don't touch
df_real['full_information'] = df_real['afterdiagnosis'] + " " + df_real['full_information']

# 5. Save the result
df_real.to_csv('/nas-ctm01/homes/fmferreira/AI4LUNGS/text_data.csv', index=False)



df = pd.read_csv('/nas-ctm01/homes/fmferreira/AI4LUNGS/text_data.csv')
df_filter= pd.read_csv('/nas-ctm01/homes/fmferreira/AI4LUNGS/clinical_metadata_with_splits.csv')
df=df[df['pid'].isin(df_filter['pid'])]

df.to_csv('/nas-ctm01/homes/fmferreira/AI4LUNGS/text_data.csv', index=False)
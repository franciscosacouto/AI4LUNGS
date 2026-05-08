import pandas as pd
import csv
dict_preclinical = {
    'age': 'Patient is {} years old, ',
    'height': 'Patient measures {} inches. ',
    'weight': 'Patient weights {} pounds. ',
    'cigar': 'Patient smokes cigar. ',
    'cigsmok': 'Patient smokes ciggaretes. ',
    'pipe': 'Patient smokes pipe. ',
    'pkyr': 'Patient has a history of {} pack years, ',
    'smokeday': ' smokes {} cigarettes a day. ',
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
    'educat': '',
    'gender':'',
    'race': '',
}


dict_education = {
    '1': '8th grade or less. ',
    '2': '9th-11th grade. ',
    '3': 'High school graduate/GED. ',
    '4': 'Post high school training (non-college). ',
    '5': 'Associate degree or some college. ',
    '6': 'Bachelors Degree. ',
    '7': 'Graduate School. ',
    '8': '',
    '95':'',
    '98': '',
    '99':'',
}

dict_gender={
    '1': 'Male. ',
    '2':'Female. '
}
dict_race={
    '1': 'White. ',
    '2': 'African-American. ',
    '3': 'Asian. ',
    '4': 'American Indian. ',
    '5': 'Native Hawaiian. ',
    '6': 'More than one race. ',
    '7': '',
    '95': '',
    '96': '',
    '98': '',
    '99': ''
}

dict_after_diagnosis = {
    'de_grade' : 'Lung cancer grade of {}. ',
    'de_stag_7thed' : '',
    'num_confirmed' : 'Has {} confirmed cancers. ',
    'lc_behav': '',
    'path_m_7thed': '',
    'path_n_7thed': '',
    'path_t_7thed': '',
    'valcsg': 'VALGCSG stage of {}. ',
    'de_type_8000.0':'Neoplasm',
    'de_type_8010.0':'Epithelial tumor',
    'de_type_8012.0':'Large Cell Carcinom',
    'de_type_8013.0':'Large cell neuroendocrine carcinoma',
    'de_type_8041.0':'Small cell carcinoma',
    'de_type_8042.0':'Oat cell carcinoma',
    'de_type_8046.0':'Non-small cell carcinoma',
    'de_type_8070.0': 'Squamous cell carcinoma',
    'de_type_8071.0':'Squamous cell carcinoma, keratinizing',
    'de_type_8140.0':'Adenoma',
    'de_type_8246.0':'Neuroendocrine carcinoma',
    'de_type_8250.0':'Pulmonary adenomatosis',
    'de_type_8252.0':'Bronchiolo-alveolar carcinoma',
    'de_type_8260.0': 'Papillary adenoma',
    'de_type_8550.0':'Acinar cell carcinoma',
    'de_type_8560.0':'Adenosquamous carcinoma',
    'lc_morph_8000':'Neoplasm',
    'lc_morph_8010':'Epithelial tumor',
    'lc_morph_8012':'Large Cell Carcinoma',
    'lc_morph_8013':'Large cell neuroendocrine carcinoma',
    'lc_morph_8041':'Small cell carcinoma',
    'lc_morph_8042':'Oat cell carcinoma',
    'lc_morph_8046':'Non-small cell carcinoma',
    'lc_morph_8070': 'Squamous cell carcinoma',
    'lc_morph_8071':'Squamous cell carcinoma, keratinizing',
    'lc_morph_8140':'Adenoma',
    'lc_morph_8246':'Neuroendocrine carcinoma',
    'lc_morph_8250':'Pulmonary adenomatosis',
    'lc_morph_8252':'Bronchiolo-alveolar carcinoma',
    'lc_morph_8260': 'Papillary adenoma',
    'lc_morph_8550':'Acinar cell carcinoma',
    'lc_morph_8560':'Adenosquamous carcinoma',
    'lc_morph_Other':'',
    'lc_morph_Missing': '',
    'lc_topog_C34.0': 'Location: Main bronchus. ', 
    'lc_topog_C34.1': 'Location: Upper lobe. ',
    'lc_topog_C34.2': 'Location: Middle lobe. ',
    'lc_topog_C34.3': 'Location: Lower lobe.',
    'lc_topog_C34.8': 'Location: Overlapping lesion of lung. ',
    'lc_topog_C34.9': 'Location: Lung. ',
    'lc_topog_C38.3': 'Location: Mediatinum. ',
}


dict_de_stag = {
    '0.0': "Stage IA. ",
    '0.1428571428571428': "Stage IB. ",
    '0.2857142857142857': "Stage IIA. ",
    '0.4285714285714285': "Stage IIB. ",
    '0.5714285714285714': "Stage IIIA. ",
    '0.7142857142857143': "Stage IIIB. ",
    '0.8571428571428571': "Stage IV. ",
    '1.0': "Occult Carcinoma. ",
}

dict_lc_behav = {
    '0.0': 'Borderline Malignancy. ',
    '0.4': 'Invasive. ',
    '1.0': 'Metastic. '

}
dict_m_7thed= {
    '0.0': 'zero metastasis',
    '0.25': 'Metastasis cannot be measured. ',
    '0.5': 'Cancer as not spread to other parts of the body. ',
    '0.75': 'Cancer has spread to other parts of the body. ',
    '1.0': 'Missing metastasis. '
}
dict_n_7thed= {
    '0.0': 'Cancer in nearby limph nodes cannot be measured. ',
    '0.25': 'There is no cancer in nearby lymph nodes. ',
    '0.5': 'Lymph nodes contain cancer. ',
    '0.75': 'Missing',
    '1.0':'nao sei'
}
dict_t_7thed= {
    '0.0': 'Main tumor cannot be measured',
    '0.1428571428571428': 'Main tumor cannot be found. ',
    '0.2857142857142857': 'Tumor has tissue',
    '0.4285714285714285': 'Small tumor was found. ',
    '0.5714285714285714': 'medium tumor was found. ',
    '0.7142857142857143': 'medium-large tumor was found. ',
    '0.8571428571428571': 'big tumor was found. ',
    '1.0': 'Missing tumor',

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
    'compcat': '',
    'comp_code_1': 'Acute respiratory failure complication. ',
    'comp_code_2': 'Allergic reactio complication. ',
    'comp_code_3': 'Anaphylaxis complication. ',
    'comp_code_5': 'Blood loss with transfusion complication. ',
    'comp_code_6': 'Bronchopulmonary fistula complication. ',
    'comp_code_7': 'Bronchospasm complication. ',
    'comp_code_8': 'Cardiac arrest complication. ',
    'comp_code_9': 'Cardiac arrhythmia requiring medical attention complication. ',
    'comp_code_10': 'Cerebral vascular accident complication. ',
    'comp_code_11': ' Congestive heart failure (CHF) complication. ',
    'comp_code_14': 'Fever requiring antibiotics complication. ',
    'comp_code_16': ' Hemothorax requiring tube placement complication. ',
    'comp_code_17': 'Hospitalization post procedure complication. ',
    'comp_code_21': ' Myocardial infarction complication. ',
    'comp_code_22': 'Pain requiring referral to a pain specialist complication. ',
    'comp_code_23': 'Pneumothorax requiring tube placement. ',
    'comp_code_25': ' Respiratory arrest complication. ',
    'comp_code_26': 'Rib fracture(s) complication. ',
    'comp_code_28': 'Vocal cord immobility/paralysis complication. ',
    'comp_code_29': ' Bronchial stump leak requiring tube thoracostomy complication. ',
    'comp_code_30': 'Empyema complication. ',
    'comp_code_31': ' Injury to vital organ or vessel complication. ',
    'comp_code_32': 'Prolonged mechanical ventilation over 48 hours post-operatively complication. ',
    'comp_code_33': 'Thromboembolic complications requiring intervention complication. ',
    'comp_code_34': 'Vaso-vagal reaction/Hypotension complication. ',
    'comp_code_35': 'other complication. ',
    'comp_code_36': 'Wound infection complication. ',
    'comp_code_37': 'Infections requiring antibiotics complication. ',
    'comp_code_40': 'Subcutaneous emphysema complication. ',
    'comp_code_41': 'Atelectasis complication. ',
    'comp_code_42': 'Pneumothorax with no chest tube complication. ',
    'comp_code_45': 'Chylous fistula complication. ',
    'comp_code_47': 'Pneumonia complication. ',
    'comp_code_48': 'Seroma complication. ',
    'comp_code_50': 'Pleural effusion complication. ',
    'comp_code_52': 'Sepsis complication. ',
    'comp_code_54': ' Splenomegaly with splenic infarcts complication. ',
    'comp_code_55': 'Parasthesias/Hypersthesias complication. ',
    'comp_code_56': 'Mucous plug requiring . ',
    'treatnum_101': 'Radiation of Primary Chest Tumor and/or Regional Nodes treatment. ',
    'treatnum_102': 'Radiation of Hilar/Mediastinal Lymph Nodes treatment. ',
    'treatnum_103': 'Radiation of Prophylactic Brain treatment. ',
    'treatnum_104': 'Radiation of Therapeutic Brain treatment. ',
    'treatnum_188': 'Radiation (other specify) treatment. ',
    'treatnum_199': 'Radiation of Unknown Site treatment. ',
    'treatnum_201': 'Exploratory Thoracotomy without Resection treatment. ',
    'treatnum_202': 'Median Sternotomy treatment. ',
    'treatnum_203': 'Lobectomy treatment. ',
    'treatnum_204': 'Bilobectomy treatment. ',
    'treatnum_205': ' Pneumonectomy treatment. ',
    'treatnum_206': 'Wedge Resection treatment. ',
    'treatnum_207': 'Segmental Resection treatment. ',
    'treatnum_208': 'Lymphadenectomy/Lymph Node Sampling treatment. ',
    'treatnum_209': 'Chest Wall Resection treatment. ',
    'treatnum_210': 'Thoracentesis treatment. ',
    'treatnum_211': ' Partial Pleurectomy treatment. ',
    'treatnum_212': 'Multiple Wedge Resections treatment. ',
    'treatnum_213': 'Multiple Segmental Resections treatment. ',
    'treatnum_214': 'Thoracotomy treatment. ',
    'treatnum_215': 'Thoracoscopy (VATS) treatment. ',
    'treatnum_216': 'Thoracoscopy (VATS) with conversion to Thoracotomy treatment. ',
    'treatnum_288': 'Surgical procedure/approach (other specify) treatment. ',
    'treatnum_299': 'Unknown Surgical procedure/approach treatment. ',
    'treatnum_401': 'Immune Therapy treatment. ',
    'treatnum_402': 'Radiofrequency Ablation treatment. ',
    'treatnum_406': 'Brachytherapy treatment. ',
    'treatnum_488': 'Other Treatment (other specify). ',
    'treat_1': 'Radiation Treatment. ',
    'treat_2': 'Surgical Treatment. ',
    'treat_3': 'Systemic Chemotherapy. ',
    'treat_4': 'Other treatment. '
}
dict_complication={
    '0.0': '',
    '0.3333333333333333':'Major .',
    '0.6666666666666666':'Intermediate. ',
    '1.0': 'Minor. ',
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
             'family_history']


family_keys_preclinical=['fambrother', 'famchild','famfather', 'fammother','famsister']




final_binary_keys= list(dict_remaining.keys())
print(type(final_binary_keys))
final_binary_keys.remove('compcat')




preclinical=r'/nas-ctm01/homes/fmferreira/AI4LUNGS/participant.data.d100517.csv'
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
                    if key== 'weight' and not(value == ''):
                        weight= int(value)
                    if key == 'height' and not(value == ''):
                        height = int(value)
            if key == 'race':
                string_values+= dict_race[value].format(value)
            if key == 'gender':
                string_values+= dict_gender[value].format(value)
                print('\n gender')
            if key == 'educat':
                print('\n educat')
                string_values+= dict_education[value].format(value)

        bmi= weight/(height*height)*703
        bmi=round(bmi,2)
        if string_family:
            string_family += 'have lung cancer. '
        global_string = string_values + string_binary + string_family
        global_string = global_string + f'Patient has a BMI of {bmi}.'
        list=[pid,global_string]
        results_preclinical.append({'pid': pid, 'preclinical': global_string})
        
        # Optional: only print every 100 rows to save console power
        print(f"ID {pid}: {global_string}")

# Create DataFrame once at the end
df = pd.DataFrame(results_preclinical)
df.to_csv('/nas-ctm01/homes/fmferreira/AI4LUNGS/text_data_new.csv', index=False)

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
            if key == 'de_stag_7thed':
                string_values += dict_de_stag[value].format(value)
            if key == 'path_n_7thed':
                string_values += dict_n_7thed[value].format(value)
            if key == 'path_t_7thed':
                string_values += dict_t_7thed[value].format(value)
            if key == 'path_m_7thed':
                string_values += dict_m_7thed[value].format(value)
            if key== 'lc_behav':
                string_values += dict_lc_behav[value].format(value)
        
        if not(string_type==''):
            string_type += ' type of cancer. '
        if not(string_morphology==''):
            string_morphology += ' type of morphology. '

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
df_ad['afterdiagnosis'] =  df_ad['afterdiagnosis'] + " " + df_ad['preclinical'] 


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

            if key == 'compcat':
                string_values+= dict_complication[value].format(value)


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
df_real['full_information'] = df_real['full_information'] + " " + df_real['afterdiagnosis']

# 5. Save the result
df_real.to_csv('/nas-ctm01/homes/fmferreira/AI4LUNGS/text_data.csv', index=False)



df = pd.read_csv('/nas-ctm01/homes/fmferreira/AI4LUNGS/text_data.csv')
df_filter= pd.read_csv('/nas-ctm01/homes/fmferreira/AI4LUNGS/clinical_metadata_with_splits.csv')
df=df[df['pid'].isin(df_filter['pid'])]

df.to_csv('/nas-ctm01/homes/fmferreira/AI4LUNGS/text_data_new.csv', index=False)
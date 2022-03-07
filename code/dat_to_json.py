# %%
import os
from glob import glob
import re
import json
from wfdb.io import rdrecord
from tqdm.contrib.concurrent import process_map
import pandas as pd
import multiprocessing as mp
import numpy as np
from functools import partial
# %%
def read_dat_to_dict(file_nm):
    
    data = rdrecord(file_nm).__dict__
    
    return data

def get_data_info(dict_data, df_table):
    
    sig_data = dict(zip(dict_data['sig_name'], [dict_data['p_signal'][:, i].tolist() for i in range(len(dict_data['sig_name']))]))
    
    data = {"ID":float(df_table['ID'].values[0]),
            "Age_group":float(df_table['Age_group'].values[0]),
            "Sex":float(df_table['Sex'].values[0]),
            "BMI":float(df_table['BMI'].values[0]),
            "Length":float(df_table['Length'].values[0]),
            "Device":float(df_table['Device'].values[0]),
            "Samp_rate":float(dict_data['fs']),
            **sig_data}
    
    return data
    

def dat_to_json(dir_path, save_path, df_table, subjects):
    
    subjects = subjects.tolist()
    
    for subject in subjects:
        
        dict_data = read_dat_to_dict(os.path.join(dir_path, subject))
        
        subject= int(subject)
        table_selected = df_table.query("ID == @subject", engine='python')
        
        data = get_data_info(dict_data, table_selected)
        
        ## Save dictionary as json
        with open(os.path.join(save_path, "subject_" + str(subject) + ".json"), 'w') as fp:
            json.dump(data, fp, indent=4)

def make_dataset(dir_path, save_path, df_table):
    PATTERN = r"\s*[/](?P<subject>\d*)[.]dat\s*"
    file_list = glob(dir_path+"/*.dat")
    
    subjects = [re.search(pattern=PATTERN, string=x).group('subject') for x in file_list]

    subjects = np.array_split(subjects, mp.cpu_count())
    
    process_map(partial(dat_to_json, dir_path, save_path, df_table), subjects, max_workers=mp.cpu_count())
            
    print("Process Finished!")
    
def main():
    DIR_PATH = "/home/lkh256/Studio/Autonomic_Aging/physionet.org/files/autonomic-aging-cardiovascular/1.0.0"
    SAVE_PATH = "/home/lkh256/Studio/Autonomic_Aging/data"
    
    master_df = pd.read_csv(os.path.join(DIR_PATH, 'subject-info.csv'))
    
    make_dataset(DIR_PATH, SAVE_PATH, master_df)

# %%
if __name__ == '__main__':
    main()
# %%

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
    """Read .dat file and return data in dictionary type

    Args:
        file_nm (str): file path to load

    Returns:
        dictionary: return data in .dat dictionary format
    """
    data = rdrecord(file_nm).__dict__
    
    return data

def get_data_info(dict_data, df_table):
    """Basic Preprocessing data: Merge dictionary with master table

    Args:
        dict_data (dict): dictionary types of data about given physionet .dat file 
        df_table (pandas.DataFrame): Master Table for subject information

    Returns:
        dict: return dictionary with basic preprocessing with merge data in master table
    """
    ## Make dictionary about given information about data and signal data 
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
    

def dat_to_json(load_path, save_path, df_table, subjects):
    """ Save dictionary to JSON on given subject list

    Args:
        load_path (str): Directory path for loading data
        save_path (str): Directory path for saving data
        df_table (pandas.DataFrame): Master Table for subject information
        subjects (numpy.array): subject list in numpy array type
    """
    subjects = subjects.tolist()
    
    for subject in subjects:
        
        ## Read .dat file and converse to dictionary type
        dict_data = read_dat_to_dict(os.path.join(load_path, subject))
        
        ## Select master table row on given subject 
        subject= int(subject)
        table_selected = df_table.query("ID == @subject", engine='python')
        
        ## Extract data from dictionary from .dat and merge data with selected table
        data = get_data_info(dict_data, table_selected)
        
        ## Save dictionary as json
        with open(os.path.join(save_path, "subject_" + str(subject) + ".json"), 'w') as fp:
            json.dump(data, fp, indent=4)

def make_dataset(load_path, save_path, df_table):
    """Generate dataset with basic preprocessing in given save_path

    Args:
        load_path (str): Directory path for loading data
        save_path (str): Directory path for saving data
        df_table (pandas.DataFrame): Master Table for subject information
    """
    ## Define Pattern and File Path
    PATTERN = r"\s*[/](?P<subject>\d*)[.]dat\s*"
    file_list = glob(load_path+"/*.dat")
    
    ## Get Subject by using regex
    subjects = [re.search(pattern=PATTERN, string=x).group('subject') for x in file_list]

    ## Split data into chunk: num_data / num_cpu
    subjects = np.array_split(subjects, mp.cpu_count())
    
    ## MultiProcessing
    process_map(partial(dat_to_json, load_path, save_path, df_table), subjects, max_workers=mp.cpu_count())
            
    print("Process Finished!")
    
def main():
    LOAD_PATH = "/home/lkh256/Studio/Autonomic_Aging/physionet.org/files/autonomic-aging-cardiovascular/1.0.0"
    SAVE_PATH = "/home/lkh256/Studio/Autonomic_Aging/data"
    
    master_df = pd.read_csv(os.path.join(LOAD_PATH, 'subject-info.csv'))
    
    make_dataset(LOAD_PATH, SAVE_PATH, master_df)

# %%
if __name__ == '__main__':
    main()
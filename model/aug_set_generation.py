import json
import numpy as np
import os
import pandas as pd

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

label_mapper = {0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:3, 
                8:3, 9:4, 10:4, 11:5, 12:5, 13:6, 14:6}

label_mapper = {0:0, 1:0, 2:1, 3:1, 4:1, 5:1, 6:1, 7:2, 
                8:2, 9:2, 10:2, 11:2, 12:3, 13:3, 14:3}

label_mapper = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3, 
                8:4, 9:4, 10:4, 11:4, 12:4, 13:4, 14:4}

label_mapper = {0:0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:1, 
                8:2, 9:2, 10:2, 11:2, 12:2, 13:2, 14:2} # 85퍼 3 Class

label_mapper = {0:0, 1:0, 2:0, 3:0, 10:1, 11:1, 12:1, 13:1, 14:1} # Binary 65세 이상 35세 미만


DATAPATH = "../output/rri_data"

def get_rri(file_nm: str, data_dir_path: str) -> np.ndarray:
    """
    Read RRI from JSON file from given directory and file name

    Args:
        file_nm (str): file names
        data_dir_path (str): data directory path

    Returns:
        np.ndarray: single RRI 
    """
    with open(os.path.join(data_dir_path, file_nm)) as json_file:
        ecg_json = json.load(json_file)

    rri = ecg_json['RRI']

    rri = np.pad(rri, (0, 1200-len(rri)), 'constant', constant_values=0)

    rri = rri.reshape(1, -1)

    return rri

def get_data_adasyn(train_x:np.ndarray, train_y:np.ndarray) -> tuple:
    """
    Perform ADASYN Oversampling

    Args:
        train_x (np.ndarray): train_x before sampling
        train_y (np.ndarray): train_y before sampling

    Returns:
        tuple: tuple of sampled train_x and train_y
    """
    adasyn_oversample = ADASYN(random_state=1004, n_jobs=-1)
    train_x_resampled, train_y_resampled = adasyn_oversample.fit_resample(train_x, train_y)
    
    return train_x_resampled, train_y_resampled

def get_data_randomover(train_x:np.ndarray, train_y:np.ndarray) -> tuple:
    """
    Perform Random Oversampling

    Args:
        train_x (np.ndarray): train_x before sampling
        train_y (np.ndarray): train_y before sampling

    Returns:
        tuple: tuple of sampled train_x and train_y
    """
    random_oversample = RandomOverSampler(random_state=1004)
    train_x_resampled, train_y_resampled = random_oversample.fit_resample(train_x, train_y)
    
    return train_x_resampled, train_y_resampled

def get_data_randomunder(train_x:np.ndarray, train_y:np.ndarray) -> tuple:
    """
    Perform Random Undersampling

    Args:
        train_x (np.ndarray): train_x before sampling
        train_y (np.ndarray): train_y before sampling

    Returns:
        tuple: tuple of sampled train_x and train_y
    """
    random_undersample = RandomUnderSampler(random_state=1004)
    train_x_resampled, train_y_resampled = random_undersample.fit_resample(train_x, train_y)
    
    return train_x_resampled, train_y_resampled


def get_data_hybrid(train_x:np.ndarray, train_y:np.ndarray) -> tuple:
    """
    Perform Hybrid Sampling Method - SMOTETomek

    Args:
        train_x (np.ndarray): train_x before sampling
        train_y (np.ndarray): train_y before sampling

    Returns:
        tuple: tuple of sampled train_x and train_y
    """
    combine_sample = SMOTETomek(random_state=1004, n_jobs=-1)
    train_x_resampled, train_y_resampled = combine_sample.fit_resample(train_x, train_y)
    
    return train_x_resampled, train_y_resampled

def test():
    master_table = pd.read_csv("../output/rri_data/master_table.csv")

    master_table = master_table.query("Age_group.isin([1, 2, 3, 4, 11, 12, 13, 14, 15])", engine='python')

    master_table = master_table.assign(label=lambda x: x['Age_group'] - 1, 
                                    label2=lambda x: x['label'].map(label_mapper))

    # Train test split
    train_table, test_table = train_test_split(master_table, test_size=0.2, random_state=1004, stratify=master_table['label2'])

    ## make dataset to array
    train_table = train_table.assign(RRI_value = lambda x: x['file_nm'].apply(lambda x: get_rri(file_nm=x, data_dir_path=DATAPATH))).reset_index(drop=True)
    test_table = test_table.assign(RRI_value = lambda x: x['file_nm'].apply(lambda x: get_rri(file_nm=x, data_dir_path=DATAPATH))).reset_index(drop=True)

    train_x = np.concatenate(train_table['RRI_value'])
    train_y = train_table['label2'].values

    test_x = np.concatenate(test_table['RRI_value'])
    test_y = test_table['label2'].values
    
    train_x_resampled, train_y_resampled = get_data_adasyn(train_x, train_y)
    print(len(train_x_resampled))
    
    train_x_resampled, train_y_resampled = get_data_randomover(train_x, train_y)
    print(len(train_x_resampled))
    
    train_x_resampled, train_y_resampled = get_data_randomunder(train_x, train_y)
    print(len(train_x_resampled))
    
    train_x_resampled, train_y_resampled = get_data_hybrid(train_x, train_y)
    print(len(train_x_resampled))


if __name__ == '__main__':
    test()

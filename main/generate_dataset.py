import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd

from functools import partial
from tqdm.contrib.concurrent import process_map

from signal_processing import extract_rri_neurokit2


def _segment_rri(rri_interp: np.ndarray) -> list:
    """
    Segment RRI in 1200 data points from whole RRI

    Args:
        rri_interp (np.ndarray): whole rri

    Returns:
        list: list of segmented rri
    """
    segment_index = [i for i in range(600, len(rri_interp), 600)]
    segmented_rri = [rri_interp[i-600:i+600] for i in segment_index]

    if (len(segmented_rri[-1]) / 1200 < 0.75):
        segmented_rri = segmented_rri[:-1]

    return segmented_rri


def _save_segmented_rri(segmented_rri: list, ecg_json: dict, subject_table: pd.DataFrame, save_path: str) -> pd.DataFrame:
    """
    Segment from the whole RRI and save segmented RRI with subject table with file names
    
    Args:
        segmented_rri (list): list of segmented rri(np.ndarray)
        ecg_json (dict): ecg database
        subject_table (pd.DataFrame): subject info from mastertable
        save_path (str): save directory path

    Returns:
        pd.DataFrame: dataframe with file names
    """
    file_names = []
    subject = subject_table['ID'].values[0]

    for index, rri in enumerate(segmented_rri):

        file_nm = f"subject_{str(subject)}_{index}.json"

        file_names.append(file_nm)

        ecg_json = {k: v for k, v in ecg_json.items(
        ) if k in ["ID", 'Age_group', 'Sex', 'BMI', 'Length', 'Device', 'fs']}

        rri_dict = {**ecg_json, 'RRI': rri.tolist()}

        ## Save dictionary as json
        with open(os.path.join(save_path, file_nm), "w") as fp:
            json.dump(rri_dict, fp, indent=4)

    subject_table = pd.merge(subject_table, pd.DataFrame(
        {'ID': subject, 'file_nm': file_names}), on='ID')

    return subject_table


def make_dataset(subject_table: pd.DataFrame, save_path: str, subject_list: np.ndarray) -> pd.DataFrame:
    """
    Preprocessing the dataset for model training with RRI segmentation and master table

    Args:
        subject_table (pd.DataFrame): master table for whole dataset
        save_path (str): save directory path
        subject_list (np.ndarray): list of subjects to process

    Returns:
        pd.DataFrame: master table for selected subjects
    """
    processed_df = pd.DataFrame()

    subject_list = subject_list.tolist()

    for idx in subject_list:

        with open(f"../output/data/subject_{str(idx)}.json") as json_file:
            ecg_json = json.load(json_file)

        if 'ECG' in ecg_json.keys():
            ecg_array = np.array(ecg_json['ECG'])
        else:
            ecg_array = np.array(ecg_json['ECG1'])

        rri_interp = extract_rri_neurokit2(
            ecg_signal=ecg_array, sampling_rate=1000)

        segmented_rri_s = _segment_rri(rri_interp)

        subject_table_info = subject_table.query('ID == @idx')

        processed_subject = _save_segmented_rri(
            segmented_rri=segmented_rri_s, ecg_json=ecg_json, subject_table=subject_table_info, save_path=save_path)

        processed_df = pd.concat((processed_df, processed_subject), axis=0)

    return processed_df.reset_index(drop=True)


def main() -> None:
    """
    main processing codews
    """
    subject_info_file = "../input/physionet.org/files/autonomic-aging-cardiovascular/1.0.0/subject-info.csv"
    
    ## Read master table
    df_orig = pd.read_csv(subject_info_file)
    df_orig = df_orig.query("Age_group.notnull() & ID != 400",
                            engine='python').reset_index(drop=True)

    SAVEPATH = "../output/rri_data"

    ## Split dataset for multi-processing
    subject_list = np.array_split(df_orig['ID'].values, mp.cpu_count())

    ## Multi-Processing
    list_processed_tables = process_map(
        partial(make_dataset, df_orig,
                SAVEPATH
                ),
        subject_list, max_workers=mp.cpu_count()
    )

    master_table = pd.concat(list_processed_tables,
                             axis=0).reset_index(drop=True)
    master_table.to_csv(os.path.join(
        SAVEPATH, "master_table.csv"), index=False, encoding='utf-8-sig')
    print("Process Finished!")


if __name__ == '__main__':
    main()

# %%
import json
import numpy as np
import multiprocessing as mp
import os
import pandas as pd
import re

from functools import partial
from glob import glob
from tqdm.contrib.concurrent import process_map
from wfdb.io import rdrecord
# %%


def _read_dat_to_dict(file_nm: str) -> dict:
  """Read .dat file and return data in dictionary type

  Args:
      file_nm (str): file path to load

  Returns:
      dict: data in .dat dictionary format
  """

  data_dict = rdrecord(file_nm).__dict__

  return data_dict


def _get_data_info(dict_data: dict, df_table: pd.DataFrame) -> dict:
  """Basic Preprocessing data: Merge dictionary with master table

  Args:
      dict_data (dict): dictionary types of data about given physionet .dat file 
      df_table (pd.DataFrame): Master Table for subject information

  Returns:
      dict: dictionary with basic preprocessing with merge data in master table
  """

  ## Make dictionary about given information about data and signal data
  sig_data_dict = dict(
      zip(
          dict_data['sig_name'],
          [
              dict_data['p_signal'][:, i].tolist()
              for i in range(len(dict_data['sig_name']))
          ]
      )
  )

  data_dict = {
      "ID": float(df_table['ID'].values[0]),
      "Age_group": float(df_table['Age_group'].values[0]),
      "Sex": float(df_table['Sex'].values[0]),
      "BMI": float(df_table['BMI'].values[0]),
      "Length": float(df_table['Length'].values[0]),
      "Device": float(df_table['Device'].values[0]),
      "Samp_rate": float(dict_data['fs']),
      **sig_data_dict
  }

  return data_dict


def _dat_to_json(
    load_path: str,
    save_path: str,
    df_table: pd.DataFrame,
    subjects: np.ndarray
) -> None:
  """Save dictionary to JSON on given subject list

  Args:
      load_path (str): Directory path for loading data
      save_path (str): Directory path for saving data
      df_table (pd.DataFrame): Master Table for subject information
      subjects (np.ndarray): subject list in numpy array type
  """
  subject_list = subjects.tolist()

  for subject in subject_list:
    ## Read .dat file and converse to dictionary type
    data_dict = _read_dat_to_dict(os.path.join(load_path, subject))

    ## Select master table row on given subject
    subject = int(subject)
    table_selected = df_table.query("ID == @subject", engine="python")

    ## Extract data from dictionary from .dat and merge data with selected table
    merged_data_dict = _get_data_info(data_dict, table_selected)

    ## Save dictionary as json
    with open(os.path.join(save_path, f"subject_{subject}.json"), "w") as fp:
        json.dump(merged_data_dict, fp, indent=4)


def make_dataset(
    load_path: str,
    save_path: str,
    df_table: pd.DataFrame,
) -> None:
  """Generate dataset with basic preprocessing in given save_path

  Args:
      load_path (str): Directory path for loading data
      save_path (str): Directory path for saving data
      df_table (pd.DataFrame): Master Table for subject information
  """

  ## Define Pattern and File Path
  PATTERN = r"\s*[/](?P<subject>\d*)[.]dat\s*"
  file_list = glob(str(load_path.joinpath("*.dat")))

  ## Get Subject by using regex
  subject_list = [re.search(pattern=PATTERN, string=x).group('subject') for x in file_list]

  ## Split data into chunk: num_data / num_cpu
  subject_list = np.array_split(subject_list, mp.cpu_count())

  ## Multi-Processing
  process_map(
      partial(
          _dat_to_json,
          load_path,
          save_path,
          df_table
      ),
      subject_list, max_workers=mp.cpu_count()
  )

  print("Process Finished!")

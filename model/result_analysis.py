# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from glob import glob
from sklearn.metrics import accuracy_score, f1_score

from my_module import *

# %%
def calculate_metric(df_log:pd.DataFrame, target:str, predict:str) -> None:
    acc_metric = df_log.groupby(["cv_num"]).apply(lambda x: accuracy_score(x[target], x[predict]))
    f1_metric = df_log.groupby(["cv_num"]).apply(lambda x: f1_score(x[target], x[predict], average='weighted'))
    print(acc_metric)
    print(f"\nAccuracy = {np.mean(acc_metric):.3f} +- {np.std(acc_metric):.3f}\n")
    print(f1_metric)
    print(f"\nWeighted F1 Score = {np.mean(f1_metric):.3f} +- {np.std(f1_metric):.3f}\n")
    return None

# %%
for file_nm in glob("../output/result/quad*_rri_hrv_*.csv"):
    print(file_nm)
    df_metric = pd.read_csv(file_nm)
    print(pd.crosstab(df_metric['label_class'], df_metric['predicted_label']))
    calculate_metric(df_metric, target='label_class', predict='predicted_label')
# %%
df_master = pd.read_csv("../output/rri_data/master_table.csv")
# %%
df_master[['ID', 'Age_group', 'Sex']].drop_duplicates()['Age_group'].value_counts()
# %%
df_master[['ID', 'Age_group', 'Sex']].drop_duplicates().groupby(['Sex'])['Age_group'].value_counts()
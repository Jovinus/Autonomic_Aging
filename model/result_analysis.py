# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from glob import glob
from sklearn.metrics import accuracy_score

from my_module import *

# %%
def calculate_metric(df_log:pd.DataFrame, target:str, predict:str) -> None:
    metric = df_log.groupby(["cv_num"]).apply(lambda x: accuracy_score(x[target], x[predict]))
    print(metric)
    print(f"\nAccuracy = {np.mean(metric):.3f} +- {np.std(metric):.3f}\n")
    return None
# %%
df_test = pd.read_csv("./result/tri_aging_women_adasyn.csv")
calculate_metric(df_test)
# %%
df_test = pd.read_csv("./binary_m_randomunder.csv")
calculate_metric(df_test)

# %%
for file_nm in glob("./result/tri*.csv"):
    print(file_nm)
    df_metric = pd.read_csv(file_nm)
    print(pd.crosstab(df_metric['label'], df_metric['predicted_label']))
    calculate_metric(df_metric)
# %%
for file_nm in glob("./result/quad*"):
    print(file_nm)
    df_metric = pd.read_csv(file_nm)
    print(pd.crosstab(df_metric['label_class'], df_metric['predicted_label']))
    calculate_metric(df_metric)
    
# %%
for file_nm in ["./result/quad_aging_adasyn.csv", "./result/quad_aging_hybrid.csv", "./result/quad_aging_randomover.csv", "./result/quad_aging_randomunder.csv"]:
    print(file_nm)
    df_metric = pd.read_csv(file_nm)
    print(pd.crosstab(df_metric['label'], df_metric['predicted_label']))
    calculate_metric(df_metric, target='label', predict='predicted_label')
    
# %%

for file_nm in glob("./result/quad_aging_all_mul*_wei*"):
    print(file_nm)
    df_metric = pd.read_csv(file_nm)
    print(pd.crosstab(df_metric['label_class'], df_metric['predicted_label']))
    calculate_metric(df_metric, target='label_class', predict='predicted_label')

# %%
for file_nm in glob("./result/quad_aging_all_mul*_wei*"):
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
# %%

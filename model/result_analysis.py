# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from glob import glob
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import resample

from my_module import *

# %%
def calculate_metric(df_log:pd.DataFrame, target:str, predict:str) -> None:
    acc_metric = df_log.groupby(["cv_num"]).apply(lambda x: accuracy_score(x[target], x[predict]))
    f1_metric = df_log.groupby(["cv_num"]).apply(lambda x: f1_score(x[target], x[predict], average='weighted'))
    f1_metric_macro = df_log.groupby(["cv_num"]).apply(lambda x: f1_score(x[target], x[predict], average='macro'))
    print(acc_metric)
    print(f"\nAccuracy = {np.mean(acc_metric):.3f} +- {np.std(acc_metric)*1.96:.3f}\n")
    print(f1_metric)
    print(f"\nWeighted F1 Score = {np.mean(f1_metric):.3f} +- {np.std(f1_metric)*1.96:.3f}\n")
    print(f1_metric_macro)
    print(f"\nMacro F1 Score = {np.mean(f1_metric_macro):.3f} +- {np.std(f1_metric_macro)*1.96:.3f}\n")
    return None


def calculate_bootstrap_metric(
    df_log:pd.DataFrame, 
    target:str, 
    predict:str, 
    n_bootstrap:int
) -> None:

    
    acc = []
    f1_macro = []
    f1_weighted = []
    

    for i in range(n_bootstrap):
        sampled_df = resample(
            df_log, 
            replace=True, 
            n_samples=300, 
            random_state=i
        )
        
        calc_acc = accuracy_score(
            sampled_df[target], 
            sampled_df[predict]
        )
        
        calc_f1_macro = f1_score(
            sampled_df[target], 
            sampled_df[predict], 
            average="macro"
        )
        
        calc_f1_weighted = f1_score(
            sampled_df[target], 
            sampled_df[predict], 
            average="weighted"
        )
        
        acc.append(calc_acc)
        f1_macro.append(calc_f1_macro)
        f1_weighted.append(calc_f1_weighted)
        
    print(f"\nAccuracy = {np.mean(acc):.3f} +- {np.std(acc)*1.96:.3f}")
    print(f"Macro F1 Score = {np.mean(f1_macro):.3f} +- {np.std(f1_macro)*1.96:.3f}")
    print(f"Weighted F1 Score = {np.mean(f1_weighted):.3f} +- {np.std(f1_weighted)*1.96:.3f}\n")
    
    metrics = {
        "accuracy": acc, 
        "f1_macro": f1_macro, 
        "f1_weighted": f1_weighted
    }
    
    metrics_df = pd.DataFrame(metrics)
        
    return metrics_df


# %%
for file_nm in glob("../output/result/quad*_1_*.csv"):
    
    print(file_nm)
    df_metric = pd.read_csv(file_nm)
    
    print(pd.crosstab(df_metric['label_class'], df_metric['predicted_label']))
    
    bootstrap_metrics = calculate_bootstrap_metric(df_metric, target='label_class', predict='predicted_label', n_bootstrap=100)
    
    save_file_nm = "../output/result/metric/bootstrap_metric_" + file_nm.split('/')[-1]
    bootstrap_metrics.to_csv(save_file_nm, encoding='utf-8', index=False)
    
    calculate_metric(df_metric, target='label_class', predict='predicted_label')
# %%

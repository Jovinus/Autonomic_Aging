# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from glob import glob
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.utils import resample

from my_module import *

pd.set_option("display.max_columns", None)

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
    auroc = []
    auprc = []
    

    for i in range(n_bootstrap):
        sampled_df = resample(
            df_log, 
            replace=True, 
            n_samples=1000, 
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
        
        calc_auroc = roc_auc_score(
            sampled_df[target], 
            sampled_df["predicted_proba"], 
        )
        
        calc_auprc = average_precision_score(
            sampled_df[target], 
            sampled_df["predicted_proba"], 
        )
        
        acc.append(calc_acc)
        f1_macro.append(calc_f1_macro)
        f1_weighted.append(calc_f1_weighted)
        auroc.append(calc_auroc)
        auprc.append(calc_auprc)
    
    metrics = {
        "accuracy": acc, 
        "f1_macro": f1_macro, 
        "f1_weighted": f1_weighted,
        "auroc":auroc,
        "auprc":auprc,
    }
    
    metrics_df = pd.DataFrame(metrics)
    
    print(f"\nAccuracy = {metrics_df['accuracy'].mean():.3f} ({metrics_df['accuracy'].quantile(0.025):.3f} - {metrics_df['accuracy'].quantile(0.975):.3f})")
    print(f"Macro F1 Score = {metrics_df['f1_macro'].mean():.3f} ({metrics_df['f1_macro'].quantile(0.025):.3f} - {metrics_df['f1_macro'].quantile(0.975):.3f})")
    print(f"Weighted F1 Score = {metrics_df['f1_weighted'].mean():.3f} ({metrics_df['f1_weighted'].quantile(0.025):.3f} - {metrics_df['f1_weighted'].quantile(0.975):.3f})")
    print(f"AUROC = {metrics_df['auroc'].mean():.3f} ({metrics_df['auroc'].quantile(0.025):.3f} - {metrics_df['auroc'].quantile(0.975):.3f})")
    print(f"AUPRC = {metrics_df['auprc'].mean():.3f} ({metrics_df['auprc'].quantile(0.025):.3f} - {metrics_df['auprc'].quantile(0.975):.3f})\n")
        
    return metrics_df


# %%
df_metric = pd.read_csv("../output/result/rri_hrv_data_osw_75_binary/binary_rri_adasyn.csv")

df_sampled = df_metric.groupby(['ID']).sample(n=1, replace=False, random_state=2)
df_sampled = df_sampled.assign(
    predicted_label = lambda dataframe: np.where(dataframe['predicted_proba'] >= 0.7, 1, 0)
)

calculate_bootstrap_metric(
    df_log=df_sampled, 
    target="label_class", 
    predict="predicted_label",
    n_bootstrap=300,
)
# %%
df_metric = pd.read_csv("../output/result/rri_hrv_data_osw_75_binary/binary_hrv_adasyn.csv")
df_sampled = df_metric.groupby(['ID']).sample(n=1, replace=False, random_state=2)
df_sampled = df_sampled.assign(
    predicted_label = lambda dataframe: np.where(dataframe['predicted_proba'] >= 0.7, 1, 0)
)
calculate_bootstrap_metric(
    df_log=df_sampled, 
    target="label_class", 
    predict="predicted_label",
    n_bootstrap=300,
)
# %%
df_metric = pd.read_csv("../output/result/rri_hrv_data_osw_75_binary/binary_combined_adasyn.csv")
df_sampled = df_metric.groupby(['ID']).sample(n=1, replace=False, random_state=2)
df_sampled = df_sampled.assign(
    predicted_label = lambda dataframe: np.where(dataframe['predicted_proba'] >= 0.7, 1, 0)
)
calculate_bootstrap_metric(
    df_log=df_sampled, 
    target="label_class", 
    predict="predicted_label",
    n_bootstrap=300,
)
# %%

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from glob import glob
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from IPython.display import display

pd.set_option("display.max_columns", None)

# %%
def calc_interp_curve(dataframe, y, score):
    
    fpr_base = np.arange(0, 1, step=0.001)
    ## Conventional
    fpr, tpr, thresholds = roc_curve(
        y_true = dataframe[y],
        y_score = dataframe[score],
        pos_label=1,
    )
    
    tpr_interp = np.interp(fpr_base, fpr, tpr)
    tpr_interp[0] = 0
    tpr_interp[-1] = 1
    threshold_interp = np.interp(fpr_base, fpr, thresholds)
    
    return fpr_base, tpr_interp, threshold_interp

def get_roc_curve(file_list):
    
    ## Model 1
    dataframe = pd.read_csv(file_list[0])
    dataframe = dataframe.groupby(['ID']).sample(n=1, replace=False, random_state=2)
    fpr_base, tpr_interp, thresholds_interp = calc_interp_curve(dataframe=dataframe, y="label_class", score="predicted_proba")
    
    svm_result = {
        'fpr':fpr_base,
        'tpr':tpr_interp,
        'thresholds':thresholds_interp,
        'model_auroc':"Model 1 (0.894, 0.867-0.919)",
    }
    svm_result = pd.DataFrame(svm_result)
    
    ## Model 2
    dataframe = pd.read_csv(file_list[1])
    dataframe = dataframe.groupby(['ID']).sample(n=1, replace=False, random_state=2)
    fpr_base, tpr_interp, thresholds_interp = calc_interp_curve(dataframe=dataframe, y="label_class", score="predicted_proba")
    
    lr_result = {
        'fpr':fpr_base,
        'tpr':tpr_interp,
        'thresholds':thresholds_interp,
        'model_auroc':"Model 2 (0.836, 0.799-0.871)",
    }
    lr_result = pd.DataFrame(lr_result)
    
    ## Model 3
    dataframe = pd.read_csv(file_list[2])
    dataframe = dataframe.groupby(['ID']).sample(n=1, replace=False, random_state=2)
    fpr_base, tpr_interp, thresholds_interp = calc_interp_curve(dataframe=dataframe, y="label_class", score="predicted_proba")
    
    ann_result = {
        'fpr':fpr_base,
        'tpr':tpr_interp,
        'thresholds':thresholds_interp,
        'model_auroc':"Model 3 (0.859, 0.824-0.889)",
    }
    ann_result = pd.DataFrame(ann_result)
    
    result = pd.concat([svm_result, lr_result, ann_result], ignore_index=True)
    
    return result

# %%
def calc_interp_prc(dataframe, y, score):
    
    recall_base = np.arange(0, 1, step=0.001)
    
    ## Conventional
    precision, recall, thresholds = precision_recall_curve(
        y_true = dataframe[y],
        probas_pred = dataframe[score],
        pos_label=1,
    )
    
    precision = np.flip(precision)
    recall = np.flip(recall)
    
    precision_interp = np.interp(recall_base, recall, precision)
    precision_interp[-1] = 0
    precision_interp[0] = 1
    
    return recall_base, precision_interp#, threshold_interp

def get_prc_curve(file_list):
    
    ## Model 1
    dataframe = pd.read_csv(file_list[0])
    dataframe = dataframe.groupby(['ID']).sample(n=1, replace=False, random_state=2)
    recall_base, precision_interp = calc_interp_prc(dataframe=dataframe, y="label_class", score="predicted_proba")
    
    svm_result = {
        'recall':recall_base,
        'precision':precision_interp,
        'model_auprc':"Model 1 (0.704, 0.634-0.770)",
    }
    svm_result = pd.DataFrame(svm_result)
    
    ## Model 2
    dataframe = pd.read_csv(file_list[1])
    dataframe = dataframe.groupby(['ID']).sample(n=1, replace=False, random_state=2)
    recall_base, precision_interp = calc_interp_prc(dataframe=dataframe, y="label_class", score="predicted_proba")
    
    lr_result = {
        'recall':recall_base,
        'precision':precision_interp,
        'model_auprc':"Model 2 (0.681, 0.611-0.745)",
    }
    lr_result = pd.DataFrame(lr_result)
    
    ## Model 3
    dataframe = pd.read_csv(file_list[2])
    dataframe = dataframe.groupby(['ID']).sample(n=1, replace=False, random_state=2)
    recall_base, precision_interp = calc_interp_prc(dataframe=dataframe, y="label_class", score="predicted_proba")
    
    ann_result = {
        'recall':recall_base,
        'precision':precision_interp,
        'model_auprc':"Model 3 (0.708, 0.635-0.768)",
    }
    ann_result = pd.DataFrame(ann_result)
    
    result = pd.concat([svm_result, lr_result, ann_result], ignore_index=True)
    
    return result
# %%
def plot_roc_curve_figure():
    
    file_list = glob("../output/result/rri_hrv_data_osw_75_binary/*csv")
    
    result = get_roc_curve(file_list=file_list)
    hue_order = [
        "Model 1 (0.894, 0.867-0.919)",
        "Model 2 (0.836, 0.799-0.871)",
        "Model 3 (0.859, 0.824-0.889)",
    ]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
    sns.lineplot(
        data=result, 
        x='fpr', 
        y='tpr', 
        hue='model_auroc', 
        hue_order=hue_order, 
        ax=ax,
        linewidth=2.5
    )
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("1 - Specificity", fontsize=20)
    plt.ylabel("Sensitivity", fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig("../figure/auroc_plot.png", dpi=500)
    plt.show()
    
    return None

plot_roc_curve_figure()
# %%
def plot_prc_curve_figure():
    
    file_list = glob("../output/result/rri_hrv_data_osw_75_binary/*csv")
    
    result = get_prc_curve(file_list=file_list)
    hue_order = [
        "Model 1 (0.704, 0.634-0.770)",
        "Model 2 (0.681, 0.611-0.745)",
        "Model 3 (0.708, 0.635-0.768)",
    ]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
    sns.lineplot(
        data=result, 
        x='recall', 
        y='precision', 
        hue='model_auprc', 
        hue_order=hue_order, 
        ax=ax,
        linewidth=2.5
    )
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig("../figure/auprc_plot.png", dpi=500)
    plt.show()
    
    return None

plot_prc_curve_figure()
# %%

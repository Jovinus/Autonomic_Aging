# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
from glob import glob

from my_dataloader import read_json_to_tensor
# %%

def hrv_static_set_generator():
    df_orig = pd.read_csv("../output/dataset/rri_hrv_data/master_table_hrv_rri.csv")

    HRV_COLUMNS = [
        'RRI', 'SDNN', 'RMSSD', 'pNN50', 'TRI', 'TINN', 'logVLF', 
        'logLF', 'LFnu','logHF', 'HFnu', 'LF_HF', 'logTot', 'ApEn', 
        'SampEn', 'a1', 'a2', 'Cordim', 'SD1', 'SD2'
    ]

    label_mapper = {
        0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:3, 
        8:3, 9:3, 10:3, 11:3, 12:3, 13:3, 14:3
    }

    hrv_df = df_orig.groupby(["ID", 'Age_group'])[HRV_COLUMNS].mean().reset_index().assign(
        age_group = lambda x: x['Age_group'].map(label_mapper)
    ).drop(columns=["Age_group"])

    hrv_df.to_csv("../output/result/hrv_df_raw.csv", index=False)

# %%
def calculate_confusion_matrix(dataframe, file_nm, dpi=500):
    
    absolute_values = pd.crosstab(
        pd.Categorical(dataframe['predicted_label'], categories=[0, 1, 2, 3]), 
        pd.Categorical(dataframe['label_class'], categories=[0, 1, 2, 3]),
        dropna=False,
    ).values.reshape(-1)
    
    percentage_values = pd.crosstab(
        pd.Categorical(dataframe['predicted_label'], categories=[0, 1, 2, 3]), 
        pd.Categorical(dataframe['label_class'], categories=[0, 1, 2, 3]), 
        dropna=False,
        normalize="columns"
    ).values
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor='white')
    ax = sns.heatmap(
        percentage_values, 
        annot=True, 
        ax=ax, 
        cbar=True,
        vmin=0,
        vmax=1,
        annot_kws={'fontsize':15}, 
        cmap="Blues", 
        xticklabels=["1 (18-29)", "2 (30-39)", "3 (40-49)", "4 (50-92)"],
        yticklabels=["1 (18-29)", "2 (30-39)", "3 (40-49)", "4 (50-92)"],
    )
    
    percentage_values = percentage_values.reshape(-1)
    
    for idx, t in enumerate(ax.texts): 
        t.set_text(f"{absolute_values[idx]:3d}\n({percentage_values[idx]*100:2.2f}%)")
    
    plt.yticks(fontsize=15, rotation=0)
    plt.xticks(fontsize=15)
    
    plt.xlabel("Ground Truth", fontsize=15)
    plt.ylabel("Prediction", fontsize=15)
    plt.savefig(file_nm, dpi=dpi)
    plt.show()
    
    return None

# %%
def plot_bar_bootstrap_metrics(dataframe, x, y, file_nm, dpi=500):
    
    x_order = ["RandomUnder", "RandomOver", "Hybrid", "ADASYN"]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='w')
    
    sns.barplot(
        x=x, y=y, 
        ax=ax, data=dataframe, 
        order=x_order, linewidth=2.5, 
        edgecolor=".2"
    )
    
    widthbars = [0.5, 0.5, 0.5, 0.5]
    for bar, newwidth in zip(ax.patches, widthbars):
        x = bar.get_x()
        width = bar.get_width()
        centre = x + width/2.
        bar.set_x(centre - newwidth/2.)
        bar.set_width(newwidth)
    
    
    if y == 'f1_weighted':
        plt.ylabel("Weighted F1 Score", fontsize=15)
    plt.xlabel("Sampling Method", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(file_nm, dpi=dpi)
    plt.show()
    
    return None

# %%
def plot_bar_bootstrap_metrics_v2(dataframe, x, y, col, hue, file_nm, dpi=500):
    
    x_order = ["HRV", "RRI", "RRI & HRV"]
    hue_order = ["RandomUnder", "RandomOver", "Hybrid", "ADASYN"]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='w')
    
    sns.barplot(
        x=x, y=y, 
        ax=ax, 
        hue=hue, 
        data=dataframe, 
        order=x_order,
        hue_order=hue_order, 
        linewidth=2.5,  
        edgecolor=".2"
    )
    
    # widthbars = [0.5]
    # for bar, newwidth in zip(ax.patches, widthbars):
    #     x = bar.get_x()
    #     width = bar.get_width()
    #     centre = x + width/2.
    #     bar.set_x(centre - newwidth/2.)
    #     bar.set_width(newwidth)
    
    
    if y == 'f1_weighted':
        plt.ylabel("Weighted F1 Score", fontsize=15)
    plt.xlabel("Data Type", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylim(0.3, 0.7)
    plt.savefig(file_nm, dpi=dpi)
    plt.show()
    
    return None

# %%
def bootstrap_result_generator():
    
    df_rri_adasyn_metrics = pd.read_csv("../output/result/metric/bootstrap_metric_quad_aging_all_multilabel_loss_weighted_1_adasyn.csv").assign(Data = "RRI", Sampling="ADASYN")
    df_rri_hybrid_metrics = pd.read_csv("../output/result/metric/bootstrap_metric_quad_aging_all_multilabel_loss_weighted_1_hybrid.csv").assign(Data = "RRI", Sampling="Hybrid")
    df_rri_randomover_metrics = pd.read_csv("../output/result/metric/bootstrap_metric_quad_aging_all_multilabel_loss_weighted_1_randomover.csv").assign(Data = "RRI", Sampling="RandomOver")
    df_rri_randomunder_metrics = pd.read_csv("../output/result/metric/bootstrap_metric_quad_aging_all_multilabel_loss_weighted_1_randomunder.csv").assign(Data = "RRI", Sampling="RandomUnder")
    
    df_hrv_adasyn_metrics = pd.read_csv("../output/result/metric/bootstrap_metric_quad_aging_all_multilabel_loss_weighted_hrv_1_adasyn.csv").assign(Data = "HRV", Sampling="ADASYN")
    df_hrv_hybrid_metrics = pd.read_csv("../output/result/metric/bootstrap_metric_quad_aging_all_multilabel_loss_weighted_hrv_1_hybrid.csv").assign(Data = "HRV", Sampling="Hybrid")
    df_hrv_randomover_metrics = pd.read_csv("../output/result/metric/bootstrap_metric_quad_aging_all_multilabel_loss_weighted_hrv_1_randomover.csv").assign(Data = "HRV", Sampling="RandomOver")
    df_hrv_randomunder_metrics = pd.read_csv("../output/result/metric/bootstrap_metric_quad_aging_all_multilabel_loss_weighted_hrv_1_randomunder.csv").assign(Data = "HRV", Sampling="RandomUnder")
    
    df_rri_hrv_adasyn_metrics = pd.read_csv("../output/result/metric/bootstrap_metric_quad_aging_all_multilabel_loss_weighted_1_rri_hrv_adasyn.csv").assign(Data = "RRI & HRV", Sampling="ADASYN")
    df_rri_hrv_hybrid_metrics = pd.read_csv("../output/result/metric/bootstrap_metric_quad_aging_all_multilabel_loss_weighted_1_rri_hrv_hybrid.csv").assign(Data = "RRI & HRV", Sampling="Hybrid")
    df_rri_hrv_randomover_metrics = pd.read_csv("../output/result/metric/bootstrap_metric_quad_aging_all_multilabel_loss_weighted_1_rri_hrv_randomover.csv").assign(Data = "RRI & HRV", Sampling="RandomOver")
    df_rri_hrv_randomunder_metrics = pd.read_csv("../output/result/metric/bootstrap_metric_quad_aging_all_multilabel_loss_weighted_1_rri_hrv_randomunder.csv").assign(Data = "RRI & HRV", Sampling="RandomUnder")
    
    df_bootstrap_metrics = pd.concat(
        [
            df_rri_adasyn_metrics, df_rri_hybrid_metrics, 
            df_rri_randomover_metrics, df_rri_randomunder_metrics,
            
            df_hrv_adasyn_metrics, df_hrv_hybrid_metrics, 
            df_hrv_randomover_metrics, df_hrv_randomunder_metrics,
            
            df_rri_hrv_adasyn_metrics, df_rri_hrv_hybrid_metrics, 
            df_rri_hrv_randomover_metrics, df_rri_hrv_randomunder_metrics,
        ], 
        axis=0
    ).reset_index(drop=True)
    
    return df_bootstrap_metrics

# %%
def make_confusion_matrix_plot():
    
    file_list = glob("../output/result/quad_*.csv")
    for file_nm in file_list:
        df_result_table = pd.read_csv(file_nm)
        figure_nm = file_nm.split("weighted_")[1].split(".")[0]
        calculate_confusion_matrix(df_result_table, file_nm=f"../figure/confusion_matrix/{figure_nm}.png", dpi=500)
        
def plot_rri():
    rri = read_json_to_tensor(datapath="../output/dataset/rri_hrv_data/subject_20_1.json")

    fig, ax = plt.subplots(1, 1, figsize=(20, 10), facecolor="w")
    plt.plot(np.arange(len(rri))/4, rri)
    plt.xlabel("Times (sec)", fontsize=20)
    plt.ylabel("RRI", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig("../figure/rri_plot.png", dpi=500)
    plt.show()

# %%

# df_result_table = pd.read_csv("../output/result/quad_aging_all_multilabel_loss_weighted_1_adasyn.csv")
# calculate_confusion_matrix(df_result_table, file_nm="../figure/rri_adasyn.png", dpi=500)

make_confusion_matrix_plot()

# %%
plot_rri()


# %%
# df_bootstrap_metrics = bootstrap_result_generator().query("Data == 'RRI'")
# plot_bar_bootstrap_metrics(
#     dataframe=df_bootstrap_metrics, 
#     x="Sampling", 
#     y="f1_weighted",
#     file_nm ="../figure/rri_bootstrap.png",
#     dpi=500,
# )

# %%
df_bootstrap_metrics = bootstrap_result_generator()
plot_bar_bootstrap_metrics_v2(
    dataframe=df_bootstrap_metrics, 
    x="Data", 
    y="f1_weighted",
    hue="Sampling",
    col=None,
    file_nm ="../figure/all_bootstrap.png",
    dpi=500,
)

# %%

# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
# %%
master_table = pd.read_csv("../output/dataset/rri_hrv_data/master_table_hrv_rri.csv")
# %%
reg_mapper = {
            0:18.5, 1:22.5, 2:27.5, 3:32.5, 4:37.5, 5:42.5, 6:47.5, 7:52.5, 
            8:57.5, 9:62.5, 10:67.5, 11:72.5, 12:77.5, 13:82.5, 14:88.5,
        }
        
label_mapper = {
    0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:3, 
    8:3, 9:3, 10:3, 11:3, 12:3, 13:3, 14:3,
}

master_table = master_table.assign(
    label=lambda x: x['Age_group'] - 1, 
    label_class=lambda x: x['label'].map(label_mapper), 
    label_reg=lambda x: x['label'].map(reg_mapper)
)

print(master_table.groupby(['ID']).head(1)['label_class'].value_counts(dropna=False))

# %%
master_table = master_table.query("label_class.isin([0, 3])", engine='python')

print(master_table.groupby(['ID']).head(1)['label_class'].value_counts(dropna=False))

master_table = master_table.assign(
        label_class = lambda x: np.where(x['label_class'] == 0, 0, 1)
).reset_index(drop=True)

print(master_table.groupby(['ID']).head(1)['label_class'].value_counts(dropna=False))
# %%
tmp_master_table = master_table[['ID', 'label_class', "label"]].drop_duplicates()
# %%
tmp_master_table['ID'].tolist()
# %%
from sklearn.model_selection import train_test_split
# %%
train_idx, test_idx = train_test_split(tmp_master_table['ID'], test_size=0.2, random_state=1004)
# %%
train_set = master_table.query("ID.isin(@train_idx)")
test_set = master_table.query("ID.isin(@test_idx)")
# %%
print(len(train_set['ID'].unique()))
print(len(test_set['ID'].unique()))
# %%
data_splitter = StratifiedKFold(n_splits=5, shuffle=True)
tmp_master_table = master_table[['ID', 'label_class', "label"]].drop_duplicates().reset_index(drop=True)

for num, (train_outer_idx, test_outer_idx) in enumerate(data_splitter.split(tmp_master_table['ID'], tmp_master_table["label"])):
    tmp = [x for x in train_outer_idx if x in test_outer_idx]
    print(f"Train Size = {len(train_outer_idx)}, Test Size = {len(test_outer_idx)}, Total Size = {len(train_outer_idx) + len(test_outer_idx)}")
    
    train_df = master_table.query("ID.isin(@train_outer_idx)").groupby(['ID']).head(1)
    test_df = master_table.query("ID.isin(@test_outer_idx)").groupby(['ID']).head(1)
    print(f"Train DF Size = {len(train_df)}, Test DF Size = {len(test_df)}, Total DF Size = {len(train_df) + len(test_df)}")
# %%

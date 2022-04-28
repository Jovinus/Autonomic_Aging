# %%
import json
import numpy as np
import os
import pandas as pd

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

label_mapper = {0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:3, 8:3, 9:4, 10:4, 11:5, 12:5, 13:6, 14:6}
label_mapper = {0:0, 1:0, 2:1, 3:1, 4:1, 5:1, 6:1, 7:2, 8:2, 9:2, 10:2, 11:2, 12:3, 13:3, 14:3}
label_mapper = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3, 8:4, 9:4, 10:4, 11:4, 12:4, 13:4, 14:4}
label_mapper = {0:0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:1, 8:2, 9:2, 10:2, 11:2, 12:2, 13:2, 14:2} # 85퍼 3 Class
label_mapper = {0:0, 1:0, 2:0, 3:0, 10:1, 11:1, 12:1, 13:1, 14:1} # Binary 65세 이상 35세 미만

# %%
master_table = pd.read_csv("../output/rri_data/master_table.csv")

master_table = master_table.query("Age_group.isin([1, 2, 3, 4, 11, 12, 13, 14, 15])", engine='python')

master_table = master_table.assign(label=lambda x: x['Age_group'] - 1, 
                                   label2=lambda x: x['label'].map(label_mapper))

# %%
train_table, test_table = train_test_split(master_table, test_size=0.2, random_state=1004, stratify=master_table['label2'])

# %%
def get_data(file_nm, DATAPATH="../output/rri_data"):
    with open(os.path.join(DATAPATH, file_nm)) as json_file:
        ecg_json = json.load(json_file)
    rri = ecg_json['RRI']
    rri = np.pad(rri, (0, 1200-len(rri)), 'constant', constant_values=0)
    # rri_array = np.array(ecg_json['RRI'], ndmin=2)
    rri = rri.reshape(1, -1)
    return rri
# %%
## make dataset to array
train_table = train_table.assign(RRI_value = lambda x: x['file_nm'].apply(get_data)).reset_index(drop=True)
test_table = test_table.assign(RRI_value = lambda x: x['file_nm'].apply(get_data)).reset_index(drop=True)

train_x = np.concatenate(train_table['RRI_value'])
# train_x = np.concatenate((train_x, train_table['Sex'].values.reshape(-1, 1)), axis=1)
train_y = train_table['label2'].values

test_x = np.concatenate(test_table['RRI_value'])
# test_x = np.concatenate((test_x, test_table['Sex'].values.reshape(-1, 1)), axis=1)
test_y = test_table['label2'].values

# %%
## Resampling using adasyn
adasyn_oversample = ADASYN(random_state=1004, n_jobs=-1)
train_x_resampled, train_y_resampled = adasyn_oversample.fit_resample(train_x, train_y)

print(len(train_x_resampled))
np.save(file="../output/train_x_adasyn.npy", arr=train_x_resampled)
np.save(file="../output/train_y_adasyn.npy", arr=train_y_resampled)

# %%
## Resampling using RandomOver
random_oversample = RandomOverSampler(random_state=1004)
train_x_resampled, train_y_resampled = random_oversample.fit_resample(train_x, train_y)

print(len(train_x_resampled))
np.save(file="../output/train_x_random_over.npy", arr=train_x_resampled)
np.save(file="../output/train_y_random_over.npy", arr=train_y_resampled)

# %%
## Resampling using random undersample
random_undersample = RandomUnderSampler(random_state=1004)
train_x_resampled, train_y_resampled = random_undersample.fit_resample(train_x, train_y)

print(len(train_x_resampled))
np.save(file="../output/train_x_random_under.npy", arr=train_x_resampled)
np.save(file="../output/train_y_random_under.npy", arr=train_y_resampled)

# %%
## Resampling using combine - SMOTETomek
combine_sample = SMOTETomek(random_state=1004, n_jobs=-1)
train_x_resampled, train_y_resampled = combine_sample.fit_resample(train_x, train_y)
print(len(train_x_resampled))

np.save(file="../output/train_x_combine_sample.npy", arr=train_x_resampled)
np.save(file="../output/train_y_combine_sample.npy", arr=train_y_resampled)

# %%
np.save(file="../output/test_x.npy", arr=test_x)
np.save(file="../output/test_y.npy", arr=test_y)
# %%

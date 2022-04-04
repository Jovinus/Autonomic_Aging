# %%
import json
import numpy as np
import os
import pandas as pd

from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split

# %%
master_table = pd.read_csv("../output/rri_data/master_table.csv")

master_table = master_table.assign(label=lambda x: x['Age_group'] - 1)

# %%
train_table, test_table = train_test_split(master_table, test_size=0.2, random_state=1004, stratify=master_table['label'])

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
train_y = train_table['label'].values

test_x = np.concatenate(test_table['RRI_value'])
test_y = test_table['label'].values

## Resampling using adasyn
adasyn_oversample = ADASYN(random_state=1004, n_jobs=-1)
train_x_resampled, train_y_resampled = adasyn_oversample.fit_resample(train_x, train_y)

# %%
np.save(file="../output/train_x_resample.npy", arr=train_x_resampled)
np.save(file="../output/train_y_resample.npy", arr=train_y_resampled)
np.save(file="../output/test_x.npy", arr=test_x)
np.save(file="../output/test_y.npy", arr=test_y)
# %%
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import json
import os
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
# %%

class CustomDataset(Dataset):
    def __init__(self, data_table, data_dir) -> None:
        super().__init__()
        
        self.master = data_table
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.master)
    
    def __getitem__(self, idx):
        
        data_path = os.path.join(self.data_dir, self.master.loc[idx, 'file_nm'])
        rri = read_json_to_tensor(datapath=data_path)
        
        rri_mean, rri_std = torch.mean(rri), torch.std(rri)
        
        ## Normalizing
        rri = (rri - rri_mean) / rri_std
        
        label = self.master.loc[idx, 'label']
        
        return  rri, label
    
class ResampleDataset_RRI_HRV(Dataset):
    def __init__(self, X_data, y_data) -> None:
        super().__init__()
        
        self.data = torch.from_numpy(X_data).view(-1, 1, 1220).type(torch.float32)
        self.label = torch.from_numpy(y_data).type(torch.LongTensor)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        rri = self.data[idx, :, :1200]
        hrv = self.data[idx, :, 1200:]
        label = self.label[idx]
        
        rri_mean, rri_std = torch.mean(rri), torch.std(rri)
        
        ## Normalizing
        rri = (rri - rri_mean) / rri_std
        
        data = torch.cat((rri, hrv.view(-1, 20)), axis=1)
        
        return  data, label
    
class ResampleDataset(Dataset):
    def __init__(self, X_data, y_data) -> None:
        super().__init__()
        
        self.data = torch.from_numpy(X_data).view(-1, 1, 1200).type(torch.float32)
        self.label = torch.from_numpy(y_data).view(-1, 2).type(torch.LongTensor)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        rri = self.data[idx, :]
        label = self.label[idx, :]
        
        rri_mean, rri_std = torch.mean(rri), torch.std(rri)
        
        ## Normalizing
        rri = (rri - rri_mean) / rri_std
        
        # print(rri.shape, label.shape)
        
        return  rri, label
    
class ResampleDataset_HRV(Dataset):
    def __init__(self, X_data:np.ndarray, y_data:np.ndarray) -> None:
        super().__init__()
        
        self.data = torch.from_numpy(X_data).view(-1, 20).type(torch.float32)
        self.label = torch.from_numpy(y_data).view(-1, 2).type(torch.LongTensor)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        hrv = self.data[idx, :]
        label = self.label[idx, :]
        
        return  hrv, label

class ResampleDataset_HRV_Single(Dataset):
    def __init__(self, X_data:np.ndarray, y_data:np.ndarray) -> None:
        super().__init__()
        
        self.data = torch.from_numpy(X_data).view(-1, 20).type(torch.float32)
        self.label = torch.from_numpy(y_data).view(-1, ).type(torch.LongTensor)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        hrv = self.data[idx, :]
        label = self.label[idx]
        
        return  hrv, label

class ResampleDataset_Single(Dataset):
    def __init__(self, X_data, y_data) -> None:
        super().__init__()
        
        self.data = torch.from_numpy(X_data).view(-1, 1, 1200).type(torch.float32)
        self.label = torch.from_numpy(y_data).view(-1, ).type(torch.LongTensor)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        rri = self.data[idx, :]
        label = self.label[idx]
        
        rri_mean, rri_std = torch.mean(rri), torch.std(rri)
        
        ## Normalizing
        rri = (rri - rri_mean) / rri_std
        
        # print(rri.shape, label.shape)
        
        return  rri, label

class ResampleDataset_RRI_HRV_Single(Dataset):
    def __init__(self, X_data, y_data) -> None:
        super().__init__()
        
        self.data = torch.from_numpy(X_data).view(-1, 1, 1220).type(torch.float32)
        self.label = torch.from_numpy(y_data).view(-1, ).type(torch.LongTensor)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        rri = self.data[idx, :, :1200]
        hrv = self.data[idx, :, 1200:]
        label = self.label[idx]
        
        rri_mean, rri_std = torch.mean(rri), torch.std(rri)
        
        ## Normalizing
        rri = (rri - rri_mean) / rri_std
        
        data = torch.cat((rri, hrv.view(-1, 20)), axis=1)
        
        return  data, label

# %%
def read_json_to_tensor(datapath):
    with open(datapath) as json_file:
            rri_json = json.load(json_file)
            
    rri = torch.from_numpy(np.array(rri_json['RRI'])).type(torch.float32)
    return rri

# %%
def padd_seq(batch):
    (x, y) = zip(*batch)
    # y = torch.LongTensor(y)
    x_pad = pad_sequence(x, batch_first=True, padding_value=0.0)
    x_pad = pad(x_pad.view(x_pad.shape[0], 1, -1), (0, 1200 - x_pad.shape[1]), "constant", 0)
    
    return x_pad, y
# %%

if __name__ == '__main__':
    DATAPATH = "../output/rri_data/"
    table = pd.read_csv("../output/dataset/rri_hrv_data/master_table_hrv_rri.csv")
    table = table.assign(label = lambda x: x['Age_group'] -1 )
# %%
    dataset = CustomDataset(data_table=table, data_dir=DATAPATH)

    for rri_1,  label in DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=padd_seq):
        print(rri_1.shape,  label.shape)
# %%
    X_data = np.load("../output/train_x_random_over.npy")
    y_data = np.load("../output/train_x_random_over.npy")
# %%
    train_dataset = ResampleDataset(X_data=X_data, y_data=y_data)
# %%
    for rri, label in DataLoader(train_dataset, batch_size=1024):
        print(rri.shape) 
        print(label.shape)
# %%

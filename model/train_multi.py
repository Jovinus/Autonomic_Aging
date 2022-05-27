import argparse
import pytorch_lightning as pl
from sympy import FunctionClass
import torch.nn as nn
import torch

from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split, StratifiedKFold
from torchmetrics import Accuracy, R2Score
from torch.utils.data import DataLoader

from aug_set_generation import *
from custom_loss import Multi_Loss
from my_module import *
from my_dataloader import *
from residual_cnn_multioutput import Residual_CNN_Model

## Define Argparser
parser = argparse.ArgumentParser(description="Autonomic Aging Classification training options")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--aug_mode", type=str, default="randomover")
parser.add_argument("--max_epoch", type=int, default=400)
parser.add_argument("--logdir", type=str, default="autonomic_aging")
parser.add_argument("--class_type", type=str, default="binary")
config = vars(parser.parse_args())

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        return bar

class Aging_Classification(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.loss = Multi_Loss()
        self.softmax = nn.Softmax(dim=1)
        self.accuracy = Accuracy()
        self.r2_score = R2Score()
        self.model = Residual_CNN_Model(output_class=4)
        
    def forward(self, x):
        pred = self.model(x)
        pred_class, pred_reg = pred[:, 0:4], pred[:, 4]
        pred_proba = self.softmax(pred_class) 
        pred = torch.cat((pred_proba, pred_reg.view(-1, 1)), dim=1)
        return pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = 1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        acc = self.accuracy(torch.argmax(pred[:, 0:4], dim=1), y[:, 0])
        r2_score = self.r2_score(pred[:, 4], y[:, 1])
        
        metrics = {'train_loss':loss, 'train_acc':acc, 'train_r2':r2_score}
        
        self.log_dict(metrics)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        
        val_acc = self.accuracy(torch.argmax(pred[:, 0:4], dim=1), y[:, 0])
        val_r2_score = self.r2_score(pred[:, 4], y[:, 1])
        
        metrics = {"val_loss":loss, "val_acc":val_acc, 'val_r2':val_r2_score}
        self.log_dict(metrics, prog_bar=True)
        
        return {'loss':loss, 'val_acc':val_acc}
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        return pred[:, 4], y[:, 1], torch.argmax(pred[:, 0:4], dim=1), y[:, 0]


def train_model(model:pl.LightningModule, 
                train_dataloaders:DataLoader, 
                val_dataloaders:DataLoader, 
                test_dataloaders:DataLoader,
                dir_name:str,
                version_name:str, 
                config:dict) -> tuple:
    
    bar = LitProgressBar()
    
    logger = TensorBoardLogger("tb_logs", name=dir_name, version=version_name)
    
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                dirpath='check_point/' + version_name, 
                                filename="residual_cnn_{epoch:03d}_{val_loss:.2f}", 
                                save_top_k=3, 
                                mode='min')
    
    trainer = pl.Trainer(logger=logger,
                        max_epochs=config['max_epoch'],
                        accelerator='gpu', 
                        devices=[config['gpu_id']], 
                        gradient_clip_val=5, 
                        log_every_n_steps=1, 
                        accumulate_grad_batches=1,
                        callbacks=[bar, checkpoint_callback], 
                        deterministic=True)
    
    trainer.fit(model, 
                train_dataloaders = train_dataloaders, 
                val_dataloaders = val_dataloaders)
    
    model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    results = trainer.predict(model, dataloaders=test_dataloaders)
    
    return model, results


def augmentation_mode(config: dict) -> FunctionClass:
    """Select Augmentation Model

    Args:
        config (dict): configuration dictionary from argparser

    Returns:
        function: selected augmentation option
    """
    if config['aug_mode'] == 'adasyn':
        aug_func = get_data_adasyn
    elif config['aug_mode'] == 'randomover':
        aug_func = get_data_randomover
    elif config['aug_mode'] == 'randomunder':
        aug_func = get_data_randomunder
    elif config['aug_mode'] == 'hybrid':
        aug_func = get_data_hybrid

    return aug_func

def main(config):
    df_con_matrix = pd.DataFrame()
            
    master_table = pd.read_csv("../output/rri_data/master_table.csv")

    # master_table = master_table.query("Age_group.isin([2, 6, 7, 8])", engine='python')
    # master_table = master_table.groupby(['ID']).head(1).reset_index(drop=True)
    # master_table = master_table.query("Sex == 1").reset_index(drop=True)
    reg_mapper = {0:18.5, 1:22.5, 2:27.5, 3:32.5, 4:37.5, 5:42.5, 6:47.5, 7:52.5, 
                  8:57.5, 9:62.5, 10:67.5, 11:72.5, 12:77.5, 13:82.5, 14:88.5}
    
    label_mapper = {0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:3, 
                8:3, 9:3, 10:3, 11:3, 12:3, 13:3, 14:3}

    master_table = master_table.assign(label=lambda x: x['Age_group'] - 1, 
                                       label_class=lambda x: x['label'].map(label_mapper), 
                                       label_reg=lambda x: x['label'].map(reg_mapper))
    
    ## stratified k-fold randomsplit (subject wise)
    data_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=1004)
    
    ## generate temporary table from data split (master_table))
    tmp_master_table = master_table[['ID', 'label_class', "label"]].drop_duplicates()
    
    for num, (train_outer_idx, test_outer_idx) in enumerate(data_splitter.split(tmp_master_table['ID'], tmp_master_table["label"])):
        
        train_table, test_table = master_table.query("ID.isin(@train_outer_idx)", engine='python'), master_table.query("ID.isin(@test_outer_idx)", engine='python').groupby(["ID"]).head(1)

        ## generate temporary table from data split (train_table))
        tmp_train_table = train_table[['ID', 'label_class', "label"]].drop_duplicates()
        
        # train test split (subject wise)
        train_inner_idx, valid_inner_idx = train_test_split(tmp_train_table['ID'], test_size=0.2, random_state=1004, stratify=tmp_train_table["label"])
        train_table, valid_table = train_table.query("ID.isin(@train_inner_idx)", engine='python'), train_table.query("ID.isin(@valid_inner_idx)", engine='python').groupby("ID").head(1)
        
        ## make dataset to array
        train_table = train_table.assign(RRI_value = lambda x: x['file_nm'].apply(lambda x: get_rri(file_nm=x, data_dir_path=DATAPATH))).reset_index(drop=True)
        valid_table = valid_table.assign(RRI_value = lambda x: x['file_nm'].apply(lambda x: get_rri(file_nm=x, data_dir_path=DATAPATH))).reset_index(drop=True)
        test_table = test_table.assign(RRI_value = lambda x: x['file_nm'].apply(lambda x: get_rri(file_nm=x, data_dir_path=DATAPATH))).reset_index(drop=True)

        train_x = np.concatenate(train_table['RRI_value'])
        train_y = train_table[["label"]].values
        
        valid_x = np.concatenate(valid_table['RRI_value'])
        valid_y = valid_table[['label_class', 'label_reg']].values

        test_x = np.concatenate(test_table['RRI_value'])
        test_y = test_table[['label_class', 'label_reg']].values
        
        aug_mode = augmentation_mode(config=config)
        train_x, train_y = aug_mode(train_x=train_x, train_y=train_y)
        train_y = np.concatenate((np.vectorize(label_mapper.get)(train_y).reshape(-1, 1), np.vectorize(reg_mapper.get)(train_y).reshape(-1, 1)), axis=1)
        
        train_dataset = ResampleDataset(X_data=train_x, y_data=train_y)
        valid_dataset = ResampleDataset(X_data=valid_x, y_data=valid_y)
        test_dataset = ResampleDataset(X_data=test_x, y_data=test_y)
        
        trainset_loader = DataLoader(train_dataset, batch_size=2**11, shuffle=True, num_workers=8)
        validset_loader = DataLoader(valid_dataset, batch_size=2**11, shuffle=False, num_workers=8)
        testset_loader = DataLoader(test_dataset, batch_size=2**11, shuffle=False, num_workers=8)
        
        model = Aging_Classification()
        
        model, results = train_model(model=model, 
                                     train_dataloaders=trainset_loader, 
                                     val_dataloaders=validset_loader, 
                                     test_dataloaders=testset_loader, 
                                     dir_name=config['logdir'], 
                                     version_name=config['class_type'] + "_" + config['aug_mode'] + "_cv_" + str(num), 
                                     config=config
                                     )

        ## log proba, prediction and label
        predicted_proba = torch.hstack([results[i][0] for i in range(len(results))]).cpu().numpy().tolist()
        labels_reg = torch.hstack([results[i][1] for i in range(len(results))]).cpu().numpy().tolist()
        predicted_labels = torch.hstack([results[i][2] for i in range(len(results))]).cpu().numpy().tolist()
        labels_class = torch.hstack([results[i][3] for i in range(len(results))]).cpu().numpy().tolist()
        
        pred_log = pd.DataFrame({'predicted_proba':predicted_proba, 'label_reg':labels_reg, 
                                 'predicted_label':predicted_labels, 'label_class':labels_class})
        pred_log['cv_num'] = num
        
        df_con_matrix = pd.concat((df_con_matrix, pred_log), axis=0)
            
    df_con_matrix.reset_index(drop=True).to_csv("./" + config['class_type'] + "_" + config['aug_mode'] + ".csv", index=False)
# %%
if __name__ == '__main__':
    
    ## set seed for reproducibility of studies
    pl.seed_everything(1004, workers=True)
    
    main(config)

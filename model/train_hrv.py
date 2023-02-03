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
from custom_loss import Cosine_Loss
from my_module import *
from my_dataloader import *
from hrv_fcl import hrv_fcl_model

## Define Argparser
parser = argparse.ArgumentParser(description="Autonomic Aging Classification training options")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--aug_mode", type=str, default="randomover")
parser.add_argument("--max_epoch", type=int, default=400)
parser.add_argument("--logdir", type=str, default="autonomic_aging")
parser.add_argument("--class_type", type=str, default="binary")
parser.add_argument("--analysis", type=str, default="main")
config = vars(parser.parse_args())

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        return bar

class Aging_Classification(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.loss = Cosine_Loss(num_class=4)
        self.softmax = nn.Softmax(dim=1)
        self.accuracy = Accuracy()
        self.model = hrv_fcl_model(in_features=20, out_class=4)
        
    def forward(self, x):
        pred = self.model(x)
        pred_proba = self.softmax(pred) 
        return pred_proba
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = 1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        acc = self.accuracy(torch.argmax(pred, dim=1), y)
        
        metrics = {'train_loss':loss, 'train_acc':acc}
        
        self.log_dict(metrics)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        
        val_acc = self.accuracy(torch.argmax(pred, dim=1), y)
        
        metrics = {"val_loss":loss, "val_acc":val_acc}
        self.log_dict(metrics, prog_bar=True)
        
        return {'loss':loss, 'val_acc':val_acc}
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        return torch.argmax(pred, dim=1), y


def train_model(
    model:pl.LightningModule, 
    train_dataloaders:DataLoader, 
    val_dataloaders:DataLoader, 
    test_dataloaders:DataLoader,
    dir_name:str,
    version_name:str, 
    config:dict
) -> tuple:
    
    bar = LitProgressBar()
    
    logger = TensorBoardLogger("../output/result/tb_logs", name=dir_name, version=version_name)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='../output/result/check_point/' + version_name, 
        filename="hrv_fcl_{epoch:03d}_{val_loss:.2f}", 
        save_top_k=3, 
        mode='min'
    )
    
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config['max_epoch'],
        accelerator='gpu', 
        devices=[config['gpu_id']], 
        gradient_clip_val=1, 
        log_every_n_steps=1, 
        accumulate_grad_batches=1,
        callbacks=[bar, checkpoint_callback], 
        deterministic=True
    )
    
    trainer.fit(
        model, 
        train_dataloaders = train_dataloaders, 
        val_dataloaders = val_dataloaders
    )
    
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
    elif config['aug_mode'] == "naive":
        aug_func = None

    return aug_func

def main(config):
    
    HRV_FEATURE = [
        'RRI','SDNN', 'RMSSD', 'pNN50', 'TRI', 
        'TINN', 'logVLF', 'logLF', 'LFnu', 
        'logHF', 'HFnu', 'LF_HF', 'logTot', 
        'ApEn', 'SampEn', 'a1', 'a2', 'Cordim', 
        'SD1', 'SD2'
    ]
    
    df_con_matrix = pd.DataFrame()
            
    master_table = pd.read_csv("../output/dataset/rri_hrv_data_no/master_table_hrv_rri.csv")
    
    if config['analysis'] == "main":

        reg_mapper = {0:18.5, 1:22.5, 2:27.5, 3:32.5, 4:37.5, 5:42.5, 6:47.5, 7:52.5, 
                    8:57.5, 9:62.5, 10:67.5, 11:72.5, 12:77.5, 13:82.5, 14:88.5}
        
        label_mapper = {0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:3, 
                    8:3, 9:3, 10:3, 11:3, 12:3, 13:3, 14:3}

        master_table = master_table.assign(
            label=lambda x: x['Age_group'] - 1, 
            label_class=lambda x: x['label'].map(label_mapper), 
            label_reg=lambda x: x['label'].map(reg_mapper)
        )
        
    else:
        
        reg_mapper = {
            0:18.5, 1:22.5, 2:27.5, 3:32.5, 4:37.5, 5:42.5, 6:47.5, 7:52.5, 
            8:57.5, 9:62.5, 10:67.5, 11:72.5, 12:77.5, 13:82.5, 14:88.5
        }
        
        label_mapper = {
            0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:3, 
            8:3, 9:3, 10:3, 11:3, 12:3, 13:3, 14:3
        }

        master_table = master_table.assign(
            label=lambda x: x['Age_group'] - 1, 
            label_class=lambda x: x['label'].map(label_mapper) - 1, 
            label_reg=lambda x: x['label'].map(reg_mapper)
        )
        
        master_table = master_table.query("label_class >= 0").reset_index(drop=True)
    
    ## stratified k-fold randomsplit (subject wise)
    data_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=1004)
    
## generate temporary table from data split (master_table))
    tmp_master_table = master_table[['ID', 'label_class', "label"]].drop_duplicates().reset_index(drop=True)
    
    for num, (train_outer_idx, test_outer_idx) in enumerate(data_splitter.split(tmp_master_table['ID'], tmp_master_table["label"])):
        
        train_id = tmp_master_table.loc[train_outer_idx, "ID"].tolist()
        test_id = tmp_master_table.loc[test_outer_idx, "ID"].tolist()
        
        trainval_table = master_table.query("ID.isin(@train_id)", engine='python')
        test_table = master_table.query("ID.isin(@test_id)", engine='python')
        
        ## generate temporary table from data split (trainval_table))
        tmp_trainval_table = trainval_table[['ID', 'label_class', "label"]].drop_duplicates()
        
        # train test split (subject wise)
        train_id, valid_id = train_test_split(tmp_trainval_table['ID'], test_size=0.2, random_state=1004, stratify=tmp_trainval_table["label"])
        
        train_table = trainval_table.query("ID.isin(@train_id)", engine='python') 
        valid_table = trainval_table.query("ID.isin(@valid_id)", engine='python')

        train_x = train_table[HRV_FEATURE].values
        train_y = train_table[["label"]].values
        
        valid_x = valid_table[HRV_FEATURE].values
        valid_y = valid_table[['label_class']].values

        test_x = test_table[HRV_FEATURE].values
        test_y = test_table[['label_class']].values
        
        aug_mode = augmentation_mode(config=config)
        
        if aug_mode is not None:
            train_x, train_y = aug_mode(train_x=train_x, train_y=train_y)

        if config['analysis'] == 'main':
            train_y = np.vectorize(label_mapper.get)(train_y).reshape(-1, 1)
        else:
            label_mapper = {
                3:0, 4:0, 5:1, 6:1, 7:2, 
                8:2, 9:2, 10:2, 11:2, 12:2, 13:2, 14:2
            }
            train_y = np.vectorize(label_mapper.get)(train_y).reshape(-1, 1)
        
        train_dataset = ResampleDataset_HRV_Single(X_data=train_x, y_data=train_y)
        valid_dataset = ResampleDataset_HRV_Single(X_data=valid_x, y_data=valid_y)
        test_dataset = ResampleDataset_HRV_Single(X_data=test_x, y_data=test_y)
        
        trainset_loader = DataLoader(train_dataset, batch_size=2**10, shuffle=True, num_workers=8)
        validset_loader = DataLoader(valid_dataset, batch_size=2**10, shuffle=False, num_workers=8)
        testset_loader = DataLoader(test_dataset, batch_size=2**10, shuffle=False, num_workers=8)
        
        model = Aging_Classification()
        
        model, results = train_model(
            model=model, 
            train_dataloaders=trainset_loader, 
            val_dataloaders=validset_loader, 
            test_dataloaders=testset_loader, 
            dir_name=config['logdir'], 
            version_name=config['class_type'] + "_" + config['aug_mode'] + "_cv_" + str(num), 
            config=config
        )

        ## log proba, prediction and label
        predicted_labels = torch.hstack([results[i][0] for i in range(len(results))]).cpu().numpy().tolist()
        labels_class = torch.hstack([results[i][1] for i in range(len(results))]).cpu().numpy().tolist()
        
        pred_log = pd.DataFrame(
            {
                'predicted_label':predicted_labels, 'label_class':labels_class
            }
        )
        pred_log['cv_num'] = num
        
        pred_log = (
            pred_log
            .pipe(lambda df: pd.concat(
                [
                    test_table.reset_index(drop=True),
                    df,
                ],
                axis=1,
                )
            )
        )
        
        df_con_matrix = pd.concat((df_con_matrix, pred_log), axis=0)
            
    df_con_matrix.reset_index(drop=True).to_csv("../output/result/" + config['class_type'] + "_" + config['aug_mode'] + ".csv", index=False)
# %%
if __name__ == '__main__':
    
    ## set seed for reproducibility of studies
    pl.seed_everything(1004, workers=True)
    
    main(config)

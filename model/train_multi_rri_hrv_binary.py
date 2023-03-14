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
from residual_cnn_1d_hrv_multioutput import Residual_CNN_Model

## Define Argparser
parser = argparse.ArgumentParser(description="Autonomic Aging Classification training options")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--aug_mode", type=str, default="randomover")
parser.add_argument("--max_epoch", type=int, default=400)
parser.add_argument("--logdir", type=str, default="autonomic_aging")
parser.add_argument("--class_type", type=str, default="binary")
parser.add_argument("--analysis", type=str, default="main")
parser.add_argument("--dataset", type=str, default="rri_hrv_data_osw_75")
config = vars(parser.parse_args())

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        return bar

class Aging_Classification(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.loss = Multi_Loss(num_class=2)
        self.softmax = nn.Softmax(dim=1)
        self.accuracy = Accuracy(task="multiclass", num_classes=2)
        self.model = Residual_CNN_Model(output_class=2)
        
    def forward(self, x):
        pred = self.model(x)
        pred_class, pred_reg = pred[:, 0:2], pred[:, 2]
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
        acc = self.accuracy(torch.argmax(pred[:, 0:2], dim=1), y[:, 0])
        
        metrics = {'train_loss':loss, 'train_acc':acc}
        
        self.log_dict(metrics)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)
        
        val_acc = self.accuracy(torch.argmax(pred[:, 0:2], dim=1), y[:, 0])
        
        metrics = {"val_loss":loss, "val_acc":val_acc}
        self.log_dict(metrics, prog_bar=True)
        
        return {'loss':loss, 'val_acc':val_acc}
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        return pred[:, 2], y[:, 1], torch.argmax(pred[:, 0:2], dim=1), y[:, 0], pred[:, 1]


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
    
    logger = TensorBoardLogger(
        f"../output/result/{config['dataset']}_binary/tb_logs", 
        name=dir_name, 
        version=version_name
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f"../output/result/{config['dataset']}_binary/check_point/" + version_name, 
        filename="residual_cnn_{epoch:03d}_{val_loss:.2f}", 
        save_top_k=3, 
        mode='min'
    )
    
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config['max_epoch'],
        accelerator='gpu', 
        devices=[config['gpu_id']], 
        gradient_clip_val=4, 
        log_every_n_steps=1, 
        accumulate_grad_batches=1,
        callbacks=[bar, checkpoint_callback], 
        deterministic=True,
        precision=16,
    )
    
    trainer.fit(
        model, 
        train_dataloaders = train_dataloaders, 
        val_dataloaders = val_dataloaders
    )
    
    model = model.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    
    results = trainer.predict(
        model, 
        dataloaders=test_dataloaders
    )
    
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
    df_con_matrix = pd.DataFrame()
    
    HRV_FEATURE = [
        'RRI','SDNN', 'RMSSD', 'pNN50', 'TRI', 
        'TINN', 'logVLF', 'logLF', 'LFnu', 
        'logHF', 'HFnu', 'LF_HF', 'logTot', 
        'ApEn', 'SampEn', 'a1', 'a2', 'Cordim', 
        'SD1', 'SD2'
    ]
    
    DATAPATH = f"../output/dataset/{config['dataset']}"        
    master_table = pd.read_csv(os.path.join(DATAPATH, "master_table_hrv_rri.csv"))
    
    if config['analysis'] == "main":

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
        
        master_table = master_table.query("label_class.isin([0, 3])", engine='python')
        
        master_table = master_table.assign(
            label_class = lambda x: np.where(x['label_class'] == 0, 0, 1)
        ).reset_index(drop=True)
    
    ## stratified k-fold randomsplit (subject wise)
    data_splitter = StratifiedKFold(
        n_splits=5, 
        shuffle=True, 
        random_state=1004
    )
    
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
        valid_table = trainval_table.query("ID.isin(@valid_id)", engine='python')#.groupby(['ID']).head(1)
        
        ## make dataset to array
        train_table = train_table.assign(RRI_value = lambda x: x['file_nm'].apply(lambda x: get_rri(file_nm=x, data_dir_path=DATAPATH))).reset_index(drop=True)
        valid_table = valid_table.assign(RRI_value = lambda x: x['file_nm'].apply(lambda x: get_rri(file_nm=x, data_dir_path=DATAPATH))).reset_index(drop=True)
        test_table = test_table.assign(RRI_value = lambda x: x['file_nm'].apply(lambda x: get_rri(file_nm=x, data_dir_path=DATAPATH))).reset_index(drop=True)
        
        train_x = np.concatenate(train_table['RRI_value'])
        train_x = np.concatenate((train_x, train_table[HRV_FEATURE].values), axis=1)
        train_y = train_table[["label"]].values
        
        valid_x = np.concatenate(valid_table['RRI_value'])
        valid_x = np.concatenate((valid_x, valid_table[HRV_FEATURE].values), axis=1)
        valid_y = valid_table[['label_class', 'label_reg']].values

        test_x = np.concatenate(test_table['RRI_value'])
        test_x = np.concatenate((test_x, test_table[HRV_FEATURE].values), axis=1)
        test_y = test_table[['label_class', 'label_reg']].values
        
        aug_mode = augmentation_mode(config=config)
        
        if aug_mode is not None:
            train_x, train_y = aug_mode(train_x=train_x, train_y=train_y)
        
        if config['analysis'] == 'main':
            label_mapper = {
                0:0, 1:0, 2:0, 7:1, 8:1, 9:1, 
                10:1, 11:1, 12:1, 13:1, 14:1
            }
            train_y = np.concatenate((np.vectorize(label_mapper.get)(train_y).reshape(-1, 1), np.vectorize(reg_mapper.get)(train_y).reshape(-1, 1)), axis=1)
        else:
            label_mapper = {
                3:0, 4:0, 5:1, 6:1, 7:2, 
                8:2, 9:2, 10:2, 11:2, 12:2, 13:2, 14:2
            }
            train_y = np.concatenate((np.vectorize(label_mapper.get)(train_y).reshape(-1, 1), np.vectorize(reg_mapper.get)(train_y).reshape(-1, 1)), axis=1)
        
        train_dataset = ResampleDataset_RRI_HRV(X_data=train_x, y_data=train_y)
        valid_dataset = ResampleDataset_RRI_HRV(X_data=valid_x, y_data=valid_y)
        test_dataset = ResampleDataset_RRI_HRV(X_data=test_x, y_data=test_y)
        
        trainset_loader = DataLoader(train_dataset, batch_size=2**11, shuffle=True, num_workers=8)
        validset_loader = DataLoader(valid_dataset, batch_size=2**11, shuffle=False, num_workers=8)
        testset_loader = DataLoader(test_dataset, batch_size=2**11, shuffle=False, num_workers=8)
        
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
        predicted_reg = torch.hstack([results[i][0] for i in range(len(results))]).cpu().numpy().tolist()
        labels_reg = torch.hstack([results[i][1] for i in range(len(results))]).cpu().numpy().tolist()
        predicted_labels = torch.hstack([results[i][2] for i in range(len(results))]).cpu().numpy().tolist()
        labels_class = torch.hstack([results[i][3] for i in range(len(results))]).cpu().numpy().tolist()
        predicted_proba = torch.hstack([results[i][4] for i in range(len(results))]).cpu().numpy().tolist()
        
        pred_log = pd.DataFrame(
            {
                'predicted_reg':predicted_reg, 
                'label_reg':labels_reg, 
                'predicted_label':predicted_labels, 
                'label_class':labels_class,
                'predicted_proba':predicted_proba
            }
        )
        pred_log['cv_num'] = num
        
        pred_log = (
            pred_log
            .pipe(lambda df: pd.concat(
                [
                    test_table.drop(columns=['RRI_value']).reset_index(drop=True),
                    df,
                ],
                axis=1,
                )
            )
        )
        
        df_con_matrix = pd.concat((df_con_matrix, pred_log), axis=0)
            
    df_con_matrix.reset_index(drop=True).to_csv(f"../output/result/{config['dataset']}_binary/" + config['class_type'] + "_" + config['aug_mode'] + ".csv", index=False)
# %%
if __name__ == '__main__':
    
    ## set seed for reproducibility of studies
    pl.seed_everything(1004, workers=True)
    
    main(config)

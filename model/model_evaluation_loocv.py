# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
import torch
from torchmetrics import Accuracy
from my_dataloader import *
from torch.utils.data import DataLoader
from residual_cnn_1d import *
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RepeatedKFold
from custom_loss import *
from my_module import *
# %%
class Depression_Detection(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        # self.loss = Cosine_Loss()
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.accuracy = Accuracy()
        self.model = Residual_CNN_Model(output_class=15)
        
    def forward(self, x):
        logits = self.model(x)
        logits = self.softmax(logits) 
        return logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = 1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        acc = self.accuracy(torch.argmax(logits, dim=1), y)
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        
        val_acc = self.accuracy(torch.argmax(logits, dim=1), y)
        metrics = {"val_loss":loss, "val_acc":val_acc}
        self.log_dict(metrics, prog_bar=True)
        
        return {'loss':loss, 'val_acc':val_acc}
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        return logits, y, torch.argmax(logits, dim=1)

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        return bar

# %%
if __name__ == '__main__':
    SIGNAL_DATAPATH = '../output/rri_data/'
    MASTER_TABLE_DATAPATH = '../output/rri_data/master_table.csv'
    
    pl.seed_everything(1004, workers=True)

    df_orig = pd.read_csv(MASTER_TABLE_DATAPATH)
    
    df_metric = pd.DataFrame()
    df_con_matrix = pd.DataFrame()
    df_fig_curve = pd.DataFrame()

    subject = df_orig.assign(label = lambda x: x['Age_group'] - 1)
    
    rkf_outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1004)
    rkf_inner = RepeatedStratifiedKFold(n_splits=4, n_repeats=1, random_state=1004)
    
    for cv_num_outer, (train_val_index, test_index) in enumerate(rkf_outer.split(subject, y=subject['label'])):
        test_data = subject.loc[test_index].reset_index(drop=True)
        train_val_data = subject.loc[train_val_index].reset_index(drop=True)
        for cv_num_inner, (train_index, valid_index) in enumerate(rkf_inner.split(train_val_data, y=train_val_data['label'])):
            
            train_data = subject.loc[train_index].reset_index(drop=True)
            valid_data = subject.loc[valid_index].reset_index(drop=True)
            
            train_dataset = CustomDataset(data_table=train_data, data_dir=SIGNAL_DATAPATH)
            valid_dataset = CustomDataset(data_table=valid_data, data_dir=SIGNAL_DATAPATH)
            test_dataset = CustomDataset(data_table=test_data, data_dir=SIGNAL_DATAPATH)
            
            trainset_loader = DataLoader(train_dataset, batch_size=2**7, shuffle=True, collate_fn=padd_seq, num_workers=8)
            validset_loader = DataLoader(valid_dataset, batch_size=2**7, shuffle=False, collate_fn=padd_seq, num_workers=8)
            testset_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=padd_seq, num_workers=8)
            
            bar = LitProgressBar()
            
            model = Depression_Detection()
            
            logger = TensorBoardLogger("tb_logs", name="aging_loocv_1", version="cross_val_outer_"+str(cv_num_outer) + '_inner_' + str(cv_num_inner))
            
            checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                        dirpath='check_point/aging_loocv_1_outer'+str(cv_num_outer)+'_inner_'+str(cv_num_inner), 
                                        filename="residual_cnn_{epoch:03d}_{val_loss:.2f}", 
                                        save_top_k=3, 
                                        mode='min')
            
            trainer = pl.Trainer(logger=logger,
                                max_epochs=400,
                                accelerator='gpu', 
                                devices=[0, 1, 2], 
                                gradient_clip_val=5, 
                                log_every_n_steps=1, 
                                accumulate_grad_batches=1,
                                callbacks=[bar, checkpoint_callback], 
                                deterministic=True)
            
            trainer.fit(model, 
                        train_dataloaders = trainset_loader, 
                        val_dataloaders = validset_loader)
            
            ## Retrieve the best model
            model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
        
            ## Log Prediction and Label
            results = trainer.predict(model, dataloaders=testset_loader)
            pred_proba =  torch.vstack([results[i][0] for i in range(len(results))]).cpu().numpy()[:, 1].tolist()
            labels = torch.hstack([results[i][1] for i in range(len(results))]).cpu().numpy().tolist()
            preds = torch.hstack([results[i][2] for i in range(len(results))]).cpu().numpy().tolist()
            pred_log = pd.DataFrame({'pred_proba':pred_proba, 'label':labels, 'pred':preds})
            pred_log['cv_num_outer'] = cv_num_outer
            pred_log['cv_num_inner'] = cv_num_inner
            
            df_con_matrix = pd.concat((df_con_matrix, pred_log), axis=0)
            
    df_con_matrix.reset_index(drop=True).to_csv("./aging_loocv_1_conf_pred_results_final.csv", index=False)
# %%

# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn
import torch
from torchmetrics import Accuracy
from my_dataloader import *
from torch.utils.data import DataLoader
from residual_cnn_1d import Residual_CNN_Model
from custom_loss import Cosine_Loss
from my_module import *
# %%
class Depression_Detection(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.loss = Cosine_Loss()
        # self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.accuracy = Accuracy()
        self.model = Residual_CNN_Model(output_class=2)
        
    def forward(self, x):
        logits = self.model(x)
        logits = self.softmax(logits) 
        return logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = 1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
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
    
    pl.seed_everything(1004, workers=True)
    
    df_metric = pd.DataFrame()
    df_con_matrix = pd.DataFrame()
    df_fig_curve = pd.DataFrame()
            
    train_x = np.load("../output/train_x_random_over.npy")
    train_y = np.load("../output/train_y_random_over.npy")
    
    valid_x = np.load("../output/test_x.npy")
    valid_y = np.load("../output/test_y.npy")
    
    train_dataset = ResampleDataset(X_data=train_x, y_data=train_y)
    valid_dataset = ResampleDataset(X_data=valid_x, y_data=valid_y)
    
    trainset_loader = DataLoader(train_dataset, batch_size=2**10, shuffle=True, num_workers=8)
    validset_loader = DataLoader(valid_dataset, batch_size=2**10, shuffle=False, num_workers=8)
    
    bar = LitProgressBar()
    
    model = Depression_Detection()
    
    logger = TensorBoardLogger("tb_logs", name="autonomic_aging", version="binary_random_over")
    
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                dirpath='check_point/binary_random_over', 
                                filename="residual_cnn_{epoch:03d}_{val_loss:.2f}", 
                                save_top_k=3, 
                                mode='min')
    
    trainer = pl.Trainer(logger=logger,
                        max_epochs=400,
                        accelerator='gpu', 
                        devices=[0], 
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
    results = trainer.predict(model, dataloaders=validset_loader)
    labels = torch.hstack([results[i][1] for i in range(len(results))]).cpu().numpy().tolist()
    preds = torch.hstack([results[i][2] for i in range(len(results))]).cpu().numpy().tolist()
    pred_log = pd.DataFrame({'label':labels, 'pred':preds})
    
    df_con_matrix = pd.concat((df_con_matrix, pred_log), axis=0)
        
    df_con_matrix.reset_index(drop=True).to_csv("./aging_binary_random_over.csv", index=False)
# %%

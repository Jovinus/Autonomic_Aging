# %% 
import torch
import torch.nn as nn
import pytorch_lightning as pl

# %%

class hrv_fcl_model(pl.LightningModule):
    def __init__(self, in_features, out_class) -> None:
        super().__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=200),
            nn.BatchNorm1d(num_features=200), 
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=200, out_features=100),
            nn.BatchNorm1d(num_features=100), 
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=100, out_features=50),
            nn.BatchNorm1d(num_features=50), 
            nn.ReLU(),
            )
        
        self.linear_class = nn.Linear(50, out_class)
        self.linear_reg = nn.Linear(50, 1)
        
    def forward(self, x):
        x_linear = self.linear(x)
        y_out_class = self.linear_class(x_linear)
        y_out_reg = self.linear_reg(x_linear)
        
        y_out = torch.concat((y_out_class, y_out_reg), dim=1)
        
        return y_out
 
# %%
if __name__ == '__main__':
    test = torch.rand((2, 20))
    
    cnn_layer = hrv_fcl_model(in_features=20, out_class=4)
    tmp = cnn_layer.forward(test.view(2, 20))
    print(tmp)
# %%

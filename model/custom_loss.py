import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class Cosine_Loss(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=torch.Tensor([0.63, 0.14, 0.13, 0.1]))
        self.cosine_similarity = nn.CosineEmbeddingLoss(reduction="mean")
        self.condition = torch.Tensor([1])

    def forward(self, logits, labels):
        loss_cross = self.cross_entropy(logits, labels)
        loss_cosine = self.cosine_similarity(logits, F.one_hot(labels, num_classes=4), 
                                             self.condition.type_as(loss_cross))
        loss = loss_cosine + 0.1 * loss_cross
        return loss

class Multi_Loss(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        # self.cross_entropy = nn.CrossEntropyLoss(weight=torch.Tensor([0.63, 0.14, 0.13, 0.1]))
        self.cross_entropy = nn.CrossEntropyLoss()
        self.cosine_similarity = nn.CosineEmbeddingLoss(reduction="mean")
        self.condition = torch.Tensor([1])
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, labels):
        pred_proba, pred_reg = outputs[:, 0:4], outputs[:, 4]
        label_class, label_reg = labels[:, 0], labels[:, 1].type_as(pred_reg)
        loss_cross = self.cross_entropy(pred_proba, label_class)
        loss_cosine = self.cosine_similarity(pred_proba, F.one_hot(label_class, num_classes=4), 
                                             self.condition.type_as(loss_cross))
        loss_mse = self.mse_loss(pred_reg, label_reg) ## Numerical Value
        loss = loss_cosine + 0.1 * loss_cross + loss_mse
        return loss
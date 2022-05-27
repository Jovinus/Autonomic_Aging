import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class Cosine_Loss(pl.LightningModule):
    def __init__(self, num_class) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.cosine_similarity = nn.CosineEmbeddingLoss(reduction="mean")
        self.condition = torch.Tensor([1])
        self.num_class = num_class

    def forward(self, logits, labels):
        loss_cross = self.cross_entropy(logits, labels)
        loss_cosine = self.cosine_similarity(logits, F.one_hot(labels, num_classes=self.num_class), 
                                             self.condition.type_as(loss_cross))
        loss = loss_cosine + 0.1 * loss_cross
        return loss

import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

class MpcsClassifier(pl.LightningModule):

    def __init__(self):
        super(MpcsClassifier, self).__init__()

        #500 отсчётов в ваттметрограмме, 3 класса в выходном слое
        self.layer_1 = nn.Linear(500, 256)

        self.layer_2 = nn.Linear(256, 3)

        self.f1_score = torchmetrics.F1Score(num_classes=3, average="macro", threshold=0.3) 

        self.save_hyperparameters()

    def forward(self, x):
        
        x = F.relu(self.layer_1(x))
        
        x = self.layer_2(x)
        
        x = F.softmax(x, dim=-1)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        # y_hat - прогноз сети , y - целевое значение
        label_int = torch.tensor(y.detach().clone(), dtype=torch.int)
        f1_metric = self.f1_score(y_hat, label_int) 

        self.log("train_f1_score", f1_metric)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)    

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        # y_hat - прогноз сети , y - целевое значение
        label_int = torch.tensor(y.detach().clone(), dtype=torch.int)
        f1_metric = self.f1_score(y_hat, label_int) 

        self.log("train_f1_score", f1_metric)
        
        print()
        #print(label_int)
        print("pred:", y_hat)
        print("target:", y)
        print("test_f1_score", f1_metric)

        return loss

            

        
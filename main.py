import data_loader
import numpy as np
from torch.utils.data import DataLoader
import MpcClassifier
import pytorch_lightning as pl
import MpcsDataset
import os


#Обучающий набор данных
(learn_labels, learn_data) = data_loader.data_loader(os.path.join(os.getcwd(), "train.csv"))
#Тестовый набор данных
(test_labels, test_data) = data_loader.data_loader(os.path.join(os.getcwd(), "test.csv"))

#Приведение данных к нужному типу
learn_data = np.array(learn_data, dtype=np.float32)
test_data = np.array(test_data, dtype=np.float32)

#Создание pytorch dataset
train_dataset = MpcsDataset.MpcsDataset(learn_data, learn_labels)
test_dataset = MpcsDataset.MpcsDataset(test_data, test_labels)

#Создание pytorch dataloader
train_dataloader = DataLoader(train_dataset, batch_size = 5)
test_dataloader = DataLoader(test_dataset, batch_size = 5)

#Создание модели
classifier = MpcClassifier.MpcsClassifier()

# Pytorch Trainer
trainer = pl.Trainer(max_epochs=10)

#Обучение модели
trainer.fit(classifier, train_dataloaders=train_dataloader)

#Проверка модели
trainer.test(classifier, dataloaders=test_dataloader)
#print((classifier(test_dataset[0][0])))
#print(train_dataset[10][0])

#y = classifier(train_dataset[0][0])
#y_hat = train_dataset[0][1]
#loss_end = F.cross_entropy(y_hat, y)

#print(loss_start, loss_end)
#print(classifier(train_dataset[10][0]))
#print(train_dataset[10][1])

import numpy as np
import torch
import torchvision.transforms as transforms
from util import CnnModel
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import time
from random import shuffle
from util import MyDataset2
import pandas as pd

normal_csv_dir = 'E:\\chenguo-code\\all\\normal\\csv'
abnormal_csv_dir = 'E:\\chenguo-code\\all\\abnormal\\csv'

normal_csv_cnt = len(os.listdir(normal_csv_dir))
abnormal_csv_cnt = len(os.listdir(abnormal_csv_dir))

normal_index_list = [i for i in range(normal_csv_cnt)]
abnormal_index_list = [i for i in range(abnormal_csv_cnt)]

shuffle(normal_index_list)
shuffle(abnormal_index_list)

train_normal_data_len = int(normal_csv_cnt * 0.6)
valid_normal_data_len = int(normal_csv_cnt * 0.2)
test_normal_data_len = normal_csv_cnt - train_normal_data_len - valid_normal_data_len

train_abnormal_data_len = int(abnormal_csv_cnt * 0.6)
valid_abnormal_data_len = int(abnormal_csv_cnt * 0.2)
test_abnormal_data_len = abnormal_csv_cnt - train_abnormal_data_len - valid_abnormal_data_len

train_normal_index_list = normal_index_list[:train_normal_data_len]
valid_normal_index_list = normal_index_list[train_normal_data_len:train_normal_data_len + valid_normal_data_len]
test_normal_index_list = normal_index_list[train_normal_data_len + valid_normal_data_len:]

train_abnormal_index_list = abnormal_index_list[:train_abnormal_data_len]
valid_abnormal_index_list = abnormal_index_list[
                            train_abnormal_data_len:train_abnormal_data_len + valid_abnormal_data_len]
test_abnormal_index_list = abnormal_index_list[train_abnormal_data_len + valid_abnormal_data_len:]

t = time.localtime(time.time())
start_time = str(t.tm_year) + str(t.tm_mon) + str(t.tm_mday) + str(t.tm_hour) + str(t.tm_min)

model_dir = os.path.join(start_time, 'model')
log_dir = os.path.join(start_time, 'log')
dataset_dir = os.path.join(start_time, 'dataset')
normal_csv = os.path.join(dataset_dir, 'normal.csv')
abnormal_csv = os.path.join(dataset_dir, 'abnormal.csv')

os.mkdir(start_time)
os.mkdir(model_dir)
os.mkdir(log_dir)
os.mkdir(dataset_dir)

df = pd.DataFrame(normal_index_list, columns=['normal'])
df.to_csv(normal_csv, index=False)
df = pd.DataFrame(abnormal_index_list, columns=['abnormal'])
df.to_csv(abnormal_csv, index=False)

writer = SummaryWriter(log_dir)

train_dataset = MyDataset2(normal_csv_dir, train_normal_index_list[:2000], 0, trans=transforms.ToTensor())

train_dataset += MyDataset2(abnormal_csv_dir, train_abnormal_index_list[:500], 1, trans=transforms.ToTensor())

train_dataset += MyDataset2(abnormal_csv_dir, train_abnormal_index_list[:500], 1,
                            trans=transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(15)]))
train_dataset += MyDataset2(abnormal_csv_dir, train_abnormal_index_list[:500], 1,
                            trans=transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(1.0)]))
train_dataset += MyDataset2(abnormal_csv_dir, train_abnormal_index_list[:500], 1, img_size=50,
                            trans=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((32, 32))]))


valid_dataset = MyDataset2(normal_csv_dir, valid_normal_index_list[:600], 0, trans=transforms.ToTensor())

valid_dataset += MyDataset2(abnormal_csv_dir, valid_abnormal_index_list[:150], 1, trans=transforms.ToTensor())
valid_dataset += MyDataset2(abnormal_csv_dir, valid_abnormal_index_list[:150], 1,
                            trans=transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(15)]))
valid_dataset += MyDataset2(abnormal_csv_dir, valid_abnormal_index_list[:150], 1,
                            trans=transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(1.0)]))
valid_dataset += MyDataset2(abnormal_csv_dir, valid_abnormal_index_list[:150], 1, img_size=50,
                            trans=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((32, 32))]))


train_data_len = len(train_dataset)
valid_data_len = len(valid_dataset)

device = torch.device("cuda:0")

train_dataloader = DataLoader(train_dataset, 32, True)
valid_dataloader = DataLoader(valid_dataset, 32)

model = CnnModel(3, 2).to(device)

loss_func = nn.CrossEntropyLoss().to(device)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

step = 0
epochs = 200

min_valid_loss = 9999
record_train_loss = []
record_valid_loss = []

for epoch in range(epochs):
    model.train()

    train_loss = 0
    train_accuracy = 0

    for images, labels in train_dataloader:
        pass
        outputs = model(images.to(device))
        labels = labels.to(device)
        loss = loss_func(outputs, labels)
        train_loss += loss.item()
        accuracy = (outputs.argmax(1) == labels).sum()
        train_accuracy += accuracy.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss.item(), step)
        step += 1

    model.eval()
    valid_loss = 0
    valid_accuracy = 0

    with torch.no_grad():
        for images, labels in valid_dataloader:
            outputs = model(images.to(device))
            labels = labels.to(device)
            loss = loss_func(outputs, labels)
            valid_loss += loss.item()
            accuracy = (outputs.argmax(1) == labels).sum()
            valid_accuracy += accuracy.item()

    print("Epochs : {}/{} ".format(epoch + 1, epochs))
    print("\t训练集 Loss : {}  训练集准确率 : {}".format(train_loss, train_accuracy / train_data_len))
    print("\t测试集 Loss : {}  测试集准确率 : {}".format(valid_loss, valid_accuracy / valid_data_len))
    writer.add_scalar("valid_loss", valid_loss, epoch)
    writer.add_scalar("train_accuracy", train_accuracy / train_data_len, epoch)
    writer.add_scalar("valid_accuracy", valid_accuracy / valid_data_len, epoch)

    record_train_loss.append(train_loss)
    record_valid_loss.append(valid_loss)

    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model, os.path.join(model_dir, 'best.pth'))

    if epoch == (epochs - 1):
        torch.save(model, os.path.join(model_dir, 'last.pth'))

train_df = pd.DataFrame(record_train_loss)
train_df.to_csv('E:\\chenguo-code\\all\\normal\\cnn_train_loss.csv')
valid_df = pd.DataFrame(record_valid_loss)
valid_df.to_csv('E:\\chenguo-code\\all\\abnormal\\cnn_valid_loss.csv')

writer.close()


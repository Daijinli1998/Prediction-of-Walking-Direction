import os
import time
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from random import shuffle
import torch.nn as nn

t = time.localtime(time.time())
start_time = str(t.tm_year) + str(t.tm_mon) + str(t.tm_mday) + str(t.tm_hour) + str(t.tm_min)

model_dir = os.path.join(start_time, 'model')
log_dir = os.path.join(start_time, 'log')

os.mkdir(start_time)
os.mkdir(model_dir)
os.mkdir(log_dir)

root_dir = 'D:\\我的\\研途\\毕业论文研\\我\\数据\\'
dir_list = []

dir_list += ['20221113_200701', '20221113_095956', '20221113_100017', '20221113_100105',
             '20221113_100153', '20221113_200207', '20221113_200635', '20221203_170630',
             '20221203_170850', '20221203_170915', '20221203_170943', '20221203_171011',
             '20221203_171047', '20221203_171110', '20221203_171138', '20221203_171211',
             '20221203_171233', '20221203_171536', '20221203_171606', '20221203_171625']

train_list = dir_list[:12]
valid_list = dir_list[12:16]

writer = SummaryWriter(log_dir)

input_dim = 10
offset = 5
output_dim = 1
sequence_len = 50
hidden_size = 224


class GruDataset(Dataset):
    def __init__(self, data, look_back, label_offset):
        self.datas = data
        self.look_back = look_back
        self.label_offset = label_offset

    def __getitem__(self, item):
        data = self.datas[item:item + self.look_back]
        label = self.datas[item + self.look_back + self.label_offset - 1]
        return data, label

    def __len__(self):
        return len(self.datas - self.look_back - self.label_offset + 1)


class GruModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(GruModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x, _ = self.gru(x)
        b, s, h = x.shape
        x = x.view(s * b, h)
        x = self.linear(x)
        x = x.view(b, s, -1)
        return x


train_dataset = []
train_label = []
valid_dataset = []
valid_label = []

for train_name in train_list:
    centroid_csv = os.path.join(root_dir, train_name, 'centroids.csv')
    centroids = pd.read_csv(centroid_csv).values

    # 归一
    # (x + 0.3) * 1.6777   ->  [0 - 1]
    # (z + 1.2) * 0.8333   ->  [0 - 1]
    centroids[:, 0] = (centroids[:, 0] + 0.3) * 1.6777
    centroids[:, 1] = (centroids[:, 1] + 1.2) * 0.8333
    centroids = np.around(centroids, 2)
    datas = centroids[:, 0] * 100 + centroids[:, 1]

    dataset = GruDataset(datas, input_dim, offset)

    cnt = 0
    datas_list = []
    label_list = []

    for data, label in dataset:
        datas_list.append(data)
        label_list.append(label)

        cnt += 1
        if cnt == sequence_len:
            train_dataset.append(datas_list)
            train_label.append(label_list)
            cnt = 0
            datas_list = []
            label_list = []

for valid_name in valid_list:
    centroid_csv = os.path.join(root_dir, valid_name, 'centroids.csv')
    centroids = pd.read_csv(centroid_csv).values

    # 归一
    # (x + 0.3) * 1.6777   ->  [0 - 1]
    # (z + 1.2) * 0.8333   ->  [0 - 1]
    centroids[:, 0] = (centroids[:, 0] + 0.3) * 1.6777
    centroids[:, 1] = (centroids[:, 1] + 1.2) * 0.8333
    centroids = np.around(centroids, 2)
    datas = centroids[:, 0] * 100 + centroids[:, 1]

    dataset = GruDataset(datas, input_dim, offset)

    cnt = 0
    datas_list = []
    label_list = []

    for data, label in dataset:
        datas_list.append(data)
        label_list.append(label)

        cnt += 1
        if cnt == sequence_len:
            valid_dataset.append(datas_list)
            valid_label.append(label_list)
            cnt = 0
            datas_list = []
            label_list = []

train_dataset = np.array(train_dataset, dtype=np.float32)
train_label = np.array(train_label, dtype=np.float32)
valid_dataset = np.array(valid_dataset, dtype=np.float32)
valid_label = np.array(valid_label, dtype=np.float32)

index_list = [i for i in range(train_dataset.shape[0])]
shuffle(index_list)

learning_rate = 0.00001
model = GruModel(input_dim, hidden_size=hidden_size, num_layers=2, output_dim=1).cuda()
loss_func = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

epochs = 2500
train_step = 0
valid_step = 0
min_loss = 9999

for epoch in range(epochs):

    train_loss_sum = 0
    valid_loss_sum = 0

    model.train()
    for index in index_list:
        data_tensor = torch.from_numpy(train_dataset[index]).cuda()
        label_tensor = torch.from_numpy(train_label[index]).cuda()
        data_tensor = torch.reshape(data_tensor, (1, -1, input_dim))
        label_tensor = torch.reshape(label_tensor, (1, -1, output_dim))

        outputs = model(data_tensor)
        loss = loss_func(outputs, label_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("train_loss", loss.item(), train_step)
        train_step += 1
        train_loss_sum += loss.item()

    model.eval()
    with torch.no_grad():
        for index in range(len(valid_dataset)):
            dataset = valid_dataset[index]
            label = valid_label[index]
            data_tensor = torch.from_numpy(dataset).cuda()
            label_tensor = torch.from_numpy(label).cuda()
            data_tensor = torch.reshape(data_tensor, (1, -1, input_dim))
            label_tensor = torch.reshape(label_tensor, (1, -1, output_dim))

            outputs = model(data_tensor)
            loss = loss_func(outputs, label_tensor)

            writer.add_scalar("valid_loss", loss.item(), valid_step)
            valid_step += 1
            valid_loss_sum += loss.item()

    writer.add_scalar("train_loss_sum", train_loss_sum, epoch)
    writer.add_scalar("valid_loss_sum", valid_loss_sum, epoch)

    if valid_loss_sum < min_loss:
        min_loss = valid_loss_sum
        torch.save(model, os.path.join(model_dir, 'best.pth'))

    if epoch == epochs - 1:
        torch.save(model, os.path.join(model_dir, 'last.pth'))

    print("Epoch : {} / {}   Train Loss : {:.6f}    Valid Loss : {:.6f}".format(epoch + 1, epochs, train_loss_sum,
                                                                                valid_loss_sum))

writer.close()

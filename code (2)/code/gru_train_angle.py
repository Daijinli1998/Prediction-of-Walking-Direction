import os
import time
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from random import shuffle
import torch.nn as nn
import pandas as pd

t = time.localtime(time.time())
start_time = str(t.tm_year) + str(t.tm_mon) + str(t.tm_mday) + str(t.tm_hour) + str(t.tm_min)

model_dir = os.path.join(start_time, 'model')
log_dir = os.path.join(start_time, 'log')

os.mkdir(start_time)
os.mkdir(model_dir)
os.mkdir(log_dir)

csv_path = 'E:\\chenguo-code\\all\\normal\\'
csv_name_list = ['forward_angle_1.csv','forward_angle_2.csv','forward_angle_3.csv',
                 'left_angle_1.csv','right_angle_1.csv','right_angle_2.csv',
                 'left_angle_2.csv','right_angle_3.csv']

train_list = csv_name_list[:6]
valid_list = csv_name_list[6:]

writer = SummaryWriter(log_dir)

input_dim = 15
offset = 8
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

for train_csv_name in train_list:
    angle_csv = os.path.join(csv_path, train_csv_name)
    angles = pd.read_csv(angle_csv).values

    # 归一
    angles = (angles + 1.57) * 0.32

    datas = angles
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

for valid_csv_name in valid_list:
    centroid_csv = os.path.join(csv_path, valid_csv_name)
    centroids = pd.read_csv(centroid_csv).values

    # 归一
    angles = (angles + 1.57) * 0.32

    datas = angles
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
model = GruModel(input_dim, hidden_size=hidden_size, num_layers=2, output_dim=output_dim).cuda()
loss_func = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

epochs = 500
train_step = 0
valid_step = 0
min_loss = 9999

record_train_loss = []
record_valid_loss = []

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
        record_train_loss.append(loss.item())
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
            record_valid_loss.append(loss.item())
            valid_step += 1
            valid_loss_sum += loss.item()

    writer.add_scalar("train_loss_sum", train_loss_sum, epoch)
    writer.add_scalar("valid_loss_sum", valid_loss_sum, epoch)

    if valid_loss_sum < min_loss:
        min_loss = valid_loss_sum
        torch.save(model, os.path.join(model_dir, 'best_angle.pth'))

    if epoch == epochs - 1:
        torch.save(model, os.path.join(model_dir, 'last_x.pth'))

    print("Epoch : {} / {}   Train Loss : {:.6f}    Valid Loss : {:.6f}".format(epoch + 1, epochs, train_loss_sum,
                                                                                valid_loss_sum))

writer.close()

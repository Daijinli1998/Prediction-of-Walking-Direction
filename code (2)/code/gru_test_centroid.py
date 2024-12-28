import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import torch.nn as nn
import pandas as pd
from matplotlib.font_manager import FontProperties  # 导入FontProperties

my_font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 设置字体

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

axis = 'x'

root_dir = 'D:\\我的\\研途\\毕业论文研\\我\\数据\\'
dir_list = []

dir_list += ['20221113_200701', '20221113_095956', '20221113_100017', '20221113_100105',
             '20221113_100153', '20221113_200207', '20221113_200635', '20221203_170630',
             '20221203_170850', '20221203_170915', '20221203_170943', '20221203_171011',
             '20221203_171047', '20221203_171110', '20221203_171138', '20221203_171211',
             '20221203_171233', '20221203_171536', '20221203_171606', '20221203_171625']

valid_list = dir_list[12:16]

cur_name = '20221113_200701'

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
        return len(self.datas) - self.look_back - self.label_offset


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


valid_dataset = []
valid_label = []

centroid_csv = os.path.join(root_dir, cur_name, 'centroids.csv')
centroids = pd.read_csv(centroid_csv).values

# 归一
# (x + 0.3) * 1.6777   ->  [0 - 1]
# (z + 1.2) * 0.8333   ->  [0 - 1]
centroids[:, 0] = (centroids[:, 0] + 0.3) * 1.6777
centroids[:, 1] = (centroids[:, 1] + 1.2) * 0.8333
centroids = np.around(centroids, 2)
datas = centroids
dataset = GruDataset(datas, input_dim, offset)
raw_index_list = [i for i in range(len(dataset) + offset)]
valid_index_list = raw_index_list[input_dim + offset:]

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

valid_dataset = np.array(valid_dataset, dtype=np.float32)
valid_label = np.array(valid_label, dtype=np.float32)

valid_dataset_x = valid_dataset[:, :, :, 0]
valid_dataset_z = valid_dataset[:, :, :, 1]
valid_label_x = valid_label[:, :, 0]
valid_label_z = valid_label[:, :, 1]

learning_rate = 0.00001
if axis == 'z':
    model = torch.load("20233132030/model/best_z.pth")
elif axis == 'x':
    model = torch.load("20233132122/model/best_x.pth")
loss_func = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

model.eval()
out_list = []
with torch.no_grad():
    for index in range(len(valid_dataset)):
        if axis == 'z':
            data = valid_dataset_z[index]
            label = valid_label_z[index]
        elif axis == 'x':
            data = valid_dataset_x[index]
            label = valid_label_x[index]
        data_tensor = torch.from_numpy(data).cuda()
        label_tensor = torch.from_numpy(label).cuda()
        data_tensor = torch.reshape(data_tensor, (1, -1, input_dim))
        label_tensor = torch.reshape(label_tensor, (1, -1, output_dim))

        outputs = model(data_tensor)
        out_list += outputs[0, :, 0].cpu().numpy().tolist()
        loss = loss_func(outputs, label_tensor)

plt.xlabel("帧序号")
plt.plot(valid_index_list, out_list, linewidth=1)
if axis == 'x':
    plt.ylabel('x')
    plt.plot(raw_index_list, centroids[0:len(dataset) + offset:, 0], linewidth=1)
    plt.title("GRU模型x轴数据原始值与预测值")
    plt.legend(["x 原始值", "x 预测值"])
elif axis == 'z':
    plt.ylabel('z')
    plt.plot(raw_index_list, centroids[0:len(dataset) + offset:, 1], linewidth=1)
    plt.title("GRU模型z轴数据原始值与预测值")
    plt.legend(["z 原始值", "z 预测值"])
plt.show()

df = pd.DataFrame(out_list)
df.to_csv("D:\\PyCharmProject\\my_project\\centroid_x.csv")
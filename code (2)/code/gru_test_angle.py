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

csv_path = 'D:\\我的\\研途\\毕业论文研\\我\\数据\\'
csv_name_list = ['forward_angle_1.csv','forward_angle_2.csv','forward_angle_3.csv',
                 'left_angle_1.csv','right_angle_1.csv','right_angle_2.csv',
                 'left_angle_2.csv','right_angle_3.csv']

cur_valid = 'right_angle_3.csv'

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

angle_csv = os.path.join(csv_path, cur_valid)
raw_angles = pd.read_csv(angle_csv).values

# 归一
# (x + 0.3) * 1.6777   ->  [0 - 1]
# (z + 1.2) * 0.8333   ->  [0 - 1]
angles = (raw_angles + 1.57) * 0.32
datas = angles
print(datas.shape)
dataset = GruDataset(datas, input_dim, offset)
print(len(dataset),len(dataset)+offset)
raw_index_list = [i for i in range(len(dataset)+offset)]

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



learning_rate = 0.00001
model = torch.load("20233132156/model/best_angle.pth")
loss_func = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

model.eval()
out_list = []
with torch.no_grad():
    for index in range(len(valid_dataset)):
        data = valid_dataset[index]
        label = valid_label[index]
        data_tensor = torch.from_numpy(data).cuda()
        label_tensor = torch.from_numpy(label).cuda()
        data_tensor = torch.reshape(data_tensor, (1, -1, input_dim))
        label_tensor = torch.reshape(label_tensor, (1, -1, output_dim))

        outputs = model(data_tensor)
        out_list += outputs[0, :, 0].cpu().numpy().tolist()
        loss = loss_func(outputs, label_tensor)


out_list = np.array(out_list) / 0.32 - 1.57

ticks = [-1.57, -1.05, -0.524, 0, 0.524, 1.05, 1.57]
ticks_label = ['-pi/2', '-pi/3', '-pi/6', '0', 'pi/6', 'pi/3', 'pi/2']
plt.yticks(ticks, ticks_label)
plt.ylim([-1.57, 1.57])
plt.xlabel("帧序号")
valid_index_list = [i for i in range(input_dim + offset,input_dim+offset+len(out_list))]
plt.plot(raw_index_list, raw_angles[0:len(raw_index_list), 0], linewidth=1)
plt.plot(valid_index_list, out_list, linewidth=1)
plt.ylabel('转向姿态角（弧度）')
plt.title("GRU模型转向姿态角原始值与预测值")
plt.legend(["转向姿态角 原始值", "转向姿态角 预测值"])

plt.show()

import torch
import torchvision.transforms as trans
from util import MyDataset
from util import CnnModel
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import time

train_list = ['20221113_095901', '20221113_095956', '20221113_100017', '20221113_100105',
              '20221113_100153', '20221113_200635', '20221203_170630', '20221203_170850',
              '20221203_170915', '20221203_170943', '20221203_171011', '20221203_171047', '20221203_171110',
              '20221203_171138', '20221203_171211', '20221203_171233', '20221203_171536', '20221203_171606',
              '20221203_171625', '20221203_171919', '20221203_171946', '20221203_172001', '20221203_172024',
              '20221203_172037', '20221203_172058', '20221203_172104', '1',
              '111', '456']

valid_list = ['20221113_200207', '20221203_172125', '345', '20221113_200701']

train_dataset = None

transforms = trans.Compose([trans.ToTensor(), trans.RandomRotation(10), trans.RandomHorizontalFlip(0.5)])
for name in train_list:
    if train_dataset is None:
        train_dataset = MyDataset('D:\\我的\\研途\\毕业论文研\\我\\数据\\{}\\'.format(name), trans=transforms)
    else:
        train_dataset = train_dataset + MyDataset('D:\\我的\\研途\\毕业论文研\\我\\数据\\{}\\'.format(name),
                                                  trans=transforms)

valid_dataset = None
for name in valid_list:
    if valid_dataset is None:
        valid_dataset = MyDataset('D:\\我的\\研途\\毕业论文研\\数据\\{}\\'.format(name), trans=trans.ToTensor())
    else:
        valid_dataset = valid_dataset + MyDataset('D:\\我的\\研途\\毕业论文研\\我\\数据\\{}\\'.format(name),
                                                  trans=trans.ToTensor())

train_data_len = len(train_dataset)
valid_data_len = len(valid_dataset)

writer = SummaryWriter(os.path.join("logs", str(int(time.time()))))

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

for epoch in range(epochs):
    model.train()

    train_loss = 0
    train_accuracy = 0

    for images, labels in train_dataloader:
        outputs = model(images.to(device))
        labels = labels[0].to(device)
        loss = loss_func(outputs, labels)
        train_loss += loss
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
            labels = labels[0].to(device)
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

    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model, "model/20221228/best.pth")

    if epoch == (epochs - 1):
        torch.save(model, "model/20221228/latest.pth")

writer.close()

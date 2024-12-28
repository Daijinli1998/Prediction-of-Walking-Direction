import pandas as pd
import torch
import util
import os
import torchvision.transforms as transforms
import cv2
import PIL.Image as Image
import numpy as np

category = ['normal', 'abnormal']

font = cv2.FONT_HERSHEY_COMPLEX
color = (0, 0, 255)

normal_img_dir = 'D:\\我的\\研途\\毕业论文研\\我\\数据\\all_in_one\\all\\normal\\img'
abnormal_img_dir = 'D:\\我的\\研途\\毕业论文研\\我\\数据\\all_in_one\\all\\abnormal\\img'

root_dir = '202212301420'
model_path = os.path.join(root_dir, 'model/best.pth')
normal_csv = os.path.join(root_dir, 'dataset/normal.csv')
abnormal_csv = os.path.join(root_dir, 'dataset/abnormal.csv')

normal_index_list = pd.read_csv(normal_csv)['normal'].values
abnormal_index_list = pd.read_csv(abnormal_csv)['abnormal'].values
start_index = -int(len(normal_index_list) * 0.2)
normal_index_list = normal_index_list[start_index:]

start_index = -int(len(abnormal_index_list) * 0.2)
abnormal_index_list = abnormal_index_list[start_index:]

err_num = 0
test_data_num = len(normal_index_list) + len(abnormal_index_list)

for i in range(len(normal_index_list)):
    img_name = '{}.jpg'.format(normal_index_list[i])
    img = Image.open(os.path.join(normal_img_dir, img_name))
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
    ])

    image = trans(img).cuda()
    model = torch.load(model_path)
    image = torch.reshape(image, (1, 3, 32, 32))

    model.eval()
    with torch.no_grad():
        output = model(image)

    category_index = output.argmax(1).item()
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    cv2.putText(img, category[category_index], (470, 25), font, 1, color, 3)
    cv2.imshow("img", img)
    if category_index == 1:
        err_num += 1
        cv2.waitKey(-1)
    else:
        cv2.waitKey(1)

    print("normal : {} / {}   准确率 : {:.6f}".format(i+1, len(normal_index_list),
                                                      (test_data_num - err_num) / test_data_num))

for i in range(len(abnormal_index_list)):
    img_name = '{}.jpg'.format(abnormal_index_list[i])
    img = Image.open(os.path.join(abnormal_img_dir, img_name))
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
    ])

    image = trans(img).cuda()
    model = torch.load(model_path)
    image = torch.reshape(image, (1, 3, 32, 32))

    model.eval()
    with torch.no_grad():
        output = model(image)

    category_index = output.argmax(1).item()
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    cv2.putText(img, category[category_index], (470, 25), font, 1, color, 3)
    cv2.imshow("img", img)
    if category_index == 1:
        cv2.waitKey(1)
    else:
        err_num += 1
        cv2.waitKey(-1)

    print("abnormal : {} / {}   准确率 : {:.6f}".format(i+1, len(abnormal_index_list),
                                                        (test_data_num - err_num) / test_data_num))

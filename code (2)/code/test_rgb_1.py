import torch
import util
import os
import torchvision.transforms as transforms
import cv2
import PIL.Image as Image
import numpy as np

category = ['normal', 'abnormal']

model_path = '202212291159/model/best.pth'
data_dir = 'D:\\我的\\研途\\毕业论文研\\我\\数据\\20221230_125307\\'
img_dir = os.path.join(data_dir, 'img')
font = cv2.FONT_HERSHEY_COMPLEX
color = (0, 0, 255)
csv_list = os.listdir(img_dir)
csv_cnt = len(csv_list)

for i in range(csv_cnt):
    img_name = '{}.jpg'.format(i)
    img = Image.open(os.path.join(img_dir, img_name))
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
    show_img = Image.open(os.path.join(img_dir, '{}.jpg'.format(i)))
    show_img = cv2.cvtColor(np.asarray(show_img), cv2.COLOR_RGB2BGR)
    show_img = cv2.flip(show_img, 0)
    cv2.putText(show_img, category[category_index], (470, 25), font, 1, color, 3)
    cv2.imshow("show_img", show_img)
    if category_index == 1:
        cv2.waitKey(-1)
    else:
        cv2.waitKey(1)


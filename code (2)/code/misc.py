import os
import pandas as pd
import shutil

dir_list = ['E:\\chenguo-code\\all\\']

all_normal_csv_dir = 'E:\\chenguo-code\\all\\normal\\csv\\'
all_normal_img_dir = 'E:\\chenguo-code\\all\\normal\\img\\'
all_abnormal_csv_dir = 'E:\\chenguo-code\\all\\abnormal\\csv\\'
all_abnormal_img_dir = 'E:\\chenguo-code\\all\\abnormal\\img\\'

normal_cnt = len(os.listdir(all_normal_csv_dir))
abnormal_cnt = len(os.listdir(all_abnormal_csv_dir))

for data_dir in dir_list:
    root_dir = 'E:\\chenguo-code\\all\\normal\\'.format(data_dir)
    csv_dir = os.path.join(root_dir, 'csv')
    img_dir = os.path.join(root_dir, 'img')
    label_dir = os.path.join(root_dir, 'dataset')
    label_csv = os.path.join(label_dir, 'label.csv')
    csv_cnt = len(os.listdir(csv_dir))

    labels = pd.read_csv(label_csv).values

    for i in range(csv_cnt):
        src_csv_name = os.path.join(csv_dir, '{}.csv'.format(i))
        src_img_name = os.path.join(img_dir, '{}.jpg'.format(i))
        if labels[i] == 0:
            dst_csv_name = os.path.join(all_normal_csv_dir, '{}.csv'.format(normal_cnt))
            dst_img_name = os.path.join(all_normal_img_dir, '{}.jpg'.format(normal_cnt))
            normal_cnt += 1
        elif labels[i] == 1:
            dst_csv_name = os.path.join(all_abnormal_csv_dir, '{}.csv'.format(abnormal_cnt))
            dst_img_name = os.path.join(all_abnormal_img_dir, '{}.jpg'.format(abnormal_cnt))
            abnormal_cnt += 1

        shutil.copy(src_csv_name, dst_csv_name)
        shutil.copy(src_img_name, dst_img_name)

print(normal_cnt)
print(abnormal_cnt)

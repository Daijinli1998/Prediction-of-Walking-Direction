import util
import os
import numpy as np
import pandas as pd
import cv2
import PIL.Image as Image

dir_list = ['20221113_200701', '20221113_095901', '20221113_095956', '20221113_100017', '20221113_100105',
            '20221113_100153', '20221113_200207', '20221113_200635', '20221203_170630', '20221203_170850',
            '20221203_170915', '20221203_170943', '20221203_171011', '20221203_171047', '20221203_171110',
            '20221203_171138', '20221203_171211', '20221203_171233', '20221203_171536', '20221203_171606',
            '20221203_171625', '20221203_171919', '20221203_171946', '20221203_172001', '20221203_172024',
            '20221203_172037', '20221203_172058', '20221203_172104', '20221203_172125', '1',
            '111', '345', '456']

dir_list += ['20221230_125343', '20221230_125307', '20221230_125227', '20221230_125040', '20221230_125008',
             '20221230_124735']

dir_list += ['20230311_230129']


def main():
    data_dir = 'E:\\chenguo-code\\all\\normal\\'

    csv_dir = os.path.join(data_dir, 'csv')
    dataset_dir = os.path.join(data_dir, 'dataset')
    img_dir = os.path.join(data_dir, 'img')
    label_csv = os.path.join(dataset_dir, 'label.csv')

    csv_list = os.listdir(csv_dir)
    csv_cnt = len(csv_list)

    label_list = []

    for i in range(csv_cnt):
        print(os.path.join(img_dir, '{}.jpg'.format(i)))
        img = Image.open(os.path.join(img_dir, '{}.jpg'.format(i)))
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.flip(img, 0)
        img = cv2.flip(img, 1)
        cv2.imshow("color", img)
        file_name = os.path.join(csv_dir, '{}.csv'.format(i))
        df = pd.read_csv(file_name)
        print(len(df))
        util.get_img(df, show=True, wait_ms=-1, img_size=255)
        # while True:
        #     c = chr(cv2.waitKey(-1))
        #     if c == 'n' or c == 'N':
        #         label_list.append(0)
        #         print("{}.csv normal".format(i))
        #         break
        #     elif c == 'a' or c == 'A':
        #         label_list.append(1)
        #         print("{}.csv abnormal".format(i))
        #         break
        #     else:
        #         print("请输入 n/N or a/A")
    #
    # labels = pd.DataFrame(label_list, columns=['flag'])
    # labels.to_csv(label_csv, index=False)


if __name__ == '__main__':
    main()

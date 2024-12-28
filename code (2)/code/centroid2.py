import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import PIL.Image as Image

import util
import cv2

root_dir = 'E:\\chenguo-code\\all\\'

dir_list = ['abnormal']

# dir_list = ['csv']

# dir_list += ['345', '456']
#
# dir_list += ['20221230_125343', '20221230_125307', '20221230_125227', '20221230_125040', '20221230_125008',
#              '20221230_124735']


def main():
    global root_dir
    global dir_list

    for dir in dir_list:
        print(os.path.join(root_dir, dir))
        csv_dir = os.path.join(root_dir, dir, 'csv')
        img_dir = os.path.join(root_dir, dir, 'img')
        file_list = os.listdir(csv_dir)
        file_count = len(file_list)

        walk_centroid_list = []
        right_centroid_list = []
        left_centroid_list = []
        idle_centroid_list = []

        for i in range(file_count):
            print(i, '.csv')

            file_name = os.path.join(csv_dir, "{}.csv".format(i))
            df = pd.read_csv(file_name)
            df = df.dropna()
            if df.empty is True:
                continue

            df = pd.DataFrame(df, columns=['x', 'y', 'z'])
            centroid = cal_centroid(df)

            img = Image.open(os.path.join(img_dir, '{}.jpg'.format(i)))
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            img = cv2.flip(img, 0)
            cv2.imshow("color", img)

            util.get_img(df, show=True, wait_ms=1, img_size=255)
            # while True:
            #     print("q\t\twalk")
            #     print("w\t\tright")
            #     print("e\t\tleft")
            #     print("r\t\tidle")
            #     c = chr(cv2.waitKey(-1))
            #     if c == 'q' or c == 'Q':
            #         print("walk")
            #         walk_centroid_list.append(centroid)
            #         break
            #     elif c == 'w' or c == 'W':
            #         print("right")
            #         right_centroid_list.append(centroid)
            #         break
            #     elif c == 'e' or c == 'E':
            #         print("left")
            #         left_centroid_list.append(centroid)
            #         break
            #     elif c == 'r' or c == 'R':
            #         print("idle")
            #         idle_centroid_list.append(centroid)
            #         break
            #     else:
            #         print("请输入 n/N or a/A")

        walk_centroids = np.array(walk_centroid_list, dtype=np.float32)
        right_centroids = np.array(right_centroid_list, dtype=np.float32)
        left_centroids = np.array(left_centroid_list, dtype=np.float32)
        idle_centroids = np.array(idle_centroid_list, dtype=np.float32)
        walk_centroids = np.around(walk_centroids, 4)
        right_centroids = np.around(right_centroids, 4)
        left_centroids = np.around(left_centroids, 4)
        idle_centroids = np.around(idle_centroids, 4)
        if len(walk_centroids) > 0:
            df = pd.DataFrame(walk_centroids, columns=['x', 'y', 'z'])
            print(os.path.join(root_dir, dir, 'walk_centroids.csv'))
            df.to_csv(os.path.join(root_dir, dir, 'walk_centroids.csv'), index=False)
        if len(right_centroids) > 0:
            df = pd.DataFrame(right_centroids, columns=['x', 'y', 'z'])
            print(os.path.join(root_dir, dir, 'right_centroids.csv'))
            df.to_csv(os.path.join(root_dir, dir, 'right_centroids.csv'), index=False)
        if len(left_centroids) > 0:
            df = pd.DataFrame(left_centroids, columns=['x', 'y', 'z'])
            print(os.path.join(root_dir, dir, 'left_centroids.csv'))
            df.to_csv(os.path.join(root_dir, dir, 'left_centroids.csv'), index=False)
        if len(idle_centroids) > 0 :
            df = pd.DataFrame(idle_centroids, columns=['x', 'y', 'z'])
            print(os.path.join(root_dir, dir, 'idle_centroids.csv'))
            df.to_csv(os.path.join(root_dir, dir, 'idle_centroids.csv'), index=False)

    return


def cal_centroid(df):
    return (df['x'].sum() / df['x'].size, df['y'].sum() / df['y'].size, df['z'].sum() / df['z'].size)


def cal_centroid_xoy(df):
    return (df['x'].sum() / df['x'].size, df['y'].sum() / df['y'].size)


def cal_centroid_xoz(df):
    return (df['x'].sum() / df['x'].size, df['z'].sum() / df['z'].size)


def cal_centroid_zoy(df):
    return (df['z'].sum() / df['z'].size, df['y'].sum() / df['y'].size)


def eucliDist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


if __name__ == "__main__":
    main()

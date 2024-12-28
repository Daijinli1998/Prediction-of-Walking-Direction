import os
import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
import pandas as pd
import cv2
import PIL.Image as Image
import numpy as np

csv_dir = 'E:\\chenguo-code\\all\\normal\\csv\\'

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

def main():
    global csv_dir
    file_list = os.listdir(csv_dir)
    file_count = len(file_list)

    plt.ion()
    fig = plt.figure(1, figsize=[14, 8])

    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1
    z_min = 0
    z_max = 1

    plt.pause(1)

    l_knee_x_list = []
    l_thigh_x_list = []
    l_ankle_x_list = []
    l_knee_y_list = []
    l_thigh_y_list = []
    l_ankle_y_list = []
    l_knee_z_list = []
    l_thigh_z_list = []
    l_ankle_z_list = []
    centroid_x_list = []
    centroid_y_list = []
    centroid_z_list = []

    for i in range(file_count):
        if i < 65:
            continue
        if i >= 450:
            break
        fig.clf()
        x_ax = fig.add_subplot(3, 1, 1)
        y_ax = fig.add_subplot(3, 1, 2)
        z_ax = fig.add_subplot(3, 1, 3)

        x_ax.set_ylim([x_min, x_max])
        x_ax.set_ylabel("x")

        y_ax.set_ylim([y_min, y_max])
        y_ax.set_ylabel('y')

        z_ax.set_ylim([z_min, z_max])
        z_ax.set_ylabel('z')
        z_ax.set_xlabel("t/s")

        file_name = csv_dir + "{}.csv".format(i)
        df = pd.read_csv(file_name)
        df.drop_duplicates(inplace=False)

        df['z'] = 1 - (df['z'].add(1.3) * 1.82)
        df['y'] = df['y'].add(0.2) * 1.53
        df['x'] = df['x'].add(0.3) * 1.67

        # df['z'] = 1 - (df['z'].add(1.15) * 1.4)
        # df['y'] = df['y'].add(0.3) * 1.78
        # df['x'] = df['x'].add(0.3) * 2

        up = y_max
        down = y_max - 0.1
        df_l_list = []
        df_r_list = []

        while up > y_min:
            df_tmp = df.loc[(df['y'] < up) & (df['y'] > down)]
            x, y, z = cal_centroid(df_tmp)
            df_l_tmp = df_tmp.loc[df_tmp['x'] < x]
            df_r_tmp = df_tmp.loc[df_tmp['x'] > x]
            df_l_list.append(df_l_tmp)
            df_r_list.append(df_r_tmp)
            up = down
            down -= 0.1

        df_l = pd.concat(df_l_list, ignore_index=True)
        df_r = pd.concat(df_r_list, ignore_index=True)

        df_l_ankle = df_l.loc[(df_l['y'] < 0.163 + 0.05) & (df_l['y'] > 0.163 - 0.05)]
        df_l_ankle['z'] = df_l_ankle['z'] - 0.15
        df_r_ankle = df_r.loc[(df_r['y'] < 0.163 + 0.05) & (df_r['y'] > 0.163 - 0.05)]
        df_r_ankle['z'] = df_r_ankle['z'] - 0.15

        df_l_knee = df_l.loc[(df_l['y'] > 0.522 - 0.05) & (df_l['y'] < 0.522 + 0.05)]
        df_l_knee['z'] = df_l_knee['z'] - 0.1
        df_r_knee = df_r.loc[(df_r['y'] > 0.522 - 0.05) & (df_r['y'] < 0.522 + 0.05)]
        df_r_knee['z'] = df_r_knee['z'] - 0.1

        df_l_thigh = df_l.loc[(df_l['y'] > 0.9)]
        df_r_thigh = df_r.loc[(df_r['y'] > 0.9)]

        l_ankle_centroid = cal_centroid(df_l_ankle)
        r_ankle_centroid = cal_centroid(df_r_ankle)
        l_knee_centroid = cal_centroid(df_l_knee)
        r_knee_centroid = cal_centroid(df_r_knee)
        l_thigh_centroid = cal_centroid(df_l_thigh)
        r_thigh_centroid = cal_centroid(df_r_thigh)

        l_knee_x_list.append(l_knee_centroid[0])
        l_ankle_x_list.append(l_ankle_centroid[0])
        l_thigh_x_list.append(l_thigh_centroid[0])

        l_knee_y_list.append(l_knee_centroid[1])
        l_ankle_y_list.append(l_ankle_centroid[1])
        l_thigh_y_list.append(l_thigh_centroid[1])

        l_knee_z_list.append(l_knee_centroid[2])
        l_ankle_z_list.append(l_ankle_centroid[2])
        l_thigh_z_list.append(l_thigh_centroid[2])

        centroid_x_list.append((l_thigh_centroid[0] + r_thigh_centroid[0]) / 2)
        centroid_y_list.append((l_thigh_centroid[1] + r_thigh_centroid[1]) / 2)
        centroid_z_list.append((l_thigh_centroid[2] + r_thigh_centroid[2]) / 2)

        x_ax.plot(l_thigh_x_list, linewidth=1)
        x_ax.plot(l_knee_x_list, linewidth=1)
        x_ax.plot(l_ankle_x_list, linewidth=1)
        x_ax.plot(centroid_x_list, linewidth=1)
        x_ax.legend(["髋关节 x", "膝关节 x", "踝关节 x", "下肢重心 x"], loc=2)
        plt.xticks([0,44, 88,132, 176,220, 264,308, 352], ['0','', '4','', '8','', '12','', '16'])

        y_ax.plot(l_thigh_y_list, linewidth=1)
        y_ax.plot(l_knee_y_list, linewidth=1)
        y_ax.plot(l_ankle_y_list, linewidth=1)
        y_ax.plot(centroid_y_list, linewidth=1)
        y_ax.legend(["髋关节 y", "膝关节 y", "踝关节 y", "下肢重心 y"], loc=2)

        z_ax.plot(l_thigh_z_list, linewidth=1)
        z_ax.plot(l_knee_z_list, linewidth=1)
        z_ax.plot(l_ankle_z_list, linewidth=1)
        z_ax.plot(centroid_z_list, linewidth=1)
        z_ax.legend(["髋关节 z", "膝关节 z", "踝关节 z", "下肢重心 z"], loc=2)

        plt.pause(0.01)

    centroid_df = pd.DataFrame(zip(centroid_x_list,centroid_y_list,centroid_z_list),columns=['x','y','z'])
    centroid_df.to_csv('E:\\chenguo-code\\all\\normal\\\centroid.csv')
    l_thigh_df = pd.DataFrame(zip(l_thigh_x_list,l_thigh_y_list,l_thigh_z_list),columns=['x','y','z'])
    l_thigh_df.to_csv('E:\\chenguo-code\\all\\normal\\l_thigh.csv')
    l_knee_df = pd.DataFrame(zip(l_knee_x_list, l_knee_y_list, l_knee_z_list), columns=['x', 'y', 'z'])
    l_knee_df.to_csv('E:\\chenguo-code\\all\\normal\\l_knee.csv')
    l_ankle_df =  pd.DataFrame(zip(l_ankle_x_list, l_ankle_y_list, l_ankle_z_list), columns=['x', 'y', 'z'])
    l_ankle_df.to_csv('E:\\chenguo-code\\all\\normal\\l_ankle.csv')

    plt.pause(100)
    return


def cal_centroid(df):
    return [df['x'].sum() / df['x'].size, df['y'].sum() / df['y'].size, df['z'].sum() / df['z'].size]


def cal_centroid_xoy(df):
    return [df['x'].sum() / df['x'].size, df['y'].sum() / df['y'].size]


def cal_centroid_xoz(df):
    return [df['x'].sum() / df['x'].size, df['z'].sum() / df['z'].size]


def cal_centroid_zoy(df):
    return [df['z'].sum() / df['z'].size, df['y'].sum() / df['y'].size]


def eucliDist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


if __name__ == "__main__":
    main()

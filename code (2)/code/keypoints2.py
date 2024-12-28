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

csv_name = 'E:\\chenguo-code\\all\\normal\\csv\\16.csv'

y_offset = 0.05


def main():
    global csv_dir

    fig = plt.figure(1, figsize=[14, 8])

    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1
    z_min = 0
    z_max = 1

    xoy_ax = fig.add_subplot(3, 2, 2)
    zoy_ax = fig.add_subplot(3, 2, 4)
    xoz_ax = fig.add_subplot(3, 2, 6)
    xyz_ax = fig.add_subplot(1, 2, 1, projection='3d', azim=90, elev=-70)
    xyz_ax.set(xlabel='X', ylabel='Y', zlabel='Z', xlim=[x_min, x_max], ylim=[y_min, y_max], zlim=[z_min, z_max])

    xoy_ax.set_xlim([x_min, x_max])
    xoy_ax.set_ylim([y_min, y_max])

    zoy_ax.set_xlim([z_min, z_max])
    zoy_ax.set_ylim([y_min, y_max])

    xoz_ax.set_xlim([x_min, x_max])
    xoz_ax.set_ylim([z_min, z_max])

    df = pd.read_csv(csv_name)
    df.drop_duplicates(inplace=False)

    df['x'] = df['x'].add(0.3) * 0.8 + 0.25
    df['y'] = df['y'].add(0.32) * 1.19
    df['z'] = 1 - df['z'].add(1.12) * 1.5

    # df_foot = df.loc[df['y'] < 0.163]
    # df_shank = df.loc[(df['y'] > 0.163) & (df['y'] < 0.522)]
    # df_thigh = df.loc[(df['y'] > 0.522) ]
    # df_foot.to_csv("D:\\PyCharmProject\\my_project\\df_foot.csv")
    # df_shank.to_csv("D:\\PyCharmProject\\my_project\\df_shank.csv")
    # df_thigh.to_csv("D:\\PyCharmProject\\my_project\\df_thigh.csv")

    df_foot = df.loc[df['y'] < 0.163 - y_offset]
    df_shank = df.loc[(df['y'] > 0.163 + y_offset) & (df['y'] < 0.522 - y_offset)]
    df_thigh = df.loc[(df['y'] > 0.522 + y_offset) & (df['y'] < 1 - 0.1)]
    df_hip = df.loc[df['y'] > 1 - 0.1]
    df_knee = df.loc[(df['y'] > 0.522 - y_offset) & (df['y'] < 0.522 + y_offset)]
    df_ankle = df.loc[(df['y'] > 0.163 - y_offset) & (df['y'] < 0.163 + y_offset)]

    df_foot.to_csv("E:\\chenguo-code\\all\\normal\\16\\df_foot_ofs.csv")
    df_shank.to_csv("E:\\chenguo-code\\all\\normal\\16\\df_shank_ofs.csv")
    df_thigh.to_csv("E:\\chenguo-code\\all\\normal\\16\\df_thigh_ofs.csv")
    df_hip.to_csv("E:\\chenguo-code\\all\\normal\\16\\df_hip_ofs.csv")
    df_knee.to_csv("E:\\chenguo-code\\all\\normal\\16\\df_knee_ofs.csv")
    df_ankle.to_csv("E:\\chenguo-code\\all\\normal\\16\\df_ankle_ofs.csv")

    # xyz_ax.scatter(df_foot['x'], df_foot['y'], df_foot['z'], s=1)
    # xyz_ax.scatter(df_shank['x'], df_shank['y'], df_shank['z'], s=1)
    # xyz_ax.scatter(df_thigh['x'], df_thigh['y'], df_thigh['z'], s=1)
    # xyz_ax.scatter(df_hip['x'], df_hip['y'], df_hip['z'], s=1)
    # xyz_ax.scatter(df_knee['x'], df_knee['y'], df_knee['z'], s=1)
    # xyz_ax.scatter(df_ankle['x'], df_ankle['y'], df_ankle['z'], s=1)



    # xoz_ax.scatter(df['x'], df['z'], s=1)
    # xoy_ax.scatter(df['x'], df['y'], s=1)
    # zoy_ax.scatter(df['z'], df['y'], s=1)

    hip_centroid = cal_centroid(df_hip)
    thigh_centroid = cal_centroid(df_thigh)
    knee_centroid = cal_centroid(df_knee)
    shank_centroid = cal_centroid(df_shank)
    ankle_centroid = cal_centroid(df_ankle)
    foot_centroid = cal_centroid(df_foot)

    df_l_hip = df_hip.loc[df_hip['x'] < hip_centroid[0]]
    df_r_hip = df_hip.loc[df_hip['x'] > hip_centroid[0]]
    l_hip_centroid = cal_centroid(df_l_hip)
    r_hip_centroid = cal_centroid(df_r_hip)
    print(l_hip_centroid, r_hip_centroid)

    df_l_thigh = df_thigh.loc[df_thigh['x'] < thigh_centroid[0]]
    df_r_thigh = df_thigh.loc[df_thigh['x'] > thigh_centroid[0]]

    df_l_knee = df_knee.loc[df_knee['x'] < knee_centroid[0]]
    df_r_knee = df_knee.loc[df_knee['x'] > knee_centroid[0]]
    l_knee_centroid = cal_centroid(df_l_knee)
    l_knee_centroid[2] = l_knee_centroid[2] - 0.025
    l_knee_centroid[0] = l_knee_centroid[0] - 0.02
    r_knee_centroid = cal_centroid(df_r_knee)
    r_knee_centroid[2] = r_knee_centroid[2] - 0.04
    print(l_knee_centroid, r_knee_centroid)

    df_l_shank = df_shank.loc[df_shank['x'] < shank_centroid[0]]
    df_r_shank = df_shank.loc[df_shank['x'] > shank_centroid[0]]

    df_l_ankle = df_ankle.loc[df_ankle['x'] < ankle_centroid[0]]
    df_r_ankle = df_ankle.loc[df_ankle['x'] > ankle_centroid[0]]
    l_ankle_centroid = cal_centroid(df_l_ankle)
    r_ankle_centroid = cal_centroid(df_r_ankle)
    print(l_ankle_centroid, r_ankle_centroid)

    df_l_foot = df_foot.loc[df_foot['x'] < foot_centroid[0]]
    df_r_foot = df_foot.loc[df_foot['x'] > foot_centroid[0]]

    xyz_ax.scatter([l_ankle_centroid[0], r_ankle_centroid[0]],
                   [l_ankle_centroid[1], r_ankle_centroid[1]],
                   [l_ankle_centroid[2], r_ankle_centroid[2]], s=100)

    xyz_ax.scatter([l_knee_centroid[0], r_knee_centroid[0]],
                   [l_knee_centroid[1], r_knee_centroid[1]],
                   [l_knee_centroid[2], r_knee_centroid[2]], s=100)

    xyz_ax.scatter([l_hip_centroid[0], r_hip_centroid[0]],
                   [l_hip_centroid[1], r_hip_centroid[1]],
                   [l_hip_centroid[2], r_hip_centroid[2]], s=100)

    xyz_ax.scatter([hip_centroid[0]],
                   [hip_centroid[1]],
                   [hip_centroid[2]], s=100)

    foot_centroid = cal_centroid(df_foot)
    df_l_foot = df_foot.loc[df_foot['x'] < foot_centroid[0]]
    df_r_foot = df_foot.loc[df_foot['x'] > foot_centroid[0]]

    xoy_ax.scatter(df_l_hip['x'], df_l_hip['y'], s=1)
    xoy_ax.scatter(df_r_hip['x'], df_r_hip['y'], s=1)
    xoy_ax.scatter(df_l_thigh['x'], df_l_thigh['y'], s=1)
    xoy_ax.scatter(df_r_thigh['x'], df_r_thigh['y'], s=1)
    xoy_ax.scatter(df_l_knee['x'], df_l_knee['y'], s=1)
    xoy_ax.scatter(df_r_knee['x'], df_r_knee['y'], s=1)
    xoy_ax.scatter(df_l_shank['x'], df_l_shank['y'], s=1)
    xoy_ax.scatter(df_r_shank['x'], df_r_shank['y'], s=1)
    xoy_ax.scatter(df_l_ankle['x'], df_l_ankle['y'], s=1)
    xoy_ax.scatter(df_r_ankle['x'], df_r_ankle['y'], s=1)

    zoy_ax.scatter(df_l_hip['z'], df_l_hip['y'], s=1)
    zoy_ax.scatter(df_r_hip['z'], df_r_hip['y'], s=1)
    zoy_ax.scatter(df_l_thigh['z'], df_l_thigh['y'], s=1)
    zoy_ax.scatter(df_r_thigh['z'], df_r_thigh['y'], s=1)
    zoy_ax.scatter(df_l_knee['z'], df_l_knee['y'], s=1)
    zoy_ax.scatter(df_r_knee['z'], df_r_knee['y'], s=1)
    zoy_ax.scatter(df_l_shank['z'], df_l_shank['y'], s=1)
    zoy_ax.scatter(df_r_shank['z'], df_r_shank['y'], s=1)
    zoy_ax.scatter(df_l_ankle['z'], df_l_ankle['y'], s=1)
    zoy_ax.scatter(df_r_ankle['z'], df_r_ankle['y'], s=1)

    # l_min_z_point = df_l_foot.loc[df_l_foot['z'] == df_l_foot['z'].min()]
    # r_min_z_point = df_r_foot.loc[df_r_foot['z'] == df_r_foot['z'].min()]

    xyz_ax.plot([l_ankle_centroid[0], l_knee_centroid[0], l_hip_centroid[0], hip_centroid[0],
                 r_hip_centroid[0], r_knee_centroid[0], r_ankle_centroid[0]],
                [l_ankle_centroid[1], l_knee_centroid[1], l_hip_centroid[1], hip_centroid[1],
                 r_hip_centroid[1], r_knee_centroid[1], r_ankle_centroid[1]],
                [l_ankle_centroid[2], l_knee_centroid[2], l_hip_centroid[2], hip_centroid[2],
                 r_hip_centroid[2], r_knee_centroid[2], r_ankle_centroid[2]],
                linewidth=5)

    xoy_ax.plot([l_ankle_centroid[0], l_knee_centroid[0], l_hip_centroid[0], hip_centroid[0],
                 r_hip_centroid[0], r_knee_centroid[0], r_ankle_centroid[0]],
                [l_ankle_centroid[1], l_knee_centroid[1], l_hip_centroid[1], hip_centroid[1],
                 r_hip_centroid[1], r_knee_centroid[1], r_ankle_centroid[1]], linewidth=2,
                color='r')

    zoy_ax.plot([l_ankle_centroid[2], l_knee_centroid[2], l_hip_centroid[2], hip_centroid[2],
                 r_hip_centroid[2], r_knee_centroid[2], r_ankle_centroid[2]],
                [l_ankle_centroid[1], l_knee_centroid[1], l_hip_centroid[1], hip_centroid[1],
                 r_hip_centroid[1], r_knee_centroid[1], r_ankle_centroid[1]], linewidth=2,
                color='r')

    # xyz_ax.plot([l_min_z_point['x'], l_ankle_centroid[0], l_knee_centroid[0], l_hip_centroid[0], hip_centroid[0],
    #              r_hip_centroid[0], r_knee_centroid[0], r_ankle_centroid[0], r_min_z_point['x']],
    #             [l_min_z_point['y'], l_ankle_centroid[1], l_knee_centroid[1], l_hip_centroid[1], hip_centroid[1],
    #              r_hip_centroid[1], r_knee_centroid[1], r_ankle_centroid[1], r_min_z_point['y']],
    #             [l_min_z_point['z'], l_ankle_centroid[2], l_knee_centroid[2], l_hip_centroid[2], hip_centroid[2],
    #              r_hip_centroid[2], r_knee_centroid[2], r_ankle_centroid[2], r_min_z_point['z']],
    #             linewidth=5)
    #
    # xoy_ax.plot([l_min_z_point['x'], l_ankle_centroid[0], l_knee_centroid[0], l_hip_centroid[0], hip_centroid[0],
    #              r_hip_centroid[0], r_knee_centroid[0], r_ankle_centroid[0], r_min_z_point['x']],
    #             [l_min_z_point['y'], l_ankle_centroid[1], l_knee_centroid[1], l_hip_centroid[1], hip_centroid[1],
    #              r_hip_centroid[1], r_knee_centroid[1], r_ankle_centroid[1], r_min_z_point['y']], linewidth=2,
    #             color='r')
    #
    # zoy_ax.plot([l_min_z_point['z'], l_ankle_centroid[2], l_knee_centroid[2], l_hip_centroid[2], hip_centroid[2],
    #              r_hip_centroid[2], r_knee_centroid[2], r_ankle_centroid[2], r_min_z_point['z']],
    #             [l_min_z_point['y'], l_ankle_centroid[1], l_knee_centroid[1], l_hip_centroid[1], hip_centroid[1],
    #              r_hip_centroid[1], r_knee_centroid[1], r_ankle_centroid[1], r_min_z_point['y']], linewidth=2,
    #             color='r')

    xoy_ax.scatter([l_ankle_centroid[0], r_ankle_centroid[0]],
                   [l_ankle_centroid[1], r_ankle_centroid[1]], s=50)
    xoy_ax.scatter([l_knee_centroid[0], r_knee_centroid[0]],
                   [l_knee_centroid[1], r_knee_centroid[1]], s=50)
    xoy_ax.scatter([l_hip_centroid[0], r_hip_centroid[0]],
                   [l_hip_centroid[1], r_hip_centroid[1]], s=50)
    xoy_ax.scatter([hip_centroid[0]],
                   [hip_centroid[1]], s=50)

    zoy_ax.scatter([l_ankle_centroid[2], r_ankle_centroid[2]],
                   [l_ankle_centroid[1], r_ankle_centroid[1]], s=50)
    zoy_ax.scatter([l_knee_centroid[2], r_knee_centroid[2]],
                   [l_knee_centroid[1], r_knee_centroid[1]], s=50)
    zoy_ax.scatter([l_hip_centroid[2], r_hip_centroid[2]],
                   [l_hip_centroid[1], r_hip_centroid[1]], s=50)
    zoy_ax.scatter([hip_centroid[2]],
                   [hip_centroid[1]], s=50)
    plt.show()
    print(hip_centroid)
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

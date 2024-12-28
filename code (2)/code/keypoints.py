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
img_dir = 'E:\\chenguo-code\\all\\normal\\img\\'


class KmeansCluster:
    def __init__(self):
        self.iter_cnt = 0
        self.pA_index_list = []
        self.pB_index_list = []
        self.pA = None
        self.pB = None

    def calculate(self, points, pA, pB):
        _pA_index_list = []
        _pB_index_list = []
        for i in range(len(points)):
            point = points[i]
            distA = (point[0] - pA[0]) ** 2
            distB = (point[0] - pB[0]) ** 2
            if distA <= distB:
                _pA_index_list.append(i)
            else:
                _pB_index_list.append(i)

        df = pd.DataFrame(points, columns=['x', 'y'])
        pA_df = pd.DataFrame(df, index=_pA_index_list, columns=['x', 'y'])
        pB_df = pd.DataFrame(df, index=_pB_index_list, columns=['x', 'y'])
        new_pA = [pA_df['x'].sum() / pA_df['x'].size, pA_df['y'].sum() / pA_df['y'].size]
        new_pB = [pB_df['x'].sum() / pB_df['x'].size, pB_df['y'].sum() / pB_df['y'].size]

        if (pA == new_pA) and (pB == new_pB):
            self.pA_index_list = _pA_index_list
            self.pB_index_list = _pB_index_list
            self.pA = new_pA
            self.pB = new_pB
            return [self.pA, self.pB]

        self.iter_cnt += 1

        return self.calculate(points, new_pA, new_pB)


def main():
    global csv_dir
    global img_dir
    file_list = os.listdir(csv_dir)
    file_count = len(file_list)

    plt.ion()
    fig = plt.figure(1, figsize=[14, 8])

    x_min = -0.3
    x_max = 0.3
    y_min = -0.19
    y_max = 0.5

    # x_min = 0
    # x_max = 1
    # y_min = 0
    # y_max = 1
    z_min = 0
    z_max = 1

    plt.pause(1)

    l_knee_z_list = []
    l_thigh_z_list = []
    l_ankle_z_list = []

    for i in range(file_count):
        if i < 65:
            continue
        fig.clf()
        thigh_ax = fig.add_subplot(3, 2, 2)
        knee_ax = fig.add_subplot(3, 2, 4)
        ankle_ax = fig.add_subplot(3, 2, 6)
        xyz_ax = fig.add_subplot(1, 2, 1, projection='3d', azim=-90, elev=120)
        xyz_ax.set(xlabel='X', ylabel='Y', zlabel='Z', xlim=[x_min, x_max], ylim=[y_min, y_max], zlim=[z_min, z_max])

        thigh_ax.set_ylim([z_min, z_max])
        thigh_ax.set_ylabel("x")

        knee_ax.set_ylim([z_min, z_max])
        knee_ax.set_ylabel('y')

        ankle_ax.set_ylim([z_min, z_max])
        ankle_ax.set_ylabel('z')
        ankle_ax.set_xlabel("frame number")

        file_name = csv_dir + "{}.csv".format(i)
        df = pd.read_csv(file_name)
        df.drop_duplicates(inplace=False)
        print(df['z'].min(),df['z'].max())

        df['z'] = 1-(df['z'].add(1.3) * 1.82)

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

        df_l_zoy = pd.DataFrame(df_l, columns=['y', 'z'])
        df_r_zoy = pd.DataFrame(df_r, columns=['y', 'z'])

        df_l_ankle = df_l.loc[df_l['y'] < -0.12]
        df_r_ankle = df_r.loc[df_r['y'] < -0.12]

        df_l_knee = df_l.loc[(df_l['y'] > 0.1) & (df_l['y'] < 0.2)]
        df_r_knee = df_r.loc[(df_r['y'] > 0.1) & (df_r['y'] < 0.2)]

        df_l_thigh = df_l.loc[(df_l['y'] > 0.4)]
        df_r_thigh = df_r.loc[(df_r['y'] > 0.4)]

        l_ankle_centroid = cal_centroid(df_l_ankle)
        r_ankle_centroid = cal_centroid(df_r_ankle)
        l_knee_centroid = cal_centroid(df_l_knee)
        r_knee_centroid = cal_centroid(df_r_knee)
        l_thigh_centroid = cal_centroid(df_l_thigh)
        r_thigh_centroid = cal_centroid(df_r_thigh)

        l_knee_z_list.append(l_knee_centroid[2])
        l_ankle_z_list.append(l_ankle_centroid[2])
        l_thigh_z_list.append(l_thigh_centroid[2])

        xyz_ax.scatter(df_l_ankle['x'], df_l_ankle['y'], df_l_ankle['z'], s=1)
        xyz_ax.scatter(df_r_ankle['x'], df_r_ankle['y'], df_r_ankle['z'], s=1)
        xyz_ax.scatter(df_l_knee['x'], df_l_knee['y'], df_l_knee['z'], s=1)
        xyz_ax.scatter(df_r_knee['x'], df_r_knee['y'], df_r_knee['z'], s=1)
        xyz_ax.scatter(df_l_thigh['x'], df_l_thigh['y'], df_l_thigh['z'], s=1)
        xyz_ax.scatter(df_r_thigh['x'], df_r_thigh['y'], df_r_thigh['z'], s=1)

        ankle_ax.plot(l_knee_z_list)
        ankle_ax.plot(l_ankle_z_list)
        ankle_ax.plot(l_thigh_z_list)

        # xyz_ax.plot([l_ankle_centroid[0],l_knee_centroid[0],l_thigh_centroid[0]],
        #             [l_ankle_centroid[1],l_knee_centroid[1],l_thigh_centroid[1]],
        #             [l_ankle_centroid[2],l_knee_centroid[2],l_thigh_centroid[2]])
        #
        # xyz_ax.plot([r_ankle_centroid[0],r_knee_centroid[0],r_thigh_centroid[0]],
        #             [r_ankle_centroid[1],r_knee_centroid[1],r_thigh_centroid[1]],
        #             [r_ankle_centroid[2],r_knee_centroid[2],r_thigh_centroid[2]])
        # xyz_ax.scatter(l_ankle_centroid[0], l_ankle_centroid[1], l_ankle_centroid[2], s=20)
        # xyz_ax.scatter(r_ankle_centroid[0], r_ankle_centroid[1], r_ankle_centroid[2], s=20)
        # xyz_ax.scatter(l_knee_centroid[0], l_knee_centroid[1], l_knee_centroid[2], s=20)
        # xyz_ax.scatter(r_knee_centroid[0], r_knee_centroid[1], r_knee_centroid[2], s=20)
        # xyz_ax.scatter(l_thigh_centroid[0], l_thigh_centroid[1], l_thigh_centroid[2], s=20)
        # xyz_ax.scatter(r_thigh_centroid[0], r_thigh_centroid[1], r_thigh_centroid[2], s=20)



        plt.pause(0.01)
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

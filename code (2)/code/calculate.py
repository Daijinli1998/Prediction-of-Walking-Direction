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
    y_min = -0.15
    y_max = 0.4
    z_min = -0.9
    z_max = -0.1
    plt.pause(3)
    last_centroids = [99, 99, 99]
    centroids = [0, 0, 0]
    centroids_list = []
    x_change = []
    y_change = []
    z_change = []
    for i in range(file_count):
        fig.clf()
        x_ax = fig.add_subplot(3, 2, 2)
        y_ax = fig.add_subplot(3, 2, 4)
        z_ax = fig.add_subplot(3, 2, 6)
        xyz_ax = fig.add_subplot(1, 2, 1, projection='3d', azim=-90, elev=120)
        # xyz_ax.set(xlabel='X', ylabel='Y', zlabel='Z', xlim=[x_min, x_max], ylim=[y_min, y_max], zlim=[z_min, z_max])
        xyz_ax.set(xlabel='X', ylabel='Y', zlabel='Z', xlim=[0, 1], ylim=[0, 1], zlim=[0, 1])
        # x_ax.set_ylim([-0.05, 0.05])
        # z_ax.set_ylim([-0.1, 0.1])
        # y_ax.set_ylim([-0.05, 0.05])

        # x_ax.set_ylim([-0.3, 0.3])
        # y_ax.set_ylim([-0.1, 0.1])
        # z_ax.set_ylim([-1.0, 0.0])

        x_ax.set_ylim([0, 1])
        y_ax.set_ylim([0, 1])
        z_ax.set_ylim([0, 1])
        z_ax.set_xlabel("frame number")
        x_ax.set_ylabel("x")
        y_ax.set_ylabel('y')
        z_ax.set_ylabel('z')

        file_name = csv_dir + "{}.csv".format(i)
        df = pd.read_csv(file_name)
        df.drop_duplicates(inplace=False)
        df = df.dropna()

        df['x'] = df['x'].add(0.3) * 1.67
        df['y'] = df['y'].add(0.15) * 1.82
        df['z'] = df['z'].add(0.9) * 1.25

        tmp_centroids = cal_centroid(df)
        centroids_list.append(tmp_centroids)
        print(tmp_centroids[0], tmp_centroids[1], tmp_centroids[2])
        if last_centroids[0] != 99:
            centroids[0] = tmp_centroids[0] - last_centroids[0]
            centroids[1] = tmp_centroids[1] - last_centroids[1]
            centroids[2] = tmp_centroids[2] - last_centroids[2]

        print(last_centroids[0], last_centroids[1], last_centroids[2])
        print(centroids[0], centroids[1], centroids[2])
        print("-------------------------------")
        last_centroids = tmp_centroids
        x_change.append(centroids[0])
        y_change.append(centroids[1])
        z_change.append(centroids[2])

        # x_ax.plot(x_change)
        # y_ax.plot(y_change)
        # z_ax.plot(z_change)
        tmp = np.array(centroids_list, dtype=float)
        x_ax.plot(tmp[:, 0])
        y_ax.plot(tmp[:, 1])
        z_ax.plot(tmp[:, 2])

        xyz_ax.scatter(df['x'], df['y'], df['z'], s=1)

        img = Image.open(os.path.join(img_dir, '{}.jpg'.format(i)))
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.flip(img, 0)
        plt.ioff()
        cv2.imshow("color", img)

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

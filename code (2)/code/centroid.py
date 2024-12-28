import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np

root_dir = 'E:\\chenguo-code\\all\\normal\\'

dir_list = []

# dir_list = ['20221113_200701', '20221113_095901', '20221113_095956', '20221113_100017', '20221113_100105',
#             '20221113_100153', '20221113_200207', '20221113_200635', '20221203_170630', '20221203_170850',
#             '20221203_170915', '20221203_170943', '20221203_171011', '20221203_171047', '20221203_171110',
#             '20221203_171138', '20221203_171211', '20221203_171233', '20221203_171536', '20221203_171606',
#             '20221203_171625', '20221203_171919', '20221203_171946', '20221203_172001', '20221203_172024',
#             '20221203_172037', '20221203_172058', '20221203_172104', '20221203_172125', '1',
#             '111']

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
        file_list = os.listdir(csv_dir)
        file_count = len(file_list)

        plt.ion()
        fig = plt.figure(1, figsize=[14, 8])

        x_min = -0.3
        x_max = 0.3
        z_min = -1.2
        z_max = 0
        plt.pause(1)

        centroid_list = []

        for i in range(file_count):
            print(i, '.csv')
            fig.clf()

            x_ax = fig.add_subplot(2, 2, 2)
            z_ax = fig.add_subplot(2, 2, 4)
            xoz_ax = fig.add_subplot(1, 2, 1)

            xoz_ax.set_xlim([x_min, x_max])
            xoz_ax.set_ylim([z_min, z_max])
            x_ax.set_ylim([x_min, x_max])
            z_ax.set_ylim([z_min, z_max])

            file_name = os.path.join(csv_dir, "{}.csv".format(i))
            df = pd.read_csv(file_name)
            df = df.dropna()
            if df.empty is True:
                continue

            xoz_df = pd.DataFrame(df, columns=['x', 'z'])

            xoz_data_list = xoz_df.values
            print(xoz_df)
            xoz_centroid = cal_centroid_xoz(xoz_df)

            centroid_list.append(xoz_centroid)
            centroids = np.array(centroid_list, dtype=np.float32)

            xoz_ax.scatter(xoz_data_list[:, 0], xoz_data_list[:, 1], s=1)
            xoz_ax.scatter([xoz_centroid[0]], [xoz_centroid[1]], s=10)

            x_ax.plot(centroids[:, 0])
            z_ax.plot(centroids[:, 1])

            plt.pause(0.03)

        centroids = np.array(centroid_list, dtype=np.float32)
        centroids = np.around(centroids, 4)
        df = pd.DataFrame(centroids, columns=['x', 'z'])
        print(os.path.join(root_dir, dir, 'centroids.csv'))
        df.to_csv(os.path.join(root_dir, dir, 'centroids.csv'), index=False)

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

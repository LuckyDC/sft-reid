from __future__ import print_function, division
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from sklearn.manifold import TSNE

from utils.misc import load_lst

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import cv2


def viz_heatmap(data):
    maximum = np.max(data)

    data = data / maximum * 255
    heat_map = cv2.applyColorMap(data.astype(np.uint8), cv2.COLORMAP_JET)[:, :, [2, 1, 0]]

    return heat_map


def color_to_indicator(x):
    x = x.squeeze()

    r = np.array([255, 0, 0])
    ind = []
    for i in range(x.shape[0]):
        if np.all(x[i] == r):
            ind.append(0)
        else:
            ind.append(1)
    return np.array(ind)


def calc_ap(x):
    tp = 0
    p = []

    for i in range(x.shape[0]):
        if x[i] == 1:
            tp += 1
            p.append(tp / (i + 1))

    return np.mean(p)


def query_plot_lst(query_idx, top_tank_gallery_idx, query_lst, gallery_lst):
    num = top_tank_gallery_idx.shape[0]

    colors = []
    for i in range(num):
        idx = top_tank_gallery_idx[i]

        if query_lst[query_idx].class_id != gallery_lst[idx].class_id:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        colors.append(np.array(color))

    colors = np.stack(colors, axis=0)[np.newaxis, :]

    return colors


if __name__ == '__main__':
    query_lst_path = "/mnt/truenas/scratch/chuanchen.luo/data/reid/duke-list/query.lst"
    gallery_lst_path = "/mnt/truenas/scratch/chuanchen.luo/data/reid/duke-list/test.lst"
    prefix = "duke"
    query_features_path = "features/query-%s.mat" % prefix
    gallery_features_path = "features/gallery-%s.mat" % prefix

    query_features = sio.loadmat(query_features_path)["feat"]
    gallery_features = sio.loadmat(gallery_features_path)["feat"]

    # Custom Parameter
    top_k = 75
    alpha = 0.95

    temps = [1.0, 0.1, 0.05]

    # Start to propagation
    query_lst = load_lst(query_lst_path)
    gallery_lst = load_lst(gallery_lst_path)

    rank_list = []
    num_query = query_features.shape[0]
    for i in tqdm(range(100)):
        q_feat = query_features[i]

        # Initial P2G affinity vector
        y_0 = np.dot(gallery_features, q_feat)

        rank_index = np.argsort(-y_0)
        top_k_index = rank_index[:top_k]

        y_0 = y_0[top_k_index]

        # G2G affinity matrix
        g_feats = gallery_features[top_k_index, :]

        A = np.dot(g_feats, g_feats.T)

        heat_maps = []
        top_rank_lists = []
        for temperature in temps:
            W = np.exp(A / temperature)
            W = W * (1 - np.identity(top_k))
            W = W / np.sum(W, axis=1, keepdims=True)

            # closed form solution
            W_ = (1 - alpha) * np.linalg.inv(np.identity(top_k) - alpha * W)

            heat_maps.append(viz_heatmap(W_))

            y = np.dot(W_, y_0)

            top_rank_lists.append(top_k_index[np.argsort(-y)])

        plt.figure(0, figsize=(20, 10))
        num_temp = len(temps)

        for t in range(num_temp):
            ax = plt.subplot(2, num_temp, t + 1)
            ax.set_title("Temp = %g" % temps[t])
            ax.imshow(heat_maps[t])

            ax = plt.subplot(2, num_temp, t + num_temp + 1)
            plot = query_plot_lst(i, top_rank_lists[t], query_lst, gallery_lst)
            plt.yticks([])
            ax.imshow(plot)

            ind = color_to_indicator(plot)
            ap = calc_ap(ind)
            ax.set_title("%.2f%%" % (ap * 100))

        plt.tight_layout()
        plt.savefig("%d.jpg" % i)
        plt.close(0)

from __future__ import print_function
from __future__ import division

import cv2
import numpy as np
import scipy.io as sio
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize


def query_plot_lst(query_img, rank_gallery_imgs, query_id, gallery_ids, query_cam_id, gallery_cam_ids, save_name,
                   display_topk=None):
    q = cv2.cvtColor(cv2.imread(query_img), cv2.COLOR_BGR2RGB)

    if display_topk is None:
        num = len(rank_gallery_imgs)
    else:
        num = display_topk

    plt.figure(0, figsize=(50, 12))
    plt.subplot(1, num + 1, 1)
    plt.imshow(cv2.resize(q, (128, 256)))
    plt.xticks([])
    plt.yticks([])

    mask = np.not_equal(gallery_cam_ids, query_cam_id)
    rank_gallery_imgs = rank_gallery_imgs[mask]
    gallery_ids = gallery_ids[mask]

    for i in range(num):
        img = rank_gallery_imgs[i]

        g = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        plt.subplot(1, num + 1, i + 2)
        plt.imshow(cv2.resize(g, (128, 256)))

        if query_id != gallery_ids[i]:
            color = "red"
        else:
            color = "green"

        plt.xticks([])
        plt.yticks([])

        ax = plt.gca()
        ax.spines['right'].set_linewidth(15)
        ax.spines['top'].set_linewidth(15)
        ax.spines['left'].set_linewidth(15)
        ax.spines['bottom'].set_linewidth(15)
        ax.spines['right'].set_color(color)
        ax.spines['top'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.spines['bottom'].set_color(color)

    plt.tight_layout()
    plt.savefig(save_name)
    plt.close(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str, choices=["baseline", "proposed", "re-rank"])

    args = parser.parse_args()
    prefix = args.prefix

    plot = True

    if prefix == "baseline":
        gallery_mat_path = "features/duke/gallery-baseline-amsoftmax0.3-relu-2-140ep.mat"
        query_mat_path = "features/duke/query-baseline-amsoftmax0.3-relu-2-140ep.mat"
    else:
        gallery_mat_path = "features/duke/gallery-baseline-gcn0.1-dsup-amsoftmax0.3-relu-2-140ep.mat"
        query_mat_path = "features/duke/query-baseline-gcn0.1-dsup-amsoftmax0.3-relu-2-140ep.mat"

    if prefix == "re-rank":
        re_rank = True
    else:
        re_rank = False

    temperature = 0.1
    top_k = 50

    g_mat = sio.loadmat(gallery_mat_path)
    q_mat = sio.loadmat(query_mat_path)

    gallery_features = normalize(g_mat["feat"])
    gallery_ids = g_mat["ids"].squeeze()
    gallery_cam_ids = g_mat["cam_ids"].squeeze()
    gallery_imgs = g_mat["imgs"].squeeze()

    query_features = normalize(q_mat["feat"])
    query_ids = q_mat["ids"].squeeze()
    query_cam_ids = q_mat["cam_ids"].squeeze()
    query_imgs = q_mat["imgs"].squeeze()

    order = np.argsort(query_imgs)
    query_features = query_features[order]
    query_ids = query_ids[order]
    query_cam_ids = query_cam_ids[order]
    query_imgs = query_imgs[order]

    # hard_example = [48, 49, 67, 138, 147, 164, 179, 184, 208, 253, 306, 314, 357, 362, 418, 430, 436, 466, 470, 479,
    #                 482, 484, 503, 512, 548, 574, 576, 577, 580, 601, 614, 629, 655, 657, 682, 784, 839, 939, 957, 958,
    #                 961, 968, 986, 1016, 1067, 1068, 1069, 1073, 1097, 1122, 1228, 1241, 1254, 1320, 1496, 1567, 1600,
    #                 1642, 1654, 1682, 1719, 1722, 1753, 1767, 1788, 1819, 1859, 1883, 1888, 1891, 1897, 1944, 1991,
    #                 2017, 2033, 2125, 2202, 2211, 2219]

    hard_example = [253]

    for i in hard_example:
        q_feat = query_features[i]
        q_cam_id = query_cam_ids[i]
        q_id = query_ids[i]
        q_img = query_imgs[i]

        dist = -np.dot(gallery_features, q_feat)
        rank_list = np.argsort(dist)
        top_k_rank_list = rank_list[:top_k]

        top_k_g_feats = gallery_features[top_k_rank_list, :]
        if re_rank:
            W = np.dot(top_k_g_feats, top_k_g_feats.T)
            W = np.exp(W / temperature)
            W = W / W.sum(axis=1, keepdims=True)
            top_k_g_feats = np.dot(W, top_k_g_feats)

            y = -np.dot(top_k_g_feats, q_feat)
            rank_list[:top_k] = top_k_rank_list[np.argsort(y)]

        if plot:
            query_plot_lst(q_img, gallery_imgs[rank_list], q_id, gallery_ids[rank_list], q_cam_id,
                           gallery_cam_ids[rank_list], save_name="retrieval_viz/%04d-%s.jpg" % (i, prefix),
                           display_topk=7)

            print(i)

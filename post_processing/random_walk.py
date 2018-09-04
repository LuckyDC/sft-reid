from __future__ import print_function, division, absolute_import
import time
import argparse
import numpy as np
import scipy.io as sio

from pprint import pprint
from easydict import EasyDict
from tqdm import tqdm
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import normalize

from utils.evaluation import eval_rank_list
from utils.misc import viz_heatmap


def random_walk_iterative(p2g, g2g, alpha, threshold=1e-3, max_iter=300):
    y_0 = p2g
    old_y = p2g
    P = np.identity(g2g.shape[0])

    num_iter = 0

    while True:
        P = alpha * np.dot(g2g, P) + (1 - alpha) * np.identity(P.shape[0])
        print(P)
        y = np.dot(P, y_0)
        time.sleep(1)

        num_iter += 1
        if np.linalg.norm(y - old_y) < threshold or num_iter > max_iter:
            return y

        old_y = y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=['cuhk', 'market', 'duke'])
    parser.add_argument("prefix", type=str)
    args = parser.parse_args()

    prefix = args.prefix
    dataset = args.dataset

    query_features_path = "features/%s/query-%s.mat" % (dataset, prefix)
    gallery_features_path = "features/%s/gallery-%s.mat" % (dataset, prefix)
    query_lst_path = "/mnt/truenas/scratch/chuanchen.luo/data/reid/%s-list/query.lst" % dataset
    gallery_lst_path = "/mnt/truenas/scratch/chuanchen.luo/data/reid/%s-list/test.lst" % dataset

    query_mat = sio.loadmat(query_features_path)
    gallery_mat = sio.loadmat(gallery_features_path)

    query_features = query_mat["feat"]
    query_ids = query_mat["ids"].squeeze()
    query_cam_ids = query_mat["cam_ids"].squeeze()

    gallery_features = gallery_mat["feat"]
    gallery_ids = gallery_mat["ids"].squeeze()
    gallery_cam_ids = gallery_mat["cam_ids"].squeeze()

    query_features = normalize(query_features)
    gallery_features = normalize(gallery_features)

    # Custom parameters
    args = EasyDict()

    args.top_k = 75
    args.alpha = 0.95
    args.temperature = 1.0

    args.closed_or_iterative = 0
    args.preserve_diag = True

    pprint(args)

    # Start to propagation
    rank_list = []
    num_query = query_features.shape[0]
    for i in tqdm(range(num_query)):
        q_feat = query_features[i]

        # Initial P2G affinity vector
        y_0 = np.dot(gallery_features, q_feat)

        rank_index = np.argsort(-y_0)
        top_k_index = rank_index[:args.top_k]

        y_0 = y_0[top_k_index]

        # G2G affinity matrix
        g_feats = gallery_features[top_k_index, :]
        # g_feats = g_feats - np.mean(g_feats, axis=0, keepdims=True)
        W = np.dot(g_feats, g_feats.T)
        W = np.exp(W / args.temperature)
        if not args.preserve_diag:
            W = W * (1 - np.identity(args.top_k))

        W /= W.sum(axis=1, keepdims=True)

        if args.closed_or_iterative == 0:
            # closed form solution
            W_ = (1 - args.alpha) * np.linalg.inv(np.identity(args.top_k) - args.alpha * W)

            # viz_heatmap(np.diag(W.sum(1))-W, "heatmap%d" % i)
            # viz_heatmap(W, "heatmap%d" % i)
            # print(np.linalg.eig(W)[0])
            # print(np.linalg.eig(W_)[0])

            y = np.dot(W_, y_0)
        else:
            # iterative solution
            y = random_walk_iterative(y_0, W, args.alpha)

        rank_index[:args.top_k] = top_k_index[np.argsort(-y)]
        rank_list.append(rank_index)

    # evaluation
    print("Evaluating...")
    eval_rank_list(rank_list, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)

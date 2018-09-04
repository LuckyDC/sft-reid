from __future__ import print_function, division, absolute_import
import argparse
import numpy as np
import scipy.io as sio

from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import normalize
from tqdm import tqdm
from easydict import EasyDict
from pprint import pprint

from utils.evaluation import eval_rank_list
from post_processing.tranform_func import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=['cuhk', 'market', 'duke', 'msmt'])
    parser.add_argument("prefix", type=str)
    args = parser.parse_args()

    prefix = args.prefix
    dataset = args.dataset

    args = EasyDict()

    args.preserve_diag = True

    args.include_query = False
    args.eu_dist = False

    args.top_k = 50

    args.temperature = 0.1
    args.eta = 0.9
    args.n = 5

    transform_func = None
    args.num_loop = 1

    if transform_func is not None:
        args.transform_func_name = transform_func.__name__
    else:
        args.transform_func_name = None

    print(dataset + "/" + prefix)
    pprint(args)
    set_transform_args(args)

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

    # Start to propagation
    rank_list = []
    for i in tqdm(range(query_features.shape[0])):
        q_feat = query_features[i]

        # Initial P2G affinity vector
        y_0 = np.dot(gallery_features, q_feat)

        rank_index = np.argsort(-y_0)
        top_k_index = rank_index[:args.top_k]

        g_feats = gallery_features[top_k_index, :]

        if args.include_query:
            g_feats = np.concatenate([q_feat.reshape([1, -1]), g_feats])

        for i in range(args.num_loop):
            # G2G affinity matrix
            W = np.dot(g_feats, g_feats.T)
            W = np.exp(W / args.temperature)

            if not args.preserve_diag:
                W = W * (1 - np.identity(W.shape[0]))
                W += np.identity(W.shape[0])

            W = W / W.sum(axis=1, keepdims=True)

            if transform_func is not None:
                W = transform_func(W)

            g_feats = np.dot(W, g_feats)

            # W = W * (1 - np.identity(W.shape[0]))
            # D = np.diag(W.sum(axis=1) ** (-0.5))
            # W = np.dot(D, W).dot(D) + np.identity(W.shape[0])
            # g_feats = normalize(np.dot(W, g_feats), axis=1) + g_feats
            # g_feats = normalize(g_feats, axis=1)

            # print(g_feats)

        if not args.include_query:
            y = np.dot(g_feats, q_feat) if not args.eu_dist else -euclidean_distances(g_feats, [q_feat]).squeeze()
        else:
            y = np.dot(g_feats[1:], g_feats[0]) if not args.eu_dist \
                else -euclidean_distances(g_feats[1:], g_feats[0].reshape(1, -1)).squeeze()

        rank_index[:args.top_k] = top_k_index[np.argsort(-y)]
        rank_list.append(rank_index)

    # evaluation
    print("Evaluating...")
    eval_rank_list(rank_list, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)

from __future__ import print_function, division, absolute_import
import argparse
import numpy as np
import scipy.io as sio

from sklearn.preprocessing import normalize
from tqdm import tqdm
from easydict import EasyDict
from pprint import pprint

from utils.evaluation import eval_rank_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=['cuhk', 'market', 'duke', 'msmt'])
    parser.add_argument("prefix", type=str)
    args = parser.parse_args()

    prefix = args.prefix
    dataset = args.dataset

    args = EasyDict()

    if dataset == "msmt":
        args.temperature = 0.02
        args.top_k = 150
    else:
        args.temperature = 0.1
        args.top_k = 50

    print(dataset + "/" + prefix)
    pprint(args)

    query_features_path = "features/%s/query-%s.mat" % (dataset, prefix)
    gallery_features_path = "features/%s/gallery-%s.mat" % (dataset, prefix)

    query_mat = sio.loadmat(query_features_path)
    gallery_mat = sio.loadmat(gallery_features_path)

    query_features = query_mat["feat"]
    query_ids = query_mat["ids"].squeeze()
    query_cam_ids = query_mat["cam_ids"].squeeze()

    gallery_features = gallery_mat["feat"]
    gallery_ids = gallery_mat["ids"].squeeze()
    gallery_cam_ids = gallery_mat["cam_ids"].squeeze()

    # l2 normalize
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

        # G2G affinity matrix
        W = np.dot(g_feats, g_feats.T)
        W = np.exp(W / args.temperature)
        W = W / W.sum(axis=1, keepdims=True)

        g_feats = np.dot(W, g_feats)

        # recompute top-k ranking list
        y = np.dot(g_feats, q_feat)

        rank_index[:args.top_k] = top_k_index[np.argsort(-y)]
        rank_list.append(rank_index)

    # evaluation
    print("Evaluating...")
    eval_rank_list(rank_list, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)

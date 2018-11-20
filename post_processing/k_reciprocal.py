from __future__ import print_function, absolute_import
import argparse
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import normalize
from sklearn.metrics import euclidean_distances

from utils.misc import euclidean_dist

from utils.re_ranking_ranklist import re_ranking
from utils.evaluation import eval_rank_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["duke", "cuhk", "market", "msmt"])
    parser.add_argument("prefix", type=str)
    args = parser.parse_args()

    prefix = args.prefix
    dataset = args.dataset
    hop2 = True
    metric = "euclidean"

    print(dataset + "/" + prefix)

    if dataset == "market":
        k1 = 20
        k2 = 6
        lambda_value = 0.3
    elif dataset == "cuhk":
        k1 = 7
        k2 = 3
        lambda_value = 0.85
    elif dataset == "duke":
        k1 = 20
        k2 = 6
        lambda_value = 0.3
    elif dataset == "msmt":
        k1 = 20
        k2 = 6
        lambda_value = 0.3
    else:
        raise ValueError("Invalid Dataset!")

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

    if metric == "euclidean":
        print("calculating G2G dists...")
        g_g_dist = euclidean_distances(gallery_features)
        print("calculating P2G dists...")
        q_g_dist = euclidean_distances(query_features, gallery_features)
        print("calculating P2P dists...")
        q_q_dist = euclidean_distances(query_features)

    elif metric == "cosine":
        query_features = normalize(query_features)
        gallery_features = normalize(gallery_features)

        print("calculating P2G dists...")
        q_g_dist = 1 - np.dot(query_features, gallery_features.T)
        print("calculating P2P dists...")
        q_q_dist = 1 - np.dot(query_features, query_features.T)
        print("calculating G2G dists...")
        g_g_dist = 1 - np.dot(gallery_features, gallery_features.T)
    else:
        raise ValueError("Invalid metric!")

    del gallery_features
    del query_features

    print("re-ranking...")
    final_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1, k2, lambda_value, hop2=hop2)

    rank_list = np.argsort(final_dist, axis=1)

    print("evaluating..")
    eval_rank_list(rank_list, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)

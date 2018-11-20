import os
import subprocess
import sys
import argparse

import numpy as np
import mxnet as mx
import scipy.io as sio
from utils.evaluation import eval_feature

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", default=0, type=int)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--distractor", action="store_true")
    args = parser.parse_args()

    model_path = args.model_path
    basename = os.path.splitext(os.path.basename(model_path))[0]
    prefix = "-".join(basename.split("-")[:-1])
    epoch_idx = int(basename.split("-")[-1])
    gpu = args.gpu
    dataset = model_path.split("/")[1]

    assert dataset in ["market", "cuhk", "duke", "msmt"]

    print("%s/%s-%d" % (dataset, prefix, epoch_idx))

    # extract feature
    cmd = "python%c -m utils.extract --prefix %s --gpu-id %d --epoch-idx %d --dataset %s" % (
        sys.version[0], prefix, gpu, epoch_idx, dataset)

    if args.distractor and dataset == "market":
        cmd += " --distractor"

    subprocess.check_call(cmd.split(" "))

    # load extracted feature
    query_features_path = 'features/%s/query-%s.mat' % (dataset, prefix)
    gallery_features_path = "features/%s/gallery-%s.mat" % (dataset, prefix)
    assert os.path.exists(query_features_path) and os.path.exists(gallery_features_path)

    query_mat = sio.loadmat(query_features_path)
    gallery_mat = sio.loadmat(gallery_features_path)

    query_features = query_mat["feat"]
    query_ids = query_mat["ids"].squeeze()
    query_cam_ids = query_mat["cam_ids"].squeeze()

    gallery_features = gallery_mat["feat"]
    gallery_ids = gallery_mat["ids"].squeeze()
    gallery_cam_ids = gallery_mat["cam_ids"].squeeze()

    eval_feature(query_features, gallery_features, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids,
                 ctx=mx.gpu(args.gpu), metric="cosine")

    if args.distractor:
        for i in range(3):
            distractor_features_path = 'features/%s/distractor%d-%s.mat' % (dataset, i + 1, prefix)
            distractor_mat = sio.loadmat(distractor_features_path)

            distractor_features = distractor_mat["feat"]
            distractor_ids = distractor_mat["ids"].squeeze()
            distractor_cam_ids = distractor_mat["cam_ids"].squeeze()

            gallery_features = np.concatenate([gallery_features, distractor_features], axis=0)
            gallery_ids = np.concatenate([gallery_ids, distractor_ids], axis=0)
            gallery_cam_ids = np.concatenate([gallery_cam_ids, distractor_cam_ids], axis=0)

            eval_feature(query_features, gallery_features, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids,
                         ctx=mx.gpu(i) if i < 2 else mx.cpu(), metric="cosine")

            del distractor_features
            del distractor_cam_ids
            del distractor_ids

    print("\n")

    cmd = "python%c -m post_processing.post_clustering %s %s" % (sys.version[0], dataset, prefix)
    subprocess.check_call(cmd.split(" "))

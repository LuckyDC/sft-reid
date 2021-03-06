from __future__ import print_function

import os
import yaml
import glob
import argparse
import numpy as np
import mxnet as mx
import scipy.io as sio

from utils.iterators import get_test_iterator
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])


def extract_feature(model, iterator):
    feature = []
    ids = []
    cam_ids = []
    imgs = []

    for batch in iterator:
        model.forward(Batch(data=batch.data))
        output = model.get_outputs()[0]
        output = output.asnumpy()

        feature.append(output)
        ids.append(batch.label[0].asnumpy())
        cam_ids.append(batch.label[1].asnumpy())
        imgs.append(batch.label[2])

    feature = np.concatenate(feature, axis=0)
    ids = np.concatenate(ids, axis=0)
    cam_ids = np.concatenate(cam_ids, axis=0)
    imgs = np.concatenate(imgs, axis=0)

    assert feature.shape[0] == ids.shape[0] == cam_ids.shape[0]

    return feature, ids, cam_ids, imgs


if __name__ == '__main__':
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--epoch-idx", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--image-size", type=str, default="256,128")
    parser.add_argument("--dataset", type=str, required=True, choices=["duke", "market", "cuhk", "msmt"])
    parser.add_argument("--distractor", action="store_true")
    args = parser.parse_args()

    config = yaml.load(open("config.yml", "r"))

    context = mx.gpu(args.gpu_id)
    image_size = tuple([int(i) for i in args.image_size.split(",")])

    # load checkpoint
    load_model_prefix = "models/%s/%s" % (args.dataset, args.prefix)
    symbol, arg_params, aux_params = mx.model.load_checkpoint(load_model_prefix, args.epoch_idx)
    flatten = symbol.get_internals()["flatten_output"]
    model = mx.mod.Module(symbol=flatten, data_names=["data"], label_names=None, context=context)
    model.bind(data_shapes=[('data', (args.batch_size, 3) + image_size)], for_training=False, force_rebind=True)
    model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True, force_init=True)

    feat_root = "features/" + args.dataset

    # extract query feature
    query = os.path.join(config[args.dataset]["root"], config[args.dataset]["query"])
    q_iterator = get_test_iterator(root=query,
                                   dataset=args.dataset,
                                   batch_size=args.batch_size,
                                   image_size=image_size,
                                   num_worker=8)

    q_feat, q_ids, q_cam_ids, q_imgs = extract_feature(model, q_iterator)
    print(q_feat.shape)

    save_name = "{}/query-{}".format(feat_root, args.prefix)
    sio.savemat(save_name, {"feat": q_feat, "ids": q_ids, "cam_ids": q_cam_ids, "imgs": q_imgs})

    # extract gallery feature
    gallery = os.path.join(config[args.dataset]["root"], config[args.dataset]["gallery"])
    g_iterator = get_test_iterator(root=gallery,
                                   dataset=args.dataset,
                                   batch_size=args.batch_size,
                                   image_size=image_size,
                                   num_worker=8)

    g_feat, g_ids, g_cam_ids, g_imgs = extract_feature(model, g_iterator)
    print(g_feat.shape)

    save_name = "{}/gallery-{}".format(feat_root, args.prefix)
    sio.savemat(save_name, {"feat": g_feat, "ids": g_ids, "cam_ids": g_cam_ids, "imgs": g_imgs})

    # extract distractor feature
    if args.distractor:
        distractor_path = config[args.dataset]["distractor"]
        for i, split in enumerate(sorted(glob.glob(os.path.join(args.distractor, "*")))):
            d_iterator = get_test_iterator(root=split,
                                           dataset=args.dataset,
                                           batch_size=args.batch_size,
                                           image_size=image_size,
                                           num_worker=8)

            d_feat, d_ids, d_cam_ids, d_imgs = extract_feature(model, d_iterator)
            print(d_feat.shape)

            save_name = "{}/distractor{}-{}".format(feat_root, i + 1, args.prefix)
            sio.savemat(save_name, {"feat": d_feat, "ids": d_ids, "cam_ids": d_cam_ids, "imgs": d_imgs})

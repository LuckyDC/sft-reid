from __future__ import print_function

import os
import glob
import argparse
import numpy as np
import mxnet as mx
import scipy.io as sio

import utils.augmentor as augmentor
from utils.iterators import EvalIterator
from collections import namedtuple

import operators.triplet_loss

Batch = namedtuple('Batch', ['data'])


def get_iterator(data, dataset, batch_size, image_size):
    transform = augmentor.Compose([augmentor.Cast("float32"), augmentor.Resize(image_size)])

    iterator = EvalIterator(data=data,
                            dataset=dataset,
                            batch_size=batch_size,
                            image_size=image_size,
                            transform=transform,
                            num_worker=16)

    return iterator


def extract_feature(model, iterator):
    feature = []
    ids = []
    cam_ids = []

    for batch in iterator:
        model.forward(Batch(data=batch.data), is_train=False)
        output = model.get_outputs()[0]
        output = output.asnumpy()

        feature.append(output)
        ids.append(batch.label[0].asnumpy())
        cam_ids.append(batch.label[1].asnumpy())

    feature = np.concatenate(feature, axis=0)
    ids = np.concatenate(ids, axis=0)
    cam_ids = np.concatenate(cam_ids, axis=0)

    assert feature.shape[0] == ids.shape[0] == cam_ids.shape[0]

    return feature, ids, cam_ids


if __name__ == '__main__':
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--epoch-idx", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--image-size", type=str, default="256,128")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--gallery", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["duke", "market", "cuhk", "msmt"])
    parser.add_argument("--distractor", type=str)
    args = parser.parse_args()

    context = mx.gpu(args.gpu_id)
    image_size = tuple([int(i) for i in args.image_size.split(",")])

    load_model_prefix = "models/%s/%s" % (args.dataset, args.prefix)

    symbol, arg_params, aux_params = mx.model.load_checkpoint(load_model_prefix, args.epoch_idx)
    flatten = symbol.get_internals()["flatten_output"]
    model = mx.mod.Module(symbol=flatten, data_names=["data"], label_names=None, context=context)
    model.bind(data_shapes=[('data', (args.batch_size, 3) + image_size)], for_training=False, force_rebind=True)

    model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    feat_root = "features/" + args.dataset

    # extract query feature
    q_iterator = get_iterator(args.query, args.dataset, args.batch_size, image_size)

    q_feat, q_ids, q_cam_ids = extract_feature(model, q_iterator)
    print(q_feat.shape)

    save_name = "{}/query-{}".format(feat_root, args.prefix)
    sio.savemat(save_name, {"feat": q_feat, "ids": q_ids, "cam_ids": q_cam_ids})

    # extract gallery feature
    g_iterator = get_iterator(args.gallery, args.dataset, args.batch_size, image_size)

    g_feat, g_ids, g_cam_ids = extract_feature(model, g_iterator)
    print(g_feat.shape)

    save_name = "{}/gallery-{}".format(feat_root, args.prefix)
    sio.savemat(save_name, {"feat": g_feat, "ids": g_ids, "cam_ids": g_cam_ids})

    # extract distractor feature
    if args.distractor is not None:
        for i, split in enumerate(sorted(glob.glob(os.path.join(args.distractor, "*")))):
            d_iterator = get_iterator(split, args.dataset, args.batch_size, image_size)

            d_feat, d_ids, d_cam_ids = extract_feature(model, d_iterator)
            print(d_feat.shape)

            save_name = "{}/distractor{}-{}".format(feat_root, i + 1, args.prefix)
            sio.savemat(save_name, {"feat": d_feat, "ids": d_ids, "cam_ids": d_cam_ids})

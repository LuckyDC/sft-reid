from __future__ import print_function

import numpy as np
import mxnet as mx

from utils.debug import *

from collections import namedtuple

Batch = namedtuple('Batch', ['data'])


def get_iterator(batch_size, data_shape, lst_path, rec_path, force_resize):
    print("FORCE_RESIZE: %s" % force_resize)
    if not force_resize:
        iterator = mx.io.ImageRecordIter(
            path_imglist=lst_path,
            path_imgrec=rec_path,
            rand_crop=False,
            rand_mirror=False,
            prefetch_buffer=8,
            preprocess_threads=4,
            shuffle=False,
            label_width=1,
            round_batch=False,
            resize=data_shape[-1],
            data_shape=data_shape,
            batch_size=batch_size)
    else:
        aug_list = [
            mx.image.CastAug(),
            mx.image.ForceResizeAug(data_shape[1:][::-1], interp=1),
            mx.image.ColorNormalizeAug(mean=[0, 0, 0], std=[1, 1, 1])
        ]

        iterator = mx.image.ImageIter(
            batch_size=batch_size,
            data_shape=data_shape,
            label_width=1,
            shuffle=False,
            path_imgrec=rec_path,
            path_imglist=lst_path,
            aug_list=aug_list
        )

    return iterator


import cv2


def extract_feature(model, iterator_or_data, dataset_size=None):
    feature = None

    if isinstance(iterator_or_data, mx.io.DataIter):
        iterator = iterator_or_data

        data_shape = iterator.provide_data[0]
        model.reshape([data_shape])

        batch_size = data_shape[1][0]
        num_iter = dataset_size // batch_size
        extra = dataset_size % batch_size

        feature = []
        iterator.reset()
        for i, batch in enumerate(iterator):
            if i < num_iter:
                data = batch.data[0]
            else:
                data_shape = ("data", (extra,) + data_shape[1][1:])
                model.reshape([data_shape])
                data = batch.data[0][:extra]

            model.forward(Batch(data=[data]), is_train=False)
            output = model.get_outputs()[0]
            output = output.asnumpy()
            feature.append(output)

        feature = np.concatenate(feature, axis=0)

    elif isinstance(iterator_or_data, mx.nd.NDArray):
        data = iterator_or_data

        if data.ndim == 3:
            data = data.expand_dims(0)

        model.reshape([("data", data.shape)])

        model.forward(Batch(data=[data]), is_train=False)
        feature = model.get_outputs()[0].asnumpy()

    return feature


if __name__ == '__main__':
    import scipy.io as sio
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", type=str)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--epoch-idx", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--query-prefix", type=str)
    parser.add_argument("--gallery-prefix", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--force-resize", action="store_true")
    args = parser.parse_args()

    force_resize = args.force_resize
    batch_size = args.batch_size
    crop_size = args.crop_size
    query_lst_path = args.query_prefix + ".lst"
    query_rec_path = args.query_prefix + ".rec"
    gallery_lst_path = args.gallery_prefix + ".lst"
    gallery_rec_path = args.gallery_prefix + ".rec"

    context = mx.gpu(args.gpu_id)

    load_model_prefix = "models/%s" % args.prefix if args.dataset is None \
        else "models/%s/%s" % (args.dataset, args.prefix)
    symbol, arg_params, aux_params = mx.model.load_checkpoint(load_model_prefix, args.epoch_idx)
    flatten = symbol.get_internals()["flatten_output"]
    model = mx.mod.Module(symbol=flatten, context=context)
    model.bind(data_shapes=[('data', (batch_size, 3, crop_size * 2, crop_size))], for_training=False,
               force_rebind=True)

    model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    # extract query feature
    q_len = len(open(query_lst_path).read().splitlines())
    print(q_len)

    q_iterator = get_iterator(batch_size=batch_size,
                              data_shape=(3, crop_size * 2, crop_size),
                              lst_path=query_lst_path,
                              rec_path=query_rec_path,
                              force_resize=force_resize)

    q_feat = extract_feature(model, q_iterator, q_len)
    print(q_feat.shape)

    feat_root = "features/" if args.dataset is None else "features/" + args.dataset

    save_name = "{}/query-{}".format(feat_root, args.prefix)
    sio.savemat(save_name, {"feat": q_feat})

    # extract gallery feature
    g_len = len(open(gallery_lst_path).read().splitlines())
    print(g_len)
    g_iterator = get_iterator(batch_size=batch_size,
                              data_shape=(3, crop_size * 2, crop_size),
                              lst_path=gallery_lst_path,
                              rec_path=gallery_rec_path,
                              force_resize=force_resize)

    g_feat = extract_feature(model, g_iterator, g_len)
    print(g_feat.shape)

    save_name = "{}/gallery-{}".format(feat_root, args.prefix)
    sio.savemat(save_name, {"feat": g_feat})

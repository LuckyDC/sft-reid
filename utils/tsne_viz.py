from __future__ import print_function, division
import os
import cv2
import yaml
import glob
import mxnet as mx
import scipy.io as sio
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from mxnet.io import DataBatch, DataIter
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from collections import namedtuple, defaultdict, OrderedDict

from utils import augmentor
from utils.misc import load_lst, Record

import operators.triplet_loss

TEMPERATURE = 1.0

Batch = namedtuple("Batch", ["data"])


class RawIterator(DataIter):
    def __init__(self, data_dir, batch_size, p_num, image_size):
        self.batch_size = batch_size
        self.image_size = tuple(image_size)
        self.data, self.label = self._process(data_dir, p_num, image_size)
        self.size = self.data.shape[0]
        self.cursor = 0

    @staticmethod
    def _process(data_dir, p_num, image_size):
        img_list = glob.glob(os.path.join(data_dir, "*"))
        img_list.sort()

        id2imgs = defaultdict(list)
        for img_name in img_list:
            idx = int(os.path.basename(img_name).split("_")[0])
            id2imgs[idx].append(img_name)

        imgs = []
        ids = []
        for i, (k, v) in enumerate(id2imgs.items()):
            if i >= p_num:
                break
            imgs.extend(v)
            ids.extend([i] * len(v))

        assert len(imgs) == len(ids)

        data = []
        for i in imgs:
            img = cv2.imread(i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img.shape[:2] != image_size:
                img = cv2.resize(img, (image_size[1], image_size[0]))

            data.append(img)
        data = np.stack(data, axis=0)
        data = np.transpose(data, axes=(0, 3, 1, 2))
        data = mx.nd.array(data)

        label = mx.nd.array(ids)
        return data, label

    @property
    def provide_data(self):
        return [('data', (self.batch_size, 3) + self.image_size)]

    @property
    def provide_label(self):
        return [('label', (self.batch_size,))]

    def next(self):
        if self.cursor >= self.size:
            raise StopIteration

        if self.cursor + self.batch_size <= self.size:
            data = self.data[self.cursor:self.cursor + self.batch_size]
            label = self.label[self.cursor:self.cursor + self.batch_size]
            pad = 0
        else:
            data = self.data[self.cursor:]
            label = self.label[self.cursor:]
            pad = self.cursor + self.batch_size - self.size

        self.cursor += self.batch_size

        return DataBatch(data=[data], label=[label], provide_data=self.provide_data, provide_label=self.provide_label,
                         pad=pad)


def tsne_viz_query2gallery(prefix, dataset, viz_range):
    if not isinstance(viz_range, (list, tuple, int)):
        raise ValueError("viz_range must be list or tuple or int")
    if isinstance(viz_range, int):
        viz_range = list(range(viz_range))

    query_features_path = 'features/%s/query-%s.mat' % (dataset, prefix)
    gallery_features_path = "features/%s/gallery-%s.mat" % (dataset, prefix)
    query_lst_path = "/mnt/truenas/scratch/chuanchen.luo/data/reid/%s-list/query.lst" % dataset
    gallery_lst_path = "/mnt/truenas/scratch/chuanchen.luo/data/reid/%s-list/test.lst" % dataset

    query_lst = np.array(load_lst(query_lst_path))
    gallery_lst = np.array(load_lst(gallery_lst_path))

    query_features = sio.loadmat(query_features_path)["feat"]
    gallery_features = sio.loadmat(gallery_features_path)["feat"]

    query_features = normalize(query_features)
    gallery_features = normalize(gallery_features)

    for i in viz_range:
        q_feat = query_features[i]
        dist = -np.dot(gallery_features, q_feat)
        rank_list = np.argsort(dist)[:150]
        g_feats = gallery_features[rank_list]

        q_record = Record(*query_lst[i])
        g_records = [Record(*item) for item in gallery_lst[rank_list]]

        same_list = [i for i in range(g_feats.shape[0]) if q_record.class_id == g_records[i].class_id]
        diff_list = [i for i in range(g_feats.shape[0]) if q_record.class_id != g_records[i].class_id]

        init_embed = TSNE(n_components=2, init="pca", perplexity=10, metric="cosine").fit_transform(g_feats)

        W = np.dot(g_feats, g_feats.T)
        W = np.exp(W / TEMPERATURE)
        W = W / np.sum(W, axis=1, keepdims=True)

        g_feats = np.dot(W, g_feats)
        res_embed = TSNE(n_components=2, init="pca", perplexity=10, metric="cosine").fit_transform(g_feats)

        plt.figure(0)
        plt.scatter(init_embed[same_list, 0], init_embed[same_list, 1], label="same")
        plt.scatter(init_embed[diff_list, 0], init_embed[diff_list, 1], label="diff")
        plt.savefig("%d_tsne_orignal.png" % i)
        plt.close(0)

        plt.figure(1)
        plt.scatter(res_embed[same_list, 0], res_embed[same_list, 1], label="same")
        plt.scatter(res_embed[diff_list, 0], res_embed[diff_list, 1], label="diff")
        plt.savefig("%d_tsne_transformed_%.2f.png" % (i, TEMPERATURE))
        plt.close(1)

        print(i)


def tsne_viz_iteration(prefix, dataset, query_id):
    query_features_path = 'features/%s/query-%s.mat' % (dataset, prefix)
    gallery_features_path = "features/%s/gallery-%s.mat" % (dataset, prefix)
    query_lst_path = "/mnt/truenas/scratch/chuanchen.luo/data/reid/%s-list/query.lst" % dataset
    gallery_lst_path = "/mnt/truenas/scratch/chuanchen.luo/data/reid/%s-list/test.lst" % dataset

    query_lst = np.array(load_lst(query_lst_path))
    gallery_lst = np.array(load_lst(gallery_lst_path))

    query_features = sio.loadmat(query_features_path)["feat"]
    gallery_features = sio.loadmat(gallery_features_path)["feat"]

    query_features = normalize(query_features)
    gallery_features = normalize(gallery_features)

    q_feat = query_features[query_id]
    dist = -np.dot(gallery_features, q_feat)
    rank_list = np.argsort(dist)[:150]
    g_feats = gallery_features[rank_list]

    q_record = Record(*query_lst[query_id])
    g_records = [Record(*item) for item in gallery_lst[rank_list]]

    same_list = [i for i in range(g_feats.shape[0]) if q_record.class_id == g_records[i].class_id]
    diff_list = [i for i in range(g_feats.shape[0]) if q_record.class_id != g_records[i].class_id]

    init_embed = TSNE(n_components=2, init="pca", perplexity=8, metric="cosine").fit_transform(g_feats)

    W = np.dot(g_feats, g_feats.T)
    W = np.exp(W / TEMPERATURE)
    W = W / np.sum(W, axis=1, keepdims=True)

    plt.figure(0)
    plt.scatter(init_embed[same_list, 0], init_embed[same_list, 1], label="same")
    plt.scatter(init_embed[diff_list, 0], init_embed[diff_list, 1], label="diff")
    plt.savefig("%d_tsne_orignal.png" % query_id)
    plt.close(0)

    for iteration in range(10):
        g_feats = np.dot(W, g_feats)
        res_embed = TSNE(n_components=2, init="pca", perplexity=8, metric="cosine").fit_transform(g_feats)

        plt.figure(iteration)
        plt.scatter(res_embed[same_list, 0], res_embed[same_list, 1], label="same")
        plt.scatter(res_embed[diff_list, 0], res_embed[diff_list, 1], label="diff")
        plt.savefig("%d_%02d_tsne_transformed_%.2f.png" % (query_id, iteration, TEMPERATURE))
        plt.close(iteration)

        print("%d-%d" % (query_id, iteration))


def tsne_viz_train(p_num, prefix, dataset, epoch, gpu, name="tsne"):
    with open("config.yml", "r") as f:
        args = yaml.load(f)

    data_dir = args[dataset]["data_dir"]
    image_size = args["image_size"]

    print(data_dir)

    dataloader = RawIterator(data_dir=os.path.join(data_dir, "bounding_box_train"), batch_size=512, p_num=p_num,
                             image_size=image_size)

    print(dataloader.size)

    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join("models", dataset, prefix), epoch)
    sym = sym.get_internals()["flatten_output"]
    sym = mx.symbol.L2Normalization(sym)

    model = mx.mod.Module(symbol=sym, context=mx.gpu(gpu))
    model.bind(data_shapes=[("data", (128, 3, 256, 128))], for_training=False, force_rebind=True)
    model.set_params(arg_params, aux_params)

    feats = []
    label = []
    for i, batch in enumerate(dataloader):
        model.forward(Batch(data=batch.data))
        feat = model.get_outputs()[0].asnumpy()
        feats.append(feat)
        label.append(batch.label[0].asnumpy())

    feats = np.concatenate(feats, axis=0)[:dataloader.size]
    label = np.concatenate(label, axis=0)[:dataloader.size]

    print("Visualizing...")

    embed = TSNE(n_components=2, perplexity=8, metric="cosine").fit_transform(feats)

    plt.figure(0)
    plt.scatter(embed[:, 0], embed[:, 1], c=label, cmap="Set1", marker=".")
    plt.savefig("%s.png" % name)
    plt.close(0)


if __name__ == '__main__':
    prefix = "baseline-140ep"
    dataset = "duke"
    TEMPERATURE = 1.0

    # tsne_viz_query2gallery(prefix, dataset, 25)
    # tsne_viz_iteration(prefix, dataset, 0)

    tsne_viz_train(p_num=32, prefix=prefix, dataset=dataset, epoch=140, gpu=0, name="nogcn")

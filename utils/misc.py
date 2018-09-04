import os
import glob
import cv2
import numpy as np
import mxnet as mx

import matplotlib as mpl
from mxnet.metric import EvalMetric, check_label_shapes

mpl.use("Agg")

import matplotlib.pyplot as plt

from collections import namedtuple

Record = namedtuple('Record', ['index', 'class_id', 'cam_id', 'img_path'])


def clean_immediate_checkpoints(model_dir, prefix, final_epoch):
    ckpts = glob.glob(os.path.join(model_dir, "%s*.params" % prefix))

    for ckpt in ckpts:
        ckpt_name = os.path.basename(ckpt)
        epoch_idx = int(ckpt_name[:ckpt_name.rfind(".")].split("-")[-1])
        if epoch_idx < final_epoch:
            os.remove(ckpt)


def euclidean_dist(x, y, eps=1e-12, squared=False):
    m, n = x.shape[0], y.shape[0]
    xx = mx.nd.power(x, 2).sum(axis=1, keepdim=True).broadcast_to([m, n])
    yy = mx.nd.power(y, 2).sum(axis=1, keepdim=True).broadcast_to([n, m]).T
    dist = xx + yy
    dist = dist - 2 * mx.nd.dot(x, y.T)
    dist = mx.nd.clip(dist, eps, np.inf)
    return dist if not squared else mx.nd.sqrt(dist)


def viz_heatmap(data, name):
    maximum = np.max(data)
    # minimun = np.min(data)

    plt.figure(0)

    data = data / maximum * 255
    heat_map = cv2.applyColorMap(data.astype(np.uint8), cv2.COLORMAP_JET)[:, :, [2, 1, 0]]
    plt.imshow(heat_map)
    plt.savefig("%s.jpg" % name)

    plt.close(0)


def load_lst(lst_path):
    lines = []
    with open(lst_path, "r") as fin:
        for line in fin:
            idx, class_id, img_path = line.strip().split('\t')

            cam_id = os.path.splitext(os.path.basename(img_path))[0].split('_')[1][1]
            idx = int(idx)
            class_id = int(class_id)
            cam_id = int(cam_id)
            lines.append(Record(idx, class_id, cam_id, img_path))
    return lines


class CustomAccuracy(EvalMetric):
    def __init__(self, name='accuracy', deep_sup=False,
                 output_names=None, label_names=None):
        super(CustomAccuracy, self).__init__(name, output_names=output_names, label_names=label_names)

        self.deep_sup = deep_sup

    def update(self, labels, preds):

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.nd.argmax(pred_label, axis=1)

            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')

            if self.deep_sup:
                pred_label = pred_label[:pred_label.shape[0] // 2]

            self.sum_metric += np.equal(pred_label, label).sum()
            self.num_inst += len(pred_label)


class CustomCrossEntropy(EvalMetric):
    def __init__(self, eps=1e-12, name='cross-entropy', deep_sup=False,
                 output_names=None, label_names=None):
        super(CustomCrossEntropy, self).__init__(name, eps=eps, output_names=output_names, label_names=label_names)
        self.eps = eps
        self.deep_sup = deep_sup

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if self.deep_sup:
                label = np.tile(label, 2)

            label = label.ravel()
            assert label.shape[0] == pred.shape[0]

            prob = pred[np.arange(label.shape[0]), np.int64(label)]
            self.sum_metric += (-np.log(prob + self.eps)).sum()
            self.num_inst += label.shape[0]

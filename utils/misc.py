from __future__ import division, print_function
import os
import glob
import numpy as np
import mxnet as mx
from mxnet.metric import EvalMetric


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def clean_immediate_checkpoints(model_dir, prefix, final_epoch):
    ckpts = glob.glob(os.path.join(model_dir, "%s*.params" % prefix))

    for ckpt in ckpts:
        ckpt_name = os.path.basename(ckpt)
        epoch_idx = int(ckpt_name[:ckpt_name.rfind(".")].split("-")[-1])
        if epoch_idx < final_epoch:
            os.remove(ckpt)


def euclidean_dist(x, y, eps=1e-12, squared=False):
    m, n = x.shape[0], y.shape[0]
    xx = mx.nd.power(x, 2).sum(axis=1, keepdims=True).broadcast_to([m, n])
    yy = mx.nd.power(y, 2).sum(axis=1, keepdims=True).broadcast_to([n, m]).T
    dist = xx + yy
    dist = dist - 2 * mx.nd.dot(x, y.T)
    dist = mx.nd.clip(dist, eps, np.inf)
    return dist if not squared else mx.nd.sqrt(dist)


class CustomAccuracy(EvalMetric):
    def __init__(self, name='accuracy', output_names=None, label_names=None):
        super(CustomAccuracy, self).__init__(name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds):

        for label, pred_label in zip(labels, preds):
            pred_label = mx.nd.argmax(pred_label, axis=1)

            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')

            if label.shape[0] != pred_label.shape[0]:
                label = np.tile(label, reps=2)

            self.sum_metric += np.equal(pred_label, label).sum()
            self.num_inst += len(pred_label)


class CustomCrossEntropy(EvalMetric):
    def __init__(self, eps=1e-12, name='cross-entropy', output_names=None, label_names=None):
        super(CustomCrossEntropy, self).__init__(name, eps=eps, output_names=output_names, label_names=label_names)
        self.eps = eps

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if label.shape[0] != pred.shape[0]:
                label = np.tile(label, reps=2)

            prob = pred[np.arange(pred.shape[0]), np.int32(label)]
            self.sum_metric += (-np.log(prob + self.eps)).sum()
            self.num_inst += label.shape[0]

from __future__ import print_function, division

import numpy as np
import mxnet as mx


def euclidean_distances(X, batch_size, squared=False):
    XX = mx.symbol.sum(mx.symbol.square(X), axis=1).expand_dims(1)

    distances = mx.symbol.dot(X, X, transpose_b=True)
    distances = mx.symbol.broadcast_add(mx.symbol.broadcast_add(- 2 * distances, XX), mx.symbol.transpose(XX))
    mx.symbol.relu(distances, out=distances)

    distances = distances * (1 - mx.symbol.eye(batch_size))
    distances = mx.symbol.clip(distances, a_min=1e-12, a_max=np.inf)

    return distances if squared else mx.symbol.sqrt(distances)


def label_to_square(label, batch_size):
    label = mx.symbol.expand_dims(label, 1)
    label = mx.symbol.broadcast_axis(label, axis=1, size=batch_size)
    aff_label = mx.symbol.broadcast_equal(label, mx.symbol.transpose(label))
    return mx.symbol.stop_gradient(aff_label)


def rw_module(data, **kwargs):
    temperature = kwargs.get("temperature", 1.0)
    batch_size = kwargs.get("batch_size", -1)
    metric = kwargs.get("metric", "cosine")

    if metric not in ["cosine", "euclidean"]:
        raise ValueError("Parameter of metric must be 'cosine' or 'euclidean'.")

    in_sim = mx.symbol.L2Normalization(data=data)

    if metric == "cosine":
        sim = mx.symbol.dot(in_sim, in_sim, transpose_b=True)
    else:
        sim = -euclidean_distances(in_sim, batch_size, squared=True)

    aff = mx.symbol.softmax(sim, temperature=temperature, axis=1)
    feat = mx.symbol.dot(aff, data)

    return feat


def classification_branch(data, trans_data, label, num_id, margin=0.5, **kwargs):
    bottleneck_dims = kwargs.get("bottleneck_dims", 512)
    use_rw = kwargs.get("use_rw", False)
    deep_sup = kwargs.get("deep_sup", False)
    norm_scale = kwargs.get("norm_scale", 1.0)

    if use_rw:
        if deep_sup:
            data = mx.symbol.Concat(data, trans_data, dim=0)
            label = mx.symbol.tile(label, reps=2)
        else:
            data = trans_data

    fc = mx.symbol.FullyConnected(data, num_hidden=bottleneck_dims, no_bias=True, name="bottleneck")

    bn = mx.sym.BatchNorm(data=fc, fix_gamma=False, momentum=0.9, eps=2e-5, name='bottleneck_bn')
    act = mx.symbol.LeakyReLU(bn, act_type="prelu")

    l2 = mx.symbol.L2Normalization(act)

    am_softmax, plain_softmax = amsoftmax(l2, label=label, num_dims=bottleneck_dims, scale=norm_scale,
                                          num_id=num_id, margin=margin)

    return am_softmax, plain_softmax


def independent_classification_branch(data, label, num_id, margin=0.5, name="", **kwargs):
    bottleneck_dims = kwargs.get("bottleneck_dims", 512)
    norm_scale = kwargs.get("norm_scale", 1.0)

    fc = mx.symbol.FullyConnected(data, num_hidden=bottleneck_dims, no_bias=True)

    bn = mx.sym.BatchNorm(data=fc, fix_gamma=False, momentum=0.9, eps=2e-5)
    act = mx.symbol.LeakyReLU(bn, act_type="prelu")

    l2 = mx.symbol.L2Normalization(act)

    am_softmax, plain_softmax = amsoftmax(l2, label=label, num_dims=bottleneck_dims, scale=norm_scale,
                                          num_id=num_id, margin=margin, postfix=name)

    return am_softmax, plain_softmax


def amsoftmax(data, label, num_dims, num_id, margin, scale, grad_scale=1.0, postfix=""):
    fc_weight = mx.symbol.Variable("fc" + postfix + "_weight", shape=(num_id, num_dims))
    fc_weight = mx.symbol.L2Normalization(fc_weight)
    fc = mx.symbol.FullyConnected(data=data, weight=fc_weight, num_hidden=num_id, no_bias=True)

    am_fc = fc - mx.sym.one_hot(label, depth=num_id, on_value=margin, off_value=0)

    am_softmax_loss = mx.sym.SoftmaxOutput(data=am_fc * scale, label=label, grad_scale=grad_scale,
                                           name="am_softmax" + postfix)
    plain_softmax_loss = mx.symbol.stop_gradient(mx.sym.softmax(data=fc * scale), name="softmax" + postfix)

    return am_softmax_loss, plain_softmax_loss


def triplet_hard_loss(data, label, margin, batch_size, grad_scale=1.0, name="triplet"):
    label = label_to_square(label, batch_size)

    eu_dist = euclidean_distances(data, batch_size)

    pos = mx.symbol.max(eu_dist * label, axis=1)
    neg = mx.symbol.min(eu_dist * (1 - label) + label * 1e8, axis=1)

    loss = mx.symbol.relu(pos - neg + margin)

    return mx.symbol.MakeLoss(loss, grad_scale=grad_scale, name=name)

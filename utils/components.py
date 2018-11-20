from __future__ import print_function, division

import numpy as np
import mxnet as mx


def euclidean_distances(X, batch_size, squared=False):
    '''
    MXNet version euclidean distance. Only support the computation of distances within a matrix.

    Parameters
    ----------
    X : Symbol
        Input of the computation.
    batch_size : Int
        The number of samples in X.
    squared : Boolean
        Whether return the squared value.

    Returns
    -------
        Symbol
        The result symbol.

    '''
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


def sft_module(data, temperature):
    '''
    The implementation of spectral feature transformation module.

    Parameters
    ----------
    data : Symbol
        The input of the module.
    temperature : Symbol
        The temperature of the softmax.

    Returns
    -------
    Symbol
        The result symbol.

    Symbol
        The result similarity matrix.
    '''
    in_sim = mx.symbol.L2Normalization(data=data)
    sim = mx.symbol.dot(in_sim, in_sim, transpose_b=True)

    aff = mx.symbol.softmax(sim, temperature=temperature, axis=1)
    feat = mx.symbol.dot(aff, data)

    return feat, sim


def my_classifier(data, num_id, bottleneck_dims):
    '''
    Our classifier with bottleneck design. Users can implement customized classifier.

    Parameters
    ----------
    data : Symbol
        The input of the classifier.
    num_id : Int
        The number of identities.
    bottleneck_dims : Int
        The dimensionality of the bottleneck FC layer.

    Returns
    -------
    Symbol
        The result symbol.
    '''
    fc = mx.symbol.FullyConnected(data, num_hidden=bottleneck_dims, no_bias=True)
    bn = mx.sym.BatchNorm(data=fc, fix_gamma=False, eps=1e-5)
    act = mx.symbol.LeakyReLU(bn, act_type="prelu")

    l2 = mx.symbol.L2Normalization(act)

    cls_weight = mx.symbol.Variable("cls_weight", shape=(num_id, bottleneck_dims))
    cls_weight = mx.symbol.L2Normalization(cls_weight)
    logits = mx.symbol.FullyConnected(l2, weight=cls_weight, num_hidden=num_id, no_bias=True)

    return logits


def AMSoftmaxOutput(data, label, num_id, margin, scale, grad_scale=1.0, name="amsoftmax"):
    '''
    The implementation of am-softmax. Since the accuracy derived from the raw output of am-softmax is not correct, we
    output an additional logits for the computation of the accuracy.

    Parameters
    ----------
    data : Symbol
        The input of the operator.
    label : Symbol
        The label of the data.
    num_id : Int
        The number of identities.
    margin : Float
        The additive margin of am-softmax.
    scale : Float
        The scaling factor of am-softmax.
    grad_scale : Float, Optional, default=1.0
        Scales the gradient by a float factor.
    name :  String, Optional
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    '''
    data_margin = data - mx.sym.one_hot(label, depth=num_id, on_value=margin, off_value=0)

    am_softmax = mx.sym.SoftmaxOutput(data=data_margin * scale, label=label, grad_scale=grad_scale, name=name)

    return am_softmax, mx.symbol.stop_gradient(data, name="logits")


def triplet_hard_loss(data, label, margin, batch_size, grad_scale=1.0, name="triplet"):
    '''
    Simplified TriHard loss.

    Parameters
    ----------
    data : Symbol
        Input data.
    label : Symbol
        The label of input.
    margin : Float
        The margin of triplet loss
    batch_size : Int
        The batch-size of input data.
    grad_scale : Float, Default=1.0
        Scales the gradient by a float factor.
    name : String, Default="triplet"
        Name of the resulting symbol.

    Returns
    -------
        Symbol
        The result symbol.

    '''
    label = label_to_square(label, batch_size)

    eu_dist = euclidean_distances(data, batch_size)

    pos = mx.symbol.max(eu_dist * label, axis=1)
    neg = mx.symbol.min(eu_dist * (1 - label) + label * 1e8, axis=1)

    loss = mx.symbol.relu(pos - neg + margin)

    return mx.symbol.MakeLoss(loss, grad_scale=grad_scale, name=name)


def normalize_cut(pred, sim, grad_scale, name):
    '''
    Normalized cut loss.

    Parameters
    ----------
    pred : Symbol
        The prediction (probability) of classification.
    sim : Symbol
        The similarity matrix of the mini-batch.
    grad_scale : Float
        Scales the gradient by a float factor.
    name : String, Default="triplet"
        Name of the resulting symbol.

    Returns
    -------
        Symbol
        The result symbol.
    '''
    affinity = mx.symbol.BlockGrad(sim)

    pred_t = mx.symbol.transpose(pred)
    pred_b_t = mx.symbol.expand_dims(pred_t, 2)

    numerator = mx.symbol.batch_dot(mx.symbol.expand_dims(mx.symbol.dot(pred_t, affinity), 1), 1 - pred_b_t)
    numerator = mx.symbol.squeeze(numerator)

    degree = mx.symbol.sum(affinity, axis=1)

    denominator = mx.symbol.dot(pred_t, degree)

    loss = mx.symbol.mean(numerator / denominator)
    return mx.symbol.MakeLoss(loss, grad_scale=grad_scale, name=name)


def nca_loss(sim, label, batch_size, grad_scale, name):
    '''
    NCA loss.

    Parameters
    ----------
    sim : Symbol
        The similarity matrix of the mini-batch.
    label : Symbol
        The label of the data.
    batch_size : Int
        The batch-size of input data.
    grad_scale : Float
        Scales the gradient by a float factor.
    name : String, Default="triplet"
        Name of the resulting symbol.

    Returns
    -------
        Symbol
        The result symbol.
    '''
    label = label_to_square(label, batch_size)
    sim = mx.symbol.exp(sim / 0.05) * (1 - mx.symbol.eye(batch_size))
    sim = mx.symbol.broadcast_div(sim, mx.symbol.sum(sim, axis=1, keepdims=True))
    p = mx.symbol.sum(label * sim, axis=1)
    return mx.symbol.MakeLoss(-mx.symbol.log(p), grad_scale=grad_scale, name=name)

from __future__ import print_function, division

import mxnet as mx
from utils.debug import forward_debug


def label_to_square(label, batch_size):
    label = mx.symbol.expand_dims(label, 1)
    label = mx.symbol.broadcast_axis(label, axis=1, size=batch_size)
    aff_label = mx.symbol.broadcast_equal(label, mx.symbol.transpose(label))
    return aff_label


def gbms_block(data, sim_normalize=True, **kwargs):
    temperature = kwargs.get("temperature", 1.0)
    batch_size = kwargs.get("batch_size", -1)
    metric = kwargs.get("metric", "cosine")

    if metric not in ["cosine", "euclidean"]:
        raise ValueError("Parameter of metric must be 'cosine' or 'euclidean'.")

    in_sim = mx.symbol.L2Normalization(data=data, name='l2_norm') if sim_normalize else data

    if metric == "cosine":
        sim = mx.symbol.dot(in_sim, in_sim, transpose_b=True)
    else:
        sim = -euclidean_distances(in_sim, batch_size, squared=True)

    aff = mx.symbol.softmax(sim, temperature=temperature, axis=1)
    feat = mx.symbol.dot(aff, data)

    return feat, sim, aff


def classification_branch(data, trans_data, label, num_id, margin=0.5, **kwargs):
    bottleneck_dims = kwargs.get("bottleneck_dims", 512)
    use_gbms = kwargs.get("use_gbms", False)
    deep_sup = kwargs.get("deep_sup", False)
    norm_scale = kwargs.get("norm_scale", 1.0)

    if use_gbms:
        if deep_sup:
            data = mx.symbol.Concat(data, trans_data, dim=0)
            label = mx.symbol.tile(label, reps=2)
        else:
            data = trans_data

    fc = mx.symbol.FullyConnected(data, num_hidden=bottleneck_dims, no_bias=True, name="bottleneck")

    bn = mx.sym.BatchNorm(data=fc, fix_gamma=False, momentum=0.9, eps=2e-5, name='bottleneck_bn')
    act = mx.symbol.LeakyReLU(bn, act_type="prelu")

    l2 = mx.symbol.L2Normalization(act)

    am_softmax, plain_softmax = amsoftmax(l2, label=label, num_dims=bottleneck_dims, scale=norm_scale, num_id=num_id,
                                          margin=margin)

    return am_softmax, plain_softmax


def euclidean_distances(X, batch_size, squared=False):
    XX = mx.symbol.sum(mx.symbol.square(X), axis=1).expand_dims(1)

    distances = mx.symbol.dot(X, X, transpose_b=True)
    distances = mx.symbol.broadcast_add(mx.symbol.broadcast_add(- 2 * distances, XX), mx.symbol.transpose(XX))
    mx.symbol.relu(distances, out=distances)

    distances = distances * (1 - mx.symbol.eye(batch_size))

    return distances if squared else mx.symbol.sqrt(distances, out=distances)


def amsoftmax(data, label, num_dims, num_id, margin, scale, grad_scale=1.0):
    fc_weight = mx.symbol.Variable("fc_weight", shape=(num_id, num_dims))
    fc_weight = mx.symbol.L2Normalization(fc_weight)
    fc = mx.symbol.FullyConnected(data=data, weight=fc_weight, num_hidden=num_id, no_bias=True)

    am_fc = fc - mx.sym.one_hot(label, depth=num_id, on_value=margin, off_value=0)

    am_softmax_loss = mx.sym.SoftmaxOutput(data=am_fc * scale, label=label, grad_scale=grad_scale, name="am_softmax")
    plain_softmax_loss = mx.symbol.stop_gradient(mx.sym.softmax(data=fc * scale), name="softmax")
    return am_softmax_loss, plain_softmax_loss


def subspace_online(data, batch_size, grad_scale=1.0, name="ssc"):
    data = mx.symbol.stop_gradient(data)

    trans_w = mx.symbol.Variable("trans_weight", shape=(batch_size, batch_size), init=mx.init.Constant(1 / 128))
    trans_w = trans_w * (1 - mx.symbol.eye(batch_size))

    data_trans = mx.symbol.dot(trans_w, data)

    loss = mx.symbol.norm(data_trans - data, ord=2) + mx.symbol.norm(trans_w, ord=1)

    return data_trans, mx.symbol.MakeLoss(loss, grad_scale=grad_scale, name=name)


def verification_branch(data, label, p_size, k_size, grad_scale=1.0, name="veri"):
    label = label_to_square(label, batch_size=p_size * k_size)
    ratio = label / k_size + (1 - label) / (k_size * (p_size - 1))
    loss = mx.symbol.sum(mx.symbol.square(label - data) * ratio, axis=1)

    return mx.symbol.MakeLoss(loss, grad_scale=grad_scale, name=name)

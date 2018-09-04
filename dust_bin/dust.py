import mxnet as mx


def non_local(data):
    sim_factor_1 = mx.sym.FullyConnected(data, num_hidden=512, no_bias=True, name="sim_factor1")
    sim_factor_2 = mx.sym.FullyConnected(data, num_hidden=512, no_bias=True, name="sim_factor2")

    sim = mx.sym.dot(sim_factor_1, sim_factor_2, transpose_b=True)
    sim = mx.sym.exp(sim)
    sim = (1 - mx.sym.eye(batch_size)) * sim
    sim = mx.symbol.broadcast_div(sim, mx.symbol.sum(sim, axis=1, keepdims=True))

    feat = mx.symbol.dot(sim, data)
    feat = mx.symbol.FullyConnected(feat, num_hidden=2048, no_bias=True)
    bn = mx.sym.BatchNorm(data=feat, fix_gamma=False, momentum=0.9, eps=2e-5)
    act = mx.sym.relu(data=bn)

    return act


def graph_regularization(data, label):
    in_sim = mx.symbol.L2Normalization(data=data, name='l2_norm')
    sim = mx.symbol.dot(in_sim, in_sim, transpose_b=True)

    label = mx.sym.broadcast_to(mx.sym.expand_dims(label, 1), [batch_size, batch_size])
    y = mx.sym.broadcast_equal(label, mx.sym.transpose(label))

    sim = sim * y
    D = mx.sym.diag(mx.sym.sum(mx.sym.abs(sim), axis=1))

    L = D - sim

    x = mx.sym.dot(mx.sym.transpose(data), L)
    x = mx.sym.dot(x, data)

    x = mx.sym.diag(x)

    return mx.sym.MakeLoss(x, grad_scale=1 / 2048)


def non_local_block(data, num_hidden):
    a = mx.sym.FullyConnected(data, num_hidden=num_hidden, no_bias=True, name="aff_a")
    b = mx.sym.FullyConnected(data, num_hidden=num_hidden, no_bias=True, name="aff_b")

    c = mx.sym.dot(a, b, transpose_b=True)
    aff = mx.sym.softmax(c, axis=1, name="affinity")
    feat = mx.symbol.dot(aff, data)

    gamma = mx.sym.Variable("gcn_bn_gamma", init=mx.init.Zero())
    feat = mx.sym.BatchNorm(data=feat, fix_gamma=False, gamma=gamma, momentum=0.9, eps=2e-5, name='gcn_bn')

    return feat + data


def subspace_cluster_block(data, num_hidden, grad_scale):
    a = mx.sym.FullyConnected(data, num_hidden=num_hidden, no_bias=True, name="aff_a")
    b = mx.sym.FullyConnected(data, num_hidden=num_hidden, no_bias=True, name="aff_b")

    c = mx.sym.dot(a, b, transpose_b=True)
    aff = mx.sym.softmax(c, axis=1, name="affinity")
    feat = mx.symbol.dot(aff, data)

    reconstruct_loss = mx.sym.MakeLoss(mx.sym.square(feat - data), grad_scale=grad_scale)
    return feat, reconstruct_loss


def pcb_classifier(data, label, num_id, num_hidden=256, postfix=""):
    fc = mx.symbol.FullyConnected(data, num_hidden=num_hidden, name="bottleneck%s" % postfix)
    bn = mx.sym.BatchNorm(data=fc, fix_gamma=False, momentum=0.9, eps=2e-5, name='bottleneck%s_bn' % postfix)
    relu = mx.sym.Activation(data=bn, act_type="relu", name='bottleneck%s_relu' % postfix)

    softmax_fc = mx.symbol.FullyConnected(relu, num_hidden=num_id, name="softmax%s_fc" % postfix)

    softmax = mx.symbol.SoftmaxOutput(data=softmax_fc, label=label, name='softmax%s' % postfix)

    return softmax


def graph_regularization(data, similarity, p_size, k_size, label, grad_scale=1.0):
    batch_size = p_size * k_size
    same_square = label_to_square(label, batch_size)

    sim_intra = same_square * (1 - similarity)
    sim_inter = (1 - same_square) * (- similarity)
    sim = sim_inter + sim_intra / (p_size - 1)

    degree = mx.sym.diag(mx.sym.sum(sim, axis=1))
    laplacian = degree - sim

    reg = mx.sym.dot(mx.sym.dot(mx.sym.transpose(data), laplacian), data)

    regularization = mx.sym.sum(mx.sym.diag(reg))

    return mx.sym.MakeLoss(regularization, grad_scale=grad_scale / k_size)


def verification_block(data, label, batch_size, margin=0.5, grad_scale=1.0):
    in_sim = mx.sym.L2Normalization(data)
    sim = mx.sym.dot(in_sim, in_sim, transpose_b=True)

    margin = 0.5

    label = label_to_square(label, batch_size)
    diff = mx.symbol.relu(label - sim + (1 - label) * margin)
    loss = mx.sym.sum(mx.sym.square(diff)) / (batch_size - 1)
    return mx.sym.MakeLoss(loss, grad_scale=grad_scale, name="verification")


def classification_block_v1(feat, embed, label, num_id, **kwargs):
    dropout_ratio = kwargs.get("dropout_ratio", 0.5)
    norm_scale = kwargs.get("norm_scale", 1.0)

    fc_feature = mx.sym.FullyConnected(feat, no_bias=True, num_hidden=256, name="fc_feature")
    fc_embedding = mx.sym.FullyConnected(embed, no_bias=True, num_hidden=256, name="fc_embedding")
    concat = mx.sym.Concat(fc_feature, fc_embedding, dim=1)
    bn = mx.sym.BatchNorm(data=concat, fix_gamma=False, momentum=0.9, eps=2e-5, name='bottleneck_bn')
    drop = mx.sym.Dropout(bn, p=dropout_ratio)
    l2 = mx.sym.L2Normalization(drop) * norm_scale

    softmax_fc_weight = mx.sym.Variable("softmax_fc_weight", shape=(num_id, 512))
    softmax_fc_weight = mx.sym.L2Normalization(softmax_fc_weight)

    softmax_fc = mx.sym.FullyConnected(l2, weight=softmax_fc_weight, num_hidden=num_id, no_bias=True, name="softmax_fc")

    return mx.symbol.SoftmaxOutput(softmax_fc, label, name="softmax")


def spatial_non_local(data, num_hidden, normalize=False):
    sim_a = mx.symbol.Convolution(data=data, num_filter=num_hidden, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                  no_bias=True)
    sim_b = mx.symbol.Convolution(data=data, num_filter=num_hidden, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                  no_bias=True)
    g = mx.symbol.Convolution(data=data, num_filter=num_hidden, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True)

    a = mx.symbol.reshape(sim_a, shape=(0, 0, -1))
    b = mx.symbol.reshape(sim_b, shape=(0, 0, -1))
    reshape_g = mx.symbol.reshape(g, shape=(0, 0, -1))

    if normalize:
        a = mx.symbol.L2Normalization(a, mode="channel")
        b = mx.symbol.L2Normalization(b, mode="channel")

    sim = mx.symbol.batch_dot(a, b, transpose_a=True)

    affinity = mx.symbol.softmax(sim, axis=2)
    trans_g = mx.symbol.batch_dot(affinity, reshape_g, transpose_b=True)

    trans_g = mx.symbol.transpose(trans_g, axes=(0, 2, 1))
    trans_g = mx.symbol.reshape_like(trans_g, g)

    z = mx.symbol.Convolution(data=trans_g, num_filter=2048, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True)

    gamma = mx.symbol.Variable("gamma", init=mx.init.Zero())
    z = mx.sym.BatchNorm(data=z, gamma=gamma, fix_gamma=False, momentum=0.9, eps=2e-5)
    z = mx.symbol.relu(z)
    return z + data


def spatial_gbms(data, num_hidden, num_output, **kwargs):
    norm_scale = kwargs.get("norm_scale", 1.0)

    g = data  # mx.symbol.Convolution(data=data, num_filter=num_hidden, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True)

    reshape_g = mx.symbol.reshape(g, shape=(0, 0, -1))

    normalized_g = mx.symbol.L2Normalization(reshape_g, mode="channel")

    sim = mx.symbol.batch_dot(normalized_g, normalized_g, transpose_a=True) * norm_scale

    affinity = mx.symbol.softmax(sim, axis=2)
    trans_g = mx.symbol.batch_dot(affinity, reshape_g, transpose_b=True)

    trans_g = mx.symbol.transpose(trans_g, axes=(0, 2, 1))
    trans_g = mx.symbol.reshape_like(trans_g, g)

    conv = mx.symbol.Convolution(data=trans_g, num_filter=num_output, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                 no_bias=True)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, momentum=0.9, eps=2e-5)
    relu = mx.symbol.Activation(bn, act_type="relu")
    return relu


def affinity_to_mixup_label(affinity, label, p_size, k_size, num_id):
    '''
    just for specific dataloader.
    TODO: generalize it
    '''
    batch_size = p_size * k_size
    reshape_aff = mx.symbol.reshape(affinity, shape=(0, p_size, k_size))
    prob = mx.symbol.sum(reshape_aff, axis=2)

    label = mx.symbol.reshape(label, shape=(p_size, k_size))
    label = mx.symbol.slice_axis(label, axis=1, begin=0, end=1)
    label = mx.symbol.broadcast_axis(mx.symbol.transpose(label), axis=0, size=batch_size)
    arange = mx.symbol.broadcast_axis(mx.symbol.arange(batch_size).expand_dims(1), axis=1, size=p_size)
    indices = mx.symbol.stack(arange, label, axis=0)
    mixup_label = mx.symbol.scatter_nd(data=prob, indices=indices, shape=(batch_size, num_id))

    return mixup_label


def classification_block(data, label, num_id, **kwargs):
    softmax_weight_normalization = kwargs.get("softmax_weight_normalization", False)
    softmax_feat_normalization = kwargs.get("softmax_feat_normalization", False)
    bottleneck_dims = kwargs.get("bottleneck_dims", 512)
    with_relu = kwargs.get("with_relu", True)
    dropout_ratio = kwargs.get("dropout_ratio", 0.5)
    norm_scale = kwargs.get("norm_scale", 1.0)
    prefix = kwargs.get("prefix", "")
    block_grad = kwargs.get("block_grad", False)
    weights = kwargs.get("weights", {})

    name = "bottleneck_weight"
    fc_w = weights[name] if name in weights else mx.sym.Variable("bottleneck" + prefix + "weight")
    weights[name] = fc_w

    if block_grad:
        fc_w = mx.sym.BlockGrad(fc_w)

    fc = mx.symbol.FullyConnected(data, num_hidden=bottleneck_dims, weight=fc_w, no_bias=True,
                                  name="bottleneck" + prefix)

    name = "bottleneck_bn_gamma"
    gamma = weights[name] if name in weights else mx.sym.Variable("bottleneck_bn" + prefix + "_gamma")
    weights[name] = gamma

    name = "bottleneck_bn_beta"
    beta = weights[name] if name in weights else mx.sym.Variable("bottleneck_bn" + prefix + "_beta")
    weights[name] = beta

    if block_grad:
        gamma = mx.sym.BlockGrad(gamma)
        beta = mx.sym.BlockGrad(beta)

    bn = mx.sym.BatchNorm(data=fc, gamma=gamma, beta=beta, fix_gamma=False, momentum=0.9, eps=2e-5,
                          name='bottleneck_bn' + prefix)

    if not with_relu:
        bn = mx.sym.Activation(data=bn, act_type='relu')

    dropout = mx.symbol.Dropout(bn, p=dropout_ratio)

    name = "softmax_weight"
    softmax_w = weights[name] if name in weights else mx.symbol.Variable("softmax" + prefix + "_weight",
                                                                         shape=(num_id, bottleneck_dims))
    weights[name] = softmax_w

    if softmax_weight_normalization:
        softmax_w = mx.symbol.L2Normalization(softmax_w)

    if softmax_feat_normalization:
        data_softmax = mx.sym.L2Normalization(dropout) * norm_scale
    else:
        data_softmax = dropout

    if block_grad:
        softmax_w = mx.sym.BlockGrad(softmax_w)

    softmax_fc = mx.symbol.FullyConnected(data_softmax, weight=softmax_w, num_hidden=num_id,
                                          no_bias=True if softmax_weight_normalization else False,
                                          name="softmax_fc" + prefix)

    softmax = mx.symbol.SoftmaxOutput(data=softmax_fc, label=label, name='softmax' + prefix)

    return softmax, weights


def dimension_reduction(data, num_filter, with_relu):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                 no_bias=True)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, momentum=0.9, eps=2e-5)

    return bn if not with_relu else mx.symbol.relu(bn)

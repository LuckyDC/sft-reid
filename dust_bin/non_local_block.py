import mxnet as mx

def non_local_spacetime_reduced_naive(insym, num_filter, t_dim=1, key_dim=0, mode='Embedded Gaussian', scale=False,
                                      non_residual=False, resample=True,spatial_pool=False, learn_scale=False, 
                                      normalize=False, non_cur_space=False,naive_only=False, adaptive_weight=False,
                                      sigmoid_weight=False, naive_conv_l=False,output_cur_only=True,
                                      prefix='',draw=False, ith=0):
    # insym: T x H x W x num_filter
    inter_filter = num_filter / 2 if num_filter >= 1024 else num_filter
    if output_cur_only:
        key_insym = mx.sym.SliceChannel(insym, axis=0, num_outputs=t_dim)[key_dim]
    else:
        key_insym = insym

    if naive_conv_l:
        theta = mx.sym.Convolution(key_insym, kernel=(1, 1), stride=(1, 1), num_filter=inter_filter,
                                   no_bias=True, name='nonlocal_conv{}%d1'.format(prefix) % ith)
    else:
        theta = mx.sym.expand_dims(key_insym, axis=1)
        theta = mx.sym.Pooling(data=theta, kernel=(2, 1, 1), stride=(2, 1, 1), pool_type='max')
        theta = mx.sym.squeeze(theta, axis=1)

    if normalize:
        theta = mx.sym.L2Normalization(data=theta, mode='channel')

    downsampled_in_sym = insym

    if non_cur_space:
        downsampled_in_sym = mx.sym.SliceChannel(data=downsampled_in_sym, axis=0, num_outputs=t_dim)
        if key_dim != 0:
            downsampled_in_sym = mx.sym.Concat(mx.sym.Concat(*downsampled_in_sym[0:key_dim], dim=0),
                                               mx.sym.Concat(*downsampled_in_sym[key_dim+1:], dim=0),
                                               dim=0)
        else:
            downsampled_in_sym = mx.sym.Concat(*downsampled_in_sym[key_dim+1:], dim=0)

    if naive_conv_l:
        phi = mx.sym.Convolution(downsampled_in_sym, kernel=(1, 1), stride=(1, 1), num_filter=inter_filter,
                                 no_bias=True, name='nonlocal_conv{}%d2'.format(prefix) % ith)
    else:
        phi = mx.sym.expand_dims(downsampled_in_sym, axis=1)
        phi = mx.sym.Pooling(data=phi, kernel=(2, 1, 1), stride=(2, 1, 1), pool_type='max')
        phi = mx.sym.squeeze(phi, axis=1)

    if normalize:
        # phi = mx.sym.L2Normalization(data=phi, mode='spatial')
        phi = mx.sym.L2Normalization(data=phi, mode='channel')


    # T x (num_filter / 2) x H x W
    # data size: 1 x HW x (num_filter / 2)
    theta = mx.sym.transpose(theta, axes=(0, 2, 3, 1)) # 3 x 38 x 50 x 512
    theta_shape = theta
    phi = mx.sym.transpose(phi, axes=(0, 2, 3, 1))
    # data size: THW x (num_filter / 2)
    theta = mx.sym.reshape(theta, shape=(-3, -2)) # (3 x 38) x 50 x 512
    theta = mx.sym.reshape(theta, shape=(-3, -2)) # (3 x 38 x 50) x 512
    phi = mx.sym.reshape(phi, shape=(-3, -2))
    phi = mx.sym.reshape(phi, shape=(-3, -2))

    # f size: THW x T(H/2)(W/2)
    # f = mx.sym.dot(lhs=theta, rhs=phi)
    # e.g., THW x 512 * TH/2W/2 x 512 -> THW x TH/2W/2
    out_theta = mx.sym.identity(theta)
    out_phi = mx.sym.identity(phi)
    theta_phi = mx.sym.dot(lhs=theta, rhs=phi, transpose_b=True, name='nonlocal_dot{}%d1'.format(prefix) % ith)

    if scale:
        if normalize:
            theta_phi = theta_phi * (inter_filter**(0.5))
        else:
            theta_phi = theta_phi * (inter_filter**(-0.5))

    p = mx.sym.softmax(theta_phi, axis=1)
    similarity = mx.sym.identity(p)

    g = downsampled_in_sym

    # g size: T x HW X (num_filter / 2)
    g = mx.sym.transpose(g, axes=(0, 2, 3, 1))
    # g size: T(H/2)(W/2) x (num_filter / 2)
    g = mx.sym.reshape(g, shape=(-3 , -2))
    g = mx.sym.reshape(g, shape=(-3 , -2))


    # e.g, g(T(H/2)(W/2), 512), p(1HW, T(H/2)(W/2)) -> (THW, 512)
    y = mx.sym.dot(lhs=p, rhs=g, name='nonlocal_dot{}%d2'.format(prefix) % ith) # 1HW x num_filter

    if output_cur_only:
        y = mx.sym.reshape(y, shape=(-4, 1, -1, 0)) # 1 x HW x num_filter
    else:
        y = mx.sym.reshape(y, shape=(-4, t_dim, -1, 0)) # 1 x HW x num_filter

    # T x num_filter x HW
    y = mx.sym.reshape_like(lhs=mx.sym.transpose(y, axes=(0, 2, 1)), rhs=key_insym)


    outsym = key_insym + y

    return outsym, mx.sym.BlockGrad(similarity)

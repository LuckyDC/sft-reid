import mxnet as mx


def get_symbol(data):
    data = mx.symbol.Variable(name="data")
    conv1_1 = mx.symbol.Convolution(name='conv1_1', data=data, num_filter=64, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                    no_bias=False)
    relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1, act_type='relu')
    conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1, num_filter=64, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2, act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu1_2, kernel=(2, 2), stride=(2, 2), pool_type='max',
                              pooling_convention="full", pad=(0, 0))

    conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1, num_filter=128, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1, act_type='relu')
    conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1, num_filter=128, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2, act_type='relu')
    pool2 = mx.symbol.Pooling(name='pool2', data=relu2_2, kernel=(2, 2), stride=(2, 2), pool_type='max',
                              pooling_convention="full", pad=(0, 0))

    conv3_1 = mx.symbol.Convolution(name='conv3_1', data=pool2, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1, act_type='relu')
    conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2, act_type='relu')
    conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3, act_type='relu')
    pool3 = mx.symbol.Pooling(name='pool3', data=relu3_3, kernel=(2, 2), stride=(2, 2), pool_type='max',
                              pooling_convention="full", pad=(0, 0))

    conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3, num_filter=512, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1, act_type='relu')
    conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1, num_filter=512, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2, act_type='relu')

    conv4_3 = mx.symbol.Convolution(name='conv4_3', data=relu4_2, num_filter=512, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu4_3 = mx.symbol.Activation(name='relu4_3', data=conv4_3, act_type='relu')
    pool4 = mx.symbol.Pooling(name='pool4', data=relu4_3, kernel=(2, 2), stride=(2, 2), pool_type='max',
                              pooling_convention="full", pad=(0, 0))

    conv5_1 = mx.symbol.Convolution(name='conv5_1', data=pool4, num_filter=512, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu5_1 = mx.symbol.Activation(name='relu5_1', data=conv5_1, act_type='relu')
   
    conv5_2 = mx.symbol.Convolution(name='conv5_2', data=relu5_1, num_filter=512, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu5_2 = mx.symbol.Activation(name='relu5_2', data=conv5_2, act_type='relu')
   
    conv5_3 = mx.symbol.Convolution(name='conv5_3', data=relu5_2, num_filter=512, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu5_3 = mx.symbol.Activation(name='relu5_3', data=conv5_3, act_type='relu')
    #pool5 = mx.symbol.Pooling(name='pool5', data=relu5_3, kernel=(2, 2), stride=(2, 2), pool_type='max',
     #                         pooling_convention="full", pad=(0, 0))
    #flat_pool5 = mx.symbol.flatten(pool5, name="flat_pool5")
    
    #fc6 = mx.symbol.FullyConnected(data=flat_pool5, num_hidden=4096, name='fc6')

    return relu5_3


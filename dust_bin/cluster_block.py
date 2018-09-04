import mxnet as mx
import numpy as np


def str_to_bool(string):
    if not isinstance(string, str):
        raise TypeError("Input type is not string type!")
    return True if string.lower() == "true" else False


class ClusterBlock(mx.operator.CustomOp):
    def __init__(self, normalize, keep_diag, temperature):
        self.normalize = normalize
        self.keep_diag = keep_diag
        self.temperature = temperature

        self.affinity = None

    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0]

        # aff_in = mx.nd.L2Normalization(data) if self.normalize else data
        # affinity = mx.nd.dot(aff_in, aff_in, transpose_b=True)
        # affinity = mx.nd.exp(affinity) / self.temperature
        #
        # if not self.keep_diag:
        #     affinity = affinity * (1 - mx.nd.eye(data.shape[0], ctx=data.context))
        #
        # affinity = affinity / mx.nd.sum(affinity, axis=1, keepdims=True)
        # self.affinity = affinity
        #
        # out = mx.nd.dot(affinity, data)

        self.assign(out_data[0], req[0], data)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        grad = out_grad[0]

        grad = mx.nd.dot(self.affinity.T, grad)
        self.assign(in_grad[0], req[0], grad)


@mx.operator.register("ClusterBlock")
class ClusterBlockProp(mx.operator.CustomOpProp):
    def __init__(self, normalize="True", keep_diag="True", temperature=1.0):
        super(ClusterBlockProp, self).__init__(need_top_grad=True)

        self.normalize = str_to_bool(normalize)
        self.keep_diag = str_to_bool(keep_diag)
        self.temperature = float(temperature)

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], []

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return ClusterBlock(self.normalize, self.keep_diag, self.temperature)

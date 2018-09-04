from __future__ import division
import mxnet as mx

from utils.ssc import sparse_coef_recovery


def str2bool(s):
    if isinstance(s, bool):
        return s
    s = s.lower()

    assert s in ["true", "false"]
    return True if s == "true" else False


class Subspace(mx.operator.CustomOp):
    def __init__(self, opt, constrain, reg_scale):
        self.opt = opt
        self.reg_scale = reg_scale
        self.constrain = constrain

        self.similarity = None

    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0]
        sim = sparse_coef_recovery(data.asnumpy().T, cst=self.constrain, Opt=self.opt, lmbda=self.reg_scale)
        sim = mx.nd.array(sim.T, ctx=data.context)

        self.similarity = sim
        out = mx.nd.dot(sim, data)

        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        grad = mx.nd.dot(self.similarity.T, out_grad[0])
        self.assign(in_grad[0], req[0], grad)


@mx.operator.register("Subspace")
class SubspaceProp(mx.operator.CustomOpProp):
    def __init__(self, opt="Lasso", constrain=True, reg_scale=0.001):
        super(SubspaceProp, self).__init__(need_top_grad=True)
        self.opt = str(opt)
        self.constrain = str2bool(constrain)
        self.reg_scale = float(reg_scale)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        return [data_shape], [data_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return Subspace(self.opt, self.constrain, self.reg_scale)

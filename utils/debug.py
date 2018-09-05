"""
Debug Op
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import marshal
import types

import mxnet as mx
import numpy as np

np.set_printoptions(threshold=np.inf, precision=4, linewidth=120)


class DebugOperator(mx.operator.CustomOp):
    def __init__(self, **kwargs):
        super(DebugOperator, self).__init__()
        self.pos = kwargs.get("pos", None)
        self.num_args = int(kwargs.get("num_args", 1))
        self.do_forward = bool(kwargs.get("do_forward", False))
        self.do_backward = bool(kwargs.get("do_backward", False))
        self.callback = None
        if "callback" in kwargs:
            callback_func_code = marshal.loads(json.loads(kwargs["callback"]).encode("latin"))
            self.callback = types.FunctionType(callback_func_code, globals())

    def forward(self, is_train, req, in_data, out_data, aux):
        if self.do_forward:
            in_data_cpu = in_data[0].asnumpy()

            if self.callback is not None:
                self.callback(in_data_cpu)

        for o, r, i in zip(out_data, req, in_data):
            self.assign(o, r, i)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if self.do_backward:
            out_grad_cpu = out_grad[0].asnumpy()
            if self.callback is not None:
                self.callback(out_grad_cpu)

        for i, r, o in zip(in_grad, req, out_grad):
            self.assign(i, r, o)


@mx.operator.register("Debug")
class DebugProp(mx.operator.CustomOpProp):
    def __init__(self, **kwargs):
        super(DebugProp, self).__init__(need_top_grad=True)
        self._kwargs = kwargs

    def list_arguments(self):
        inputs = ['data']
        num_args = int(self._kwargs.get("num_args", 1))
        if num_args > 1:
            for i in range(1, num_args):
                inputs.append("data%d" % i)
        return inputs

    def list_outputs(self):
        return ['output']

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return out_grad

    def infer_shape(self, in_shape):
        return in_shape, in_shape[:1]

    def create_operator(self, ctx, shapes, dtypes):
        return DebugOperator(**self._kwargs)


def Debug(data, type="nchw", num_args=1, **kwargs):
    kwargs.update({"pos": type, "data": data, "num_args": num_args, "do_forward": True})
    return mx.sym.Custom(op_type="Debug", **kwargs)


def forward_debug(data, **kwargs):
    kwargs.update({"data": data, "do_forward": True})
    if "callback" in kwargs:
        callback = kwargs["callback"]
        callback_code = marshal.dumps(callback.__code__)
        kwargs["callback"] = json.dumps(callback_code.decode("latin"))
    return mx.sym.Custom(op_type="Debug", **kwargs)


def backward_debug(data, **kwargs):
    kwargs.update({"data": data, "do_backward": True})
    if "callback" in kwargs:
        callback = kwargs["callback"]
        callback_code = marshal.dumps(callback.__code__)
        kwargs["callback"] = json.dumps(callback_code.decode("latin"))
    return mx.sym.Custom(op_type="Debug", **kwargs)

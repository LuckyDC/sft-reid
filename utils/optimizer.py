from __future__ import print_function
from __future__ import division

import mxnet as mx
from mxnet.optimizer import SGD, clip
from mxnet.ndarray import NDArray

import logging


@mx.optimizer.register
class CustomSGD(SGD):
    def __init__(self, **kwargs):
        super(CustomSGD, self).__init__(**kwargs)

    def update(self, index, weight, grad, state):
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad *= self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        grad += wd * weight

        state[:] *= self.momentum
        state[:] += grad

        weight[:] += -lr * state

    def update_multi_precision(self, index, weight, grad, state):
        self.update(index, weight, grad, state)

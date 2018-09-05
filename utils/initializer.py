import mxnet as mx


@mx.init.register
# @alias("init_with_array")
class InitWithArray(mx.init.Initializer):
    def __init__(self, array):
        super(InitWithArray, self).__init__(array=array)
        self.array = mx.nd.array(array)

    def _init_weight(self, name, arr):
        arr[:] = self.array[:]

    def _init_bias(self, _, arr):
        arr[:] = self.array[:]

from __future__ import division
import logging
from mxnet.lr_scheduler import LRScheduler


class WarmupFactorScheduler(LRScheduler):

    def __init__(self, step, factor=1, warmup=False, mode="constant", warmup_lr=0, warmup_step=0, stop_factor_lr=1e-8):
        super(WarmupFactorScheduler, self).__init__()
        if step < 1:
            raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        if mode not in ["constant", "gradual"]:
            raise ValueError("Mode must be \"gradual\" or \"constant\"")
        if warmup_step >= step:
            raise ValueError("Parameter warm_up_step must be smaller than parameter step")

        self.step = step
        self.factor = factor
        self.mode = mode
        self.warmup = warmup
        self.warmup_lr = warmup_lr
        self.warmup_step = warmup_step
        self.stop_factor_lr = stop_factor_lr
        self.count = warmup_step if self.warmup else 0

    def __call__(self, num_update):
        # warming up
        if self.warmup and num_update < self.warmup_step:
            if self.mode == "constant":
                return self.warmup_lr
            if self.mode == "gradual":
                return self.warmup_lr + (self.base_lr - self.warmup_lr) / self.warmup_step * num_update

        # NOTE: use while rather than if  (for continuing training via load_epoch)
        while num_update > self.count + self.step:
            self.count += self.step
            self.base_lr *= self.factor
            if self.base_lr < self.stop_factor_lr:
                self.base_lr = self.stop_factor_lr
                logging.info("Update[%d]: now learning rate arrived at %0.5e, will not "
                             "change in the future" % (num_update, self.base_lr))
            else:
                logging.info("Update[%d]: Change learning rate to %0.5e" % (num_update, self.base_lr))
        return self.base_lr


class WarmupMultiFactorScheduler(LRScheduler):

    def __init__(self, step, factor=1.0, warmup=False, mode="constant", warmup_lr=0.0, warmup_step=0,
                 stop_factor_lr=1e-8):
        super(WarmupMultiFactorScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, step_ in enumerate(step):
            if i != 0 and step[i] <= step[i - 1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if step_ < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        if mode not in ["constant", "gradual"]:
            raise ValueError("Mode must be \"gradual\" or \"constant\"")
        if warmup_step >= step[0]:
            raise ValueError("Parameter warm_up_step must be smaller than parameter step")

        self.step = step
        self.factor = factor
        self.mode = mode
        self.warmup = warmup
        self.warmup_lr = warmup_lr
        self.warmup_step = warmup_step
        self.stop_factor_lr = stop_factor_lr
        self.cur_step_ind = 0
        self.count = warmup_step if self.warmup else 0

    def __call__(self, num_update):
        # warming up
        if self.warmup and num_update < self.warmup_step:
            if self.mode == "constant":
                return self.warmup_lr
            if self.mode == "gradual":
                return self.warmup_lr + (self.base_lr - self.warmup_lr) / self.warmup_step * num_update

        # NOTE: use while rather than if  (for continuing training via load_epoch)
        while self.cur_step_ind < len(self.step):
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.base_lr *= self.factor
                self.cur_step_ind += 1
                if self.base_lr < self.stop_factor_lr:
                    self.base_lr = self.stop_factor_lr
                    logging.info("Update[%d]: now learning rate arrived at %0.5e, will not "
                                 "change in the future" % (num_update, self.base_lr))
                else:
                    logging.info("Update[%d]: Change learning rate to %0.5e" % (num_update, self.base_lr))
            else:
                return self.base_lr
        return self.base_lr


class ExponentialScheduler(LRScheduler):

    def __init__(self, base_lr, exp, start_step, end_step):
        super(ExponentialScheduler, self).__init__()
        assert isinstance(start_step, int)
        if end_step < start_step:
            raise ValueError("The value of end_step must be larger than start_step")
        self.start_step = start_step
        self.end_step = end_step
        self.exp = exp
        self.base_lr = base_lr
        self.base_lr_orig = base_lr

    def __call__(self, num_update):
        if num_update < self.start_step:
            self.base_lr = self.base_lr_orig
        elif num_update < self.end_step:
            self.base_lr = self.base_lr_orig * self.exp ** (
                    (num_update - self.start_step) / (self.end_step - self.start_step))
        else:
            self.base_lr = self.base_lr_orig * self.exp

        return self.base_lr


class WarmupMultiFactorScheduler_v1(LRScheduler):

    def __init__(self, steps, iters_per_epoch, factor=0.1, warmup=False, mode="constant", warmup_begin_lr=0.0,
                 warmup_epoch=0):
        super(WarmupMultiFactorScheduler_v1, self).__init__()
        assert isinstance(steps, list) and len(steps) >= 1
        for i, step_ in enumerate(steps):
            if i != 0 and steps[i] <= steps[i - 1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if step_ < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        if mode not in ["constant", "gradual"]:
            raise ValueError("Mode must be \"gradual\" or \"constant\"")
        if warmup_epoch >= steps[0]:
            raise ValueError("Parameter warm_up_step must be smaller than parameter step")

        self.steps = steps
        self.factor = factor
        self.iters_per_epoch = iters_per_epoch
        self.mode = mode
        self.warmup = warmup
        self.warmup_begin_lr = warmup_begin_lr
        self.warmup_epoch = warmup_epoch

        self.cur_step_ind = 0

    def __call__(self, num_update):
        epoch = num_update // self.iters_per_epoch
        # warming up
        if self.warmup and epoch < self.warmup_epoch:
            if self.mode == "constant":
                return self.warmup_begin_lr
            if self.mode == "gradual":
                return self.warmup_begin_lr + (self.base_lr - self.warmup_begin_lr) / self.warmup_epoch * epoch

        # NOTE: use while rather than if  (for continuing training via load_epoch)
        while self.cur_step_ind < len(self.steps):
            if epoch >= self.steps[self.cur_step_ind]:
                self.base_lr *= self.factor
                self.cur_step_ind += 1

                logging.info("Update[%d]: Change learning rate to %0.5e" % (num_update, self.base_lr))
            else:
                return self.base_lr
        return self.base_lr


if __name__ == '__main__':
    lr_sheduler = WarmupMultiFactorScheduler_v1([30, 50],
                                                factor=0.1,
                                                warmup=True,
                                                mode="gradual",
                                                warmup_begin_lr=0.001,
                                                warmup_epoch=10,
                                                iters_per_epoch=5)
    lr_sheduler.base_lr = 0.1
    print(lr_sheduler.base_lr)
    for i in range(1000):
        print("{} : {}".format(i, lr_sheduler(i)))

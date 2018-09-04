from __future__ import print_function, division

import sys
import os
import logging
import yaml
import importlib
import subprocess
import mxnet as mx
from easydict import EasyDict
from pprint import pprint

import utils.augmentor as augmentor
from utils.sub_module import *
from utils.misc import clean_immediate_checkpoints
from utils.misc import CustomAccuracy, CustomCrossEntropy
from utils.group_iterator import GroupIterator
from utils.plain_iterator import PlainIterator
from utils.memonger import search_plan
from utils.lr_scheduler import WarmupMultiFactorScheduler, ExponentialScheduler

# operators
import operators.triplet_loss


def build_network(symbol, num_id, p_size, **kwargs):
    triplet_normalization = kwargs.get("triplet_normalization", False)
    use_triplet = kwargs.get("use_triplet", False)
    use_softmax = kwargs.get("use_softmax", False)
    triplet_margin = kwargs.get("triplet_margin", 0.5)
    softmax_margin = kwargs.get("softmax_margin", 0.5)
    use_verification = kwargs.get("use_verification", False)

    label = mx.symbol.Variable(name="softmax_label")

    group = [label]

    pooling = mx.symbol.Pooling(data=symbol, kernel=(1, 1), global_pool=True, pool_type='avg', name='global_pool')
    flatten = mx.symbol.Flatten(data=pooling, name='flatten')

    flatten_gbms, sim, aff = gbms_block(flatten, sim_normalize=True, **kwargs)

    # triplet loss
    if use_triplet:
        data_triplet = mx.sym.L2Normalization(flatten, name="triplet_l2") if triplet_normalization else flatten
        triplet = mx.symbol.Custom(data=data_triplet, p_size=p_size, margin=triplet_margin, op_type='TripletLoss',
                                   name='triplet')
        group.append(triplet)

    # softmax cross entropy loss
    if use_softmax:
        softmax = classification_branch(flatten, flatten_gbms, label=label, num_id=num_id, margin=softmax_margin,
                                        **kwargs)
        group.extend(softmax)

    if use_verification:
        veri = verification_branch(sim, label, p_size, k_size, name="veri")
        group.append(veri)

    return mx.symbol.Group(group)


def get_iterators(data_dir, p_size, k_size, image_size, aug_dict, seed):
    random_mirror = aug_dict.get("random_mirror", False)
    random_crop = aug_dict.get("random_crop", False)
    random_erasing = aug_dict.get("random_erasing", False)
    resize_size = aug_dict.get("resize_size", image_size)
    pad_to_size = aug_dict.get("pad_to_size", None)
    color_jitter = aug_dict.get("color_jitter", False)

    transforms = []
    if color_jitter:
        transforms.append(augmentor.ColorJitter(0.1, 0.1, 0.1, 0))

    transforms.append(augmentor.Cast("float32"))

    resize_size = tuple(resize_size)
    transforms.append(augmentor.Resize(resize_size))

    if pad_to_size is not None:
        transforms.append(augmentor.PadTo(pad_to_size, fill_value=127))

    if random_crop:
        transforms.append(augmentor.RandomCrop(image_size))

    if random_mirror:
        transforms.append(augmentor.RandomHorizontalFlip())

    if random_erasing:
        transforms.append(augmentor.RandomErase())

    transform = augmentor.Compose(transforms)

    train = GroupIterator(data_dir=data_dir, p_size=p_size, k_size=k_size,
                          transform=transform, num_worker=8, image_size=image_size, random_seed=seed)

    # val = GroupIterator(data_dir=os.path.join(data_dir, "bounding_box_test"), p_size=p_size, k_size=k_size,
    #                     image_size=image_size, random_seed=seed)

    return train, None


if __name__ == '__main__':
    random_seed = 0
    mx.random.seed(random_seed)

    # load configuration
    args = yaml.load(open("config.yml", "r"))
    selected_dataset = args["dataset"]
    datasets = ["duke", "market", "cuhk", "msmt"]
    args["prefix"] = selected_dataset + "/" + args["prefix"]
    for dataset in datasets:
        dataset_config = args.pop(dataset)
        if dataset == selected_dataset:
            args.update(dataset_config)

    args = EasyDict(args)
    pprint(args)

    model_load_prefix = args.model_load_prefix
    model_load_epoch = args.model_load_epoch
    network = args.network
    gpus = args.gpus
    data_dir = args.data_dir
    p_size = args.p_size
    k_size = args.k_size
    lr_step = args.lr_step
    optmizer = args.optimizer
    lr = args.lr
    wd = args.wd
    num_epoch = args.num_epoch
    image_size = args.image_size
    prefix = args.prefix
    batch_size = p_size * k_size
    use_softmax = args.use_softmax
    use_gbms = args.use_gbms
    bottleneck_dims = args.bottleneck_dims
    num_id = args.num_id
    use_triplet = args.use_triplet
    triplet_margin = args.triplet_margin
    aug = args.aug
    deep_sup = args.deep_sup
    begin_epoch = args.begin_epoch
    norm_scale = args.norm_scale
    temperature = args.temperature
    softmax_margin = args.softmax_margin
    use_verification = args.use_verification

    # config logger
    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename='log/%s/%s.log' % (selected_dataset, os.path.basename(prefix)), filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.info(args)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    _, arg_params, aux_params = mx.model.load_checkpoint('%s' % model_load_prefix, model_load_epoch)

    devices = [mx.gpu(i) for i in gpus]

    train, val = get_iterators(data_dir=data_dir, p_size=p_size, k_size=k_size, image_size=image_size, aug_dict=aug,
                               seed=random_seed)

    lr_scheduler = WarmupMultiFactorScheduler(step=[s * train.size for s in lr_step], factor=0.1, warmup=True,
                                              warmup_lr=lr / 100, warmup_step=train.size * 20, mode="gradual")
    # lr_scheduler = ExponentialScheduler(base_lr=lr, exp=0.001, start_step=150 * train.size, end_step=300 * train.size)

    init = mx.initializer.MSRAPrelu(factor_type='out', slope=0.0)

    optimizer_params = {"learning_rate": lr,
                        "wd": wd,
                        "lr_scheduler": lr_scheduler,
                        "rescale_grad": 1.0 / batch_size,
                        "begin_num_update": begin_epoch * train.size}

    if optmizer in ["sgd", "nag"]:
        optimizer_params["momentum"] = 0.9

    symbol = importlib.import_module('symbols.symbol_' + network).get_symbol()

    net = build_network(symbol=symbol, num_id=num_id, batch_size=batch_size, p_size=p_size,
                        softmax_margin=softmax_margin, norm_scale=norm_scale, deep_sup=deep_sup,
                        use_gbms=use_gbms, bottleneck_dims=bottleneck_dims, use_softmax=use_softmax,
                        use_triplet=use_triplet, temperature=temperature, triplet_margin=triplet_margin,
                        use_verification=use_verification)

    if batch_size > 128:
        net = search_plan(net, data=(batch_size, 3) + tuple(image_size), softmax_label=(batch_size,))

    # Metric
    metric_list = []
    if use_softmax:
        acc = CustomAccuracy(output_names=["softmax_output"], label_names=["softmax_label"], deep_sup=deep_sup,
                             name="acc")
        ce_loss = CustomCrossEntropy(output_names=["softmax_output"], label_names=["softmax_label"],
                                     deep_sup=deep_sup, name="ce")
        metric_list.extend([acc, ce_loss])

    if use_triplet:
        triplet_loss = mx.metric.Loss(output_names=["triplet_output"], name="triplet")
        metric_list.append(triplet_loss)

    if use_verification:
        veri_loss = mx.metric.Loss(output_names=["veri_output"], name="veri")
        metric_list.append(veri_loss)

    # ssc = mx.metric.Loss(output_names=["ssc_output"], name="ssc")
    # metric_list.append(ssc)

    metric = mx.metric.CompositeEvalMetric(metrics=metric_list)

    model = mx.mod.Module(symbol=net, data_names=train.data_names, label_names=train.label_names, context=devices,
                          logger=logger)

    model.fit(train_data=train,
              eval_data=None,
              eval_metric=metric,
              validation_metric=metric,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              initializer=init,
              optimizer=optmizer,
              optimizer_params=optimizer_params,
              num_epoch=num_epoch,
              begin_epoch=begin_epoch,
              batch_end_callback=mx.callback.Speedometer(batch_size=batch_size, frequent=5),
              epoch_end_callback=mx.callback.do_checkpoint("models/" + prefix, period=10),
              kvstore='device')

    clean_immediate_checkpoints("models", prefix, num_epoch)

    # cmd = "python{} eval.py {} {}".format(sys.version[0], gpus[0], "models/" + prefix + "-%04d.params" % num_epoch)
    # subprocess.check_call(cmd.split(" "))

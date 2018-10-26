from __future__ import print_function, division

import os
import logging
import yaml
import importlib

from pprint import pprint

from utils.sub_module import *
from utils.misc import DotDict
from utils.misc import clean_immediate_checkpoints
from utils.misc import CustomAccuracy, CustomCrossEntropy
from utils.iterators import get_train_iterator
from utils.memonger import search_plan


def build_network(symbol, num_id, **kwargs):
    use_triplet = kwargs.get("use_triplet", False)
    use_softmax = kwargs.get("use_softmax", False)
    triplet_margin = kwargs.get("triplet_margin", 0.5)
    softmax_margin = kwargs.get("softmax_margin", 0.5)
    batch_size = kwargs.get("batch_size", 0)

    label = mx.symbol.Variable(name="softmax_label")

    group = [label]

    pooling = mx.symbol.Pooling(data=symbol, kernel=(1, 1), global_pool=True, pool_type='avg', name='global_pool')
    flatten = mx.symbol.Flatten(data=pooling, name='flatten')

    flatten_rw = rw_module(flatten, sim_normalize=True, **kwargs)

    # triplet loss
    if use_triplet:
        triplet = triplet_hard_loss(flatten, label, margin=triplet_margin, batch_size=batch_size)

        group.append(triplet)

    # softmax cross entropy loss
    if use_softmax:
        softmax = classification_branch(flatten, flatten_rw, label=label, num_id=num_id, margin=softmax_margin,
                                        **kwargs)
        group.extend(softmax)

    return mx.symbol.Group(group)


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

    args = DotDict(args)
    pprint(args)

    model_load_prefix = args.model_load_prefix
    model_load_epoch = args.model_load_epoch
    network = args.network
    gpus = args.gpus
    data_dir = args.data_dir
    p_size = args.p_size
    k_size = args.k_size
    num_epoch = args.num_epoch
    image_size = args.image_size
    prefix = args.prefix
    batch_size = p_size * k_size
    use_softmax = args.use_softmax
    use_rw = args.use_rw
    bottleneck_dims = args.bottleneck_dims
    num_id = args.num_id
    use_triplet = args.use_triplet
    triplet_margin = args.triplet_margin
    deep_sup = args.deep_sup
    norm_scale = args.norm_scale
    temperature = args.temperature
    softmax_margin = args.softmax_margin
    memonger = args.memonger

    # config logger
    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename='log/%s/%s.log' % (selected_dataset, os.path.basename(prefix)), filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info(args)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    devices = [mx.gpu(i) for i in gpus]

    train = get_train_iterator(root=data_dir,
                               p_size=p_size,
                               k_size=k_size,
                               image_size=image_size,
                               random_erase=args.random_erase,
                               random_mirror=args.random_mirror,
                               random_crop=args.random_crop,
                               num_worker=8,
                               seed=random_seed)

    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(factor=0.1,
                                                        base_lr=args.lr,
                                                        step=[s * train.size for s in args.lr_step],
                                                        warmup_steps=train.size * 20,
                                                        warmup_begin_lr=args.lr / 100,
                                                        warmup_mode='linear')

    init = mx.initializer.MSRAPrelu(factor_type='out', slope=0.0)

    if args.optimizer == "sgd":
        optimizer = mx.optimizer.SGD(learning_rate=args.lr,
                                     wd=args.wd,
                                     momentum=0.9,
                                     lr_scheduler=lr_scheduler,
                                     rescale_grad=1 / batch_size,
                                     begin_num_update=args.begin_epoch * train.size)
    else:
        optimizer = mx.optimizer.Adam(learning_rate=args.lr,
                                      wd=args.wd,
                                      lr_scheduler=lr_scheduler,
                                      rescale_grad=1 / batch_size,
                                      begin_num_update=args.begin_epoch * train.size)

    _, arg_params, aux_params = mx.model.load_checkpoint('%s' % model_load_prefix, model_load_epoch)
    symbol = importlib.import_module('symbols.symbol_' + network).get_symbol()

    net = build_network(symbol=symbol, num_id=num_id, batch_size=batch_size,
                        softmax_margin=softmax_margin, norm_scale=norm_scale, deep_sup=deep_sup,
                        use_rw=use_rw, bottleneck_dims=bottleneck_dims, use_softmax=use_softmax,
                        use_triplet=use_triplet, temperature=temperature, triplet_margin=triplet_margin)

    if memonger:
        net = search_plan(net, data=(batch_size, 3) + tuple(image_size), softmax_label=(batch_size,))

    # Metric
    metric_list = []
    if use_softmax:
        acc = CustomAccuracy(batch_size=batch_size, output_names=["softmax_output"], label_names=["softmax_label"],
                             name="acc")
        ce_loss = CustomCrossEntropy(batch_size=batch_size, output_names=["am_softmax_output"],
                                     label_names=["softmax_label"], name="ce")
        metric_list.extend([acc, ce_loss])

    if use_triplet:
        triplet_loss = mx.metric.Loss(output_names=["triplet_output"], name="triplet")
        metric_list.append(triplet_loss)

    metric = mx.metric.CompositeEvalMetric(metrics=metric_list)

    model = mx.mod.Module(symbol=net, data_names=train.data_names, label_names=train.label_names, context=devices,
                          logger=logger)

    model.fit(train_data=train,
              eval_metric=metric,
              validation_metric=metric,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              initializer=init,
              optimizer=optimizer,
              num_epoch=num_epoch,
              begin_epoch=args.begin_epoch,
              batch_end_callback=mx.callback.Speedometer(batch_size=batch_size, frequent=10),
              epoch_end_callback=mx.callback.do_checkpoint("models/" + prefix, period=10),
              kvstore='device')

    clean_immediate_checkpoints("models", prefix, num_epoch)

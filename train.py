from __future__ import print_function, division

import os
import logging
import yaml
import importlib

from pprint import pprint

from utils.components import *
from utils.misc import DotDict
from utils.misc import clean_immediate_checkpoints
from utils.iterators import get_train_iterator
from utils.memonger import search_plan
from utils.misc import CustomCrossEntropy, CustomAccuracy


def build_network(symbol, **kwargs):
    num_id = kwargs.get("num_id", -1)
    use_triplet = kwargs.get("use_triplet", False)
    use_softmax = kwargs.get("use_softmax", False)
    triplet_margin = kwargs.get("triplet_margin", 0.5)
    softmax_margin = kwargs.get("softmax_margin", 0.5)
    batch_size = kwargs.get("batch_size", 0)
    use_sft = kwargs.get("use_sft", False)
    temperature = kwargs.get("temperature", 1.0)
    deep_sup = kwargs.get("deep_sup", False)
    bottleneck_dims = kwargs.get("bottleneck_dims", 512)
    norm_scale = kwargs.get("norm_scale", 10.0)

    label = mx.symbol.Variable(name="softmax_label")

    pooling = mx.symbol.Pooling(data=symbol, kernel=(1, 1), global_pool=True, pool_type='avg', name='global_pool')
    flatten = mx.symbol.Flatten(data=pooling, name='flatten')

    flatten_sft, sim = sft_module(flatten, temperature=temperature)

    # softmax loss
    if not use_sft:
        logits = my_classifier(flatten, num_id=num_id, bottleneck_dims=bottleneck_dims)
        amsoftmax = AMSoftmaxOutput(logits, label=label, num_id=num_id, margin=softmax_margin, scale=norm_scale,
                                    name="amsoftmax")
    else:
        if deep_sup:
            concat = mx.symbol.concat(flatten, flatten_sft, dim=0)
            logits = my_classifier(concat, num_id=num_id, bottleneck_dims=bottleneck_dims)

            amsoftmax = AMSoftmaxOutput(logits, mx.symbol.tile(label, reps=2), num_id=num_id, margin=softmax_margin,
                                        scale=norm_scale, name="amsoftmax")

        else:
            logits = my_classifier(flatten_sft, num_id=num_id, bottleneck_dims=bottleneck_dims)

            amsoftmax = AMSoftmaxOutput(logits, label, num_id=num_id, margin=softmax_margin, scale=norm_scale,
                                        name="amsoftmax")

    # triplet loss
    triplet = triplet_hard_loss(flatten, label, margin=triplet_margin, batch_size=batch_size, name="triplet")

    # output group
    group = []
    if use_softmax:
        group.extend(amsoftmax)
    if use_triplet:
        group.append(triplet)

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

    p_size = args.p_size
    k_size = args.k_size
    batch_size = p_size * k_size
    image_size = args.image_size
    prefix = args.prefix

    # Config logger
    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename='log/%s/%s.log' % (selected_dataset, os.path.basename(prefix)), filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info(args)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    devices = [mx.gpu(i) for i in args.gpus]

    # Data loader
    data_dir = os.path.join(args.root, args.train)
    train = get_train_iterator(root=data_dir,
                               p_size=p_size,
                               k_size=k_size,
                               image_size=image_size,
                               random_erase=args.random_erase,
                               random_mirror=args.random_mirror,
                               random_crop=args.random_crop,
                               num_worker=8,
                               seed=random_seed)

    # Optimizer
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(factor=0.1,
                                                        base_lr=args.lr,
                                                        step=[s * train.size for s in args.lr_step],
                                                        warmup_steps=train.size * 20,
                                                        warmup_begin_lr=args.lr / 100,
                                                        warmup_mode='linear')

    optimizer = args.optimizer

    optimizer_params = {"learning_rate": args.lr,
                        "wd": args.wd,
                        "momentum": 0.9,
                        "lr_scheduler": lr_scheduler,
                        "rescale_grad": 1 / batch_size,
                        "begin_num_update": args.begin_epoch * train.size}

    # Metric
    metric_list = []
    if args.use_softmax:
        acc = CustomAccuracy(name="acc", output_names=["logits_output"], label_names=["softmax_label"])
        ce_loss = CustomCrossEntropy(name="ce", output_names=["amsoftmax_output"], label_names=["softmax_label"])

        metric_list.extend([acc, ce_loss])

    if args.use_triplet:
        triplet_loss = mx.metric.Loss(output_names=["triplet_output"], name="triplet")
        metric_list.append(triplet_loss)

    metric = mx.metric.CompositeEvalMetric(metrics=metric_list)

    # Load pre-trained model
    _, arg_params, aux_params = mx.model.load_checkpoint('%s' % args.model_load_prefix, args.model_load_epoch)
    symbol = importlib.import_module('symbols.symbol_' + args.network).get_symbol()

    net = build_network(symbol=symbol, batch_size=batch_size, **args)

    model = mx.mod.Module(symbol=net, data_names=train.data_names, label_names=train.label_names, context=devices,
                          logger=logger)

    init = mx.initializer.MSRAPrelu(factor_type='out', slope=0.0)

    if args.memonger:
        net = search_plan(net, data=(batch_size, 3) + tuple(image_size), softmax_label=(batch_size,))

    model.fit(train_data=train,
              eval_metric=metric,
              validation_metric=metric,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              initializer=init,
              optimizer=optimizer,
              optimizer_params=optimizer_params,
              num_epoch=args.num_epoch,
              begin_epoch=args.begin_epoch,
              batch_end_callback=mx.callback.Speedometer(batch_size=batch_size, frequent=10),
              epoch_end_callback=mx.callback.do_checkpoint("models/" + prefix, period=10),
              kvstore='device')

    # Clean immediate checkpoints
    clean_immediate_checkpoints("models", prefix, args.num_epoch)

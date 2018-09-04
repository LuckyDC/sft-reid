from __future__ import print_function

import os
import glob
import shutil
import argparse

from collections import defaultdict


def split_trainval(data_dir):
    imgs = glob.glob(data_dir + "*.jpg") + glob.glob(data_dir + "*.png")

    imgs.sort()

    id2imgs = defaultdict(list)
    for img in imgs:
        idx = int(os.path.basename(img).split("_")[0])
        id2imgs[idx].append(img)

    train_split = os.path.join(data_dir, "train")
    val_split = os.path.join(data_dir, "val")
    os.mkdir(train_split)
    os.mkdir(val_split)

    for v in id2imgs.values():
        shutil.copy(v[0], val_split)

        for f in v[1:]:
            shutil.copy(f, train_split)


if __name__ == '__main__':
    data_dirs = dict(duke="/mnt/truenas/scratch/chuanchen.luo/data/reid/DukeMTMC-reID",
                     market="/mnt/truenas/scratch/chuanchen.luo/data/reid/Market-1501-v15.09.15",
                     cuhk="/mnt/truenas/scratch/chuanchen.luo/data/reid/cuhk03-np/labeled")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", required=True, choices=["cuhk", "duke", "market"], type=str)
    args = parser.parse_args()

    dataset = args.dataset
    split_trainval(data_dirs[dataset])

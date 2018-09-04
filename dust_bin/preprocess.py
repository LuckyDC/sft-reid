from __future__ import print_function, division
import os
import csv
import sys
import random
import glob
import yaml
import subprocess

from easydict import EasyDict


def preprocess(name, data_dir, root_dir, output_dir, shuffle=True, id_map=None):
    '''
    Pre-process the raw data, and generate a .lst file
    :param name: The name of the target .lst file
    :param data_dir: The directory of raw data.
    :param root_dir: The root directory of the data.
    :param output_dir: The directory for saving processed .lst file.
    :param id_map: The provided identity map.
    :return: The processed identity map
    '''
    img_lst = []
    cnt = 0

    img_dir = os.path.join(root_dir, data_dir)
    images = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    print("The number of samples in %s is: %d." % (data_dir, len(images)))

    if id_map is None:
        id_map = {}
        need_map = True
    else:
        need_map = False

    num_id = 0
    for image in images:
        k = image.rfind('/')
        if image[k + 1] == '-':
            label = -1
        else:
            label = int(image[k + 1:k + 5])

        if need_map:
            if label not in id_map:
                id_map[label] = num_id
                num_id += 1
        elif label not in id_map:
            raise ValueError("The identity of this sample is not in the identity map!")

        img_lst.append((cnt, id_map[label], image))
        cnt += 1

    print("The number of identities in %s is: %d.\n" % (data_dir, len(id_map)))

    save(img_lst, lst_name=name, lst_dir=output_dir, shuffle=shuffle)

    return id_map


def even_shuffle(lst):
    idx = list(range(0, len(lst), 2))
    random.shuffle(idx)
    ret = []

    for i in idx:
        ret.append(lst[i])
        ret.append(lst[i + 1])

    return ret


def save(img_lst, lst_name, lst_dir, shuffle=True):
    if not os.path.exists(lst_dir):
        os.makedirs(lst_dir)

    # raw
    fname = os.path.join(lst_dir, lst_name + ".lst")
    fo = csv.writer(open(fname, "w+"), delimiter='\t', lineterminator='\n')
    if shuffle:
        random.shuffle(img_lst)
    for item in img_lst:
        fo.writerow(item)

    # CSV writer for *-even.lst
    writer = csv.writer(open('%s/%s-even.lst' % (lst_dir, lst_name), "w"), delimiter='\t', lineterminator='\n')

    img_lst = sorted(img_lst, key=lambda x: x[1])  # sort the img_lst by id

    # organize the data in thre form of pair
    lst = []
    idx = -1  # the identity of the current pairs
    now = 0
    p = -1
    cnt = 0
    for item in img_lst:
        if item[1] != idx:
            if now % 2 == 1:
                lst.append((cnt, lst[p][1], lst[p][2]))
                cnt += 1
            p = len(lst)
            idx = item[1]
            now = 0

        lst.append((cnt, item[1], item[2]))
        cnt += 1
        now += 1

    if now % 2 == 1:
        lst.append((cnt, lst[p][1], lst[p][2]))
        cnt += 1

    lst = even_shuffle(lst)
    for item in lst:
        writer.writerow(item)

    # CSV writer for *-rand.lst
    writer = csv.writer(open('%s/%s-rand.lst' % (lst_dir, lst_name), "w+"), delimiter='\t', lineterminator='\n')

    random.shuffle(lst)
    for item in lst:
        writer.writerow(item)


if __name__ == '__main__':
    # load configuration
    args = yaml.load(open("config.yml", "r"))
    selected_dataset = "cuhk"
    lst_dir = "/mnt/truenas/scratch/chuanchen.luo/data/reid/%s-list" % selected_dataset

    datasets = ["duke", "market", "cuhk"]
    for dataset in datasets:
        dataset_config = args.pop(dataset)
        if dataset == selected_dataset:
            args.update(dataset_config)
    args = EasyDict(args)
    random.seed(0)

    print(args.data_dir)
    preprocess('train', data_dir='bounding_box_train', root_dir=args.data_dir, output_dir=lst_dir)

    id_map = preprocess('test', data_dir='bounding_box_test', shuffle=False, root_dir=args.data_dir,
                        output_dir=lst_dir)
    preprocess('query', data_dir='query', root_dir=args.data_dir, shuffle=False, output_dir=lst_dir, id_map=id_map)
    # preprocess('gt-query', data_dir='gt_query', root_dir=args.data_dir, output_dir=args.lst_dir)

    print("Training examples: %d. \n" % len(open(os.path.join(lst_dir, "train-even" + ".lst")).readlines()))
    # clean useless .lst file
    useless_lsts = ["query-even", "query-rand", "train"]
    for f in useless_lsts:
        os.system("rm -v %s" % (os.path.join(lst_dir, f + ".lst")))

    # im2rec
    lst_files = glob.glob(os.path.join(lst_dir, "*.lst"))
    ver = sys.version[0]
    pool = []
    for lst in lst_files:
        rec = lst.replace(".lst", ".rec")
        cmd = "python%c utils/im2rec.py %s %s --pass-through" % (ver, lst, rec)
        pool.append(subprocess.Popen(cmd.split(" ")))

    for p in pool:
        p.wait()

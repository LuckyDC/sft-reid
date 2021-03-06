from __future__ import print_function, division, absolute_import
import os
import fnmatch
import cv2
import random
import glob

import mxnet as mx
import numpy as np

from threading import Thread
from collections import defaultdict

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from mxnet.io import DataBatch, DataIter
import utils.augmentor as augmentor


def get_train_iterator(root, p_size, k_size, image_size, random_mirror=False, random_crop=False, random_erase=False,
                       num_worker=4, seed=None):
    transform = []
    transform.append(augmentor.Cast("float32"))

    transform.append(augmentor.Resize(image_size))

    if random_crop:
        transform.append(augmentor.RandomCrop(image_size))

    if random_mirror:
        transform.append(augmentor.RandomHorizontalFlip())

    if random_erase:
        transform.append(augmentor.RandomErase(mean=[127, 127, 127]))

    transform = augmentor.Compose(transform)

    train = TrainIterator(data_dir=root, p_size=p_size, k_size=k_size,
                          transform=transform, num_worker=num_worker, image_size=image_size, random_seed=seed)

    return train


def get_test_iterator(dataset, root, batch_size, image_size, num_worker=4):
    transform = [augmentor.Cast("float32"), augmentor.Resize(image_size)]
    transform = augmentor.Compose(transform)

    train = EvalIterator(data=root, batch_size=batch_size, image_size=image_size, dataset=dataset,
                         transform=transform, num_worker=num_worker)

    return train


class TrainIterator(DataIter):
    def __init__(self, data_dir, p_size, k_size, image_size, transform=None, num_worker=4, random_seed=None):

        if random_seed is None:
            random_seed = random.randint(0, 2 ** 32 - 1)
        np.random.RandomState(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.random_seed = random_seed

        self.p_size = p_size
        self.k_size = k_size
        self.batch_size = p_size * k_size
        self.image_size = image_size
        self.transform = transform
        self.num_worker = num_worker
        self.img_list = None
        self.cursor = 0
        self.size = 0

        print("Data loading..")
        self.id2imgs = self._preprocess(data_dir)
        print("Data loaded!")

        self.num_id = len(self.id2imgs)
        print(self.num_id)

        # multi-thread primitive
        self.result_queue = Queue(maxsize=8 * num_worker)
        self.index_queue = Queue()
        self.workers = None

        self._thread_start(num_worker)

        self.reset()

    def _preprocess(self, data_dir):
        img_list = [os.path.join(root, f) for root, dirs, files in os.walk(data_dir) for f in
                    fnmatch.filter(files, "*.png") + fnmatch.filter(files, "*.jpg")]

        img_list.sort()
        self.img_list = img_list

        assert len(img_list) != 0

        id2imgs = defaultdict(list)
        for img_name in img_list:
            idx = int(os.path.basename(img_name).split("_")[0])
            id2imgs[idx].append(img_name)

        id2imgs_organized = {}
        for i, v in enumerate(id2imgs.values()):
            id2imgs_organized[i] = v

        return id2imgs_organized

    def _insert_queue(self):
        data = []

        pids = list(self.id2imgs.keys()).copy()
        random.shuffle(pids)

        sample_range = list(range(0, len(pids), self.p_size))[:-1]
        self.size = len(sample_range)

        for i in sample_range:
            start = i
            stop = start + self.p_size

            for pid in pids[start:stop]:
                if len(self.id2imgs[pid]) >= self.k_size:
                    imgs = np.random.choice(self.id2imgs[pid], replace=False, size=self.k_size)
                else:
                    imgs = np.random.choice(self.id2imgs[pid], replace=True, size=self.k_size)

                data.extend(imgs)

            label = np.array(pids[start:stop]).repeat(self.k_size)

            self.index_queue.put([data.copy(), label])

            assert len(data) == self.batch_size
            assert len(label) == self.batch_size

            del data[:]

    def _thread_start(self, num_worker):
        self.workers = [Thread(target=self._worker, args=(self.random_seed + i,)) for i in range(num_worker)]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def _worker(self, seed):
        np.random.RandomState(seed)
        np.random.seed(seed)
        random.seed(seed)

        while True:
            indices = self.index_queue.get()
            result = self._get_batch(indices=indices)

            if result is None:
                return

            self.result_queue.put(result)

    def _get_batch(self, indices):
        img_paths = indices[0]
        label = indices[1]

        # Loading
        data = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                img = self.transform(img)

            data.append(img)

        data = np.stack(data, axis=0).transpose([0, 3, 1, 2])
        data = mx.nd.array(data, dtype=np.float32)
        label = mx.nd.array(label)

        assert data.shape[0] == label.shape[0] == self.batch_size

        return [data], [label]

    @property
    def provide_data(self):
        return [('data', (self.batch_size, 3, self.image_size[0], self.image_size[1]))]

    @property
    def provide_label(self):
        return [('softmax_label', (self.batch_size,))]

    @property
    def data_names(self):
        return list(zip(*self.provide_data))[0]

    @property
    def label_names(self):
        return list(zip(*self.provide_label))[0]

    def reset(self):
        self.cursor = 0
        self.index_queue.queue.clear()

        self._insert_queue()

    def next(self):
        if self.cursor >= self.size:
            raise StopIteration

        data, label = self.result_queue.get()

        self.cursor += 1

        return DataBatch(data=data, label=label, provide_data=self.provide_data, provide_label=self.provide_label)


class EvalIterator(DataIter):
    def __init__(self, data, batch_size, image_size, dataset=None, transform=None, num_worker=4):
        self.batch_size = batch_size
        self.image_size = tuple(image_size)
        self.transform = transform

        assert dataset in ["market", "duke", "cuhk", "msmt"]

        self.dataset = dataset

        self.imgs, self.ids, self.cam_ids = self._preprocess(data)
        self.size = len(self.imgs)
        self.num_iters = int(np.ceil(self.size / self.batch_size))
        print(self.size)

        self.cursor = 0

        # multi-thread primitive
        self.result_queue = Queue(maxsize=8 * num_worker)
        self.index_queue = Queue()
        self.workers = None

        self._thread_start(num_worker)
        self._insert_queue()

        self.reset()

    def _preprocess(self, data):
        if self.dataset == "msmt":
            assert os.path.isfile(data)
            imgs = list(np.loadtxt(data, delimiter=" ", dtype=np.str)[:, 0])
            imgs = [os.path.join(os.path.dirname(data), "test", img) for img in imgs]
        else:
            assert os.path.isdir(data)
            print(data)
            imgs = glob.glob(os.path.join(data, "*.jpg")) + glob.glob(os.path.join(data, "*.png"))
            imgs.sort()

        assert len(imgs) != 0
        ids = []
        cam_ids = []
        for img in imgs:
            splits = os.path.basename(img).split("_")
            ids.append(int(splits[0]))

            if self.dataset == "msmt":
                cam_id = int(splits[2])
            else:
                cam_id = int(splits[1][1])

            cam_ids.append(cam_id)

        assert len(imgs) == len(ids) == len(cam_ids)
        return imgs, ids, cam_ids

    @property
    def provide_data(self):
        return [("data", (self.batch_size, 3) + self.image_size)]

    @property
    def provide_label(self):
        return [("id", (self.batch_size,)), ("cam_id", (self.batch_size,)), ("imgs", (self.batch_size,))]

    @property
    def data_names(self):
        return list(zip(*self.provide_data))[0]

    @property
    def label_names(self):
        return list(zip(*self.label_names))[0]

    def _insert_queue(self):
        imgs = []
        ids = []
        cam_ids = []

        for i in range(0, self.size, self.batch_size):
            imgs.extend(self.imgs[i:i + self.batch_size])
            ids.extend(self.ids[i:i + self.batch_size])
            cam_ids.extend(self.cam_ids[i:i + self.batch_size])

            self.index_queue.put([imgs.copy(), ids.copy(), cam_ids.copy()])

            imgs.clear()
            ids.clear()
            cam_ids.clear()

    def _thread_start(self, num_worker):
        self.workers = [Thread(target=self._worker) for _ in range(num_worker)]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

    def _worker(self):
        while True:
            indices = self.index_queue.get()
            if indices is None:
                return

            result = self._get_batch(indices)
            self.result_queue.put(result)

    def _get_batch(self, indices):
        data = []
        for img_path in indices[0]:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.transform(img)

            data.append(img)

        data = np.stack(data, axis=0)
        data = np.transpose(data, axes=[0, 3, 1, 2])

        data = mx.nd.array(data)
        ids = mx.nd.array(indices[1])
        cam_ids = mx.nd.array(indices[2])
        imgs = np.array(indices[0])

        return data, ids, cam_ids, imgs

    def next(self):
        if self.cursor >= self.num_iters:
            raise StopIteration

        self.cursor += 1
        data, ids, cam_ids, imgs = self.result_queue.get()

        if data.shape[0] < self.batch_size:
            pad = self.batch_size - data.shape[0]
        else:
            pad = 0

        return DataBatch(data=[data], label=[ids, cam_ids, imgs], pad=pad, provide_data=self.provide_data,
                         provide_label=self.provide_label)

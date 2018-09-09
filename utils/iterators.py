from __future__ import print_function, division, absolute_import
import os
import fnmatch
import cv2
import random

import mxnet as mx
import numpy as np

from threading import Thread
from collections import defaultdict

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from mxnet.io import DataBatch, DataIter


def pop(x, size):
    return [x.pop(0) for _ in range(size)]


class GroupIterator(DataIter):
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

        id2imgs.pop(0, None)
        id2imgs.pop(-1, None)

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


if __name__ == '__main__':
    import time
    import utils.augmentor as augmentor

    transforms = [augmentor.Cast(), augmentor.PadTo((288, 144)), augmentor.RandomCrop((256, 128))]
    transform = augmentor.Compose(transforms)
    data_dir = "/mnt/truenas/scratch/chuanchen.luo/data/reid/cuhk03-np/labeled"
    train = GroupIterator(data_dir=os.path.join(data_dir, "bounding_box_train"),
                          p_size=16, k_size=4, image_size=(256, 128), transform=transform, num_worker=8)

    tic = time.time()
    tmp = []
    for i in range(100):
        print(i)
        batch = train.next()
        tmp.append(batch.label[0].asnumpy().tolist())
        print(batch.data[0].shape)
        print(batch.label[0].shape)
        print(batch.label[0])
        imgs = batch.data[0].transpose([0, 2, 3, 1]).asnumpy()

        print(imgs.dtype)
        for j in range(imgs.shape[0]):
            img = imgs[j]
            cv2.imwrite("%d-%d.jpg" % (i, j), img[:, :, [2, 1, 0]].astype(np.uint8))

    print((time.time() - tic) / 100)

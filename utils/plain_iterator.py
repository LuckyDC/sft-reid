from __future__ import print_function, division, absolute_import
import os
import cv2
import glob
import random

import mxnet as mx
import numpy as np

from threading import Thread

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from mxnet.io import DataBatch, DataIter


class PlainIterator(DataIter):
    def __init__(self, data_dir, batch_size, image_size, transform=None, num_worker=4, random_seed=None):

        if random_seed is None:
            random_seed = random.randint(0, 2 ** 32 - 1)
        np.random.RandomState(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.random_seed = random_seed

        self.batch_size = batch_size
        self.image_size = image_size
        self.transform = transform
        self.num_worker = num_worker
        self.cursor = 0
        self.num_id = 0

        print("Data loading..")
        self.img_list = self._preprocess(data_dir)
        self.size = len(self.img_list) // self.batch_size
        print("Data loaded!")
        print("Number of samples: %d." % len(self.img_list))
        print("Number of iterations in an epoch: %d." % self.size)
        print("Number of identification: %d." % self.num_id)

        # multi-thread primitive
        self.result_queue = Queue(maxsize=8 * num_worker)
        self.index_queue = Queue()
        self.workers = None

        self._thread_start(num_worker)

        self.reset()

    def _preprocess(self, data_dir):
        img_list = glob.glob(os.path.join(data_dir, "*.jpg")) + glob.glob(os.path.join(data_dir, "*.png"))
        img_list.sort()

        assert len(img_list) != 0

        label_list = [int(os.path.basename(img).split("_")[0]) for img in img_list]
        self.num_id = len(set(label_list))

        id_map = dict(zip(list(set(label_list)), range(self.num_id)))
        label_list = [id_map[i] for i in label_list]

        img_list = list(zip(img_list, label_list))
        return img_list

    def _insert_queue(self):
        random.shuffle(self.img_list)

        sample_range = list(range(0, len(self.img_list), self.batch_size))[:-1]

        for i in sample_range:
            data = self.img_list[i:i + self.batch_size]

            self.index_queue.put(list(zip(*data)))

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

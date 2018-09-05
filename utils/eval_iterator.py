from __future__ import print_function, division
import cv2
import os
import glob
import mxnet as mx
import numpy as np

from mxnet.io import DataBatch, DataIter
from threading import Thread
from collections import defaultdict

try:
    from queue import Queue
except ImportError:
    from Queue import Queue


class EvalIterator(DataIter):
    def __init__(self, data, batch_size, image_size, dataset=None, transform=None, num_worker=4):
        super(EvalIterator, self).__init__(self)
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
        return [("id", (self.batch_size,)), ("cam_id", (self.batch_size,))]

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

        return data, ids, cam_ids

    def next(self):
        if self.cursor >= self.num_iters:
            raise StopIteration

        self.cursor += 1
        data, ids, cam_ids = self.result_queue.get()

        if data.shape[0] < self.batch_size:
            pad = self.batch_size - data.shape[0]
        else:
            pad = 0

        return DataBatch(data=[data], label=[ids, cam_ids], pad=pad, provide_data=self.provide_data,
                         provide_label=self.provide_label)

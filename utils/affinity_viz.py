from __future__ import division, print_function

import cv2
import os
import numpy as np
import scipy.io as sio

from collections import Counter
from sklearn.preprocessing import normalize

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt


def viz_affinity_matrix(mat_path, num_id, id_list=None, name=None):
    assert len(id_list) == num_id
    if name is None:
        name = os.path.splitext(os.path.basename(mat_path))[0]

    mat = sio.loadmat(mat_path)

    features = mat["feat"]
    ids = mat["ids"].squeeze()

    assert ids.shape[0] == features.shape[0]

    counter = Counter(ids)
    for k, v in list(counter.items()):
        if v > 30 or v < 15:
            counter.pop(k)

    if id_list is None:
        selected_id = np.random.choice(list(counter.keys()), size=num_id, replace=False)
    else:
        selected_id = np.array(id_list, dtype=np.int32)

    mask = np.in1d(ids, selected_id)
    features = features[mask, :]
    features = normalize(features)
    ids = ids[mask]
    ids = ids.astype(np.int32)

    print("selected ids:", selected_id)

    # reorganize data
    data = []
    id_list = []
    for i, idx in enumerate(selected_id):
        mask = np.equal(ids, idx)

        feat = features[mask, :]
        data.append(feat)

        id_list.extend([i] * feat.shape[0])

    ids = np.array(id_list)
    ids_1 = np.broadcast_to(ids[:, np.newaxis], [ids.shape[0]] * 2)
    ids_2 = np.broadcast_to(ids[np.newaxis, :], [ids.shape[0]] * 2)
    eq_mask = np.equal(ids_1, ids_2)

    data = np.concatenate(data, axis=0)
    affinity = np.dot(data, data.T)
    # affinity = affinity ** 4

    plt.figure(0, figsize=(10, 10))
    print("max neg:", affinity[np.logical_not(eq_mask)].max())
    print("min pos:", affinity[eq_mask].min())
    im = plt.imshow(affinity, vmin=0, vmax=1, cmap=plt.cm.jet)
    plt.xticks([])
    plt.yticks([])
    # plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join("viz", name + ".pdf"))
    plt.close(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mat_path", type=str)

    args = parser.parse_args()

    np.random.seed(0)

    # id_list = [#324, 4729, 111, 434, 30, 323, 4280, 214, 230, 332, 46, 794,
    #            791, 4723, 2471, 593, 211, 409, 459, 690, 254, 4117, 188, 4102]
    #             # 76, 823, 316, 344, 777, 4128, 427, 676, 376, 816, 2488, 499,
    #             # 4422, 1228, 1893, 341]

    id_list = [4422, 4723, 211, 409, 690, 254]
    viz_affinity_matrix(args.mat_path, len(id_list), id_list)

from __future__ import division, print_function
import os
import numpy as np
import scipy.io as sio

from collections import Counter
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt


def viz_tsne_feature(mat_path, num_id, id_list=None, name=None):
    assert len(ids_list) == num_id
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

    if ids_list is None:
        selected_id = np.random.choice(list(counter.keys()), size=num_id, replace=False)
    else:
        selected_id = np.array(ids_list, dtype=np.int32)

    mask = np.in1d(ids, selected_id)
    features = features[mask, :]
    features = normalize(features)
    ids = ids[mask]
    ids = ids.astype(np.int32)

    print("selected ids:", selected_id)
    # reorganize ids
    for i, idx in enumerate(selected_id):
        mask = np.equal(ids, idx)
        ids[mask] = i

    embed = TSNE(n_components=2, metric="cosine", init="pca", random_state=0).fit_transform(features)

    plt.figure(0, figsize=(10, 10))
    plt.scatter(embed[:, 0], embed[:, 1], c=ids, cmap="tab10", marker=".")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join("viz", name + ".pdf"))
    plt.close(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mat_path", type=str)
    parser.add_argument("--num-id", type=int, default=40)

    args = parser.parse_args()

    np.random.seed(0)

    ids_list = [324, 4729, 111, 434, 30, 323, 4280, 214, 230, 332, 46, 794,
                791, 4723, 2471, 593, 211, 409, 459, 690, 254, 4117, 188, 4102,
                76, 823, 316, 344, 777, 4128, 427, 676, 376, 816, 2488, 499,
                4422, 1228, 1893, 341]
    viz_tsne_feature(args.mat_path, args.num_id, ids_list)

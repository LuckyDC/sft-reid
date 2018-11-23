# Spectral Feature Transformation for Person Re-identification

### Preparation

1. Satisfy the dependency of the program.
    The program is developed on MXNet 1.3.1 and Python 3.5.
    You can install dependency as follow:
    ```bash
    pip3 -r requirement.txt
    ```

2. Create directories to store logs or checkpoints.
    ```bash
    mkdir models log features pretrain_models
    mkdir models/duke models/msmt models/market models/cuhk
    mkdir log/duke log/msmt log/market log/cuhk
    mkdir features/duke features/msmt features/market features/cuhk
    ```
3. Download datasets [CUHK03-NP](https://pan.baidu.com/s/1RNvebTccjmmj1ig-LVjw7A),
[MSMT17](https://docs.google.com/forms/d/e/1FAIpQLScIGhLvB2GzIXjX1oFW0tNUWxkbK2l0fYG5Q9vX93ls2BVsQw/viewform?usp=sf_link),
[Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?usp=sharing),
[DukeMTMC-reID](https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O),
[Market-1501 Distractors](https://drive.google.com/file/d/0B8-rUzbwVRk0cGtxWmFFVDZkNUE/view?usp=sharing), and
decompress them.
You should separate Market-1501 distractors into 3 subsets(100k,100k,300k) before evaluating on it.
Download pretrained [resnet-50](http://data.dmlc.ml/mxnet/models/imagenet/resnet/50-layers/) to `pretrain_models/` directory. 

### Training
You can modify the configuration file `config.yml` before training.
Then run
```bash
python3 train.py
```

### Evaluation
We implement GPU version evaluation code which is much faster.
```bash
python3 eval.py GPU_ID MODEL_PATH
```
GPU_ID is the id of gpu used for evaluation.
MODEL_PATH is the path of model file.

There is an example:
```bash
python3 eval.py 2 models/cuhk/baseline-sft0.1-dsup-0140.params
```
This command will also give re-ranked results in default.

### Post-processing
You can perform post-processing independently.
```bash
python3 -m post_processing.post_clustering DATASET PREFIX
```
DATASET is the name of the dataset.
PREFIX is the prefix model file.
There is an exmaple:
```bash
python3 -m post_processing.post_clustering market baseline-gcn0.1-dsup-amsoftmax0.3-relu-140ep
```

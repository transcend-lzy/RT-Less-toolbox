# SurfEmb

## Install

Download surfemb:

```shell
$ git clone https://github.com/rasmushaugaard/surfemb.git
$ cd surfemb
```

All following commands are expected to be run in the project root directory.

[Install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
, create a new environment, *surfemb*, and activate it:

```shell
$ conda env create -f environment.yml
$ conda activate surfemb
```

## Download dataset

Download and extract datasets from the [RTL site](http://www.zju-rtl.cn/RTL/).

The original data format of RTL does not support training and testing of Surfemb, you need to use generate_bop.py to convert the original data format to the BOP standard format.

Extract the datasets under ```data/bop``` (or make a symbolic link).

## Model

Train a model:

```shell
$ python -m surfemb.scripts.train [dataset] --gpus [gpu ids]
```

For example, to train a model on *T-LESS* on *cuda:0*

```shell
$ python -m surfemb.scripts.train tless --gpus 0
```

## Inference data

### Detection results

We use the random_bbox as detections. For ease of use, this data can be downloaded and extracted at [RTL](http://www.zju-rtl.cn/RTL/)

## Inference 

Inference is run on the (real) test images with random_bbox detections:

```shell
$ python -m surfemb.scripts.infer [model_path] --device [device]
```

## Reference

[1]R. L. Haugaard and A. G. Buch, “SurfEmb: Dense and Continuous Correspondence Distributions for Object Pose Estimation with Learnt Surface Embeddings,” *ArXiv211113489 Cs*, Apr. 2022, Accessed: Apr. 06, 2022. [Online]. Available: http://arxiv.org/abs/2111.13489

[2][github of suremb](https://github.com/rasmushaugaard/surfemb)
# coding: utf-8
import os
import gc
import time
import random
import numpy as np
from basic.datacenter import DataCenter
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from model.sama import SAMA
from basic.utils import around


seed = 50   # set seed for reproduction
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
print(" " * 10 + "SAMA model" + "\n<" + "=" * 30 + ">")


def parse_args():
    # dir
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    parser.add_argument("--root_dir", type=str, default="/home/ramon/datasets/NE/hashtag/preprocessed")
    parser.add_argument("--dataset", type=str, default="2011")
    parser.add_argument("--results_dir", help="the results dir (embeddings)", type=str, default="results")
    parser.add_argument("--models_dir", help="the saved models dir", type=str, default="models")
    parser.add_argument("--model_name", type=str, default="hashtag2vec")
    # data
    parser.add_argument("--edgelist", type=str, default="edgelist.pkl")    # 4种
    parser.add_argument("--empirical", type=str, default="empirical.pkl")   # 也是提前构造好的四种
    parser.add_argument("--test_ratio", type=float, default=0.2)     # for downstream task
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch_num", type=int, default=20)
    # network parameters
    parser.add_argument("--dim", help="the dimention of the embedding", type=int, default=50)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=5)    # deepwalk的参数
    parser.add_argument("--walks_node", type=int, default=30)
    parser.add_argument("--walk_len", type=int, default=40)
    config = parser.parse_args()
    return config


def main(args):
    pass


if __name__ == "__main__":
    main(parse_args())


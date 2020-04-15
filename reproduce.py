# coding: utf-8
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
    parser.add_argument("--root_dir", type=str, default="/home/rleating/datasets/SAMA")
    parser.add_argument("--models_dir", help="the pretrained models dir", type=str, default="pretrained_models")
    parser.add_argument("--model_name", type=str, default="pretrained_SAMA")
    # optimizer
    parser.add_argument("--batch_size", type=int, default=5)
    # network parameters
    parser.add_argument("--hiddendim", help="the dimention of the embedding", type=int, default=400)
    parser.add_argument("--worddim", help="the dimention of the word embedding", type=int, default=100)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--max_len_skill", type=int, default=30)
    parser.add_argument("--max_len_req", type=int, default=150)
    parser.add_argument("--lambda", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=1.4)
    config = parser.parse_args()
    return config


def main(args):
    pretrained_models_path = os.path.join(args.root_dir, args.models_dir, args.model_name)
    dc = DataCenter(args)
    dc.load_dataset()
    model = SAMA(dc, args)
    model.load_state_dict(torch.load(pretrained_models_path))



if __name__ == "__main__":
    main(parse_args())

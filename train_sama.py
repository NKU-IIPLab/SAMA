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
from basic.utils import around, dataPreprocess, vocab, globalVocab
from tqdm import tqdm


seed = 50   # set seed for reproduction
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


def parse_args():
    # dir
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    parser.add_argument("--root_dir", type=str, default="/home/rleating/datasets/SAMA")
    parser.add_argument("--models_dir", help="the pretrained models dir", type=str, default="pretrained_models")
    parser.add_argument("--model_name", type=str, default="pretrained_SAMA")
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch_num", type=int, default=15)
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
    dc = DataCenter(args)
    dc.load_dataset()
    model = SAMA(dc, args)   # the model part
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    param_count = 0   # counting the parameters
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print('total number of parameters of complete model: %d\n' % param_count)
    start = time.time()
    # training
    for e in range(args.epoch_num):
        train_batches = dc.get_batches(getattr(dc, "trainset"))  # get all the batches for training
        avg_loss = []
        model.train()
        with tqdm(total=len(train_batches)) as t:
            for train_batch in train_batches:
                model.zero_grad()
                t.set_description("EPOCH [{}]".format(e))
                loss = model.loss(train_batch)
                loss.backward()
                t.set_postfix(loss=loss.item())
                optim.step()
                avg_loss.append(loss.item())
                t.update()
        print("FINISHED training EPOCH [{} / {}], cost [{}] mins."
              .format(e, np.mean(avg_loss), around((time.time() - start) / 60)))

        # evaluating
        avg_loss.clear()
        gc.collect()


if __name__ == "__main__":
    print(" " * 10 + "SAMA model" + "\n<" + "=" * 30 + ">")
    main(parse_args())

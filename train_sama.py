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
# from torch.utils.tensorboard import SummaryWriter
# from apex import amp

# writer = SummaryWriter("./results/SAMA_training")
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
    parser.add_argument("--models_dir", help="the pretrained models dir", type=str, default="./pretrained_models")
    parser.add_argument("--model_name", type=str, default="pretrained_SAMA")
    parser.add_argument("--results_name", type=str, default="generation.txt")
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch_num", type=int, default=15)
    parser.add_argument("--train_bs", type=int, default=5)
    parser.add_argument("--test_bs", type=int, default=100)
    # network parameters
    parser.add_argument("--hiddendim", help="the dimention of the embedding", type=int, default=400)
    parser.add_argument("--worddim", help="the dimention of the word embedding", type=int, default=100)
    parser.add_argument("--enc_layers", help="the layers of the encoder", type=int, default=1)
    parser.add_argument("--dropout", help="the dropout rate", type=float, default=0.3)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--max_len_skill", type=int, default=30)
    parser.add_argument("--skill_len", type=int, default=500)
    parser.add_argument("--max_len_req", type=int, default=150)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=1.4)
    config = parser.parse_args()
    return config


def main(args):
    dc = DataCenter(args)
    dc.load_dataset()
    model = SAMA(dc, args).cuda(args.device)   # the model part
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    # model, optim = amp.initialize(model, optim, opt_level="O1")
    param_count = 0   # counting the parameters
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print('total number of parameters of complete model: %d\n' % param_count)
    start = time.time()
    # training
    # running_loss = 0
    for e in range(args.epoch_num):
        train_batches = dc.get_batches(getattr(dc, "trainset"), args.train_bs)  # get all the batches for training
        avg_loss = []
        model.train()
        with tqdm(total=len(train_batches)) as t:
            for i, train_batch in enumerate(train_batches):
                model.zero_grad()
                t.set_description("EPOCH [{}]".format(e))
                loss = model.loss(train_batch)
                # with amp.scale_loss(loss, optim) as scaled_loss:
                #     scaled_loss.backward()
                loss.backward()
                t.set_postfix(loss=loss.item())
                optim.step()
                avg_loss.append(loss.item())
                # if i % 500 == 0:
                #     writer.add_scalar("train loss", running_loss / 500, e * len(train_batches) * args.batch_size + i)
                #     running_loss = 0
                t.update()
        print("FINISHED training EPOCH [{} / {}], cost [{}] mins."
              .format(e, np.mean(avg_loss), around((time.time() - start) / 60)))
        model_name = args.model_name + "_{}".format(e)
        torch.save(model.state_dict(), os.path.join(args.models_dir, model_name))

        # evaluating
        model.eval()
        model.evaluate(dc.get_batches(getattr(dc, "testset"), args.test_bs))
        avg_loss.clear()
        gc.collect()


if __name__ == "__main__":
    print(" " * 10 + "SAMA model" + "\n<" + "=" * 30 + ">")
    main(parse_args())

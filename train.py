#-*-coding:utf-8-*-
import torch
import numpy as np
import random
import time
import os
from util.processdataall import dataPreprocess, vocab, globalVocab
from util.makedata import build_pretrain_embedding, buildDataperBatch
from util.evaluate import evaluate
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gc
from model.sama import sama
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # remove GPU cache
    torch.backends.cudnn.deterministic = True

setup_seed(421)
time_above = 0

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    parser.add_argument("--file_dir", type=str, default="/home/rleating/datasets/SAMA")
    parser.add_argument("--train", type=str, default="train.pt")
    parser.add_argument("--test", type=str, default="test.pt")
    parser.add_argument("--vocab", type=str, default="vocab.pt")
    parser.add_argument("--model_dir", help="the model dir", type=str, default="./trained_model")
    parser.add_argument("--result_dir", help="the result dir", type=str, default="./results/training")
    parser.add_argument("--pretrain_tgt_embedding", type=str, default="pretrained_w2v")
    parser.add_argument("--pretrain_src_embedding", type=str, default="pretrained_w2v")
    parser.add_argument("--ifGPU", help="whether use gpu", type=bool, default=True)
    parser.add_argument("--encoder_bidirectional", type=bool, default=True)
    parser.add_argument("--encoder_layer", type=int, default=1)
    parser.add_argument("--max_decoder_len", type=int, default=150)
    parser.add_argument("--dropout", help="dropout rate", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--src_embedding_dim", type=int, default=100)
    parser.add_argument("--tgt_embedding_dim", type=int, default=100)
    parser.add_argument("--src_encoder", type=str, default="LSTM")
    parser.add_argument("--src_decoder", type=str, default="LSTM")
    parser.add_argument("--tgt_encoder", type=str, default="LSTM")
    parser.add_argument("--tgt_decoder", type=str, default="LSTM")
    parser.add_argument("--src_encoder_hidden_dim", type=int, default=400)
    parser.add_argument("--generatealpha", type=float, default=1.4)
    parser.add_argument("--topicGenLamda", type=float, default=0.05)
    parser.add_argument("--max_skill_len", type=int, default=30)
    parser.add_argument("--skill_len", type=int, default=500)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # read config
    data = parse_args()
    if not os.path.exists(data.model_dir):
        os.makedirs(data.model_dir)
    if not os.path.exists(data.result_dir):
        os.makedirs(data.result_dir)

    # load vocb
    vocab_dir = os.path.join(data.file_dir, data.vocab)
    data.gVocab = torch.load(vocab_dir)
    # load data
    train, test = os.path.join(data.file_dir, data.train), os.path.join(data.file_dir, data.test)
    data.train_dataset = torch.load(train)
    data.test_dataset = torch.load(test)

    # load pretrain embedding
    pretrain_src_path = os.path.join(data.file_dir, data.pretrain_src_embedding)
    data.pretrain_src_embedding, data.src_embedding_dim, ukn_src_count = build_pretrain_embedding(
        pretrain_src_path, data.gVocab, "src")
    src_embedding = torch.tensor(data.pretrain_src_embedding)
    print("src unknown words: " + str(ukn_src_count))
    pretrain_tgt_path = os.path.join(data.file_dir, data.pretrain_tgt_embedding)
    data.pretrain_tgt_embedding, data.tgt_embedding_dim, ukn_tgt_count = build_pretrain_embedding(
        pretrain_tgt_path, data.gVocab, "tgt")
    tgt_embedding = torch.tensor(data.pretrain_tgt_embedding)
    print("tgt unknown words: " + str(ukn_tgt_count))

    # build model
    model = sama(data)
    param_count = 0   # counting the parameters
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print('total number of parameters of complete model: %d\n' % param_count)

    # optimizer
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad is True, model.parameters()), lr=data.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=data.learning_rate)
    best_BLEU4 = -10
    best_epoch = -1
    for idx_ in range(int(data.epoch)):
        idx = idx_ + 1
        epoch_start = time.time()
        temp_start = epoch_start
        sample_loss = 0
        total_loss = 0
        model.train()
        model.zero_grad()
        random.shuffle(data.train_dataset)
        batch_size = int(data.batch)
        train_num = len(data.train_dataset)
        total_batch = train_num // batch_size + 1
        with tqdm(total=total_batch) as t:
            for batch_id in range(total_batch):
                t.set_description("EPOCH [{}/{}]".format(idx, data.epoch))
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                instance = data.train_dataset[start:end]
                if len(instance) == 0:
                    continue
                src_tensor, tgt_tensor, skilltgt_tensor, skillnet_tensor, src_lengths, tgt_lengths, \
                    skill_tgt_lengths, skill_net_lengths = buildDataperBatch(instance, data.device, data.ifGPU)
                loss = model.neg_log_likelihood_loss(
                    src_tensor, src_lengths, tgt_tensor, skilltgt_tensor, skillnet_tensor, skill_net_lengths)
                t.set_postfix(loss=loss.item())
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                model.zero_grad()
                t.update()

        epoch_end = time.time()
        epoch_cost = epoch_end - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s, total loss: %s" % (
            idx, epoch_cost, train_num / epoch_cost, total_loss))

        # evaluate
        if idx >= 0:
            BLUE_score_1, BLUE_score_2, BLUE_score_3, BLEU4 = evaluate(
                data, model, data.result_dir + "/output_" + str(idx))
            if BLEU4 > best_BLEU4:
                best_epoch = idx
                time_above = 0
                print("best BLEU4 score\n BLEU1 %f, BLEU2 %f BLEU3 %f BLEU4 %f, epoch %d"
                      % (BLUE_score_1, BLUE_score_2, BLUE_score_3, BLEU4, best_epoch),
                      file=open(data.result_dir + "/BLEU_" + str(idx), "w"))
                model_name = data.result_dir + "/modelFinal_{}.model".format(BLEU4)
                torch.save(model.state_dict(), model_name)
                best_BLEU4 = BLEU4
            else:
                time_above = time_above + 1
        gc.collect()

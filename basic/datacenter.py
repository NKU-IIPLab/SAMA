# coding: utf-8
import os
import numpy as np
from numpy.random import uniform
import torch
import random
from basic.utils import load_pretrain_emb, normalization


class DataCenter(object):
    def __init__(self, config):
        self.config = config
        self.root_path = config.root_dir

    def load_dataset(self):
        trainset_path = os.path.join(self.root_path, "train.pt")
        testset_path = os.path.join(self.root_path, "test.pt")
        vocab_path = os.path.join(self.root_path, "vocab.pt")
        trainset, testset, vocabset = torch.load(trainset_path), torch.load(testset_path), torch.load(vocab_path)
        src_embedding, src_unknown_count = self.build_pretrain_embedding(vocabset, "src")
        tgt_embedding, tgt_unknown_count = self.build_pretrain_embedding(vocabset, "tgt")
        print("Finished loading src and tgt pretrained embeddings \nUnknown word count(src/tgt) : [{}/{}]"
              .format(src_unknown_count, tgt_unknown_count))
        src_embedding, tgt_embedding = torch.Tensor(src_embedding), torch.Tensor(tgt_embedding)
        setattr(self, "src_embedding", src_embedding), setattr(self, "tgt_embedding", tgt_embedding)
        setattr(self, "trainset", trainset), setattr(self, "testset", testset), setattr(self, "vocabset", vocabset)

    def build_pretrain_embedding(self, vocab, thetype):
        """
        :param vocab: global vocabulary. including the vocabs of skill, src, and tgt
        :param thetype: either "src" or "tgt"
        :return: tgt_emb, ukn_count
        """
        embedding, ukn_count = load_pretrain_emb(os.path.join(self.root_path, "pretrained_w2v")), 0
        scale = np.sqrt(3 / self.config.worddim)
        if thetype == "src":
            thevocab = vocab.src_vocab
        else:
            thevocab = vocab.tgt_vocab

        vocab_size = len(thevocab.word2id)
        emb = np.zeros([vocab_size, self.config.worddim], dtype='float32')
        for word, wordid in thevocab.word2id.items():
            if word in embedding:
                emb[int(wordid), :] = normalization(embedding[word])
            elif word.lower() in embedding:
                emb[int(wordid), :] = normalization(embedding[word.lower()])
            elif word != "<PAD>":
                ukn_count += 1
                emb[int(wordid), :] = uniform(-scale, scale, size=(self.config.worddim,)).astype('float32')
        return emb, ukn_count

    def get_batches(self, dataset):
        """
        get all the batches for training and testing.
        :param dataset: trainset((trainset and devset)) or testset.
        :return: all the batches (CPU device)
        """
        random.shuffle(dataset)
        all_batches = []
        bs, datalen = self.config.batch_size, len(dataset)
        batch_num = datalen // bs if datalen % bs == 0 else datalen // bs + 1
        for batch_id in range(batch_num):
            start = batch_id * bs
            end = (batch_id + 1) * bs
            if end > datalen:  # the last batch
                end = datalen
            batch_data = dataset[start: end]
            all_batches.append(self.make_batch(batch_data))
        return all_batches

    @staticmethod
    def make_batch(batch_data):
        bs, srcs, skilltgt, tgts, skillnet = len(batch_data), [], [], [], []
        for sample in batch_data:
            srcs.append(sample.srcid), skilltgt.append(sample.skilltgtid)
            tgts.append(sample.tgtid), skillnet.append(sample.skillnetid)

        src_lens, tgt_lens, skill_tgt_lens = list(map(len, srcs)), list(map(len, tgts)), list(map(len, skilltgt))
        skill_net_lens = list(map(len, skillnet))
        max_src_len, max_tgt_len, max_skilltgt_len = max(src_lens), max(tgt_lens), max(skill_tgt_lens)

        for idx in range(bs):
            srcs[idx].extend([1 for _ in range(max_src_len - src_lens[idx])])
            tgts[idx].extend([1 for _ in range(max_tgt_len - tgt_lens[idx])])
            skilltgt[idx].extend([0 for _ in range(max_skilltgt_len - skill_tgt_lens[idx])])
            skillnet[idx].extend([0 for _ in range(500 - skill_net_lens[idx])])

        return srcs, tgts, skilltgt, skillnet, src_lens, tgt_lens, skill_tgt_lens, skill_net_lens

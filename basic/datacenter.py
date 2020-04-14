# coding: utf-8
import os
import numpy as np
from numpy.random import uniform
import torch
from basic.utils import load_pretrain_emb, normalization, globalVocab


class DataCenter(object):
    def __init__(self, config):
        self.config = config
        self.root_path = config.root_path
        self.pretrained_w2v_path = os.path.join(self.root_path, "pretrained_w2v")


    def build_pretrain_embedding(self, vocab, thetype):
        """
        :param vocab: global vocabulary. including the vocabs of skill, src, and tgt
        :param thetype: either "src" or "tgt"
        :return: tgt_emb, vec_dim, ukn_count
        """
        embedding, ukn_count = load_pretrain_emb(self.pretrained_w2v_path), 0
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

# coding: utf-8
import os
import numpy as np
from numpy.random import uniform
import torch
from basic.utils import load_pretrain_emb, normalization, globalVocab, dataPreprocess


class DataCenter(object):
    def __init__(self, config):
        self.config = config
        self.root_path = config.root_path

    def load_dataset(self):
        trainset_path = os.path.join(self.root_path, "train.pt")
        testset_path = os.path.join(self.root_path, "test.pt")
        vocab_path = os.path.join(self.root_path, "vocab.pt")
        trainset, testset, vocabset = torch.load(trainset_path), torch.load(testset_path), torch.load(vocab_path)
        src_embedding, src_unknown_count = self.build_pretrain_embedding(vocabset, "src")
        tgt_embedding, tgt_unknown_count = self.build_pretrain_embedding(vocabset, "tgt")
        print("Finished loading src and tgt pretrained embeddings \n Unknown word count(src/tgt) : [{}/{}]"
              .format(src_unknown_count, tgt_unknown_count))
        setattr(self, "src_embedding", src_embedding), setattr(self, "tgt_embedding", tgt_embedding)


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


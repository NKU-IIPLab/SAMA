# coding: utf-8
import numpy as np
import codecs


class dataPreprocess(object):
    def __init__(self):
        self.src, self.skilltgt, self.skillnet, self.tgt, self.srcid = [], [], [], [], []
        self.skillnetid, self.tgtid, self.skilltgtid = [], [], []

    def setValue(self, name, data):
        if name == "src": self.src = data
        elif name == "skilltgt": self.skilltgt = data
        elif name == "tgt": self.tgt = data
        elif name == "skillnet": self.skillnet = data
        elif name == 'srcid': self.srcid = data
        elif name == 'skilltgtid': self.skilltgtid = data
        elif name == 'tgtid': self.tgtid = data
        elif name == 'skillnetid': self.skillnetid = data
        else: print("error key for dataProcess!!")


class vocab(object):
    def __init__(self):
        self.id2word, self.word2id = {}, {}

    def getIndex(self, word):
        if word in self.word2id: return int(self.word2id[word])
        else: return int(self.word2id["<UNK>"])

    def getWord(self, id):
        if str(id) not in self.id2word:
            return "<UNK>"
        return self.id2word[str(id)]

    def addWord(self, word):
        if word not in self.word2id:
            self.word2id[word] = str(len(self.id2word))
            self.id2word[str(len(self.id2word))] = word


class globalVocab(object):
    def __init__(self):
        self.src_vocab = vocab()
        self.skilltgt_vocab = vocab()
        self.tgt_vocab = vocab()


def normalization(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(path):
    file_r = codecs.open(path, "rb", "utf-8")
    embedding = dict()
    lines = file_r.readlines()[1:]
    for line in lines:
        items = line.split(" ")
        item = items[0]
        try:
            vec = np.array(items[1:], dtype="float32")
        except Exception:
            item = " "
            vec = np.array(items[2:], dtype="float32")
        embedding[item] = vec
    return embedding


def around(number, deci=4):
    return np.around(number, decimals=deci)

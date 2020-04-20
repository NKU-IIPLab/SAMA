# -*-coding:utf-8-*-


class dataPreprocess:
    def __init__(self):
        self.src = []
        self.skilltgt = []
        self.skillnet = []
        self.tgt = []
        self.srcid = []
        self.skilltgtid = []
        self.skillnetid = []
        self.tgtid = []

    def setValue(self, name, data):
        if name == "src":
            self.src = data
        elif name == "skilltgt":
            self.skilltgt = data
        elif name == "tgt":
            self.tgt = data
        elif name == "skillnet":
            self.skillnet = data
        elif name == 'srcid':
            self.srcid = data
        elif name == 'skilltgtid':
            self.skilltgtid = data
        elif name == 'tgtid':
            self.tgtid = data
        elif name == 'skillnetid':
            self.skillnetid = data
        else:
            print("error key for dataProcess!!")

class vocab:
    def __init__(self):
        self.id2word = {}
        self.word2id = {}

    def getIndex(self, word):
        if word in self.word2id:
            return int(self.word2id[word])
        else:
            return int(self.word2id["<UNK>"])

    def getWord(self, id):
        if str(id) not in self.id2word:
            return "<UNK>"
        return self.id2word[str(id)]

    def addWord(self, word):
        if word not in self.word2id:
            self.word2id[word] = str(len(self.id2word))
            self.id2word[str(len(self.id2word))] = word

class globalVocab:
    def __init__(self):
        self.src_vocab = vocab()
        self.skilltgt_vocab = vocab()
        self.tgt_vocab = vocab()


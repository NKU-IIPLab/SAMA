# code : utf-8
import numpy as np
import codecs
import torch


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square


def load_pretrain_emb(path):
    file_r = codecs.open(path, "rb", "utf-8")
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(" "))
    embedding = dict()
    line = file_r.readline()
    while line:
        items = line.split(" ")
        item = items[0]

        try:
            vec = np.array(items[1:], dtype="float32")
        except Exception:
            item = " "
            vec = np.array(items[2:], dtype="float32")

        embedding[item] = vec
        line = file_r.readline()
    return embedding, vec_dim


def build_pretrain_embedding(path, gVocab, type):
    embedding, vec_dim = load_pretrain_emb(path)
    ukn_count = 0
    scale = np.sqrt(3.0/vec_dim)
    if type == "src":
        src_vocab_size = len(gVocab.src_vocab.word2id)
        src_emb = np.zeros([src_vocab_size,vec_dim], dtype='float32')
        for word, id in gVocab.src_vocab.word2id.items():
            if word in embedding:
                src_emb[int(id), :] = norm2one(embedding[word])
            elif word.lower() in embedding:
                src_emb[int(id), :] = norm2one(embedding[word.lower()])
            elif word != "<PAD>":
                ukn_count += 1
                random_vec = np.random.uniform(-scale, scale, size=(vec_dim,)).astype('float32')
                src_emb[int(id), :] = random_vec
        return src_emb, vec_dim, ukn_count

    elif type == "tgt":
        tgt_vocab_size = len(gVocab.tgt_vocab.word2id)
        tgt_emb = np.zeros([tgt_vocab_size, vec_dim])
        for word, id in gVocab.tgt_vocab.word2id.items():
            if word in embedding:
                tgt_emb[int(id), :] = norm2one(embedding[word])
            elif word.lower() in embedding:
                tgt_emb[int(id), :] = norm2one(embedding[word.lower()])
            elif word!="<PAD>":
                ukn_count += 1
                tgt_emb[int(id), :] = np.random.uniform(-scale, scale, size=(vec_dim,)).astype('float32')
        return tgt_emb, vec_dim, ukn_count
    else:
        print("Error type!")
        exit(0)

def buildDataperBatch(data, device, ifGPU=True):
    batch_size = len(data)
    # get SRC/TGT/SKILLtgt/skillnet from data
    srcs = []
    skilltgt = []
    tgts = []
    skillnet = []

    for sample in data:
        srcs.append(sample.srcid)
        skilltgt.append(sample.skilltgtid)
        tgts.append(sample.tgtid)
        skillnet.append(sample.skillnetid)

    #statistic of len(src)
    src_lengths = torch.LongTensor(list(map(len, srcs)))
    max_src_len = int(np.percentile(src_lengths, 100))

    #statistic of len(tgt)
    tgt_lengths = torch.LongTensor(list(map(len, tgts)))
    max_tgt_len = int(np.percentile(tgt_lengths, 100))

    #statistic of len(skilltgt)
    skill_tgt_lengths = torch.LongTensor(list(map(len, skilltgt)))
    max_skilltgt_len = skill_tgt_lengths.max()

    # statistic of len(skillnet)
    skill_net_lengths = torch.LongTensor(list(map(len, skillnet)))
    max_skillnet_len = skill_net_lengths.max()

    # define Tensors
    src_tensor = torch.ones((batch_size, max_src_len), dtype = torch.long)
    skilltgt_tensor = torch.zeros((batch_size, max_skilltgt_len), dtype=torch.long)
    tgt_tensor = torch.ones((batch_size, max_tgt_len), dtype = torch.long)
    skillnet_tensor = torch.zeros((batch_size, 500), dtype=torch.long)

    # make value per batch
    for idx, (src, tgt, skill, sknet, src_sl, tgt_sl, skill_sl, skill_nl) in enumerate(
       zip(srcs, tgts, skilltgt, skillnet, src_lengths, tgt_lengths, skill_tgt_lengths, skill_net_lengths)):
        src_tensor[idx, :src_sl] = torch.LongTensor(src)
        skilltgt_tensor[idx, :skill_sl] = torch.LongTensor(skill)
        tgt_tensor[idx, :tgt_sl] = torch.LongTensor(tgt)
        skillnet_tensor[idx, :skill_nl] = torch.LongTensor(sknet)

    # make cuda
    if ifGPU:
        src_tensor = src_tensor.cuda(device)
        skilltgt_tensor = skilltgt_tensor.cuda(device)
        tgt_tensor = tgt_tensor.cuda(device)
        skillnet_tensor = skillnet_tensor.cuda(device)

    return src_tensor, tgt_tensor, skilltgt_tensor, skillnet_tensor, src_lengths, \
           tgt_lengths, skill_tgt_lengths, skill_net_lengths
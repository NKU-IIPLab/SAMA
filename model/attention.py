# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as func


class AttentionOneThree(nn.Module):
    def __init__(self, config):
        super(AttentionOneThree, self).__init__()
        self.hiddendim = config.hiddendim
        self.w = nn.Parameter(torch.zeros(self.hiddendim, self.hiddendim))
        init.xavier_uniform_(self.w)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, dec_hidden, enc_hidden):
        bs = dec_hidden.shape[0]
        submm = func.linear(dec_hidden, self.w).view(bs, 1, self.hiddendim)   # the intermediate result
        attn = torch.bmm(submm, enc_hidden.transpose(2, 1))
        attn = self.softmax(attn)
        sumresult = torch.bmm(attn, enc_hidden).view(bs, self.hiddendim)
        return attn, sumresult


class AttentionTwo(nn.Module):
    def __init__(self, config):
        super(AttentionTwo, self).__init__()
        self.config = config
        self.w = nn.Parameter(torch.zeros(config.worddim, config.hiddendim))
        init.xavier_uniform_(self.w)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, dec_hidden, enc_hidden, skillnet_lens):
        bs = dec_hidden.size(0)
        submm = func.linear(dec_hidden, self.w, None).view(bs, 1, self.config.worddim)
        attn = torch.bmm(submm, enc_hidden.transpose(2, 1))
        mask = torch.ones((bs, 1, 500), dtype=torch.bool).to(self.config.device)
        for i in range(bs):
            mask[i, :, :skillnet_lens[i]] = torch.zeros((1, 1, skillnet_lens[i]), dtype=torch.bool)
        attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        sumresult = torch.bmm(attn, enc_hidden).view(bs, self.config.worddim)
        return attn, sumresult


class AttentionFour(nn.Module):
    def __init__(self, config):
        super(AttentionFour, self).__init__()
        self.config = config
        self.w = nn.Parameter(torch.zeros(config.worddim, config.hiddendim), requires_grad=True)
        init.xavier_uniform_(self.w)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, dec_hidden, enc_hidden):
        bs = dec_hidden.size(0)
        submm = func.linear(dec_hidden, self.w, None).view(bs, 1, self.config.worddim)
        attn = torch.bmm(submm, enc_hidden.transpose(2, 1))
        attn = self.softmax(attn)
        sumresult = torch.bmm(attn, enc_hidden).view(bs, self.config.worddim)
        return attn, sumresult

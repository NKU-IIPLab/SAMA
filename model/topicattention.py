import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


class TopicAttention(nn.Module):
    def __init__(self, data):
        super(TopicAttention, self).__init__()
        self.data = data
        self.encoderHiddenDim = data.src_embedding_dim
        self.decoderHiddendim = data.src_encoder_hidden_dim
        self.W = nn.Parameter(torch.Tensor(int(self.encoderHiddenDim), int(self.decoderHiddendim))).cuda(self.data.device)
        init.xavier_uniform_(self.W)
        self.softmax = nn.Softmax(dim=2)
        batchSize = data.batch
        self.attn_1 = torch.zeros((batchSize, 1, 500), dtype=torch.float).cuda(self.data.device)

    def calculateWithMatrix(self, decoderFeature, encoderFeature):
        batch_size = decoderFeature.size(0)
        seq_len = decoderFeature.size(1)
        topicWord_len = encoderFeature.size(2)
        subResult = F.linear(decoderFeature, self.W, None).view(-1,1, self.encoderHiddenDim)
        encoderFeature = encoderFeature.view(-1, topicWord_len, self.encoderHiddenDim)
        attention_alpha = torch.bmm(subResult, encoderFeature.transpose(2,1))
        attention_alpha = self.softmax(attention_alpha)
        sum_result = torch.bmm(attention_alpha, encoderFeature)
        attention_alpha = attention_alpha.view(batch_size, seq_len, topicWord_len)
        sum_result = sum_result.view(batch_size, seq_len, self.encoderHiddenDim)
        return attention_alpha, sum_result

    def forward(self, decoderFeature, encoderFeature, skill_net_lengths):
        batchSize = decoderFeature.size(0)
        subResult = F.linear(decoderFeature, self.W, None).view(batchSize, 1, self.encoderHiddenDim)
        attn = torch.bmm(subResult, encoderFeature.transpose(2,1))
        # mask = torch.ones((batchSize, 1, 500), dtype=torch.uint8).cuda(self.data.device)
        mask = torch.ones((batchSize, 1, 500), dtype=torch.bool).cuda(self.data.device)
        for i in range(batchSize):
            mask[i,:, :skill_net_lengths[i]] = torch.zeros((1, 1, skill_net_lengths[i]), dtype=torch.bool).cuda(self.data.device)
        # src_tensor[idx, :src_sl] = torch.LongTensor(src)
        attn = attn.masked_fill(mask, -np.inf)
        # attn = attn * mask
        attn = self.softmax(attn)
        # for i in range(batchSize):
        #     self.attn_1[i,:, :skill_net_lengths[i]] = self.softmax(self.attn_1[i,:, :skill_net_lengths[i]])
        # attn_2 = self.attn_1
        sumResult = torch.bmm(attn, encoderFeature).view(batchSize, self.encoderHiddenDim)
        return attn, sumResult

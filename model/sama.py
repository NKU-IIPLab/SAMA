import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.attention import AttentionLayer
from model.attention3 import AttentionLayer3
from model.topicattention import TopicAttention


class sama(nn.Module):
    def __init__(self, data):
        super(sama, self).__init__()
        self.gpu = data.ifGPU
        self.data = data
        #word embedding
        self.wordEmbeddingDim = data.src_embedding_dim
        self.embedding = nn.Embedding(len(data.gVocab.src_vocab.word2id), self.wordEmbeddingDim)
        if data.pretrain_src_embedding is not None:
            self.embedding.weight = nn.Parameter(torch.Tensor(data.pretrain_src_embedding))
        self.embedding.weight.requires_grad = True

        self.worddicdim = len(data.gVocab.tgt_vocab.word2id)
        self.skilldicdim = len(data.gVocab.skilltgt_vocab.word2id)

        self.sktgtembed = nn.Embedding(self.skilldicdim, self.wordEmbeddingDim)
        if data.pretrain_tgt_embedding is not None:
            self.sktgtembed.weight = nn.Parameter(torch.Tensor(data.pretrain_tgt_embedding))
        self.sktgtembed.weight.requires_grad = True

        self.tgtembed = nn.Embedding(self.worddicdim, self.wordEmbeddingDim)
        if data.pretrain_tgt_embedding is not None:
            self.tgtembed.weight = nn.Parameter(torch.Tensor(data.pretrain_tgt_embedding))
        self.tgtembed.weight.requires_grad = True

        #encoder
        self.encoderInputdim = self.wordEmbeddingDim
        self.encoderHiddendim = int(data.src_encoder_hidden_dim)
        self.encoderBidirectional = data.encoder_bidirectional
        self.encoderLayerNum = int(data.encoder_layer)
        self.encoderExtractor = data.src_encoder
        self.decoderHiddendim = int(data.src_encoder_hidden_dim)
        self.maxDecoderLength = int(data.max_decoder_len)

        if self.encoderBidirectional:
            self.encoderHiddendim = self.encoderHiddendim // 2
        if self.encoderExtractor == "LSTM":
            self.encoder = nn.LSTM(self.encoderInputdim, self.encoderHiddendim, num_layers=self.encoderLayerNum,
                                   batch_first=True, bidirectional=self.encoderBidirectional)
        else:
            print("Error encoder extractor type!!!")
            exit(0)

        self.genealpha = float(data.generatealpha)

        #skill decoder
        self.dropout = nn.Dropout(float(data.dropout))
        # self.attention = AttentionLayer(data)
        self.attention1 = AttentionLayer(data)
        self.attention2 = AttentionLayer(data)
        self.attention3 = AttentionLayer3(data)
        self.topicAttention = TopicAttention(data)


        self.skilldecoderCell = nn.LSTMCell(self.wordEmbeddingDim + self.decoderHiddendim, self.decoderHiddendim)
        self.skilldecoderLinearTanh = nn.Linear(2 * self.decoderHiddendim, self.decoderHiddendim)
        self.skilldecoderLinear = nn.Linear(self.decoderHiddendim, self.skilldicdim)
        self.skilldecoderSoftmax = nn.Softmax(dim=1)

        self.decoderCell = nn.LSTMCell(self.wordEmbeddingDim, self.decoderHiddendim)
        self.decoderLinearTanh = nn.Linear(2 * self.decoderHiddendim + 2 * self.wordEmbeddingDim, self.decoderHiddendim)
        self.decoderLinear = nn.Linear(self.decoderHiddendim, self.worddicdim)
        self.decoderSoftmax = nn.Softmax(dim=1)

        #skill aware
        self.topicGenLamda = float(data.topicGenLamda)
        self.TopicLinearTanh = nn.Linear(2 * self.decoderHiddendim + self.wordEmbeddingDim, self.decoderHiddendim)
        self.topiclinear = nn.Linear(self.decoderHiddendim, int(data.skill_len))
        self.genSoftmax = nn.Softmax(dim=2)

        if self.gpu:
            self.embedding = self.embedding.cuda(self.data.device)
            self.tgtembed = self.tgtembed.cuda(self.data.device)
            self.sktgtembed = self.sktgtembed.cuda(self.data.device)
            self.encoder = self.encoder.cuda(self.data.device)
            self.dropout = self.dropout.cuda(self.data.device)
            self.attention1 = self.attention1.cuda(self.data.device)
            self.attention2 = self.attention2.cuda(self.data.device)
            self.attention3 = self.attention3.cuda(self.data.device)
            self.topicAttention = self.topicAttention.cuda(self.data.device)
            self.skilldecoderCell = self.skilldecoderCell.cuda(self.data.device)
            self.skilldecoderLinearTanh = self.skilldecoderLinearTanh.cuda(self.data.device)
            self.skilldecoderLinear = self.skilldecoderLinear.cuda(self.data.device)
            self.skilldecoderSoftmax = self.skilldecoderSoftmax.cuda(self.data.device)
            self.decoderCell = self.decoderCell.cuda(self.data.device)
            self.decoderLinear = self.decoderLinear.cuda(self.data.device)
            self.decoderSoftmax = self.decoderSoftmax.cuda(self.data.device)
            self.decoderLinearTanh = self.decoderLinearTanh.cuda(self.data.device)
            self.TopicLinearTanh = self.TopicLinearTanh.cuda(self.data.device)
            self.topiclinear = self.topiclinear.cuda(self.data.device)
            self.genSoftmax = self.genSoftmax.cuda(self.data.device)

    def mainforTrain(self, src_tensor, src_lengths, tgt_tensor, skilltgt_tensor, skillnet_tensor, skill_net_lengths):
        
        batch_size = src_tensor.size(0)

        #max tgt len
        tgt_len = tgt_tensor.size(1)
        skilltgt_len = skilltgt_tensor.size(1)

        #encoder
        embed = self.embedding(src_tensor)
        packed_words = pack_padded_sequence(embed, src_lengths, batch_first=True, enforce_sorted=False)
        hidden = None
        lstm_out, hidden = self.encoder(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        encoderFeature = self.dropout(lstm_out)

        #concat hx and cx
        hx = hidden[0].transpose(1,0).contiguous().view(batch_size, -1)
        cx = hidden[1].transpose(1,0).contiguous().view(batch_size, -1)

        #skill decoder
        skilldecoderInputinit = torch.zeros((batch_size, 1, self.wordEmbeddingDim)).cuda(self.data.device)
        skilldecoderInput = self.sktgtembed(skilltgt_tensor)
        skilldecoderInput = torch.cat([skilldecoderInputinit, skilldecoderInput], 1)

        sdhx = hx
        sdcx = cx
        skillFeature = []
        skillHidden = []
        skseq = []

        for idx in range(skilltgt_len):
            skillHidden.append(sdhx.view(batch_size, 1, -1))
            attn, attnFeature = self.attention1(sdhx, encoderFeature)
            sdhx, sdcx = self.skilldecoderCell(torch.cat([skilldecoderInput[:, idx, :], attnFeature], 1), (sdhx, sdcx))

            skilloutFeature = torch.cat([sdhx, attnFeature], 1)
            skilloutFeature = self.skilldecoderLinearTanh(skilloutFeature)
            skilloutFeature = torch.tanh(skilloutFeature)
            skilloutFeature = self.skilldecoderLinear(skilloutFeature)
            skillFeature.append(skilloutFeature.view(batch_size, 1, -1))

            skdecoderOutput = self.skilldecoderSoftmax(skilloutFeature)
            skillgeneratePro = torch.log(skdecoderOutput)
            topSkillGenPro, topSkillGenPos = torch.topk(skillgeneratePro, 1)
            skilldecoderfoTest = self.sktgtembed(topSkillGenPos).view(batch_size, self.wordEmbeddingDim)
            skseq.append(skilldecoderfoTest.view(batch_size, 1, self.wordEmbeddingDim))

        skillHiddenFeature = torch.cat(skillHidden, 1)
        skillOutFeature = torch.cat(skillFeature, 1)
        skillseq = torch.cat(skseq, 1)

        hx = skillHiddenFeature[:, -1, :]
        #decoder
        decoderInputinit = torch.zeros((batch_size, 1, self.wordEmbeddingDim)).cuda(self.data.device)
        decoderInput = self.tgtembed(tgt_tensor)
        decoderInput = torch.cat([decoderInputinit, decoderInput], 1)

        decoderOutput = []
        topicfeature = []
        hiddenoutput = []


        for idx in range(tgt_len):
            # skill-aware generator
            skillFeatureWordEmbedding = self.tgtembed(skillnet_tensor)
            skillword_len = skillFeatureWordEmbedding.size(1)
            forcusSkillWords, forcusResult = self.topicAttention(hx, skillFeatureWordEmbedding, skill_net_lengths)

            attn, attnFeature = self.attention2(hx, encoderFeature)
            # bateattn, bateatFeature = self.attention2(hx, skillHiddenFeature)
            gamaattn, gamaFeature = self.attention3(hx, skillseq)

            topicFeature = torch.cat([hx, attnFeature, forcusResult], 1)
            topicFeature = self.TopicLinearTanh(topicFeature)
            topicFeature = torch.tanh(topicFeature)
            topicFeature = self.topiclinear(topicFeature)
            topicfeature.append(topicFeature.view(batch_size, 1, -1))

            hx, cx = self.decoderCell(decoderInput[:,idx,:],(hx, cx))
            # hx, cx = self.decoderCell(torch.cat([decoderInput[:, idx, :], attnFeature, gamaFeature], 1), (hx, cx))
            hiddenoutput.append(hx.view(batch_size, 1, self.decoderHiddendim))

            outputFeature = torch.cat([decoderInput[:,idx,:], hx, attnFeature, gamaFeature], 1)
            outputFeature = self.decoderLinearTanh(outputFeature)
            outputFeature = torch.tanh(outputFeature)
            outputFeature = self.decoderLinear(outputFeature)
            decoderOutput.append(outputFeature.view(batch_size, 1, -1))

        topicfeature = torch.cat(topicfeature, 1)
        outPutFeature = torch.cat(decoderOutput, 1)

        # union dim
        skillnet_tensor = skillnet_tensor.view(batch_size, 1, -1).repeat(1, tgt_len, 1)
        topicGenPro = torch.zeros(batch_size, tgt_len, self.worddicdim)
        batchIndex = []
        decoderLenINdex = []
        for i in range(batch_size):
            batchIndex.append(i)
        for i in range(tgt_len):
            decoderLenINdex.append(i)
        batchIndexTensor = torch.LongTensor(batchIndex)
        decoderIndexTensor = torch.LongTensor(decoderLenINdex)
        if self.gpu:
            topicGenPro = topicGenPro.cuda(self.data.device)
            batchIndexTensor = batchIndexTensor.cuda(self.data.device)
            decoderIndexTensor = decoderIndexTensor.cuda(self.data.device)

        decoderIndexTensor = decoderIndexTensor.view(1, tgt_len, 1).repeat(batch_size, 1, skillword_len)
        batchIndexTensor = batchIndexTensor.view(batch_size, 1, 1).repeat(1, tgt_len, skillword_len)
        batchIndexTensor = batchIndexTensor.view(-1)
        decoderIndexTensor = decoderIndexTensor.view(-1)
        skilldimIndexTensor = skillnet_tensor.view(-1)
        wordProbValue = topicfeature.view(-1)
        topicGenPro = topicGenPro.index_put_((batchIndexTensor, decoderIndexTensor, skilldimIndexTensor), wordProbValue)
        generatePro = outPutFeature + self.topicGenLamda * topicGenPro
        return generatePro, skillOutFeature

    def mainforTest(self, src_tensor, src_lengths, skillnet_tensor, skill_net_lengths):

        batch_size = src_lengths.size(0)
        decoder_len = self.maxDecoderLength
        skill_len = 20
        skillnet_len = skillnet_tensor.size(1)

        embed = self.embedding(src_tensor)
        pack_words = pack_padded_sequence(embed, src_lengths, batch_first=True, enforce_sorted=False)

        hidden = None
        lstm_out, hidden = self.encoder(pack_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        #encoderFeature = self.dropout(lstm_out)
        encoderFeature = lstm_out

        # concat hx and cx
        hx = hidden[0].transpose(1, 0).contiguous().view(batch_size, -1)
        cx = hidden[1].transpose(1, 0).contiguous().view(batch_size, -1)

        #decoder
        decoderInputInit = torch.zeros((batch_size, self.wordEmbeddingDim)).cuda(self.data.device)
        skilldecoderInputInit = torch.zeros((batch_size, self.wordEmbeddingDim)).cuda(self.data.device)

        output = []
        skhx = hx
        skcx = cx

        skillFeature = []
        skillHidden = []
        skseq = []
        skill = []

        decoderInputforTest = decoderInputInit
        skilldecoderfoTest = skilldecoderInputInit

        for idx in range(skill_len):
            skillHidden.append(skhx.view(batch_size, 1, -1))
            attn, attnFeature = self.attention1(skhx, encoderFeature)
            skhx, skcx = self.skilldecoderCell(torch.cat([skilldecoderfoTest[:,:], attnFeature], 1), (skhx, skcx))

            skilloutFeature = torch.cat([skhx, attnFeature], 1)
            skilloutFeature = self.skilldecoderLinearTanh(skilloutFeature)
            skilloutFeature = torch.tanh(skilloutFeature)
            # skilloutFeature = self.active(skilloutFeature)
            skilloutFeature = self.skilldecoderLinear(skilloutFeature)
            skillFeature.append(skilloutFeature.view(batch_size, 1, -1))

            skdecoderOutput = self.skilldecoderSoftmax(skilloutFeature)
            skillgeneratePro = torch.log(skdecoderOutput)
            topSkillGenPro, topSkillGenPos = torch.topk(skillgeneratePro, 1)
            skill.append(topSkillGenPos)
            skilldecoderfoTest = self.sktgtembed(topSkillGenPos).view(batch_size, self.wordEmbeddingDim)
            skseq.append(skilldecoderfoTest.view(batch_size, 1, self.wordEmbeddingDim))

        skillseqemb = torch.cat(skseq, 1)
        skillHiddenFeature = torch.cat(skillHidden, 1)
        skill = torch.cat(skill, 1)

        hx = skillHiddenFeature[:, -1, :]

        for idx in range(decoder_len):
            attn, attnFeature = self.attention2(hx, encoderFeature)
            # bateattn, bateatFeature = self.attention2(hx, skillHiddenFeature)
            gamaattn, gamaFeature = self.attention3(hx, skillseqemb)

            # skill word
            skillFeatureWordEmbedding = self.tgtembed(skillnet_tensor)
            forcusTopicWords, forcusResult = self.topicAttention(hx, skillFeatureWordEmbedding, skill_net_lengths)
            topicFeature = torch.cat([hx, attnFeature, forcusResult], 1)
            topicFeature = self.TopicLinearTanh(topicFeature)
            topicFeature = torch.tanh(topicFeature)
            topicFeature = self.topiclinear(topicFeature)

            hx, cx = self.decoderCell(decoderInputforTest[:, :], (hx, cx))
            # hx, cx = self.decoderCell(torch.cat([decoderInputforTest[:, :], attnFeature, gamaFeature], 1), (hx, cx))

            decoderOutput = torch.cat([decoderInputforTest[:, :], hx, attnFeature, gamaFeature], 1)
            decoderOutput = self.decoderLinearTanh(decoderOutput)
            decoderOutput = torch.tanh(decoderOutput)
            decoderOutput = self.decoderLinear(decoderOutput)

            topicGenPro = torch.zeros(batch_size, self.worddicdim)
            batchIndex = []
            for i in range(batch_size):
                batchIndex.append(i)
            batchIndexTensor = torch.LongTensor(batchIndex)
            if self.gpu:
                topicGenPro = topicGenPro.cuda(self.data.device)
                batchIndexTensor = batchIndexTensor.cuda(self.data.device)

            batchIndexTensor = batchIndexTensor.view(batch_size, 1).repeat(1, skillnet_len)
            batchIndexTensor = batchIndexTensor.view(-1)
            wordDimIndexTensor = skillnet_tensor.view(-1)
            wordProbValue = topicFeature.view(-1)
            topicGenPro = topicGenPro.index_put_((batchIndexTensor, wordDimIndexTensor), wordProbValue)

            generatePro = decoderOutput +  self.topicGenLamda * topicGenPro
            generatePro = self.decoderSoftmax(generatePro)
            generatePro = torch.log(generatePro)
            topGenProb, topGenpos = torch.topk(generatePro, 1)
            output.append(topGenpos)
            decoderInputforTest = self.tgtembed(topGenpos).view(batch_size, self.wordEmbeddingDim)

        output = torch.cat(output, dim=1)
        return output, skill

    def neg_log_likelihood_loss(self, src_tensor, src_lengths, tgt_tensor, skilltgt_tensor, skillnet_tensor, skill_net_lengths):

        generatePro, skillPro = self.mainforTrain(src_tensor, src_lengths, tgt_tensor, skilltgt_tensor, skillnet_tensor, skill_net_lengths)
        lossFunc = nn.CrossEntropyLoss()

        #skill loss
        skillFeature = skillPro.view(-1, self.skilldicdim)
        standardSkill = skilltgt_tensor.view(-1)
        skillLoss = lossFunc(skillFeature, standardSkill)

        #sequence loss
        generateFeature = generatePro.view(-1, self.worddicdim)
        standardOutput = tgt_tensor.view(-1)
        generateLoss = lossFunc(generateFeature, standardOutput)

        tatolloss =  self.genealpha * generateLoss +  skillLoss
        # tatolloss = generateLoss

        return tatolloss

    def forward(self, src_tensor, src_lengths, skillnet_tensor, skill_net_lengths):
        output, skill = self.mainforTest(src_tensor, src_lengths, skillnet_tensor, skill_net_lengths)

        return output, skill

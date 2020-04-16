# coding: utf-8
import torch
import torch.nn as nn
from model.attention import AttentionOneThree, AttentionTwo, AttentionFour


class SAMA(nn.Module):
    """
    the sama model, including four parts: job description encoder, skill prediciton, skill
    refinement, job requirement generation.
    """
    def __init__(self, dc, config):
        super(SAMA, self).__init__()
        self.config = config
        # the embedding (including the pretrained one)
        self.src_embedding = nn.Embedding.from_pretrained(dc.src_embedding, freeze=False)
        self.tgt_embedding = nn.Embedding.from_pretrained(dc.tgt_embedding, freeze=False)
        self.skilltgt_embedding = nn.Embedding(len(dc.vocabset.skilltgt_vocab.word2id), config.worddim)
        nn.init.uniform_(self.skilltgt_embedding)
        enc_hiddendim = config.hiddendim // 2   # bidirectional
        # the multi-attention part
        self.atten_1 = AttentionOneThree(config)
        self.atten_2 = AttentionTwo(config)
        self.atten_3 = AttentionOneThree(config)
        self.atten_4 = AttentionFour(config)
        # the job description encoder
        self.dropout = nn.Dropout(config.dropout).to(config.device)
        self.enc = nn.LSTM(config.worddim, enc_hiddendim, num_layers=config.enc_layers,
                           batch_first=True, bidirectional=True)
        # the decoder, and fully connected for skill pred
        self.skill_dec = nn.LSTMCell(config.worddim + config.hiddendim, config.hiddendim)
        self.skill_fc_1 = nn.Linear(2 * config.hiddendim, config.hiddendim)
        self.skill_fc_2 = nn.Linear(config.hiddendim, len(dc.vocabset.skilltgt_vocab.word2id))
        self.skill_softmax = nn.Softmax(dim=1)
        # for sequence generation
        self.dec = nn.LSTMCell(config.worddim, config.hiddendim)
        self.fc_1 = nn.Linear(2 * (config.hiddendim + config.worddim), config.hiddendim)
        self.fc_2 = nn.Linear(config.hiddendim, config.worddim)
        self.gene_softmax = nn.Softmax(dim=2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, train_batch):
        # for training process
        srcs, tgts, skilltgt, skillnet, src_lens, tgt_lens, skilltgt_lens, skillnet_lens = \
            [torch.LongTensor(i).to(self.config.device) for i in train_batch]
        bs = srcs.shape[0]
        max_tgt_len, max_skilltgt_len = tgts.shape[1], skilltgt.shape[1]
        # encode the job description
        embeded_srcs = self.src_embedding(srcs)
        packed_words = nn.utils.rnn.pack_padded_sequence(embeded_srcs, src_lens, True, False)
        lstm_out, hidden = self.enc(packed_words, None)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        enc_hidden = self.dropout(lstm_out)
        hidden_state = hidden[0].transpose(1, 0).contiguous().view(bs, -1)
        cell_state = hidden[1].transpose(1, 0).contiguous().view(bs, -1)
        # skill prediction part (using lstm)
        skilldec_input = torch.cat(
            [torch.zeros((bs, 1, self.config.worddim)).to(self.config.device), self.skilltgt_embedding(skilltgt)], 1)
        skillfeat, skillhidden, skillseq = [], [], []
        for idx in range(max_skilltgt_len):    # decode the skill one by one when training
            skillhidden.append(hidden_state.view(bs, 1, -1))
            att_alpha, c_first = self.atten_1(hidden_state, enc_hidden)
            hidden_state, cell_state = self.skill_dec(   # here is the hidden and cell state of skill
                torch.cat([skilldec_input[:, idx, :], c_first], 1), (hidden_state, cell_state))
            skill_outfeat = self.skill_fc_2(torch.tanh(self.skill_fc_1(torch.cat([hidden_state, c_first], 1))))
            skillfeat.append(skill_outfeat)   # the equation 6 in paper
            maxprob_index = torch.argmax(self.skill_softmax(skill_outfeat))   # argmax
            skill_dec = self.skilltgt_embedding(maxprob_index).view(bs, 1, self.config.worddim)
            skillseq.append(skill_dec)
        skillfeat, skillhidden, skillseq = torch.cat(skillfeat, 1), torch.cat(skillhidden, 1), torch.cat(skillseq, 1)
        hidden_state = skillfeat[:, -1, :]   # the last skill dec hidden state is to init the text decoder
        # text decoder, including the skill graph part
        dec_input = torch.cat(
            [torch.zeros((bs, 1, self.config.worddim)).to(self.config.device), self.tgt_embedding(tgts)], 1)
        






        return seq, skill

    def loss(self, train_batch):
        # calculate the negative log likelihood loss that has two part: the generation loss and the pred loss
        generate_seq, pred_skill = self.forward(train_batch)

        nll_loss = self.config.mu * gene_loss + pred_loss
        return nll_loss

    def inference(self, src, src_lens, skillnet, skillnet_lens):
        # for inference process
        pass

    def evaluate(self):
        pass

# coding: utf-8
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import codecs
import os


def makeresult(config, dc, geneseq):
    file = os.path.join("./results/generation", config.results_name)
    golden_seq = [instence.tgt for instence in dc.testset]
    lens = geneseq.shape[0]
    tgt_vocab = dc.vocabset.tgt_vocab
    geneseq = list(geneseq)
    gene_words, gold_words = [], []
    for idx in range(lens):
        pred = []
        gold = golden_seq[idx][: -1]
        for id_pred in range(config.max_len_req):   # the pred seq
            word_idx = geneseq[idx][id_pred]
            if word_idx in [1, 2]:
                break
            pred.append(tgt_vocab.getWord(word_idx))
        gene_words.append(pred), gold_words.append([gold])

    # write to file
    outputfile = codecs.open(file, "w", "utf-8")
    for idx in range(lens):
        for idy in range(len(gold_words[idx][0])):
            outputfile.write("".join(gold_words[idx][0][idy]))
        outputfile.write("\n")
        for idy in range(len(gene_words[idx])):
            outputfile.write("".join(gene_words[idx][idy]))
        outputfile.write("\n")
    outputfile.close()

    return gene_words, gold_words


def getbleu(gold_words, gene_words):
    bleu1 = corpus_bleu(gold_words, gene_words, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(gold_words, gene_words, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(gold_words, gene_words, weights=(1/3, 1/3, 1/3, 0))
    bleu4 = corpus_bleu(gold_words, gene_words, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1, bleu2, bleu3, bleu4

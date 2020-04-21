# code : utf-8
import time
import codecs
from nltk.translate.bleu_score import corpus_bleu
from util.makedata import buildDataperBatch


def makeResult(data, tgtOutput, instance):
    decoder_seq_len = tgtOutput.size(1)
    batch_size = tgtOutput.size(0)
    tgtVocab = data.gVocab.tgt_vocab

    tgtOutput = list(tgtOutput)
    pred_words = []
    gold_words = []
    for idx in range(batch_size):
        pred = []
        gold = instance[idx].tgt[:-1]
        for idy in range(decoder_seq_len):
            if tgtOutput[idx][idy] in [1,2]:
                    break
            word = tgtVocab.getWord(int(tgtOutput[idx][idy]))
            pred.append(word)
        pred_words.append(pred)
        gold_words.append([gold])
    return pred_words, gold_words

def evaluate(data, model, path):
    instances = data.test_dataset
    model.eval()
    print("evaluating ... ")
    batchSize = int(data.batch)
    start_time = time.time()
    instances_num = len(instances)
    total_batch = instances_num // batchSize + 1
    pred_result = []
    gold_result = []
    # make batch
    for batch_idx in range(total_batch):
        start = batch_idx * batchSize
        end = (batch_idx + 1) * batchSize
        if end > instances_num:
            end = instances_num
        intance = instances[start:end]
        if len(intance) == 0:
            continue
        # build data tensor
        src_tensor, tgt_tensor, skilltgt_tensor, skillnet_tensor, src_lengths, tgt_lengths, \
        skill_tgt_lengths, skill_net_lengths = buildDataperBatch(intance, data.device, data.ifGPU)
        # model calculate result tensor
        tgtOutput, skill = model(src_tensor, src_lengths, skillnet_tensor, skill_net_lengths)
        # tensor to text
        pred_words, gold_words = makeResult(data, tgtOutput, intance)

        pred_result += pred_words
        gold_result += gold_words

    decode_time = time.time() - start_time
    speed = instances_num/decode_time
    # write to file
    outputFile = codecs.open(path, "w", "utf-8")
    for idx in range(instances_num):
        for idy in range(len(gold_result[idx][0])):
            outputFile.write("".join(gold_result[idx][0][idy]))
        outputFile.write("\n")
        for idy in range(len(pred_result[idx])):
            outputFile.write("".join(pred_result[idx][idy]))
        outputFile.write("\n")
        outputFile.write("\n")
    outputFile.close()

    # evaluate
    bleu_score_1 = corpus_bleu(gold_result, pred_result, weights=(1, 0, 0, 0))
    bleu_score_2 = corpus_bleu(gold_result, pred_result, weights=(0.5, 0.5, 0, 0))
    bleu_score_3 = corpus_bleu(gold_result, pred_result, weights=(0.33, 0.33, 0.33, 0))
    bleu_score_4 = corpus_bleu(gold_result, pred_result, weights=(0.25, 0.25, 0.25, 0.25))
    print("time: %.2fs, speed: %.2fst/s, bleu1: %f, bleu2: %f, bleu3: %f, bleu4: %f"
          % (decode_time, speed, bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4))
    return bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4

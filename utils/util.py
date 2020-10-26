#__author__ = 'chuntingzhou'
from __future__ import division
import dynet as dy
import numpy as np
from collections import defaultdict
import gzip
import cPickle as pkl
import codecs
import math
import random
from random import shuffle
import os
random.seed(448)
np.random.seed(1)
import operator
import re
from copy import deepcopy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_CHAR_LENGTH = 45

# Regular expressions used to normalize digits.
DIGIT_RE = re.compile(br"\d")


# word = utils.DIGIT_RE.sub(b"0", tokens[0]) if normalize_digits else tokens[0]


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def get_entity(label):
    entities = []
    i = 0
    while i < len(label):
        if label[i] != 'O':
            e_type = label[i][2:]
            j = i + 1
            while j < len(label) and label[j] == 'I-' + e_type:
                j += 1
            entities.append((i, j, e_type))
            i = j
        else:
            i += 1
    return entities


def evaluate_ner(pred, gold):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(pred)):
        pred_entities = get_entity(pred[i])
        gold_entities = get_entity(gold[i])
        temp = 0
        for entity in pred_entities:
            if entity in gold_entities:
                tp += 1
                temp += 1
            else:
                fp += 1
        fn += len(gold_entities) - temp
    precision = 1.0 * tp / (tp + fp)
    recall = 1.0 * tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


def get_pretrained_emb(fixedVocab, path_to_emb, word_to_id, char_to_id, dim):
    word_emb = []
    print("Loading pretrained embeddings from %s." % (path_to_emb))
    print("length of dict: %d" % len(word_to_id))

    pretrain_word_emb = {}
    pretrain_vocab = []
    for line in codecs.open(path_to_emb, "r", "utf-8", errors='replace'):
        items = line.strip().split()
        if len(items) == dim + 1:
            try:
                pretrain_word_emb[items[0]] = np.asarray(items[1:]).astype(np.float32)
                pretrain_vocab.append(items[0])
            except ValueError:
                continue

    for _ in range(len(word_to_id)):
        word_emb.append(np.random.uniform(-math.sqrt(3.0 / dim), math.sqrt(3.0 / dim), size=dim))

    not_covered = 0


    for word, id in word_to_id.iteritems():
        if word in pretrain_word_emb:
            word_emb[id] = pretrain_word_emb[word]
        elif word.lower() in pretrain_word_emb:
            word_emb[id] = pretrain_word_emb[word.lower()]
        else:
            not_covered += 1
    
    if fixedVocab:
        #Take top 100000 from the word embeddings
        num = 0
        for word in pretrain_vocab:
            if num > 400000:
                break
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
                word_emb.append(pretrain_word_emb[word])
                num +=1
	
    else:
        for word in pretrain_word_emb.keys():
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
                word_emb.append(pretrain_word_emb[word])

    emb = np.array(word_emb, dtype=np.float32)

    print("Word number not covered in pretrain embedding: %d" % not_covered)
    return emb, word_to_id, char_to_id

def loadPretrainedEmbedding(path_to_emb, dim, label2idx):


  if path_to_emb is None or not os.path.exists(path_to_emb):
      return None
  print("Loading pretrained embeddings from %s." % (path_to_emb))
  print(label2idx)
  lines = open(path_to_emb, "r").readlines()
  length = len(label2idx.keys())
  pretrained_emb = np.zeros((length, dim))
  for i in range(length):
      pretrained_emb[i] = np.random.uniform(-math.sqrt(3.0 / dim), math.sqrt(3.0 / dim), size=dim)

  for line in lines:
    items = line.strip().split()
    embedding = [float(emb.replace(",","").replace("[","").replace("]","")) for emb in items[1:]]
    assert len(embedding) == dim
    if items[0] in label2idx:
        new_index = label2idx[items[0]]
        pretrained_emb[new_index] = np.asarray(embedding).astype(np.float32)

  return pretrained_emb

def pkl_dump(obj, path):
    with open(path, "wb") as fout:
        pkl.dump(obj, fout)


def pkl_load(path):
    with open(path, "rb") as fin:
        obj = pkl.load(fin)
    return obj


def log_sum_exp_dim_0(x):
    # numerically stable log_sum_exp
    dims = x.dim()
    max_score = dy.max_dim(x, 0)  # (dim_1, batch_size)
    if len(dims[0]) == 1:
        max_score_extend = max_score
    else:
        max_score_reshape = dy.reshape(max_score, (1, dims[0][1]), batch_size=dims[1])
        max_score_extend = dy.concatenate([max_score_reshape] * dims[0][0])
    x = x - max_score_extend
    exp_x = dy.exp(x)
    # (dim_1, batch_size), if no dim_1, return ((1,), batch_size)
    log_sum_exp_x = dy.log(dy.mean_dim(exp_x, d=[0], b=False) * dims[0][0])
    return log_sum_exp_x + max_score


def data_iterator(data_pair, batch_size):
    batches = make_bucket_batches(data_pair, batch_size)
    for batch in batches:
        yield batch


def make_bucket_batches(data_collections, feature_wise_tgt_tags,feature_wise_known_tags, batch_size):
    # Data are bucketed according to the length of the first item in the data_collections.
    buckets = defaultdict(list)
    tot_items = len(data_collections[0])
    for data_item in data_collections:
        src = data_item[0]
        buckets[len(src)].append(data_item)

    batches = []
    # np.random.seed(2)
    for src_len in buckets:
        bucket = buckets[src_len]
        np.random.shuffle(bucket)

        num_batches = int(np.ceil(len(bucket) * 1.0 / batch_size))
        for i in range(num_batches):
            cur_batch_size = batch_size if i < num_batches - 1 else len(bucket) - batch_size * i
            batch = [[bucket[i * batch_size + j][k] for j in range(cur_batch_size)] for k in range(tot_items)]

            if feature_wise_known_tags is not None and feature_wise_tgt_tags is not None:
                batch_tgt_tags = defaultdict(list)
                batch_known_tgt_tags = defaultdict(list)
                for feat, all_tgt_tags in feature_wise_tgt_tags.items():
                    for sent_index in batch[-1]:
                        batch_tgt_tags[feat].append(all_tgt_tags[sent_index])

                for feat, known_tags in feature_wise_known_tags.items():
                    for sent_index in batch[-1]:
                        batch_known_tgt_tags[feat].append(known_tags[sent_index])

                batch.append(batch_tgt_tags)
                batch.append(batch_known_tgt_tags)

            batches.append(batch)


    np.random.shuffle(batches)
    return batches


def transpose_input(seq, padding_token=0):
    # input seq: list of samples [[w1, w2, ..], [w1, w2, ..]]
    max_len = max([len(sent) for sent in seq])
    seq_pad = []
    seq_mask = []
    for i in range(max_len):
        pad_temp = [sent[i] if i < len(sent) else padding_token for sent in seq]
        mask_temp = [1.0 if i < len(sent) else 0.0 for sent in seq]
        seq_pad.append(pad_temp)
        seq_mask.append(mask_temp)

    return seq_pad, seq_mask


def transpose_discrete_features(feature_batch):
    # Discrete features are zero-one features
    # TODO: Other integer features, create lookup tables
    # tgt_batch: [[[feature of word 1 of sent 1], [feature of word 2 of sent 2], ]]
    # return: [(feature_num, batchsize)]
    max_sent_len = max([len(s) for s in feature_batch])
    feature_num = len(feature_batch[0][0])
    batch_size = len(feature_batch)
    features = []  # each: (feature_num, batch_size)
    for i in range(max_sent_len):
        w_i_feature = [dy.inputTensor(sent[i], batched=True) if i < len(sent) else dy.zeros(feature_num) for sent in feature_batch]
        w_i_feature = dy.reshape(dy.concatenate(w_i_feature, d=1), (feature_num,), batch_size=batch_size)
        features.append(w_i_feature)

    return features


def transpose_and_batch_embs(input_embs, emb_size):
    # input_embs: [[w1_emb, w2_emb, ]], embs are dy.expressions
    max_len = max(len(sent) for sent in input_embs)
    batch_size = len(input_embs)
    padded_seq_emb = []
    seq_masks = []
    for i in range(max_len):
        w_i_emb = [sent[i] if i < len(sent) else dy.zeros(emb_size) for sent in input_embs]
        w_i_emb = dy.reshape(dy.concatenate(w_i_emb, d=1), (emb_size,), batch_size=batch_size)
        w_i_mask = [1.0 if i < len(sent) else 0.0 for sent in input_embs]
        padded_seq_emb.append(w_i_emb)
        seq_masks.append(w_i_mask)

    return padded_seq_emb, seq_masks

def get_lang_code_dicts(path):
  """
   Returns lang_to_code, code_to_lang dictionaries
  """
  lang_to_code = {}
  bad_chars = ",''"
  rgx = re.compile('[%s]' % bad_chars)

  with open(path) as f:
      data = f.read()
      lines = data.split("\n")
      split_line = [line.split() for line in lines]
      for line in split_line[:-2]:
          lang = rgx.sub('', line[0])
          code = rgx.sub('', line[2])
          lang_to_code[lang] = code
      code_to_lang = {v: k for k, v in lang_to_code.iteritems()}
  return lang_to_code, code_to_lang

def transpose_char_input(tgt_batch, padding_token):
    # The tgt_batch may not be padded with <sow> and <eow>
    # tgt_batch: [[[<sow>, <sos>, <eow>], [<sow>, s,h,e, <eow>],
    # [<sow>, i,s, <eow>], [<sow>, p,r,e,t,t,y, <eow>], [<sow>, <eos>, <eow>]], [[],[],[]]]
    max_sent_len = max([len(s) for s in tgt_batch])
    sent_w_batch = []  # each is list of list: max_word_len, batch_size
    sent_mask_batch = []  # each is list of list: max_word_len, batch_size
    max_w_lens = []
    SOW_PAD = 0
    EOW_PAD = 1
    EOS_PAD = 2
    for i in range(max_sent_len):
        max_len_w = max([len(sent[i]) for sent in tgt_batch if i < len(sent)])
        max_w_lens.append(max_len_w)
        w_batch = []
        mask_batch = []
        for j in range(0, max_len_w):
            temp_j_w = []
            for sent in tgt_batch:
                if i < len(sent) and j < len(sent[i]):
                    temp_j_w.append(sent[i][j])
                elif i >= len(sent):
                    if j == 0:
                        temp_j_w.append(SOW_PAD)
                    elif j == max_len_w - 1:
                        temp_j_w.append(EOW_PAD)
                    else:
                        temp_j_w.append(EOS_PAD)
                else:
                    temp_j_w.append(EOW_PAD)
            # w_batch = [sent[i][j] if i < len(sent) and j < len(sent[i]) else self.EOW for sent in tgt_batch]
            # print "temp: ", temp_j_w
            w_batch.append(temp_j_w)
            mask_batch.append([1. if i < len(sent) and j < len(sent[i]) else 0.0 for sent in tgt_batch])
        sent_w_batch.append(w_batch)
        sent_mask_batch.append(mask_batch)
    return sent_w_batch, sent_mask_batch, max_sent_len, max_w_lens

def get_vocab_from_set(a_set, shift=0):
    vocab = {}
    for i, elem in enumerate(a_set):
        vocab[elem] = i + shift

    return vocab


def unfreeze_dict(feats):
    dicti = {}
    for feat in feats.split("|"):
        info = feat.split("=")
        dicti[info[0]] = info[1]
    return dicti


SEPARATOR = "|"

def set_equal(str1, str2):
    set1 = set(str1.split(SEPARATOR))
    set2 = set(str2.split(SEPARATOR))
    return set1 == set2


def computeF1(hyps, golds):
    """
    hyps: List of dicts for predicted morphological tags
    golds: List of dicts for gold morphological tags
    """

    f1_precision_scores = {}
    f1_precision_total = {}
    f1_recall_scores = {}
    f1_recall_total = {}
    f1_average = 0.0
    morph_acc = 0.0

    orig_hyps = deepcopy(hyps)
    orig_golds = deepcopy(golds)
    hyps = [unfreeze_dict(h) for h in hyps]
    golds = [unfreeze_dict(t) for t in golds]

    # calculate precision
    for i, word_tags in enumerate(hyps, start=0):


        morph_acc += set_equal(orig_hyps[i], orig_golds[i])


        for k, v in word_tags.items():
            if v == "NULL":
                continue
            if k not in f1_precision_scores:
                f1_precision_scores[k] = 0
                f1_precision_total[k] = 0
            if k in golds[i]:
                if v == golds[i][k]:
                    f1_precision_scores[k] += 1
            f1_precision_total[k] += 1

    f1_micro_precision = sum(f1_precision_scores.values()) / sum(f1_precision_total.values())

    for k in f1_precision_scores.keys():
        f1_precision_scores[k] = f1_precision_scores[k] / f1_precision_total[k]

    # calculate recall
    for i, word_tags in enumerate(golds, start=0):
        for k, v in word_tags.items():
            if v == "NULL":
                continue
            if k not in f1_recall_scores:
                f1_recall_scores[k] = 0
                f1_recall_total[k] = 0
            if k in hyps[i]:
                if v == hyps[i][k]:
                    f1_recall_scores[k] += 1
            f1_recall_total[k] += 1

    f1_micro_recall = sum(f1_recall_scores.values()) / sum(f1_recall_total.values())

    f1_scores = {}
    for k in f1_recall_scores.keys():
        f1_recall_scores[k] = f1_recall_scores[k] / f1_recall_total[k]

        if f1_recall_scores[k] == 0 or k not in f1_precision_scores:
            f1_scores[k] = 0
        else:
            f1_scores[k] = 2 * (f1_precision_scores[k] * f1_recall_scores[k]) / (
                        f1_precision_scores[k] + f1_recall_scores[k])

        f1_average += f1_recall_total[k] * f1_scores[k]
        #print(k, f1_recall_scores[k] )

    f1_average /= sum(f1_recall_total.values())
    f1_micro_score = 2 * (f1_micro_precision * f1_micro_recall) / (f1_micro_precision + f1_micro_recall)

    # if write_results:
    #     print("Writing F1 scores...")
    #     with open(prefix + '_results_f1.txt', 'ab') as file:
    #         file.write(pickle.dumps(f1_scores))
    #         file.write("\nMacro-averaged F1 Score: " + str(f1_average))
    #         file.write("\nMicro-averaged F1 Score: " + str(f1_micro_score))

    return f1_average, f1_micro_score, morph_acc * 1.0 / len(orig_golds)

def plot_heatmap(weights, id_to_tag, tag):
    font = {'family': 'normal',
            'size': 14,
            'weight': 'bold'}

    matplotlib.rc('font', **font)

    # weights is a ParameterList
    tag_labels = [id_to_tag[id] for id in range(len(weights))]
    plt.figure(figsize=(20, 18), dpi=80)
    plt.xticks(range(0, len(tag_labels)), tag_labels, rotation=45)
    plt.yticks(range(0, len(tag_labels)), tag_labels)
    plt.tick_params(labelsize=40)
    plt.xlabel(tag, fontsize=50)
    plt.ylabel(tag, fontsize=50)
    plt.imshow(weights, cmap='Greys', interpolation='nearest')
    plt.savefig("./" + tag + "_" + tag + ".png", bbox_inches='tight')
    plt.close()

def printTransitionMatrix(model, dataloader, tag):

    transition_matrix = dy.parameter(model.crf_decoders[tag].transition_matrix).npvalue() #(from, to)
    print(dataloader.id2tags[tag])
    plot_heatmap(transition_matrix[:-2,:-2], dataloader.id2tags[tag], tag)

    #np.save("pos_transition_matrix",transition_matrix)
    exit(-1)

def outputWordEmbedding(model, sentence_index):
    word_weights = []
    print("Number: ",len(model.word_embedding_weights))
    for sent_num in sentence_index:
        char_word_emb  = model.word_embedding_weights[sent_num]
        for char_emb in char_word_emb:
            word_weights.append(char_emb)
    return word_weights

def outputLanguageEmbedding(model, dataloader):
    with codecs.open("./learned_typolgoical_vectors.vec", "w", encoding='utf-8') as fvec:
        fvec.write("Learned typology weights \n")
        for lang, weight in model.typology_encoder.W.items():
            feature_vector = dataloader.pre_computed_features[lang.replace("<","").replace(">","")]
            W = dy.parameter(weight).npvalue()
            b= dy.parameter(model.typology_encoder.b[lang]).npvalue()
            print(W.shape, b.shape, feature_vector.shape)
            vector = np.dot(W, feature_vector) + b
            print(vector.shape)
            fvec.write(lang + "\t"  + " ".join(map(str,vector)) + "\n")
            lang_id = [dataloader.word_to_id[lang]]
            lang_emb = model.word_lookup.encode([lang_id])[0].npvalue()
            fvec.write(lang + "\t" + " ".join(map(str,lang_emb)))
            fvec.write("\n")

def getTagPred(data_loader, best_path, key, one_prediction):
    for token_num, pred_tag_feat in enumerate(best_path):
        #if data_loader.id2tags[key][pred_tag_feat] == "_":
        #    continue
        tag_name = data_loader.id2tags[key][pred_tag_feat]
        one_prediction[token_num].append(tag_name)

if __name__ == "__main__":

    dim = 100
    # 9 1000
    path_to_emb = "/Users/zct/Downloads/tir1.emb"
    pretrain_word_emb = {}
    i = 1
    for line in codecs.open(path_to_emb, "r", 'utf-8', errors='replace'):
        items = line.strip().split()
        if len(items) == dim + 1:
            try:
                pretrain_word_emb[items[0]] = np.asarray(items[1:]).astype(np.float32)
            except ValueError:
                continue
            print items[0], i, pretrain_word_emb[items[0]][:3]
        i += 1


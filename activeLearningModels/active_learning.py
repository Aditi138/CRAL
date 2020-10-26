__author__ = 'aditichaudhary'
import codecs
import math
from collections import defaultdict
import numpy as np
np.random.seed(1)
import os
from sklearn.metrics import pairwise_distances_argmin_min
from copy import deepcopy
import pickle

def getTokenIndexInfo(data_loader, gold_dict, model, output_dict, sentence_index, sentence_index_sent,
                      sentence_index_tokens, token_index):
    sent_index = model.token_index_sent_index_map[token_index]
    sentence_index.append(sent_index)
    sent = model.sent_index_sent_map[sent_index]
    sent = [data_loader.id_to_word[int(token)] for token in sent]
    gold_path = gold_dict[" ".join(sent)]
    sentence_index_sent[sent_index] = (sent, gold_path, output_dict[" ".join(sent)])
    sentence_index_tokens[sent_index].append(model.token_index_relative_idx_map[token_index])
    return sent, model.token_index_relative_idx_map[token_index]

def createAnnotationOutput_mimic_oracle(args, model, data_loader, gold_dict, output_dict):
    sentence_index = []
    sentence_index_sent = {}
    sentence_index_tokens = defaultdict(list)

    typeTag, typeTagIndex, tokenIndex_map, sentIndex_map, _ = readTrain(args.test_path)
    data = {}
    with codecs.open(args.to_annotate, "w", encoding='utf-8') as fout, codecs.open(args.debug, "w", encoding='utf-8') as fdebug:
        sorted_type = sorted(model.approxTypeErrors.items(), key=lambda kv: kv[1], reverse=True )[:args.k]
        fdebug.write("TOKEN\tTYPE\tGOLD\tPRED\tPREDPROB\tERRORS\n")
        for (type, error_percent) in sorted_type:
            token_pred_error = model.predTypeErrors[type]
            token_tag_error = model.approxTokenClassErrors[type]
            sorted_token_tag_error = sorted(token_tag_error.items(),key= lambda kv: kv[1], reverse=True)
            errors = []
            maxTag = sorted_token_tag_error[0][0]
            for (tagId, error) in sorted_token_tag_error:
                tag = data_loader.id2tags["POS"][tagId]
                errors.append(tag + "=" + str(error))

            predErrors = []
            sorted_tag_error = sorted(token_pred_error.items(),key= lambda kv: kv[1], reverse=True)
            for (tagId, error) in sorted_tag_error:
                tag = data_loader.id2tags["POS"][tagId]
                predErrors.append(tag + "=" + str(error))


            token_indices = list(model.type_tokenIndices[type])
            required_embeddings, gold_token_tags, pred_token_tags = [], [],[]
            for token_index in token_indices:
                embedding = model.token_embeddings[token_index]
                gammaVal= model.token_gamma_key[token_index][maxTag]
                prob = np.exp(gammaVal)
                required_embeddings.append(embedding * prob)
                (token_, tag_, sent_index_, relative_index_) = tokenIndex_map[token_index]
                one_sent_ = sentIndex_map[sent_index_]
                pred_path_ = output_dict[" ".join(one_sent_)]
                gold_path_ = gold_dict[" ".join(one_sent_)]
                pred_token_tags.append(pred_path_[relative_index_])
                gold_token_tags.append(gold_path_[relative_index_])

            cluster_center = centeroidnp(np.array(required_embeddings))
            closest, _ = pairwise_distances_argmin_min(np.array([cluster_center]), required_embeddings)
            centroid = token_indices[closest[0]]

            (token, tag, sent_index, relative_index) = tokenIndex_map[centroid]

            one_sent = sentIndex_map[sent_index]

            sentence_index.append(sent_index)
            pred_path = output_dict[" ".join(one_sent)]
            gold_path = gold_dict[" ".join(one_sent)]
            sentence_index_sent[sent_index] = (one_sent, gold_path, pred_path)
            sentence_index_tokens[sent_index].append(relative_index)
            data[token] = {"tokenindices":token_indices, "weighted":required_embeddings, "centroid_center":cluster_center, "pred":pred_token_tags,  "gold":gold_token_tags}
            fdebug.write( str(centroid) + "\t" + data_loader.id_to_word[type] + "\t" + gold_path[relative_index] + "\t" + pred_path[relative_index] + "\t" + "@".join(predErrors) +"\t" + "@".join(errors) + "\n")

        covered = set()
        count = 0

        with open("./" + args.model_name + "approx.pkl", "wb") as f:
            pickle.dump(data, f)

        with codecs.open(args.to_annotate, "w", encoding='utf-8') as fout:

            for sent_index in sentence_index:
                if sent_index not in covered:
                    covered.add(sent_index)
                    (sent, gold_path, pred_path) = sentence_index_sent[sent_index]
                    path = deepcopy(pred_path)
                    for token_index in sentence_index_tokens[sent_index]:
                        path[token_index] = "UNK"

                    for token, tag_label, gold_tag in zip(sent, path, gold_path):
                        fout.write(token + "\t" + tag_label + "\t" + gold_tag + "\n")
                        if tag_label == "UNK":
                            count += 1

                    fout.write("\n")



def centeroidnp(arr):
    length, dim = arr.shape
    return np.array([np.sum(arr[:, i])/length for i in range(dim)])

def readTrain(input):
    typeTag = {}
    typeTagIndex = {}
    tokenIndex_map = {}
    sentIndex_map = {}
    one_sent = []
    output_dict = {}
    gold_path = []
    with codecs.open(input, "r", encoding='utf-8') as fin:
        token_index = 0
        sent_index = 0
        relative_index = 0
        for line in fin:
            if line.startswith("#"):
                continue
            if line == "" or line == "\n":
                sentIndex_map[sent_index] = one_sent
                sent_index += 1
                relative_index = 0
                output_dict[" ".join(one_sent)] = gold_path
                gold_path = []
                one_sent = []

            else:
                line = line.strip().split("\t")
                token, tag = line[1], line[5]
                one_sent.append(token)
                gold_path.append(tag)
                if token not in typeTag:
                    typeTag[token] = defaultdict(lambda:0)
                    typeTagIndex[token] = defaultdict(list)
                typeTag[token][tag] += 1
                typeTagIndex[token][tag].append(token_index)
                tokenIndex_map[token_index] = (token, tag, sent_index, relative_index)
                relative_index += 1
                token_index += 1

        return typeTag, typeTagIndex, tokenIndex_map, sentIndex_map, output_dict

def createAnnotationOutput_SPAN_wise(args, model, data_loader, gold_dict, output_dict, score_sentence):
    sentence_index = []
    reverse = True

    # Order the sentences by entropy of the spans
    fout = codecs.open(args.to_annotate, "w", encoding='utf-8')
    sorted_spans = sorted(model.most_uncertain_entropy_spans, key=lambda k: model.most_uncertain_entropy_spans[k],
                          reverse=reverse)
    print("Total unique spans: {0}".format(len(sorted_spans)))
    count_span = args.k
    count_tokens = args.k

    # DEBUG Print Span Entropy
    fdebug = codecs.open(args.debug, "w", encoding='utf-8')

    # Accumulate tokens by sentence
    sentence_index_sent = {}
    sentence_index_tokens = defaultdict(list)
    spans_index = 0
    for sorted_span in sorted_spans:
        # Debug
        spans_index += 1
        span_words = []
        span_in_text = data_loader.id_to_word[int(sorted_span)]

        if count_tokens <= 0:
            break

        (span_entropy, sentence_key, start, end, best_path, instance_entropy, sent_index, token_index) = \
        model.most_uncertain_entropy_spans[sorted_span]
        sent = sentence_key.split()
        sent = [data_loader.id_to_word[int(token)] for token in sent]
        gold_path = gold_dict[" ".join(sent)]

        if args.normdata and span_in_text.lower() in data_loader.word_to_id:
            sorted_span = data_loader.word_to_id[span_in_text.lower()]

        tokenEntropy = model.typeHeap[str(sorted_span)]
        tokenInfo = model.heapInfo[str(sorted_span)]

        sorted_tokenEntropy, sorted_tokenInfo = zip(*(sorted(zip(tokenEntropy, tokenInfo), reverse=True)))
        (t_sentence_key, t_start, t_sent_index) = sorted_tokenInfo[0]
        t_sent = [data_loader.id_to_word[int(token)] for token in t_sentence_key.split()]
        t_gold_path = gold_dict[" ".join(t_sent)]
        sentence_index.append(t_sent_index)
        sentence_index_sent[t_sent_index] = (t_sent, t_gold_path, output_dict[" ".join(t_sent)])
        sentence_index_tokens[t_sent_index].append(t_start)
        span_words.append(t_sent[t_start])
        pred_path = output_dict[" ".join(t_sent)]
        fdebug.write(
            str(t_sent[t_start]) + "\t" + " ".join(span_words) + "\t" + str(span_entropy) + "\t" + pred_path[
                t_start] + "\t" + t_gold_path[t_start] + "\n")
        count_tokens -= 1

    # print remaining cluster stats
    remaining_spans = sorted_spans[spans_index:]
    for span in remaining_spans:

        (span_entropy, sentence_key, start, end, best_path, instance_entropy, sent_index, token_index) = \
        model.most_uncertain_entropy_spans[span]
        sent = sentence_key.split()
        sent = [data_loader.id_to_word[int(token)] for token in sent]
        gold_path = gold_dict[" ".join(sent)]
        span_words = []
        span_in_text = data_loader.id_to_word[int(span)]
        sentence_index_sent[sent_index] = (sent, gold_path, output_dict[" ".join(sent)])
        fdebug.write(str(token_index) + "\t" + " ".join([sent[start]]) + "\t" + str(span_entropy) + "\n")

    covered = set()
    count = 0
    for sent_index in sentence_index:
        if sent_index not in covered:
            covered.add(sent_index)
            (sent, gold_path, path) = sentence_index_sent[sent_index]
            for token_index in sentence_index_tokens[sent_index]:
                path[token_index] = "UNK"

            for token, tag_label, gold_tag in zip(sent, path, gold_path):
                fout.write(token + "\t" + tag_label + "\t" + gold_tag + "\n")
                if tag_label == "UNK":
                    count += 1

            fout.write("\n")

    print("Total unique spans for exercise: {0}".format(args.k))
    print("Total spans for exercise: {0}".format(count))

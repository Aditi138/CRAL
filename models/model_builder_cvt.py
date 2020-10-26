__author__ = 'chuntingzhou,aditichaudhary'
from encoders import *
from decoders import *
from collections import defaultdict
from copy import deepcopy
import codecs
from itertools import combinations
#np.set_printoptions(threshold='nan')
minx = -1
maxx = 1


class CRF_Model_CVT(object):
    def __init__(self, args, data_loader):
        self.save_to = args.save_to_path
        self.load_from = args.load_from_path
        if not args.monolithic:
            tag_to_ids = data_loader.tag_to_ids
            self.tag_to_ids = tag_to_ids
        else:
            self.tag_to_ids = data_loader.tag_to_id
        self.constraints = None
        self.data_loader = data_loader


        #partial CRF
        self.use_partial = args.use_partial
        self.id_to_tags = data_loader.id2tags
        self.B_UNK = data_loader.B_UNK

        #active learning for partial annotations
        self.k = args.k
        self.cluster_nums = args.clusters
        self.activeLearning = args.activeLearning

        self.entropy_spans = defaultdict(lambda: 0)
        self.most_uncertain_entropy_spans = {}
        self.full_sentences = defaultdict(list)
        self.avg_spans_in_sent_entropy = defaultdict(list)
        self.token_count = defaultdict(lambda:0)
        self.typeHeap = defaultdict(list)
        self.heapInfo = defaultdict(list)
        self.clusterDetails = args.clusterDetails
        self.token_index_entropy_map = {}
        self.to_index =  0
        self.gammas = []
        self.approxTokenClassErrors = {}
        self.approxTypeErrors = defaultdict(lambda :0)
        self.type_tokenIndices = defaultdict(set)
        self.token_gamma_key = {}
        self.token_gamma_key_generic = {}
        self.predTypeErrors = {}
        self.model_conf = {}
        self.fout = codecs.open(args.model_name  + "_featureWiseEntropy.txt","w", encoding='utf-8')

    def forward(self, sents, char_sents, feats, bc_feats, training=True):
        raise NotImplementedError

    def save(self):
        if self.save_to is not None:
            self.model.save(self.save_to)
        else:
            print('Save to path not provided!')

    def load(self, path=None):
        if path is None:
            path = self.load_from
        if self.load_from is not None or path is not None:
            print('Load model parameters from %s!' % path)
            self.model.populate(path)
        else:
            print('Load from path not provided!')

    def cal_loss(self, sents, char_sents, tags, feats, bc_feats, known_tags, langs=None, training=True):
        birnn_outputs = self.forward(sents, char_sents, feats, bc_feats, langs, training=training)
        if self.args.monolithic:#Relevant for morphological analysis where all individual features are considered as one tagset
            crf_loss = self.classifier.decode_loss(birnn_outputs, tags, known_tags, self.data_loader.tag_to_id, self.B_UNK)

        else:
            first = True
            for key, gold_tags in tags.items():
                if first:
                    neg_log_likelihoods = self.crf_decoders[key].decode_loss(birnn_outputs, gold_tags,self.use_partial, known_tags[key], self.tag_to_ids[key], self.B_UNK, self.B_UNK)
                    first = False
                else:
                    neg_log_likelihoods += self.crf_decoders[key].decode_loss(birnn_outputs, gold_tags, self.use_partial,
                                                                             known_tags[key], self.tag_to_ids[key], self.B_UNK,
                                                                             self.B_UNK)

            neg_log_likelihoods = neg_log_likelihoods / len(tags)
            crf_loss = dy.sum_batches(neg_log_likelihoods) / len(sents)
        return crf_loss#, sum_s, sent_s

    def cal_cvt_loss(self, sents, char_sents, known_tags,  training=True, type="dev"):
        birnn_outputs, fwd_outputs, bwd_outputs = self.cvt_forward(sents, char_sents, training=training, type=type)
        batch_size = len(sents)

        first = True
        for key, _ in  self.cvt_crf_decoders.items():
            teacher_gammas = self.crf_decoders[key].decode_forward_scores(birnn_outputs, fixed = True)


            # prediction modules
            fwd_loss = self.computeTokenKL(fwd_outputs, key, teacher_gammas, batch_size)
            bwd_loss = self.computeTokenKL(bwd_outputs, key, teacher_gammas, batch_size)

            future = []
            for i in range(1, len(fwd_outputs)):
                future.append(fwd_outputs[i])
            future.append(fwd_outputs[0])
            future_ = self.computeTokenKL(future,  key, teacher_gammas , batch_size)

            past = [bwd_outputs[-1]]
            for i in range(len(bwd_outputs) -1 ):
                past.append(bwd_outputs[i])
            past_ = self.computeTokenKL(past,  key, teacher_gammas , batch_size)

            if first:
                unsupervised = fwd_loss + bwd_loss + past_ + future_
                first= False
            else:
                unsupervised += fwd_loss + bwd_loss + past_ + future_

        loss = unsupervised / len(self.cvt_crf_decoders.keys())
        return -loss

    def computeTokenKL(self, encoder_outputs, key, teacher_gammas, batch_size):
        prediction_gammas = self.cvt_crf_decoders[key].decode_forward_scores(encoder_outputs, fixed =False)

        first = True
        for stud_token_distr, teacher_token_distr in zip(prediction_gammas, teacher_gammas):# KL per token
            prob_teacher = dy.exp(teacher_token_distr)
            kls = dy.cmult(prob_teacher, (stud_token_distr - teacher_token_distr))
            if first:
                token_losses = dy.sum_batches(kls) / batch_size
                token_loss  = dy.mean_dim(token_losses, d=[0], b=False)
                first = False
            else:
                token_losses = dy.sum_batches(kls) / batch_size
                token_loss += dy.mean_dim(token_losses, d=[0], b=False)


        return token_loss / len(encoder_outputs)

    def eval_scores(self, sents, char_sents, feats, bc_feats, training=False):
        birnn_outputs = self.forward(sents, char_sents, feats, bc_feats, training=training)
        tag_scores, transit_score = self.crf_decoder.get_crf_scores(birnn_outputs)
        return tag_scores, transit_score

    def eval(self, sents, char_sents, feats, bc_feats, langs=None,
             training=False, type="dev", tgt_tags=None):
        birnn_outputs = self.forward(sents, char_sents, feats, bc_feats, langs,
                                     training=training, type=type)
        best_scores, best_paths = {}, {}
        featureEntropies = {}
        tokenEntropy = {}
        tokenInfo = {}
        sent = sents[0]
        # For each feature decoder, currenlty only feature=POS, else we construct independent CRF decoder for each feature
        for key, crf_decoder in self.crf_decoders.items():

            best_score, best_path, tag_scores = self.crf_decoders[key].decoding(birnn_outputs)
            best_scores[key] = best_score
            best_paths[key] = best_path
            best_path_copy = deepcopy(best_path)
            featureWiseEntropy = defaultdict(lambda: 0)

            if type == "test" and self.activeLearning:
                alpha_value, alphas = crf_decoder.forward_alg(tag_scores)
                beta_value, betas = crf_decoder.backward_one_sequence(tag_scores)
                # print("Alpha:{0} Beta:{1}".format(alpha_value.value(), beta_value.value()))

                gammas = []
                for i in range(len(sent)):
                    gammas.append(alphas[i] + betas[i] - alpha_value)

                if key == "POS":
                    self.gammas += [g.npvalue() for g in gammas[1:-1]]

                # Different active learning strategies
                if self.args.oracle: #ORACLE_UNS
                    gold_tags = tgt_tags[key][crf_decoder.sent_index][1:-1]
                    gamma = gammas[1:-1]
                    i = 0
                    for token, gold_tag, g in zip(sent[1:-1], gold_tags, gamma):
                        log_p = g.npvalue()[gold_tag]
                        H = -1.0 * np.exp(log_p) * log_p
                        span = str(token)
                        span_text = self.data_loader.id_to_word[token]
                        if self.args.normdata and span_text.lower() in self.data_loader.word_to_id: #norm data lowercases all data, but didn't observe significant gains with or without
                            span = str(self.data_loader.word_to_id[span_text.lower()])

                        sent_ = " ".join([str(x) for x in sent[1:-1]])
                        self.entropy_spans[span] += H

                        if i in tokenEntropy:
                            tokenEntropy[i] += H  # Accumulating Entropy per feature
                        else:
                            tokenEntropy[i] = H
                            tokenInfo[i] = (sent_, i, crf_decoder.sent_index)

                        if span in self.most_uncertain_entropy_spans:
                            (_, existing_sent, existing_i, _, existing_best_path, existing_H, _, _) = \
                            self.most_uncertain_entropy_spans[span]
                            if tokenEntropy[i] > existing_H:
                                self.most_uncertain_entropy_spans[span] = (
                                self.entropy_spans[span], sent_, i, i + 1, best_path, tokenEntropy[i],
                                crf_decoder.sent_index, i)
                        else:
                            self.most_uncertain_entropy_spans[span] = (
                            self.entropy_spans[span], sent_, i, i + 1, best_path, tokenEntropy[i],
                            crf_decoder.sent_index, i)
                        i += 1
                    crf_decoder.sent_index += 1
                # Entropy based method by accumulating only on exact match.
                else: #UNS
                    crf_decoder.get_uncertain_subsequences_UNS(sent[1:-1], gammas[1:-1], best_path_copy,
                                                                 self.entropy_spans,
                                                                 self.most_uncertain_entropy_spans,
                                                                 featureWiseEntropy, tokenEntropy, tokenInfo)

                featureEntropies[key] = featureWiseEntropy

        if type == "test" and self.activeLearning and not self.args.label:
            for token_index, token in enumerate(sent[1:-1]):
                token_entropy = tokenEntropy[token_index]
                token_info = tokenInfo[token_index]
                self.token_index_entropy_map[self.to_index] = token_entropy

                token_text = self.data_loader.id_to_word[token]

                if self.args.normdata and token_text.lower() in self.data_loader.word_to_id:
                    token = self.data_loader.word_to_id[token_text.lower()]
                self.typeHeap[str(token)].append(token_entropy)
                self.heapInfo[str(token)].append(token_info)

                self.token_count[str(token)] += 1
                self.to_index += 1

        return best_scores, best_paths

    def eval_cvt(self, sents, char_sents, training=False, type="dev", tgt_tags = None):
        birnn_outputs, fwd_outputs_, bwd_outputs_ = self.cvt_forward(sents, char_sents, training=training, type=type)
        best_scores, best_paths = {}, {}
        fwd_best_paths, bwd_best_paths, futback_best_paths = {},{},{}


        #print("Using partial encoding")
        if self.args.eval_cvt_fwd:
            birnn_outputs = [dy.concatenate([c, w]) for c, w in
                                 zip(fwd_outputs_, fwd_outputs_)]
        elif self.args.eval_cvt_bwd:
            birnn_outputs = [dy.concatenate([c, w]) for c, w in
                                 zip(bwd_outputs_, bwd_outputs_)]
        elif self.args.eval_cvt_future:
            future = []
            for i in range(1, len(fwd_outputs_)):
                future.append(fwd_outputs_[i])
            future.append(fwd_outputs_[0])
            birnn_outputs = [dy.concatenate([c, w]) for c, w in
                                 zip(future, future)]
        elif self.args.eval_cvt_back:
            past = [bwd_outputs_[-1]]
            for i in range(len(bwd_outputs_) - 1):
                past.append(bwd_outputs_[i])
            birnn_outputs = [dy.concatenate([c, w]) for c, w in
                                 zip(past,past)]
        elif self.args.eval_cvt_futback:
            future = []
            for i in range(1, len(fwd_outputs_)):
                future.append(fwd_outputs_[i])
            future.append(fwd_outputs_[0])
            past = [bwd_outputs_[-1]]
            for i in range(len(bwd_outputs_) - 1):
                past.append(bwd_outputs_[i])
            birnn_outputs = [dy.concatenate([c, w]) for c, w in
                                 zip(future,past)]


        if not (type == "test" and self.activeLearning):
            for key, crf_decoder in self.crf_decoders.items():
                best_score, best_path, tag_scores = self.crf_decoders[key].decoding(birnn_outputs)
                best_scores[key] = best_score
                best_paths[key] = best_path
            return best_scores, best_paths, deepcopy(best_paths), deepcopy(best_paths), deepcopy(best_paths)


        # For each feature decoder
        tokenEntropy = defaultdict(lambda:0)
        tokenInfo = {}

        for key, _ in self.cvt_crf_decoders.items():
            sent_index = self.crf_decoders[key].sent_index

            feature_tgt_tags = tgt_tags[key][sent_index]
            gold_path = np.array(feature_tgt_tags[1:-1])

            # using entire view
            teacher_best_score, teacher_best_path, teacher_scores = self.crf_decoders[key].decoding(birnn_outputs)
            teacher_gammas =  self.crf_decoders[key].getGamma(teacher_scores, fixed=True)
            best_scores[key] = teacher_best_score
            best_paths[key] = teacher_best_path

            teacher_best_path = np.array(teacher_best_path[1:-1])

            # for each prediction module
            fwd_outputs, bwd_outputs, future_outputs, back_outputs, fut_back  = self.getAuxiliaryEncoder(fwd_outputs_, bwd_outputs_)

            _, stud_best_path, stud_scores_fwd = self.crf_decoders[key].decoding(fwd_outputs)
            stud_fwd_gammas = self.crf_decoders[key].getGamma(stud_scores_fwd, fixed=True)
            fwd_kl = self.getKLDivergence(teacher_gammas, stud_fwd_gammas, key)
            stud_fwd_best_path = np.array(stud_best_path[1:-1])
            fwd_best_paths[key] = deepcopy(stud_best_path)

            _, stud_best_path, stud_scores_bwd = self.crf_decoders[key].decoding(bwd_outputs)
            stud_bwd_gammas = self.crf_decoders[key].getGamma(stud_scores_bwd, fixed=True)
            bwd_kl = self.getKLDivergence(teacher_gammas, stud_bwd_gammas, key)
            stud_bwd_best_path = np.array(stud_best_path[1:-1])
            bwd_best_paths[key] = deepcopy(stud_best_path)


            sent = sents[0][1:-1]
            sent_text = " ".join([str(x) for x in sent])


            for token_index, token_id in enumerate(sent):
                token_text = self.data_loader.id_to_word[int(token_id)]
                if self.args.normdata and token_text.lower() in self.data_loader.word_to_id:
                    token_id = self.data_loader.word_to_id[token_text.lower()]
                if self.args.al_mode == "qbc":
                    pred = defaultdict(lambda :0)

                    if self.args.oracle:#ORACLE
                        pred[teacher_best_path[token_index]] +=1
                        pred[gold_path[token_index]] += 1
                        sorted_pred = sorted(pred.items(), key=lambda kv: kv[1], reverse=True)[0]
                        num_disagree = 2 - sorted_pred[1]
                    else:
                        pred[teacher_best_path[token_index]] +=1
                        pred[stud_fwd_best_path[token_index]] +=1
                        pred[stud_bwd_best_path[token_index]] += 1
                        #pred[stud_futback_best_path[token_index]] += 1

                        sorted_pred = sorted(pred.items(), key=lambda kv: kv[1], reverse=True)[0]
                        num_disagree = 3 - sorted_pred[1]

                    if num_disagree > 0:
                        self.entropy_spans[token_id] += num_disagree
                        self.most_uncertain_entropy_spans[token_id] = (self.entropy_spans[token_id], sent_text, token_index, token_index + 1, teacher_best_path,num_disagree, sent_index, self.to_index)
                    tokenEntropy[token_index] += num_disagree
                    tokenInfo[token_index] = (sent_text, token_index, sent_index)


            self.crf_decoders[key].sent_index +=1

        for token_index, token in enumerate(sents[0][1:-1]):
            token_kl = tokenEntropy[token_index]
            token_info = tokenInfo[token_index]
            self.token_index_entropy_map[self.to_index] = token_kl
            token_text = self.data_loader.id_to_word[token]

            if self.args.normdata and token_text.lower() in self.data_loader.word_to_id:
                token = self.data_loader.word_to_id[token_text.lower()]
            self.typeHeap[str(token)].append(token_kl)
            self.heapInfo[str(token)].append(token_info)
            self.token_count[str(token)] += 1
            self.to_index += 1


        return best_scores, best_paths, fwd_best_paths, bwd_best_paths, best_paths

    def eval_mimic_oracle(self, sents, char_sents, training=False, type="dev", tgt_tags = None):
        birnn_outputs, fwd_outputs_, bwd_outputs_ = self.cvt_forward(sents, char_sents, training=training, type=type)
        best_scores, best_paths = {}, {}


        for key, _ in self.cvt_crf_decoders.items():

            sent_index = self.crf_decoders[key].sent_index

            feature_tgt_tags = tgt_tags[key][sent_index]
            gold_path = np.array(feature_tgt_tags[1:-1])

            # using entire view
            teacher_best_score, teacher_best_path, teacher_scores = self.crf_decoders[key].decoding(birnn_outputs)
            teacher_gammas =  self.crf_decoders[key].getGamma(teacher_scores, fixed=True)[1:-1]
            best_scores[key] = teacher_best_score
            best_paths[key] = teacher_best_path
            teacher_best_path = np.array(teacher_best_path[1:-1])

            if type == "test":

                assert len(teacher_gammas) == len(sents[0][1:-1])
                assert len(teacher_best_path) == len(teacher_gammas)
                token_index = deepcopy(self.to_index) #Ger the token index and reset it for every crf

                for token, gamma, pred in zip(sents[0][1:-1], teacher_gammas, teacher_best_path):
                    if token not in self.approxTokenClassErrors:
                        self.approxTokenClassErrors[token] = defaultdict(lambda :0)
                        self.predTypeErrors[token] = defaultdict(lambda:0)

                    gamma = gamma.npvalue()[:-2]


                    for tag_id, log_prob in enumerate(gamma):
                        if tag_id == pred:
                            self.predTypeErrors[token][tag_id] += np.exp(log_prob)
                            continue

                        prob = np.exp(log_prob)
                        self.approxTokenClassErrors[token][tag_id] += prob
                        self.approxTypeErrors[token] += prob

                    self.token_gamma_key[token_index] =  gamma
                    self.type_tokenIndices[token].add(token_index)
                    self.model_conf[token_index] = np.exp(gamma[pred])
                    token_index += 1



        self.to_index += len(sents[0][1:-1])
        return best_scores, best_paths



    def getAuxiliaryEncoder(self, fwd_outputs, bwd_outputs):
        future = []
        for i in range(1, len(fwd_outputs)):
            future.append(fwd_outputs[i])
        future.append(fwd_outputs[0])
        future_outputs = [dy.concatenate([c, w]) for c, w in
                           zip(future, future)]


        past = [bwd_outputs[-1]]
        for i in range(len(bwd_outputs) - 1):
            past.append(bwd_outputs[i])
        back_outputs = [dy.concatenate([c, w]) for c, w in
                           zip(past, past)]

        fut_back = [dy.concatenate([c, w]) for c, w in
                           zip(future, past)]
        fwd = [dy.concatenate([c, w]) for c, w in
                    zip(fwd_outputs, fwd_outputs)]
        bwd = [dy.concatenate([c, w]) for c, w in
                    zip(bwd_outputs, bwd_outputs)]
        return fwd, bwd, future_outputs, back_outputs, fut_back

    def getKLDivergence(self, teacherGammas, student_gammas, key):
        token_level = []
        for teacher_token, stud_token in zip(teacherGammas, student_gammas):
            prob_teacher = dy.exp(teacher_token)
            kls = dy.cmult(prob_teacher, (teacher_token - stud_token))
            kl = dy.mean_dim(kls, d=[0], b=False) * self.data_loader.tag_vocab_sizes[key]
            token_level.append(kl)
        return token_level
    
    
class BiRNN_ATTN_CRF_model_CVT(CRF_Model_CVT):
    ''' The same as above, except that we replace the cnn layer for characters with BiRNN layer. '''
    def __init__(self, args, data_loader, lm_data_loader=None):
        self.model = dy.Model()
        self.args = args
        super(BiRNN_ATTN_CRF_model_CVT, self).__init__(args, data_loader)

        tag_vocab_sizes = data_loader.tag_vocab_sizes

        char_vocab_size = data_loader.char_vocab_size
        word_vocab_size = data_loader.word_vocab_size
        word_padding_token = data_loader.word_padding_token

        char_emb_dim = args.char_emb_dim
        word_emb_dim = args.word_emb_dim
        tag_emb_dim = args.tag_emb_dim

        birnn_input_dim = args.char_hidden_dim * 2

        hidden_dim = args.hidden_dim
        char_hidden_dim = args.char_hidden_dim
        self.char_hidden_dim = args.char_hidden_dim * 2
        src_ctx_dim = args.hidden_dim * 2

        output_dropout_rate = args.output_dropout_rate
        emb_dropout_rate = args.emb_dropout_rate
        
        self.char_birnn_encoder = BiRNN_Encoder(self.model,
                 char_emb_dim,
                 char_hidden_dim,
                 emb_dropout_rate=0.0,
                 output_dropout_rate=0.0,
                 vocab_size=char_vocab_size,
                 emb_size=char_emb_dim)

        self.proj1_W = self.model.add_parameters((char_hidden_dim, char_hidden_dim * 2))
        self.proj1_b = self.model.add_parameters(char_hidden_dim)
        self.proj1_b.zero()
        self.proj2_W = self.model.add_parameters((char_hidden_dim, char_hidden_dim * 2))
        self.proj2_b = self.model.add_parameters(char_hidden_dim)
        self.proj2_b.zero()

        if args.pretrain_emb_path is None:
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token)
        else:
            print("Using pretrained word embedding!")
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token, data_loader.pretrain_word_emb)


        birnn_input_dim = birnn_input_dim + args.char_hidden_dim * 2

        self.char_birnn_modeling = BiRNN_Encoder(self.model,
                                                 args.char_hidden_dim * 4,
                                                 args.char_hidden_dim * 2,
                                                emb_dropout_rate=emb_dropout_rate,
                                                output_dropout_rate=output_dropout_rate)

    
        if args.use_token_langid:
            birnn_input_dim = birnn_input_dim + word_emb_dim

    
        self.birnn_encoder = BiRNN_Encoder(self.model,
                                           birnn_input_dim,
                                           hidden_dim,
                                           emb_dropout_rate=emb_dropout_rate,
                                           output_dropout_rate=output_dropout_rate)

        if args.simple or args.monolithic:
            self.classifier = classifier(self.model, src_ctx_dim, data_loader.tag_vocab_sizes)
        else:
            self.crf_decoders = {}

            self.cvt_crf_decoders = {}
            for key, tag_size in tag_vocab_sizes.items():
                self.crf_decoders[key] = chain_CRF_decoder(args, self.model, self.data_loader, src_ctx_dim, tag_emb_dim, tag_size, constraints=self.constraints)
                self.cvt_crf_decoders[key] =  chain_CRF_decoder(args, self.model, self.data_loader, src_ctx_dim / 2, tag_emb_dim, tag_size, constraints=self.constraints)
        self.attention_weights = []
        self.word_embedding_weights = []
        self.token_embeddings = []
        self.tokens = []
        self.token_gold = []
        self.id_to_char = data_loader.id_to_char
        self.token_index_sent_index_map = {}
        self.t_index, self.s_index,self.sent_index_sent_map, self.token_index_relative_idx_map  = 0,0, {},{}

    def forward(self, sents, char_sents, feats, bc_feats, b_langs, training=True, type="dev"):
        char_embs = self.char_birnn_encoder.encode_char(char_sents, training=training)
        proj1_W = dy.parameter(self.proj1_W)
        proj1_b = dy.parameter(self.proj1_b)
        proj2_W = dy.parameter(self.proj2_W)
        proj2_b = dy.parameter(self.proj2_b)
        attended_sents = []
        for i,batch in enumerate(char_embs):
            attended_sent = []
            for word_num,word in enumerate(batch):
                E = np.ones((len(word), len(word)), dtype=float) - np.eye(len(word))
                attn_keys = [dy.tanh(dy.affine_transform([proj1_b, proj1_W, w_attn])) for w_attn in word]
                attn_values = dy.concatenate_cols([dy.tanh(dy.affine_transform([proj2_b, proj2_W, w_attn])) for w_attn in word])
                attn_weights = [dy.softmax(dy.transpose(key) * attn_values, d=1) for key in attn_keys]
                word_representation = dy.concatenate_cols(word)

                if not training:
                    self.attention_weights.append([a.npvalue() for a in attn_weights])
                    #orig_word = char_sents[i][word_num]
                    #self.attn_fout.write("".join(self.id_to_char[id] for id in orig_word) + "\t" + " ".join(map(str,attn_weights_value)) + "\n")


                maskedAttention =[]
                for j in range(len(word)):
                    masking = dy.cmult(dy.inputTensor(np.concatenate([E[j] for _ in range(self.char_hidden_dim)]).reshape((self.char_hidden_dim, len(word)))),
                         word_representation)
                    maskedAttention.append(masking * dy.transpose(attn_weights[j]))

                attended_sent.append([dy.concatenate([h, ha]) for h, ha in zip(word, maskedAttention)])

            attended_sents.append(attended_sent)
        #print("Char Rep", len(attended_sents), len(char_embs))
        char_embs = self.char_birnn_modeling.encode(attended_sents, training=training, char=True, model_char=True)
        if not training and type == "test":
            self.word_embedding_weights.append([c.npvalue() for c in char_embs[1:-1]])
            if not self.args.tokenRep:
                for w in char_embs[1:-1]:
                    self.token_embeddings.append(w.npvalue())
            for rel_idx, token in enumerate(sents[0][1:-1]):
                self.tokens.append(token)
                self.token_index_sent_index_map[self.t_index] = self.s_index
                self.token_index_relative_idx_map[self.t_index] = rel_idx
                self.t_index +=1

            self.sent_index_sent_map[self.s_index] = sents[0][1:-1]
            self.s_index += 1



        concat_inputs = [dy.concatenate([c]) for c in char_embs]
        birnn_outputs, f, b = self.birnn_encoder.encode(concat_inputs, training=training)
        if not training and type=="test" and self.args.tokenRep:
            for word_emb in birnn_outputs[1:-1]:
                self.token_embeddings.append(word_emb.npvalue())


        return birnn_outputs


    def cvt_forward(self, sents, char_sents, training=True, type="dev"):
        char_embs = self.char_birnn_encoder.encode_char(char_sents, training=training)
        proj1_W = dy.parameter(self.proj1_W)
        proj1_b = dy.parameter(self.proj1_b)
        proj2_W = dy.parameter(self.proj2_W)
        proj2_b = dy.parameter(self.proj2_b)
        attended_sents = []
        for i,batch in enumerate(char_embs):
            attended_sent = []
            for word_num,word in enumerate(batch):
                E = np.ones((len(word), len(word)), dtype=float) - np.eye(len(word))
                attn_keys = [dy.tanh(dy.affine_transform([proj1_b, proj1_W, w_attn])) for w_attn in word]
                attn_values = dy.concatenate_cols([dy.tanh(dy.affine_transform([proj2_b, proj2_W, w_attn])) for w_attn in word])
                attn_weights = [dy.softmax(dy.transpose(key) * attn_values, d=1) for key in attn_keys]
                word_representation = dy.concatenate_cols(word)

                if not training:
                    self.attention_weights.append([a.npvalue() for a in attn_weights])
                    #orig_word = char_sents[i][word_num]
                    #self.attn_fout.write("".join(self.id_to_char[id] for id in orig_word) + "\t" + " ".join(map(str,attn_weights_value)) + "\n")


                maskedAttention =[]
                for j in range(len(word)):
                    masking = dy.cmult(dy.inputTensor(np.concatenate([E[j] for _ in range(self.char_hidden_dim)]).reshape((self.char_hidden_dim, len(word)))),
                         word_representation)
                    maskedAttention.append(masking * dy.transpose(attn_weights[j]))

                attended_sent.append([dy.concatenate([h, ha]) for h, ha in zip(word, maskedAttention)])
                # attended_sent.append(dy.mean_dim(
                #     dy.concatenate_cols([dy.concatenate([h, ha]) for h, ha in zip(word, maskedAttention)]),
                #     d=[1], b=False))
            attended_sents.append(attended_sent)
        #print("Char Rep", len(attended_sents), len(char_embs))
        char_embs = self.char_birnn_modeling.encode(attended_sents, training=training, char=True, model_char=True)
        if not training and type == "test":
            self.word_embedding_weights.append([c.npvalue() for c in char_embs[1:-1]])
            if not self.args.tokenRep:
                for w in char_embs[1:-1]:
                    self.token_embeddings.append(w.npvalue())
            for rel_idx, token in enumerate(sents[0][1:-1]):
                self.tokens.append(token)
                self.token_index_sent_index_map[self.t_index] = self.s_index
                self.token_index_relative_idx_map[self.t_index] = rel_idx
                self.t_index +=1

            self.sent_index_sent_map[self.s_index] = sents[0][1:-1]
            self.s_index += 1



        concat_inputs = [dy.concatenate([c]) for c in char_embs]
        birnn_outputs, fwd_outputs, bwd_outputs = self.birnn_encoder.encode(concat_inputs, training=training)

        if not training and type=="test" and self.args.tokenRep:
            #concat_word_char = [dy.concatenate([c, w]) for c, w in zip(char_embs, birnn_outputs)]
            for word_emb in birnn_outputs[1:-1]:
                self.token_embeddings.append(word_emb.npvalue())


        return birnn_outputs, fwd_outputs, bwd_outputs


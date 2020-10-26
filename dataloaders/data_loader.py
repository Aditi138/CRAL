__author__ = 'chuntingzhou'
import os
from utils.util import *
#tagset = ['B-LOC','B-PER','B-MISC', 'B-ORG','I-LOC','I-PER','I-MISC', 'I-ORG','O']
#tagset = ['B-LOC','B-PER','B-GPE', 'B-ORG','I-LOC','I-PER','I-GPE', 'I-ORG','O']

class DataLoader():
    def __init__(self, args, special_normal=False):
        # This is data loader as well as feature extractor!!
        '''Data format: id word pos_tag syntactic_tag NER_tag'''
        ''' TODO: 1. normalizing all digits
                  2. Using full vocabulary from GloVe, when testing, lower case first'''
        self.args = args
        self.train_path = args.train_path
        self.test_path = args.test_path
        self.dev_path = args.dev_path
        self.args = args
        self.use_langid = args.use_langid

        self.pretrained_embedding_path = args.pretrain_emb_path
        self.use_discrete_feature = args.use_discrete_features
        self.use_brown_cluster = args.use_brown_cluster
        self.multilingual = args.multilingual

        self.train_senttypes = self.dev_senttypes = self.test_senttypes = None
        self.brown_cluster_dicts = None

        print("Generating vocabs from training file ....")
        if args.multilingual:
            self.lang_to_code, self.code_to_lang = get_lang_code_dicts(args.lang_codes)
            paths_to_read = []
            langs =  args.langs.split("/")
            if args.augVocablang is not None:
                langs += args.augVocablang.split("/")
            for lang in langs:
                input_folder = args.input_folder + "/" + "UD_" + self.code_to_lang[lang] + "//"
                if args.monolithic:
                    self.filestart = lang
                else:
                    self.filestart = "udmap"
                for [path, dir, files] in os.walk(input_folder):
                    if self.args.use_sort:
                        files.sort()
                    for file in files:
                        if file.startswith(self.filestart):
                            path = input_folder + file
                            print("Reading vocab from ", path)
                            paths_to_read.append((path, lang))
                    break

            if args.monolithic: #Using monolithic tagset with softmax as classifier
                self.tag_to_id, self.word_to_id, self.char_to_id = self.read_files_morph_monolithic(paths_to_read)
            else:
                self.tag_to_ids, self.word_to_id, self.char_to_id = self.read_files_morph(paths_to_read)


        # FIXME: Remember dictionary value for char and word has been shifted by 1
        print("Size of vocab before: %d" % len(self.word_to_id))
        self.word_to_id['<unk>'] = len(self.word_to_id) + 1
        self.char_to_id['<unk>'] = len(self.char_to_id) + 1

        self.word_to_id['<\s>'] = 0
        self.char_to_id['<pad>'] = 0
        print("Size of vocab after: %d" % len(self.word_to_id))
        # pkl_dump(self.tag_to_id, self.tag_vocab_path)
        # pkl_dump(self.char_to_id, self.char_vocab_path)
        # pkl_dump(self.word_to_id, self.word_vocab_path)

        self.word_padding_token = 0
        self.char_padding_token = 0

        if self.pretrained_embedding_path is not None:
            self.pretrain_word_emb, self.word_to_id, self.char_to_id = get_pretrained_emb(self.args.fixedVocab, self.pretrained_embedding_path,
                                                                         self.word_to_id, self.char_to_id, args.word_emb_dim)

        # for char vocab and word vocab, we reserve id 0 for the eos padding, and len(vocab)-1 for the <unk>
        if args.monolithic:
            self.id2tags = {}
            self.id2tags = {v: k for k, v in self.tag_to_id.iteritems()}
            self.tag_vocab_sizes = len(self.tag_to_id)
            print("Tagset size: {0}".format(self.tag_vocab_sizes))
        else:
            self.id2tags= {}
            self.tag_vocab_sizes = {}
            for key, tag2id in self.tag_to_ids.items():
                self.id2tags[key] = {v: k for k, v in tag2id.iteritems()}
                self.tag_vocab_sizes[key] = len(tag2id)
                print("Feat: {0} Size: {1}".format(key, len(tag2id)))
                print(self.tag_to_ids[key])

        self.id_to_word = {v: k for k, v in self.word_to_id.iteritems()}
        self.id_to_char = {v: k for k, v in self.char_to_id.iteritems()}

        self.word_vocab_size = len(self.id_to_word)
        self.char_vocab_size = len(self.id_to_char)

        self.cap_ratio_dict = None
        #Partial CRF
        self.B_UNK = 100

        print("Size of vocab after: %d" % len(self.word_to_id))
        print("Word vocab size=%d, Char Vocab size=%d" % ( self.word_vocab_size, self.char_vocab_size))


    def getTypologyFeatures(self, path):
        features = {}
        with open(path, "r") as fin:
            lines = fin.readlines()[1:]
            for line in lines:
                line = line.strip().split()
                vector = line[1:]
                features[line[0]] = np.asarray(vector).astype(np.float32)
                self.feature_emb_dim = len(vector)
        print("Loaded typology features from {0}: {1}".format(path, self.feature_emb_dim))
        return features

    def read_out_cap_ratio(self, path):
        word_ratio = dict()
        with codecs.open(path, "r", "utf-8") as fin:
            for line in fin:
                word, ratio = line.strip().split()
                if word not in self.word_to_id:
                    continue
                if ratio < 0.4:
                    word_ratio[word] = 0
                elif  0.4 <= ratio < 0.6:
                    word_ratio[word] = 1
                elif 0.6 <= ratio < 0.8:
                    word_ratio[word] = 2
                elif ratio >= 0.8:
                    word_ratio[word] = 3
        return word_ratio

    def read_out_file_names(self, path):
        if path is None:
            return None

        types = []
        with open(path, "r") as fin:
            for line in fin:
                types.append(line.strip().split("_")[1])

        return None

    @staticmethod
    def exists(path):
        return os.path.exists(path)



    def read_one_line_morph(self, line, tag_set, word_dict, char_set, fdebug, lang):
        write = False
        for w in line:
            fields = w.split("\t")
            if len(fields) != 10:
                print("ERROR")
                print(fields)
                exit(0)
            word = fields[1]
            feats = fields[5]
            if feats == "":
                print("ERROR: the feature is blank, re-run pretrain_pos")
                write=True
            # print(word)
            # print(ner_tag)
            for c in word:
                char_set.add(c)
            if feats != "B-UNK":
                for feat in feats.split("|"):
                    feat_info = feat.split("=")
                    tag_set[feat_info[0]].add(feat_info[-1])
            word_dict[word] += 1 
        if write:
            for w in line:
                fdebug.write(w + "\n")
            fdebug.write("\n")
            exit(-1)

    def read_one_line_morph_monolithic(self, line, tag_set, word_dict, char_set, fdebug, lang):
        write = False
        for w in line:
            fields = w.split("\t")
            if len(fields) != 10:
                print("ERROR")
                print(fields)
                exit(0)
            word = fields[1]
            feats = fields[5]
            if feats == "":
                print("ERROR: the feature is blank, re-run pretrain_pos")
                write = True
            # print(word)
            # print(ner_tag)
            for c in word:
                char_set.add(c)
            if feats != "B-UNK":
                feats = feats.split(";")
                feats.sort()
                feats = ";".join(feats)
                tag_set.add(feats)

            word_dict[word] += 1
        if write:
            for w in line:
                fdebug.write(w + "\n")
            fdebug.write("\n")
            exit(-1)

    def get_vocab_from_set(self, a_set, shift=0):
        vocab = {}
        for i, elem in enumerate(a_set):
            vocab[elem] = i + shift

        return vocab

    def get_vocab_from_dict(self, a_dict, shift=0, remove_singleton=False):
        vocab = {}
        i = 0
        self.singleton_words = set()

        #Sort the defaultdict
        sortedDict = sorted(a_dict.iteritems(), key=lambda (k, v): v, reverse=True)
        for (k,v) in sortedDict:

        #for k, v in a_dict.iteritems():
            if v == 1:
                self.singleton_words.add(i + shift)
            if remove_singleton:
                if v > 1:
                    # print k, v
                    vocab[k] = i + shift
                    i += 1
            else:
                vocab[k] = i + shift
                i += 1
        print("Singleton words number: %d" % len(self.singleton_words))
        return vocab

    def read_files_morph(self, paths):
        # word_list = []
        # char_list = []
        # tag_list = []
        word_dict = defaultdict(lambda: 0)
        char_set = set()
        tag_vocab = defaultdict(set)


        def _read_a_file(path, lang):
            fdebug = codecs.open("./debug.txt","w", encoding='utf-8')
            with codecs.open(path, "r", "utf-8") as fin:
                to_read_line = []
                if self.use_langid:
                    line = ["_" for  _ in range(10)]
                    line[1] = "<" + lang + ">"
                    to_read_line.append("\t".join(line))
                for line in fin:
                    if line.startswith("#"):
                        continue
                    if line.strip() == "":
                        if self.use_langid:
                            line = ["_" for _ in range(10)]
                            line[1] = "<" + lang + ">"
                            to_read_line.append("\t".join(line))
                        self.read_one_line_morph(to_read_line, tag_vocab, word_dict, char_set,fdebug, lang)
                        to_read_line = []
                        if self.use_langid:
                            line = ["_" for _ in range(10)]
                            line[1] = "<" + lang + ">"
                            to_read_line.append("\t".join(line))
                    else:
                        to_read_line.append(line.strip())
                self.read_one_line_morph(to_read_line, tag_vocab, word_dict, char_set,fdebug, lang)

        for (path, lang) in paths:
            _read_a_file(path, lang)

        tag2ids = {}
        for key, tag_set in tag_vocab.items():
            if key == "_":
                continue
            tag_set.add("_")
            tag2ids[key] = self.get_vocab_from_set(tag_set)
        word_vocab = self.get_vocab_from_dict(word_dict, 1, self.args.remove_singleton)
        char_vocab = self.get_vocab_from_set(char_set, 1)

        return tag2ids, word_vocab, char_vocab

    def read_files_morph_monolithic(self, paths):
        # word_list = []
        # char_list = []
        # tag_list = []
        word_dict = defaultdict(lambda: 0)
        char_set = set()
        tag_vocab = set()

        def _read_a_file(path, lang):
            fdebug = codecs.open("./debug.txt", "w", encoding='utf-8')
            with codecs.open(path, "r", "utf-8") as fin:
                to_read_line = []
                if self.use_langid:
                    line = ["_" for _ in range(10)]
                    line[1] = "<" + lang + ">"
                    to_read_line.append("\t".join(line))
                for line in fin:
                    if line.startswith("#"):
                        continue
                    if line.strip() == "":
                        if self.use_langid:
                            line = ["_" for _ in range(10)]
                            line[1] = "<" + lang + ">"
                            to_read_line.append("\t".join(line))
                        self.read_one_line_morph_monolithic(to_read_line, tag_vocab, word_dict, char_set, fdebug, lang)
                        to_read_line = []
                        if self.use_langid:
                            line = ["_" for _ in range(10)]
                            line[1] = "<" + lang + ">"
                            to_read_line.append("\t".join(line))
                    else:
                        to_read_line.append(line.strip())
                self.read_one_line_morph_monolithic(to_read_line, tag_vocab, word_dict, char_set, fdebug, lang)

        for (path, lang) in paths:
            _read_a_file(path, lang)

        tag2ids = {}
        tag_vocab.add("_")
        tag2ids = self.get_vocab_from_set(tag_vocab)
        word_vocab = self.get_vocab_from_dict(word_dict, 1, self.args.remove_singleton)
        char_vocab = self.get_vocab_from_set(char_set, 1)

        return tag2ids, word_vocab, char_vocab

    def get_data_set_morph(self, path, lang, source="train"):
        sents = []
        char_sents = []
        tgt_tags = defaultdict(list)
        discrete_features = []
        bc_features = []
        known_tags = defaultdict(list)

        if source == "train":
            sent_types = self.train_senttypes
        else:
            sent_types = self.dev_senttypes

        def add_sent(one_sent, type):
            temp_sent = []
            temp_feats = defaultdict(list)
            temp_char = []
            temp_bc = []
            sent = []
            temp_known_tag = defaultdict(list)
            if self.use_langid:
                line = ["_" for _ in range(10)]
                line[1] = "<"  + lang  + ">"
                one_sent  = ["\t".join(line)] + one_sent + ["\t".join(line)] # Adding language tag before and after each sequence
            for w in one_sent:
                fields = w.split("\t")
                assert len(fields) == 10
                word = fields[1]
                sent.append(word)
                feats = fields[5]
                if feats == "_": #No Morph tags, for each feature assingn "_"
                    for key in self.tag_to_ids.keys():
                        temp_feats[key].append(self.tag_to_ids[key]["_"])
                        temp_known_tag[key].append([1])
                elif feats == "B-UNK":
                    for key in self.tag_to_ids.keys():
                        temp_feats[key].append(self.B_UNK)
                        temp_known_tag[key].append([0])
                else:
                    addedFeats = set()
                    keyvalue_set = list(set(feats.split("|")))
                    for feat in keyvalue_set:
                        key_info = feat.split("=")
                        key = key_info[0]
                        if key in addedFeats:
                            continue
                        if key == "V":
                            break
                        temp_feats[key].append(self.tag_to_ids[key][key_info[-1]])
                        addedFeats.add(key)
                    for key in self.tag_to_ids.keys():
                        temp_known_tag[key].append([1])
                        if key not in addedFeats:#Adding Null char for features absent in one token
                            temp_feats[key].append(self.tag_to_ids[key]["_"])


                if self.args.fixedVocab:
                    if word in self.word_to_id:
                        temp_sent.append(self.word_to_id[word])
                    elif word.lower() in self.word_to_id:
                        temp_sent.append(self.word_to_id[word.lower()])
                    else:
                        temp_sent.append(self.word_to_id["<unk>"])
                else:
                    temp_sent.append(self.word_to_id[word] if word in self.word_to_id else self.word_to_id["<unk>"])

                temp_char.append(
                    [self.char_to_id[c] if c in self.char_to_id else self.char_to_id["<unk>"] for c in word])

            sents.append(temp_sent)
            char_sents.append(temp_char)
            for key, one_sent_feature_wise_tags in temp_feats.items():
                tgt_tags[key].append(one_sent_feature_wise_tags)
                known_tags[key].append(temp_known_tag[key])

            bc_features.append(temp_bc)


            discrete_features.append([])


            # print len(discrete_features[-1])

        with codecs.open(path, "r", "utf-8") as fin:
            i = 0
            one_sent = []
            for line in fin:
                if line.startswith("#"):
                    continue
                if line.strip() == "":
                    if len(one_sent) > 0:
                        add_sent(one_sent, sent_types[i] if sent_types is not None else None)
                        i += 1
                        if i % 1000 == 0:
                            print("Processed %d training data." % (i,))
                    one_sent = []
                else:
                    one_sent.append(line.strip())

            if len(one_sent) > 0:
                add_sent(one_sent, sent_types[i] if sent_types is not None else None)
                i += 1

        if sent_types is not None:
            assert i == len(sent_types), "Not match between number of sentences and sentence types!"

        if self.use_discrete_feature:
            self.num_feats = len(discrete_features[0][0])
        else:
            self.num_feats = 0

        # debug_sents = [sents[0] for _ in range(10)]
        # debug_charsents = [char_sents[0] for _ in range(10)]
        # debug_tgt_tags = {}
        # for key, tags in tgt_tags.items():
        #     debug_tgt_tags[key] = [tags[0] for _ in range(10)]
        #
        # return debug_sents, debug_charsents, debug_tgt_tags, discrete_features, bc_features, known_tags

        return sents, char_sents, tgt_tags, discrete_features, bc_features, known_tags

    def get_data_set_morph_monolithic(self, path, lang, source="train"):
        sents = []
        char_sents = []
        tgt_tags = []
        discrete_features = []
        bc_features = []
        known_tags = []

        if source == "train":
            sent_types = self.train_senttypes
        else:
            sent_types = self.dev_senttypes

        def add_sent(one_sent, type):
            temp_sent = []
            temp_feats = []
            temp_char = []
            temp_bc = []
            sent = []
            temp_known_tag = []
            if self.use_langid:
                line = ["_" for _ in range(10)]
                line[1] = "<"  + lang  + ">"
                one_sent  = ["\t".join(line)] + one_sent + ["\t".join(line)] # Adding language tag before and after each sequence
            for w in one_sent:
                fields = w.split("\t")
                assert len(fields) == 10
                word = fields[1]
                sent.append(word)
                feats = fields[5]
                if feats == "_": #No Morph tags, for each feature assingn "_"
                    temp_feats.append(self.tag_to_id["_"])
                    temp_known_tag.append(1)
                elif feats == "B-UNK":
                    temp_feats.append(0) #place_holder
                    temp_known_tag.append(0)
                else:
                    feats = feats.split(";")
                    feats.sort()
                    feats  = ";".join(feats)
                    temp_feats.append(self.tag_to_id[feats])
                    temp_known_tag.append(1)

                if self.args.fixedVocab:
                    if word in self.word_to_id:
                        temp_sent.append(self.word_to_id[word])
                    elif word.lower() in self.word_to_id:
                        temp_sent.append(self.word_to_id[word.lower()])
                    else:
                        temp_sent.append(self.word_to_id["<unk>"])
                else:
                    temp_sent.append(self.word_to_id[word] if word in self.word_to_id else self.word_to_id["<unk>"])

                temp_char.append(
                    [self.char_to_id[c] if c in self.char_to_id else self.char_to_id["<unk>"] for c in word])

            sents.append(temp_sent)
            char_sents.append(temp_char)
            tgt_tags.append(temp_feats)
            known_tags.append(temp_known_tag)
            bc_features.append(temp_bc)

            if not self.args.isLr:
                discrete_features.append([])
            else:
                discrete_features.append(get_feature_sent(lang, sent, self.args, self.cap_ratio_dict, type=type))

            # print len(discrete_features[-1])

        with codecs.open(path, "r", "utf-8") as fin:
            i = 0
            one_sent = []
            for line in fin:
                if line.startswith("#"):
                    continue
                if line.strip() == "":
                    if len(one_sent) > 0:
                        add_sent(one_sent, sent_types[i] if sent_types is not None else None)
                        i += 1
                        if i % 1000 == 0:
                            print("Processed %d training data." % (i,))
                    one_sent = []
                else:
                    one_sent.append(line.strip())

            if len(one_sent) > 0:
                add_sent(one_sent, sent_types[i] if sent_types is not None else None)
                i += 1

        if sent_types is not None:
            assert i == len(sent_types), "Not match between number of sentences and sentence types!"

        if self.use_discrete_feature:
            self.num_feats = len(discrete_features[0][0])
        else:
            self.num_feats = 0

        # debug_sents = [sents[0] for _ in range(10)]
        # debug_charsents = [char_sents[0] for _ in range(10)]
        # debug_tgt_tags = {}
        # for key, tags in tgt_tags.items():
        #     debug_tgt_tags[key] = [tags[0] for _ in range(10)]
        #
        # return debug_sents, debug_charsents, debug_tgt_tags, discrete_features, bc_features, known_tags

        return sents, char_sents, tgt_tags, discrete_features, bc_features, known_tags

    def get_pos_data_set(self, path, tagset,isDev=False):
        #tagset = set()
        pos_tags = []
        one_pos_tag = []
        if self.use_langid:
            one_pos_tag.append("_")  
        with codecs.open(path, "r", encoding='utf-8') as fin:
            for line in fin:
                if line == "" or line == "\n":
                    if self.use_langid:
                        one_pos_tag.append("_")
                    pos_tags.append(one_pos_tag)
                    one_pos_tag = []
                    if self.use_langid:
                        one_pos_tag.append("_")
                else:
                    line = line.strip().split("\t")
                    if not isDev:
                        one_pos_tag.append(line[-1])
                        tagset.add(line[-1])
                    else:
                        one_pos_tag.append(line[5])
                        tagset.add(line[5])

        #    if len(one_pos_tag) > 0:
        #        pos_tags.append(one_pos_tag)

        return pos_tags

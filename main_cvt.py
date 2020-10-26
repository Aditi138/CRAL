__author__ = 'aditichaudhary'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import evaluate_2019_task2

def evaluate(data_loader, path, model, model_name, type="dev"):
    sents, char_sents, tgt_tags, discrete_features, bc_feats, _ = data_loader.get_data_set_morph(path, args.lang,
                                                                                               source="dev")
    prefix = model_name + "_" + str(uid)
    predictions, predictions_fwd, predictions_bwd, predictions_futback = [], [], [], []
    sentences = []
    i = 0
    gold_file = args.gold_file
    score_sent = {}
    num = 0
    langid = data_loader.word_to_id["<" + args.lang + ">"]

    model.tokens = []
    model.to_index = 0
    #Currently only the token and character features are used the discrete_features and brown-cluster features are empty.
    #For some methods we have included the provision to add these additional features, feel free to extend them to the proposed method
    for sent, char_sent, discrete_feature, bc_feat in zip(sents, char_sents, discrete_features, bc_feats):
        dy.renew_cg()
        sent, char_sent, discrete_feature, bc_feat, b_lang = [sent], [char_sent], [discrete_feature], [bc_feat], [ [langid for _ in range(len(sent))]]
        if args.al_mode == "qbc": #QBC
            best_scores, best_paths, fwd_best_paths, bwd_best_paths, futback_best_paths= model.eval_cvt(sent, char_sent, training=False, type=type, tgt_tags=tgt_tags)

        elif args.al_mode == "entropy": #Entropy
            best_scores, best_paths = model.eval(sent, char_sent, discrete_feature, bc_feat, langs=b_lang,  training=False,type=type, tgt_tags=tgt_tags)
            fwd_best_paths, bwd_best_paths, futback_best_paths= best_paths, best_paths, best_paths  #dummy variables since we don't use all the partial views here

        else: #Proposed method CRAL
            best_scores, best_paths = model.eval_mimic_oracle(sent, char_sent,
                                                 training=False, type=type, tgt_tags=tgt_tags)


        one_prediction = [[] for _ in range(len(sent[0]))]
        one_prediction_fwd = [[] for _ in range(len(sent[0]))]
        one_prediction_bwd = [[] for _ in range(len(sent[0]))]
        one_prediction_futback = [[] for _ in range(len(sent[0]))]

        sent_score = 0.0
        #The  code has been written for easy extensibility to include other morphological features such as Gender, Number etc,
        #Currently we only use the POS feature hence best_paths is a dictionary with only one key: POS.
        for key, gold_tgt_tags in tgt_tags.items():
            getTagPred(data_loader, best_paths[key], key, one_prediction)
            assert len(best_paths[key]) == len(gold_tgt_tags[i])
            sent_score += best_scores[key]
            if args.cvt: #To retrieve predictions from the different views of cross-view training
                getTagPred(data_loader, fwd_best_paths[key], key, one_prediction_fwd)
                getTagPred(data_loader, bwd_best_paths[key], key, one_prediction_bwd)
                getTagPred(data_loader, futback_best_paths[key], key, one_prediction_futback)


        sent_score = sent_score / len(best_scores)
        sentence_key = " ".join([str(x) for x in sent[0]])
        score_sent[sentence_key] = sent_score
        predictions.append(one_prediction)
        if args.cvt:
            predictions_fwd.append(one_prediction_fwd)
            predictions_bwd.append(one_prediction_bwd)
            predictions_futback.append(one_prediction_futback)

        sentences.append(sent)
        i += 1
        if i % 1000 == 0:
            print("Testing processed %d lines " % i)
    num +=1


    if type ==  "dev":
        pred_output_fname = os.path.dirname(path) + "/" + prefix + "dev_pred.conllu"
    elif type == "test":
        pred_output_fname = os.path.dirname(path) +"/" + prefix + "test_pred.conllu"
        gold_file = args.gold_test_file
    else:
        pred_output_fname = "%s/%s_pred_output.txt" % (args.eval_folder,prefix)

    printPredictions(data_loader, pred_output_fname, predictions, sentences)
    
    if args.cvt and type=="test":
        printPredictions(data_loader, pred_output_fname + ".fwd", predictions_fwd, sentences)
        printPredictions(data_loader, pred_output_fname + ".bwd", predictions_bwd, sentences)
#        printPredictions(data_loader, pred_output_fname + ".futback", predictions_futback, sentences)
        #Use the SIGMORPHON released evaluation file to evaluate (it can be used for full morphological analysis  in the unimorph format)
        acc, f1, precision, recall, _, _, _, _ = evaluate_2019_task2.main(out_file=pred_output_fname + ".fwd", ref_file=gold_file)
        print(acc, f1)

        acc, f1, precision, recall, _, _, _, _ = evaluate_2019_task2.main(out_file=pred_output_fname + ".bwd",
                                                                          ref_file=gold_file)
        print(acc, f1)

    acc, f1,  precision, recall, prediction_pairs, gold_pairs, gold_token_predictions, token_predictions= evaluate_2019_task2.main(out_file=pred_output_fname, ref_file=gold_file)
    # if type == "dev":
    #     os.system("rm %s" % (pred_output_fname,))
    return acc, precision, recall, f1, prediction_pairs, gold_pairs, score_sent, gold_token_predictions,token_predictions

def printPredictions(data_loader, pred_output_fname, predictions, sentences):

    with open(pred_output_fname, "w") as fout:
        for sent, pred in zip(sentences, predictions):
            info = ["_" for _ in range(10)]
            index = 1
            if args.use_langid:
                sent = sent[0][1:-1]
                pred = pred[1:-1]
            else:
                sent = sent[0]
            for s, p in zip(sent, pred):
                new_p = []
                for fe in p:
                    if fe != "_":
                        new_p.append(fe)
                p = new_p
                if len(p) == 0:
                    p = "_"
                elif len(p) > 1:
                    p = ";".join(p)
                else:
                    p = p[0]
                info[0], info[1], info[5] = str(index), data_loader.id_to_word[int(s)], p
                fout.write("\t".join(info) + "\n")
                index += 1

                # fout.write(data_loader.id_to_word[int(s)] + " " + data_loader.id_to_tag[g] + " " + data_loader.id_to_tag[p] + "\n")
            fout.write("\n")

def replace_singletons(data_loader, sents, replace_rate):
    new_batch_sents = []
    for sent in sents:
        new_sent = []
        for word in sent:
            if word in data_loader.singleton_words:
                new_sent.append(word if np.random.uniform(0., 1.) > replace_rate else data_loader.word_to_id["<unk>"])
            else:
                new_sent.append(word)
        new_batch_sents.append(new_sent)
    return new_batch_sents

def main(args):
    prefix = args.model_name + "_" + str(uid)
    print("PREFIX: %s" % prefix)
    pos_data_loader = DataLoader(args)

    if args.multilingual:
        sents, char_sents, tgt_tags, discrete_features, bc_features, known_tags, langs, typological_features = [], [], defaultdict(list), [], [], defaultdict(list), [], []
        all_langs = args.langs.split("/")
        print(all_langs)
        for lang in all_langs:
            input_folder = args.input_folder + "/" + "UD_" + pos_data_loader.code_to_lang[lang]  + "//"
            print("Reading files from folder", input_folder)
            train_path = None
            if args.train_path is not None:
                train_path = args.train_path
            else:
                for [path, dir, files] in os.walk(input_folder):
                    for file in files:
                        if file.startswith(pos_data_loader.filestart) and file.endswith("train.conllu"):
                            train_path = input_folder + file
                            break
                    break
            if not os.path.exists(train_path):
                print("Train Feature file not exists", train_path)
                continue
            print("Reading from, ", train_path)
            lang_sents, lang_char_sents, lang_tgt_tags, lang_discrete_features, lang_bc_features, lang_known_tags = pos_data_loader.get_data_set_morph(
                train_path, lang)
            #Concatenating datasets across related languages
            sents += lang_sents
            char_sents += lang_char_sents
            for key, lang_tags in lang_tgt_tags.items():
                tgt_tags[key] += lang_tags
                known_tags[key] += lang_known_tags[key]
            discrete_features += lang_discrete_features
            bc_features += lang_bc_features
            langs += ["<"+lang+">" for _ in range(len(lang_sents))]


        if args.fineTune: #Used for the active learning phase where the pre-trained model is fine-tuned on data acquired for AL
            input_folder = args.input_folder + "/" + "UD_" + pos_data_loader.code_to_lang[args.lang] + "//"
            if args.train_path is not None:
                train_path = args.train_path
            else:
                for [path, dir, files] in os.walk(input_folder):
                    for file in files:
                        if file.startswith(pos_data_loader.filestart) and file.endswith("train.conllu"):
                            train_path = input_folder + file
                            break
                    break

            print("FineTune: Reading from, ", train_path)
            lang_sents, lang_char_sents, lang_tgt_tags, lang_discrete_features, lang_bc_features, lang_known_tags = pos_data_loader.get_data_set_morph(
                train_path, args.lang)

            sents = lang_sents[:args.sent_count]
            char_sents = lang_char_sents[:args.sent_count]
            for key, lang_tags in lang_tgt_tags.items():
                tgt_tags[key] = lang_tags[:args.sent_count]
                known_tags[key] = lang_known_tags[key][:args.sent_count]
            discrete_features = lang_discrete_features[:args.sent_count]
            bc_features = lang_bc_features[:args.sent_count]
            langs = ["<"+args.lang+">" for _ in range(len(sents))]

    print("Data set size (train): %d" % len(sents))
    print("Number of discrete features: ", pos_data_loader.num_feats)

    epoch = bad_counter = updates = tot_example = cum_loss = 0
    cum_semi_sup_loss = 0
    tot_unlabeled_example = 0
    patience = args.patience

    display_freq = 100
    valid_freq = args.valid_freq
    batch_size = args.batch_size
    print("Using Char Birnn Attn model!")
    model = BiRNN_ATTN_CRF_model_CVT(args, pos_data_loader)

    inital_lr = args.init_lr

    if args.fineTune:
        print("Loading pre-trained model!")
        model.load()
        if args.special_lr: #Learning rate for AL experiments
             inital_lr = args.init_lr + (0.001 * len(lang_sents) / args.k )
             inital_lr = min(inital_lr, 0.015)
        else:
             inital_lr = args.init_lr

    print("learning at rate: {0}".format(inital_lr))
    trainer = dy.MomentumSGDTrainer(model.model, inital_lr, 0.9)

    def _check_batch_token(batch, id_to_vocab, fout):
        for line in batch:
            fout.write(" ".join([id_to_vocab[i] for i in line]) + "\n")

    def _check_batch_tags(batch, id2tags, fout):
        for feat, sent_tags in batch.items():
            fout.write("Printing tags for feature: {0}".format( feat))
            for line in sent_tags:
                fout.write(" ".join([id2tags[feat][i] for i in line]) + "\n")

    def _check_batch_char(batch, id_to_vocab):
        for line in batch:
            print([u" ".join([id_to_vocab[c] for c in w]) for w in line])

    lr_decay = args.decay_rate
    valid_history = []
    best_results = [0.0, 0.0, 0.0, 0.0]
    sent_index = [i for i in range(len(sents))]

    while epoch <= args.tot_epochs:
        #Get the batches for unlabeled sentences
        unlabeled_sents, unlabeled_char_sents, unlabeled_tgt_tags, _, _, unlabeled_known_tags= pos_data_loader.get_data_set_morph(args.unlabeled_path, args.lang)
        unlabeled_langs = ["<" + args.lang + ">" for _ in range(len(unlabeled_sents))]
        unlabeled_sent_index = [i for i in range(len(unlabeled_sents))]
        unlabeled_batches = make_bucket_batches(zip(unlabeled_sents, unlabeled_char_sents, unlabeled_langs, unlabeled_sent_index), unlabeled_tgt_tags, unlabeled_known_tags, batch_size)

        batches = make_bucket_batches(
            zip(sents, char_sents, discrete_features, bc_features, langs, sent_index), tgt_tags,known_tags, batch_size)
        supervised_batch_index = 0
        alpha = 1
        #Train the semi-supervised setting
        for unlabeled_b_sents, unlabeled_b_char_sents, unlabeled_b_langs, _,  unlabeled_tgt_tags, unlabeled_known_tags in unlabeled_batches:
            dy.renew_cg()
            token_size = len(unlabeled_b_sents[0])
            unlabeled_lang_batch = []
            for _ in range(len(unlabeled_b_sents)): #Creating language-id list
                unlabeled_lang_batch.append([pos_data_loader.word_to_id[unlabeled_b_langs[0]] for _ in range(token_size)])

            #Training the CVT model
            loss_cvt = model.cal_cvt_loss(unlabeled_b_sents, unlabeled_b_char_sents, unlabeled_known_tags,
                                              training=True)
            loss_cvt = loss_cvt * alpha
            loss_val = loss_cvt.value()
            cum_semi_sup_loss += loss_val * len(unlabeled_b_sents)
            tot_unlabeled_example += len(unlabeled_b_sents)
            updates += 1
            loss_cvt.backward()
            trainer.update()

            #Train the supervised setting
            b_sents, b_char_sents, b_feats, b_bc_feats, b_langs, _, b_tgt_tags, b_known_tags = batches[supervised_batch_index]
            dy.renew_cg()
            if args.replace_unk_rate > 0.0:
                b_sents = replace_singletons(pos_data_loader, b_sents, args.replace_unk_rate)
            token_size = len(b_sents[0])
            lang_batch = []
            for _ in range(len(b_sents)):
                lang_batch.append([pos_data_loader.word_to_id[b_langs[0]] for _ in range(token_size)])

            loss = model.cal_loss(b_sents, b_char_sents, b_tgt_tags, b_feats, b_bc_feats, b_known_tags, langs=lang_batch,
                                  training=True)
            loss_val = loss.value()
            cum_loss += loss_val * len(b_sents)
            tot_example += len(b_sents)
            updates += 1
            loss.backward()
            trainer.update()
            supervised_batch_index +=1

            if supervised_batch_index == len(batches):
                supervised_batch_index = 0
                epoch += 1
                alpha = alpha + 10
                if epoch > 10:
                    alpha = 100


            if updates % display_freq == 0:
                print("Epoch = %d, Updates = %d, Accumulative Loss=%f. Accumulative SSL Loss=%f." % (
                epoch, updates, cum_loss * 1.0 / tot_example, cum_semi_sup_loss * 1.0 / tot_unlabeled_example))

            if updates % valid_freq == 0:
                acc, precision, recall, f1, _, _,_ ,_,_= evaluate(pos_data_loader, args.dev_path, model, args.model_name)
                print(acc, f1)

                if len(valid_history) == 0 or acc > max(valid_history):
                    bad_counter = 0
                    best_results = [acc, precision, recall, f1]
                    if updates > 0:
                        print("Saving the best model so far.......", model.save_to)
                        model.save()
                else:
                    bad_counter += 1
                    if args.lr_decay and bad_counter >= 3 and os.path.exists(args.save_to_path):
                        bad_counter = 0
                        model.load()
                        lr = inital_lr / (1 + epoch * lr_decay)
                        print("Epoch = %d, Learning Rate = %f." % (epoch, lr))
                        trainer = dy.MomentumSGDTrainer(model.model, lr)

                if bad_counter > patience:
                    print("Early stop!")
                    print("Best on validation: acc=%f, prec=%f, recall=%f, f1=%f" % tuple(best_results))
                    model.load_from = args.save_to_path
                    print("Loading best model from", model.load_from)
                    model.load()
                    acc, precision, recall, f1 = finalEval(args, f1, model, pos_data_loader)
                    exit(0)
                valid_history.append(acc)
            if epoch >= args.tot_epochs:
                print("All epochs done of supervised setting")
                break
        epoch += 1

    model.load_from = args.save_to_path
    print("Loading best model from", model.load_from)
    model.load()
    acc, precision, recall, f1 = finalEval(args, f1, model, pos_data_loader)
    print("All Epochs done.")


def finalEval(args, f1, model, pos_data_loader):
    acc, precision, recall, f1, _, _,_ ,_,_= evaluate(pos_data_loader, args.dev_path, model,
                                                args.model_name,
                                                type="dev")
    print("Dev: Acc: {0} Prec: {1} Recall: {2} F1: {3}".format(acc, precision, recall, f1))
    model.word_embedding_weights = []
    testacc, testprecision, testrecall, testf1,  gold_dict, output_dict, score_sentence, gold_token_predictions, token_predictions = evaluate(pos_data_loader,args.test_path,
                                                                model,
                                                                args.model_name,
                                                                type="test")
    print("Test: Acc: {0} Prec: {1} Recall: {2} F1: {3}".format(testacc, testprecision, testrecall, testf1))
    if args.activeLearning:
        if args.al_mode == "cral":
            createAnnotationOutput_mimic_oracle(args, model, pos_data_loader, gold_dict, output_dict)
        else:
            createAnnotationOutput_SPAN_wise(args, model, pos_data_loader,  gold_dict, output_dict, score_sentence)
    return acc, precision, recall, f1



def test_single_model(args):
    pos_data_loader = DataLoader(args)

    model = BiRNN_ATTN_CRF_model_CVT(args, pos_data_loader)
    model.load()
    if args.visualize:
        #printTransitionMatrix(model, pos_data_loader, "POS")
        outputLanguageEmbedding(model, pos_data_loader)

    acc, precision, recall, f1  = finalEval(args, None, model, pos_data_loader)
    np.save(args.model_name, np.array(model.token_embeddings))
    print("Output word embeddings: ",args.model_name + ".npy")


from args import init_config

args = init_config()
from models.model_builder_cvt import *
from activeLearningModels.active_learning import *
import uuid
from dataloaders.data_loader import *

uid = uuid.uuid4().get_hex()[:6]

if __name__ == "__main__":
    if args.mode == "train":
        if args.load_from_path is not None:
            args.load_from_path = args.load_from_path
	else:
            args.load_from_path = args.save_to_path
        main(args)
    elif args.mode == "test":
        test_single_model(args)
    else:
        raise NotImplementedError

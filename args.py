def init_config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynet-mem", default=1000, type=int)
    parser.add_argument("--dynet-seed", default=5783287, type=int)
    parser.add_argument("--dynet-gpu")

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--eval_folder", type=str, default="../eval")
    parser.add_argument("--lang", default=None, help="the target language")
    parser.add_argument("--train_path", default=None, type=str)
    parser.add_argument("--dev_path", default="../datasets/english/eng.dev.bio.conll", type=str)
    parser.add_argument("--test_path", default="../datasets/english/eng.test.bio.conll", type=str)
    parser.add_argument("--save_to_path", default="../saved_models/")
    parser.add_argument("--load_from_path", default=None)

    parser.add_argument("--tag_emb_dim", default=50, type=int)
    parser.add_argument("--pos_emb_dim", default=64, type=int)
    parser.add_argument("--char_emb_dim", default=30, type=int)
    parser.add_argument("--word_emb_dim", default=100, type=int)
    parser.add_argument("--cnn_filter_size", default=30, type=int)
    parser.add_argument("--cnn_win_size", default=3, type=int)
    parser.add_argument("--rnn_type", default="lstm", choices=['lstm', 'gru'], type=str)
    parser.add_argument("--hidden_dim", default=200, type=int, help="token level rnn hidden dim")
    parser.add_argument("--char_hidden_dim", default=25, type=int, help="char level rnn hidden dim")
    parser.add_argument("--layer", default=1, type=int)


    parser.add_argument("--replace_unk_rate", default=0.0, type=float, help="uses when not all words in the test data is covered by the pretrained embedding")
    parser.add_argument("--remove_singleton", default=False, action="store_true")
    parser.add_argument("--map_pretrain", default=False, action="store_true")
    parser.add_argument("--map_dim", default=100, type=int)
    parser.add_argument("--pretrain_fix", default=False, action="store_true")

    parser.add_argument("--output_dropout_rate", default=0.5, type=float, help="dropout applied to the output of birnn before crf")
    parser.add_argument("--emb_dropout_rate", default=0.3, type=float, help="dropout applied to the input of token-level birnn")
    parser.add_argument("--valid_freq", default=500, type=int)
    parser.add_argument("--tot_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--init_lr", default=0.015, type=float)
    parser.add_argument("--lr_decay", default=False, action="store_true")
    parser.add_argument("--special_lr", default=False, action="store_true")
    parser.add_argument("--decay_rate", default=0.05, action="store", type=float)
    parser.add_argument("--patience", default=3, type=int)

    parser.add_argument("--pretrain_emb_path", type=str, default=None)
    parser.add_argument("--feature_birnn_hidden_dim", default=50, type=int, action="store")
    parser.add_argument("--use_discrete_features", default=False, action="store_true", help="indicator features")
    parser.add_argument("--feature_dim", type=int, default=10, help="dimension of discrete features")

    parser.add_argument("--use_brown_cluster", default=False, action="store_true")
    parser.add_argument("--brown_cluster_path", action="store", type=str, help="path to the brown cluster features")
    parser.add_argument("--brown_cluster_num", default=0, type=int, action="store")
    parser.add_argument("--brown_cluster_dim", default=30, type=int, action="store")

    parser.add_argument("--gold_file", default=None, type=str, help="Gold annotations of dev path")
    parser.add_argument("--gold_test_file", default=None, type=str, help="Gold annotations for test path")

    # CRF decoding
    parser.add_argument("--interp_crf_score", default=False, action="store_true", help="if True, interpolate between the transition and emission score.")
    # post process arguments
    parser.add_argument("--label_prop", default=False, action="store_true")
    # Use trained model to test
    parser.add_argument("--mode", default="train", type=str, choices=["train", "test"],
                        help="test: use one model")

    #Using monolithic tagset
    parser.add_argument("--monolithic", action="store_true", default=False)

    #Using a simple classifer
    parser.add_argument("--simple", action="store_true", default=False)

    # Partial CRF
    parser.add_argument("--use_partial", default=False, action="store_true")

    # Active Learning
    parser.add_argument("--ngram", default=2, type=int)
    parser.add_argument("--to_annotate", type=str,default="./annotate.txt")
    parser.add_argument("--entropy_threshold", type=float, default=None)
    parser.add_argument("--use_CFB", default=False, action="store_true")
    parser.add_argument("--sumType",default=False, action="store_true")
    parser.add_argument("--use_similar_label", default=False, action="store_true")
    parser.add_argument("--SPAN_wise", default=False, action="store_true", help="get span wise scores, even if there are duplicates.")
    parser.add_argument("--k", default=200, type=int, help="fixed number of spans to annotate")
    parser.add_argument("--clusters", default=400, type=int, help="fixed number of spans to annotate")
    parser.add_argument("--debug", type=str)
    parser.add_argument("--clusterDetails", type=str, default="temp.txt")
    parser.add_argument("--activeLearning", action="store_true", default=False)
    parser.add_argument("--use_centroid", action="store_true", default=False)
    parser.add_argument("--selectUniq",action="store_true", default=False)
    parser.add_argument("--tokenRep",action="store_true", default=False)
    parser.add_argument("--label", action="store_true", default=False)
    parser.add_argument("--cluster_information", type=str)


    # Format of test output
    parser.add_argument("--test_conll", default=False, action="store_true")
    parser.add_argument("--fixedVocab", default=False, action="store_true", help="for loading pre-trained model")
    parser.add_argument("--fineTune", default=False, action="store_true", help="for loading pre-trained model")
    parser.add_argument("--run",default=0, type=int)

    #Add multiple languages
    parser.add_argument("--input_folder", default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/myNRF/data", type=str)
    parser.add_argument("--pos_folder", default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/myNRF/data/POS_Folder", type=str)

    parser.add_argument("--lang_codes",
                        default="/Users/aditichaudhary/Documents/CMU/Lorelei/LORELEI_NER/utils/lang_codes.txt",
                        type=str)
    parser.add_argument("--langs", type=str, default="en/hi")
    parser.add_argument("--augVocablang", type=str, default=None)
    parser.add_argument("--use_langid", action="store_true", default=False)
    parser.add_argument("--use_token_langid", action="store_true", default=False)
    parser.add_argument("--use_char_attention", action="store_true", default=False)
    parser.add_argument("--use_lang_specific_decoder", action="store_true", default=False)
    parser.add_argument("--multilingual", default=False, action="store_true") #TO use data from from multiple languages, currently supported for sigmorph
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--sent_count", type=int, default=10000000)

    #Normalizing all tokens to lowercase
    parser.add_argument("--normdata", action="store_true", default=False)


    #Add the cross-view training
    parser.add_argument("--cvt", action="store_true", default=False)
    parser.add_argument("--kl", action="store_true", default=False)
    parser.add_argument("--oracle", action="store_true", default=False)
    parser.add_argument("--unlabeled_path",type=str)
    parser.add_argument("--eval_cvt_fwd", action="store_true", default=False)
    parser.add_argument("--eval_cvt_bwd", action="store_true", default=False)
    parser.add_argument("--eval_cvt_future", action="store_true", default=False)
    parser.add_argument("--eval_cvt_back", action="store_true", default=False)
    parser.add_argument("--weightKL", action="store_true", default=False)
    parser.add_argument("--eval_cvt_futback", action="store_true", default=False)
    parser.add_argument("--al_mode", default="kl", choices=["entropy", "naive_kl", "cral", "qbc"], type=str)
    parser.add_argument("--rand", action="store_true", default=False)
    parser.add_argument("--singular", action="store_true", default=False)
    parser.add_argument("--use_sort", action="store_true", default=False)
    parser.add_argument("--entropyInfo", type=str)
    parser.add_argument("--confidence", type=str)


    args = parser.parse_args()
    args.save_to_path = args.save_to_path + args.model_name + ".model"
    print(args)
    return args

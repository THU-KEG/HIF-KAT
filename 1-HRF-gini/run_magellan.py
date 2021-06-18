from utils import *
from repre import *
from reason import *
import torchtext
import dill
import json
import traceback
import argparse
import datetime
import time
# torch.set_printoptions(precision=16)

parser = argparse.ArgumentParser()
parser.add_argument("--co_train", type=bool, default = False, help="Whether to add the geo loss")
parser.add_argument("--tree_load", type=str, default = "0", help="use which name to load the tree")
parser.add_argument("--tree_save", type=str, default = "0", help="use which name to save the tree")
parser.add_argument("--mask_rate", type=float, default = "0.4", help = "Mask rate for attributes before training")

parser.add_argument("--knn", type=int, default = 0, help="KNN graph")
parser.add_argument("--pow", type=float, default = 8., help="Reduce the weight by the power")
parser.add_argument("--res", type=bool, default = False, help="Residual Connection")

parser.add_argument("--rep", type=str, default="gcn2", help="rep")

# dataset information


parser.add_argument("--dataname", type=str, default = "music")
parser.add_argument("--dataset", type=str, default = "itunes-amazon")
parser.add_argument("--train_split", type=float, default = "0.6")
parser.add_argument("--drop_rate", type=int, default = "0")
parser.add_argument("--no", type=int, default = "0")
parser.add_argument("--obligate", type=str, default = "Song_Name")
parser.add_argument("--attr_len", type=int, default = 32)
parser.add_argument("--title_len", type=int, default = 64)
parser.add_argument("--batch_num", type=int, default = 4)


args = parser.parse_args()
print("Co-Training Flag", args.co_train)
print("File name for tree to Load", args.tree_load)
print("File name for tree to Save", args.tree_save)

epoch = 4000
learning_rate = 0.01
eta = 0.1     # 0.1 performs best
early_stop = 500
weight_decay = 1e-5
title_len = args.title_len
attri_len = args.attr_len
running_threshold = 200             # Epoch larger than this number will activate the early stop mechanism
mask_rate = args.mask_rate          # 0.4 is optimal, for now



data_path = "../dataset/magellan_dataset/{}:{}".format(args.dataname, args.dataset)
data_file = "{}".format(args.dataname)

adj_path = "../dataset/magellan_dataset/{}:{}".format(args.dataname, args.dataset)
adj_file = "../dataset/magellan_dataset/{}:{}".format(args.dataname, args.dataset)

obligated_attribute = args.obligate

index_path = data_path
index_file = data_file

train_rate = args.train_split
val_rate = 0.2
test_rate = 1 - train_rate - val_rate


attr_drop_rate = args.drop_rate
data_table_name = "{}.{}".format(attr_drop_rate, data_file)






dataset = Dataset(
    data_path = data_path, 
    data_file = "{}.csv".format(data_table_name),
    title_len = title_len, 
    attri_len = attri_len,

    embedding_type = "fasttext",
    embedding_path = "~/.vector_cache/fasttext",
    embedding_model = "fasttext.wiki.en.300d",

    init_token = "<<<",
    eos_token = ">>>"

    # embedding_type = "tencent",
    # embedding_path = "~/.vector_cache/tencent",
    # embedding_model = "Tencent_AILab_ChineseEmbedding",

    # init_token = "<s>",
    # eos_token = "</s>"
)

pair_maker = PairMaker(
    init_token = dataset.init_token_num,
    end_token = dataset.end_token_num,
    pad_token = dataset.pad_token_num,
    text_fields = dataset.text_attributes,
    obligate_fields = [obligated_attribute,],
    batch_size = len(dataset.examples),
    vocab_size = dataset.vocab_size,
    mask_rate = mask_rate
)

val_pair_maker = PairMaker(
    init_token = dataset.init_token_num,
    end_token = dataset.end_token_num,
    pad_token = dataset.pad_token_num,
    text_fields = dataset.text_attributes,
    obligate_fields = [obligated_attribute, ],
    batch_size = len(dataset.examples),
    vocab_size = dataset.vocab_size,
    mask_rate = 0.0
)

model = RepresentationModel(
    training_fields = [x for x in dataset.text_attributes if x != obligated_attribute],
    title_field = obligated_attribute,
    index_field = "index",
    embedding_weights = dataset.embedding_weight_matrix,
    embedding_trainable = False,
    vocab_size = dataset.vocab_size,
    title_seq_len = title_len,
    other_seq_len = attri_len,
    summarize_para_share = False,
    attention_para_share = False,
    graphconv_para_share = False,
    activation = F.relu,

    # summarize_dimension = [64, ],
    # attention_dimension = [48, ],
    # gcn_dimension = [32, 64]
)

### Add 2020-03-25 ###
### Change The Split ###
interface = Interface(
    attributes = [x for x in dataset.text_attributes if x != obligated_attribute] + [obligated_attribute],
    index_path = index_path,
    index_file = index_file,
    train_rate = train_rate, 
    val_rate = val_rate, 
    test_rate = test_rate,
    perm = False
)

tree = GradientBoostingDecisionTree()
if args.co_train:
    co_train_tree = GradientBoostingDecisionTree()
    co_train_tree.load_model("{}.{}.tree".format(data_file, args.tree_load))
    geo_loss_attribute_weight = co_train_tree.get_attribute_weight(fmap = "feature_map.txt", importance_type = "gain")

dataset_iter = torchtext.data.Iterator(dataset=dataset, batch_size = int(len(dataset.examples) / args.batch_num + 1), shuffle=False, sort_within_batch=False, device="cuda")
adam = torch.optim.Adam(model.parameters(), lr  = learning_rate, weight_decay = weight_decay)


best_val_f1 = 0

patience = 0
test_f1 = 0
test_prec = 0
test_rec = 0

class TData:
    def __init__(self):
        pass

mag_H, mag_train, mag_valid, mag_test = pkl.load(open('../dataset/magellan_dataset/{}/{}.{}.{}.mag.pkl'.format(args.dataname, args.drop_rate, args.train_split, args.dataname), 'rb'))

mag_valid = mag_valid.values[:,:-1]
mag_test = mag_test.values[:,:-1]

mag_train = np.array(mag_train)[:,:-1]
mag_name = list(mag_H.columns)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

for i in range(epoch):
    if patience == early_stop:
        break
    
    reps = {attrx: [] for attrx in dataset.text_attributes}
    val_used_data = {attrx: [] for attrx in dataset.text_attributes}
    whole_train_loss = []
    train_start = time.time()
    for iter_num, dat in enumerate(dataset_iter):
        # for attribute_ in dataset.text_attributes:
        #     getattr(train_used_data, attribute_).append(getattr(dat, attribute_))
        for attrx in val_used_data:
            val_used_data[attrx].append(getattr(dat, attrx))

        bm, bi, bo, = pair_maker.make_pairs(dat, iter_num)

        adam.zero_grad()
        model.train()
        out, loss_dict, mlm_loss, all_output = model(bi, bo, bm)
        rep = all_output[args.rep]

        for attribute_ in dataset.text_attributes:
            reps[attribute_].append(rep[attribute_])

        loss = 0
        loss += mlm_loss

        loss.backward()
        adam.step()
        adam.zero_grad()
        whole_train_loss.append(loss.item())

    rep = {attrx: torch.cat(reps[attrx], 0) for attrx in rep}


    loss_scores, scores, score_name = interface.convert(rep, mode = "train")
    # print(scores.shape, mag_train.shape)
    # scores = mag_train
    scores = np.concatenate((scores, mag_train), axis = 1)
    score_name = score_name + mag_name
    # print(scores.shape)

    print("epoch %5d:%3d, total_loss %2.6f, mlm_loss %2.6f" % (i, patience, np.mean(whole_train_loss), np.mean(whole_train_loss)), end="")



    val_data = TData()
    for attrx in val_used_data:
        # after_cat = torch.cat(val_used_data[attrx], dim = 1)
        setattr(val_data, attrx, torch.cat(val_used_data[attrx], dim=1))


    del val_used_data
    del rep
    del reps

    with torch.no_grad():
        ### Note that val pair maker would not add || MASK || ###
        ### The validation process are mainly prepared for xgboost ###
        bm, bi, bo, = val_pair_maker.make_pairs(val_data, iter_num + 1)
        model.eval()
        ### Representations without dropout ###
        all_output = model(bi, train = False)
        rep = all_output[args.rep]

        loss_scores, scores, score_name = interface.convert(rep, mode = "train")
        # print(scores.shape, mag_train.shape)
        # scores = mag_train
        scores = np.concatenate((scores, mag_train), axis = 1)
        score_name = score_name + mag_name
        # print(scores.shape)

        if i == 0:
            feature_map_file = open("feature_map/{}.{}.map.txt".format(args.dataset, args.dataname), "w+")
            for score_pos, scn in enumerate(score_name):
                print(str(score_pos) + "\t" + str(scn) + "\t" + "q", file = feature_map_file)
            feature_map_file.close()
        
        tree.fit(scores, interface.train_labels.reshape(-1))
        train_f1, train_prec, train_rec = tree.eval(scores, interface.train_labels.reshape(-1))

        train_end = time.time()
        print(" ||", "%.7f" % (train_end - train_start), "train seconds", "||", end = '')

        loss_scores, scores, score_name = interface.convert(rep, mode = "val")
        # print(scores.shape, mag_valid.shape)
        # scores = mag_valid
        scores = np.concatenate((scores, mag_valid), axis = 1)
        score_name = score_name + mag_name
        # print(scores.shape)


        val_f1, val_prec, val_rec = tree.eval(scores, interface.val_labels.reshape(-1))

        print(", train %2.4f%%, val %2.4f%%, best_val %2.4f%%, cur_test %2.4f%%" % (
            train_f1 * 100, 
            val_f1 * 100, 
            best_val_f1 * 100, 
            test_f1 * 100
            ), end="")

        if i > running_threshold and val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = 1

            test_start = time.time()
            loss_scores, scores, score_name = interface.convert(rep, mode = "test")
            # print(scores.shape, mag_test.shape)
            # scores = mag_test
            scores = np.concatenate((scores, mag_test), axis = 1)
            score_name = score_name + mag_name
            # print(scores.shape)

            tree_x, tree_y = scores, interface.test_labels.reshape(-1)
            test_f1, test_prec, test_rec = tree.eval(scores, interface.test_labels.reshape(-1), case = False)

            test_end = time.time()
            print("||", test_end - test_start, "test seconds||")

            # tree.tree.dump_model("dump/{}.{}.{}.{}.dump.txt".format(attr_drop_rate, train_rate, args.dataset, args.dataname), "feature_map/{}.{}.map.txt".format(args.dataset, args.dataname))
            print(", test_f1 %2.4f%%" % (test_f1 * 100))

            # ff = open("dump.txt", "a")
            # print(file = ff)
            # print("test_f1 = %.6f" % test_f1, file = ff)
            # ff.close()

            loss_scores, scores, score_name = interface.convert(rep, mode = "train")
            # print(scores.shape, mag_train.shape)
            # scores = mag_train
            scores = np.concatenate((scores, mag_train), axis = 1)
            score_name = score_name + mag_name
            # print(scores.shape)


            tree_x, tree_y = scores, interface.train_labels.reshape(-1)
            
            tree.fit(scores, interface.train_labels.reshape(-1), depth = 6)
            # tree.tree.save_model("{}.{}.tree".format(data_table_name, args.tree_save))

            # error_pair = [(interface.test_pairs[e[0]], e[1]) for e in error_case]
            # error_output_file = open("error_pair.txt", "w+")
            # for e in error_pair:
            #     print(e[0][0].item(), e[0][1].item(), e[1], file = error_output_file)
            # error_output_file.close()

        elif i > running_threshold:
            patience += 1
            print()
        else:
            print()

# log_file = open("logs/Log_k={}_pow={}_rep={}.txt".format(k_NN, weight_inv, args.rep), "a+")

log_file = open("logs/{},{},{}.txt".format(attr_drop_rate, train_rate, data_file), "a+")
print("test_f1 = %.7f, test_prec = %.7f, test_rec = %.7f @ val_f1 = %.7f" % (test_f1, test_prec, test_rec, best_val_f1), file = log_file)
log_file.close()
print("test_f1 = %.7f, test_prec = %.7f, test_acc = %.7f @ val_f1 = %.7f" % (test_f1, test_prec, test_rec, best_val_f1))
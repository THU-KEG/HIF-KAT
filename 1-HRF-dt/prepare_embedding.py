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


parser.add_argument("--dataname", type=str, default = "toner")
parser.add_argument("--dataset", type=str, default = "toner")
parser.add_argument("--train_split", type=float, default = "0.6")
parser.add_argument("--drop_rate", type=int, default = "0")
parser.add_argument("--no", type=int, default = "0")
parser.add_argument("--obligate", type=str, default = "Song_Name")
parser.add_argument("--attr_len", type=int, default = 32)
parser.add_argument("--batch_num", type=int, default = 4)


args = parser.parse_args()
print("Co-Training Flag", args.co_train)
print("File name for tree to Load", args.tree_load)
print("File name for tree to Save", args.tree_save)

epoch = 4000
learning_rate = 0.01
eta = 0.1     # 0.1 performs best
early_stop = 200
weight_decay = 1e-5
title_len = 64
attri_len = args.attr_len
running_threshold = 100             # Epoch larger than this number will activate the early stop mechanism
mask_rate = args.mask_rate          # 0.4 is optimal, for now

### Graph Hyper-parameters
k_NN = args.knn
weight_inv = args.pow
residual = args.res



data_path = "../dataset/taobao_dataset/{}".format(args.dataname, args.dataset)
data_file = "{}".format(args.dataname)

adj_path = "../dataset/taobao_dataset/{}".format(args.dataname, args.dataset)
adj_file = "../dataset/taobao_dataset/{}".format(args.dataname, args.dataset)

obligated_attribute = args.obligate

index_path = data_path
index_file = data_file

train_rate = args.train_split
val_rate = (1 - train_rate) / 2
test_rate = (1 - train_rate) / 2

attr_drop_rate = args.drop_rate
data_table_name = "{}.{}".format(attr_drop_rate, data_file)






dataset = Dataset(
    data_path = data_path, 
    data_file = "{}.csv".format(data_table_name),
    title_len = title_len, 
    attri_len = attri_len,

    embedding_type = "tencent",
    # embedding_path = "./",
    # embedding_model = "embedding",
    embedding_path = "~/.vector_cache/tencent",
    embedding_model = "Tencent_AILab_ChineseEmbedding",

    init_token = "<s>",
    eos_token = "</s>"

    # embedding_type = "fasttext", 
    # embedding_path = "~/.vector_cache/fasttext",
    # embedding_model = "fasttext.ch.300d",

    # embedding_type = "fasttext",
    # embedding_path = "~/.vector_cache/fasttext",
    # embedding_model = "fasttext.wiki.en.300d"

    # init_token = "<<<"
    # eos_token = ">>>"
)

fetch = FetchTopConfidence(dataset.vocab.itos)

word_embedding = dataset.embedding_weight_matrix.detach().cpu().numpy()
word_embedding_file = open("{}.embedding.txt".format(args.dataname), "w+")
print(word_embedding.shape[0], word_embedding.shape[1], file = word_embedding_file)
# print(1, word_embedding.shape[1], file = word_embedding_file)

# for r in range(1):
for r in range(word_embedding.shape[0]):
    tok = dataset.vocab.itos[r]
    emb = word_embedding[r].tolist()
    print(tok, file=word_embedding_file, end=" ")
    for i, t in enumerate(emb):
        if i != len(emb) - 1:
            print(t, file = word_embedding_file, end=" ")
        else:
            print(t, file = word_embedding_file)

word_embedding_file.close()

from gensim.models import KeyedVectors
wv = KeyedVectors.load_word2vec_format("{}.embedding.txt".format(args.dataname), binary=False)

# for r in range(1, word_embedding.shape[0]):
#     tok = dataset.vocab.itos[r]
#     emb = word_embedding[r].tolist()
#     wv[tok] = emb
print(wv["."][:3])
wv.save("{}.embedding.bin".format(args.dataname))
wv = gensim.models.KeyedVectors.load("{}.embedding.bin".format(args.dataname), mmap = "r")
print(wv["."][:3])

import py_entitymatching as em
import os
import pandas as pd
import numpy as np
from sklearn import metrics
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type = int, default = 1)
parser.add_argument("--dataset", type = str, default = "citation")
parser.add_argument("--drop_rate", type = int, default = 0)
parser.add_argument("--train_rate", type = float, default = 0.01)
args = parser.parse_args()
seed = args.seed

dataset = args.dataset
train_rate = args.train_rate
drop_rate = args.drop_rate



# if dataset == 'dmusic' or dataset == 'music':
#     train_rate = 0.1


model = dict()

path_A = '{}/{}.{}.A.csv'.format(dataset, drop_rate, train_rate)
path_B = '{}/{}.{}.B.csv'.format(dataset, drop_rate, train_rate)
path_labeled_data = '{}/{}.{}.S.csv'.format(dataset, drop_rate, train_rate)

A = em.read_csv_metadata(path_A, key='id')
B = em.read_csv_metadata(path_B, key='id')
# Load the pre-labeled data
S = em.read_csv_metadata(path_labeled_data, key='_id', ltable=A, rtable=B, fk_ltable='ltable_id', fk_rtable='rtable_id')


index = [x for x in range(len(S))]

val_rate = 0.2
test_rate = 1 - train_rate - val_rate

idxs = dict()

idxs["train"] = index[ : int(train_rate * len(index))]
idxs["test"] = index[int(train_rate * len(index)) : int((train_rate + test_rate) * len(index))]
idxs["val"] = index[int((train_rate + test_rate) * len(index)) : ]





F = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)
# F = F.iloc[4:]
print(F)

H = em.extract_feature_vecs(S, feature_table = F, attrs_after = 'label', show_progress = False)
H = H.astype(np.float64)
H = H.fillna(H.mean())
# H = em.impute_table(H, exclude_attrs = ['_id', 'ltable_id', 'rtable_id', 'label'], strategy = 'most_frequent')

train = H.iloc[idxs['train']]
valid = H.iloc[idxs['val']]
test = H.iloc[idxs['test']]

pkl.dump((H, train, valid, test), open('{}/{}.{}.{}.mag.pkl'.format(dataset, drop_rate, train_rate, dataset), 'wb+'))

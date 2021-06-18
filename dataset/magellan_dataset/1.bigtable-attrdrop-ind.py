import pandas as pd
import numpy as np
import pickle as pkl
import random

drop_rate_max = 1

# dataset = "music:itunes-amazon"
# dataname = "music"
# obligated_attribute = "Song_Name"

# dataset = "citation:dblp-scholar"
# dataname = "citation"
# obligated_attribute = "title"

# dataset = "citeacm:dblp-acm"
# dataname = "citeacm"
# obligated_attribute = "title"



# dataset = "dmusic:itunes-amazon"
# dataname = "dmusic"
# obligated_attribute = "Song_Name"

# dataset = "dcitation:dblp-scholar"
# dataname = "dcitation"
# obligated_attribute = "title"

dataset = "dciteacm:dblp-acm"
dataname = "dciteacm"
obligated_attribute = "title"





tableA = pd.read_csv("{}/tableA.csv".format(dataset))
tableB = pd.read_csv("{}/tableB.csv".format(dataset))

id_pos = [x for x in tableA.columns].index("id")
tableA_hash = {
    row[id_pos]: row for row in tableA.values
}
tableB_hash = {
    row[id_pos]: row for row in tableB.values
}
Aid2ind = dict()
Bid2ind = dict()
ind2Aid = dict()
ind2Bid = dict()
cur_ind = 0

for col in tableA.values:
    col_id = col[id_pos]
    Aid2ind[col_id] = cur_ind
    ind2Aid[cur_ind] = col_id
    cur_ind += 1
for col in tableB.values:
    col_id = col[id_pos]
    Bid2ind[col_id] = cur_ind
    ind2Aid[cur_ind] = col_id
    cur_ind += 1

table_hash = dict()
for key in tableA_hash:
    col = tableA_hash[key]
    key = Aid2ind[key]
    table_hash[key] = col
for key in tableB_hash:
    col = tableB_hash[key]
    key = Bid2ind[key]
    table_hash[key] = col


train_table = pd.read_csv("{}/train.csv".format(dataset))[["ltable_id", "rtable_id", "label"]]
val_table = pd.read_csv("{}/valid.csv".format(dataset))[["ltable_id", "rtable_id", "label"]]
test_table = pd.read_csv("{}/test.csv".format(dataset))[["ltable_id", "rtable_id", "label"]]

train_pair_inds = [(Aid2ind[triples[0]], Bid2ind[triples[1]], triples[2]) for triples in train_table.values]
val_pair_inds = [(Aid2ind[triples[0]], Bid2ind[triples[1]], triples[2]) for triples in val_table.values]
test_pair_inds = [(Aid2ind[triples[0]], Bid2ind[triples[1]], triples[2]) for triples in test_table.values]
full_inds = train_pair_inds + val_pair_inds + test_pair_inds

pkl.dump(full_inds, open("{}/{}.pair.pkl".format(dataset, dataname), "wb+"))

supervised_ind = set([x[0] for x in full_inds] + [x[1] for x in full_inds])

ori_ind_ct = 0
cur_ind_ct = 0
ori_ind2cur_ind = dict()
def sample(x):
    global ori_ind_ct
    global cur_ind_ct
    global ori_ind2cur_ind
    # if x in supervised_ind or random.randint(0, 20) == 0:
    if x in supervised_ind:
        ori_ind2cur_ind[ori_ind_ct] = cur_ind_ct
        ori_ind_ct += 1
        cur_ind_ct += 1
        return True
    else:
        ori_ind_ct += 1
        return False

training_table = np.array([table_hash[r] for r in range(cur_ind) if sample(r)])
training_table = pd.DataFrame(training_table, columns = tableA.columns)
training_table.to_csv("{}/{}.csv".format(dataset, dataname), index = True, index_label = "index")

new_full_ind = [(ori_ind2cur_ind[x[0]], ori_ind2cur_ind[x[1]], x[2]) for x in full_inds]
pkl.dump(new_full_ind, open("{}/{}.pair.pkl".format(dataset, dataname), "wb+"))



def make_drop(x, attr, rate):
    global obligated_attribute
    if attr == obligated_attribute:
        return x[attr]
    elif random.random() < rate:
        return None
    else:
        return x[attr]

for droprate in range(0, drop_rate_max):
    rate = droprate / 10
    drop_table = training_table.copy()
    for att in drop_table.columns:
        drop_table[att] = drop_table.apply(lambda x: make_drop(x, att, rate), axis = 1)
    drop_table.to_csv("{}/{}.{}.csv".format(dataset, droprate, dataname), index = True, index_label = "index")
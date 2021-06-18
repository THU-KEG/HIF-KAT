import pandas as pd
import numpy as np
import pickle as pkl
import os

# dataset = "music:itunes-amazon"
# dataname = "music"
# train_rate_range = [0.6, 0.1]

# dataset = "citation:dblp-scholar"
# dataname = "citation"
# train_rate_range = [0.6, 0.01]

# dataset = "citeacm:dblp-acm"
# dataname = "citeacm"
# train_rate_range = [0.6, 0.01]




# dataset = "dmusic:itunes-amazon"
# dataname = "dmusic"
# train_rate_range = [0.6, 0.1]

# dataset = "dcitation:dblp-scholar"
# dataname = "dcitation"
# train_rate_range = [0.6, 0.01]

dataset = "dciteacm:dblp-acm"
dataname = "dciteacm"
train_rate_range = [0.6, 0.01]




index = pkl.load(open("{}/{}.pair.pkl".format(dataset, dataname), "rb"))

for drop_rate in range(0, 1):
    big_table_data_frame = pd.read_csv("{}/{}.{}.csv".format(dataset, drop_rate, dataname))
    
    print(train_rate_range)
    for train_rate in train_rate_range:

        val_rate = 0.2
        test_rate = 1 - train_rate - val_rate

        columns = ["label",] + ["left_" + x for x in big_table_data_frame.columns] + ["right_" + x for x in big_table_data_frame.columns]
        idxs = dict()

        idxs["train"] = index[ : int(train_rate * len(index))]
        idxs["test"] = index[int(train_rate * len(index)) : int((train_rate + test_rate) * len(index))]
        idxs["val"] = index[int((train_rate + test_rate) * len(index)) : ]

        idx_left = dict()
        idx_right = dict()
        labels = dict()

        for split in ["train", "val", "test"]:
            idx_left[split] = [x[0] for x in idxs[split]]
            idx_right[split] = [x[1] for x in idxs[split]]
            labels[split] = [x[2] for x in idxs[split]]

        data_values = big_table_data_frame.values
        left_data_values = dict()
        right_data_values = dict()
        label_values = dict()

        for split in ["train", "val", "test"]:
            left_data_values[split] = data_values[idx_left[split]]
            right_data_values[split] = data_values[idx_right[split]]
            label_values[split] = np.array(labels[split]).reshape((-1, 1)).astype(np.int)

            pair_data = np.concatenate((label_values[split], left_data_values[split], right_data_values[split]), axis=1)
            pair_data_frame = pd.DataFrame(pair_data, columns = columns)

            pair_data_frame.to_csv("{}/{},{},{},{}.csv".format(dataset, drop_rate, train_rate, dataname, split), index = True, index_label = "id")
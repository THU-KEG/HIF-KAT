import pandas as pd
import pickle as pkl
import numpy as np




# dataset = 'music'
# dataname = 'itunes-amazon'
# train_rate = 0.1

# dataset = 'citation'
# dataname = 'dblp-scholar'
# train_rate = 0.01

# dataset = 'citeacm'
# dataname = 'dblp-acm'
# train_rate = 0.01

# dataset = 'dmusic'
# dataname = 'itunes-amazon'
# train_rate = 0.1

# dataset = 'dcitation'
# dataname = 'dblp-scholar'
# train_rate = 0.01

dataset = 'dciteacm'
dataname = 'dblp-acm'
train_rate = 0.01


train_rate = 0.6            # Fully Supervised


drop_rate = 0

big_table_data_frame = pd.read_csv('{}:{}/{}.{}.csv'.format(dataset, dataname, drop_rate, dataset))
big_table_data_frame = big_table_data_frame[[x for x in big_table_data_frame.columns[:1]] + [x for x in big_table_data_frame.columns[2:]]]

print(big_table_data_frame.columns)

index = pkl.load(open('{}:{}/{}.pair.pkl'.format(dataset, dataname, dataset), 'rb'))

columns = ["ltable_" + x for x in big_table_data_frame.columns] + ["rtable_" + x for x in big_table_data_frame.columns] + ["label",]

print(columns)

val_rate = 0.2
test_rate = 1 - train_rate - val_rate



idx_left = [x[0] for x in index]
idx_right = [x[1] for x in index]
labels = [x[2] for x in index]

data_values = big_table_data_frame.values
left_data_values = dict()
right_data_values = dict()
label_values = dict()

left_data_values = data_values[idx_left]
right_data_values = data_values[idx_right]
label_values = np.array(labels).reshape((-1, 1)).astype(np.int)

pair_data = np.concatenate((left_data_values, right_data_values, label_values), axis=1)
pair_data_frame = pd.DataFrame(pair_data, columns = columns)

print(pair_data_frame.head(5))





A_idx = sorted(list(set(idx_left)))
B_idx = sorted(list(set(idx_right)))
A_values = data_values[A_idx]
B_values = data_values[B_idx]
A_data_frame = pd.DataFrame(A_values, columns = big_table_data_frame.columns)
B_data_frame = pd.DataFrame(B_values, columns = big_table_data_frame.columns)

print(A_data_frame.head(5))
print(B_data_frame.head(5))

A_data_frame = A_data_frame.rename(columns = {'index' : 'id'})
B_data_frame = B_data_frame.rename(columns = {'index' : 'id'})
pair_data_frame = pair_data_frame.rename(columns =  {
    'ltable_index': 'ltable_id',
    'rtable_index': 'rtable_id'
})

print(A_data_frame.columns)

pair_data_frame.to_csv("{}/{}.{}.S.csv".format(dataset, drop_rate, train_rate), index = True, index_label = "_id")
A_data_frame.to_csv("{}/{}.{}.A.csv".format(dataset, drop_rate, train_rate), index = False)
B_data_frame.to_csv("{}/{}.{}.B.csv".format(dataset, drop_rate, train_rate), index = False)
import pickle as pkl

# dataset = "itunes-amazon"
# dataname = "music"
# train_rate_range = [x / 20 for x in range(12, 1, -1)]

dataname = "citation"
dataset = "dblp-scholar"
train_rate_range = [x / 200 for x in range(5, 0, -1)]




index = pkl.load(open("{}:{}/{}.pair.pkl".format(dataname, dataset, dataname), "rb"))

for train_rate in train_rate_range:
    val_rate = 0.2
    test_rate = 1 - train_rate - val_rate

    idxs = dict()
    idxs["train"] = index[ : int(train_rate * len(index))]
    idxs["test"] = index[int(train_rate * len(index)) : int((train_rate + test_rate) * len(index))]
    idxs["val"] = index[int((train_rate + test_rate) * len(index)) : ]
    
    train_pos = 0
    train_neg = 0

    for x in idxs["train"]:
        if x[2]:
            train_pos += 1
        else:
            train_neg += 1

    val_pos = 0
    val_neg = 0

    for x in idxs["val"]:
        if x[2]:
            val_pos += 1
        else:
            val_neg += 1

    test_pos = 0
    test_neg = 0


    for x in idxs["test"]:
        if x[2]:
            test_pos += 1
        else:
            test_neg += 1

    total = train_pos + train_neg + val_pos + val_neg + test_pos + test_neg

    
    print("train_rate = %.4f, train_pos = %d, train_neg = %d, val_pos = %d, val_neg = %d, test_pos = %d, test_neg = %d, total = %d" % (train_rate, train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, total))

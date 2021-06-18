import numpy as np
import pandas as pd
import scipy.sparse as sp
import gensim
import os
import pickle as pkl
import copy
from tqdm import tqdm
from multiprocessing import Pool

word_embedding = None

def cal_dis(idcs_range_titles):
    global word_embedding
    i = idcs_range_titles[0]
    rg = idcs_range_titles[1]
    titles = idcs_range_titles[2]

    dis = np.zeros(rg)
    for j in range(rg):
        dis[j] = word_embedding.wmdistance(titles[i], titles[j])
    return i, dis

def createGraph(
    file_path="../dataset/taobao_raw/lipstick.csv",
    index = "商品id",
    title = "商品标题"
):
    data_frame = pd.read_csv(file_path)
    data_frame = data_frame[[index, title]]
    global word_embedding
    if word_embedding is None:
        print("Loading Word Embedding")
        embedding_file = os.path.expanduser("~/.vector_cache/Tencent_AILab_ChineseEmbedding.bin")
        word_embedding = gensim.models.KeyedVectors.load(embedding_file, mmap='r')
        print("Loading Word Embedding Complete")
    node_id = data_frame[index].values
    titles = data_frame[title].values.tolist()

    adj = np.zeros([node_id.shape[0], node_id.shape[0]])
    idcs_range_titles = [(x, copy.deepcopy(int(node_id.shape[0])), copy.deepcopy(titles)) for x in range(node_id.shape[0])]

    print(len(idcs_range_titles))
    pool = Pool(processes=64)
    for i, dis in tqdm(pool.imap(cal_dis, idcs_range_titles, chunksize = 2)):
        adj[i] = dis
    pool.close()
    
    adj_cache_file = file_path.split("/")[-1].split(".")[0]
    adj_cache_file = open("adj/" + adj_cache_file + ".adj.pkl", "wb+")
    pkl.dump(adj, adj_cache_file)
    adj_cache_file.close()
    return node_id, adj

def loadGraph(
    adj_path = "./adj",
    data_path = None,
    dataset = "mobile", 
    index = "商品id",
    k_nearest = 10, 
    inv_max_sim = 2
):
    if k_nearest > 0:
        adj_file_path = dataset + ".adj.pkl"
        adj_file_path = os.path.join(adj_path, adj_file_path)
        assert os.path.isfile(adj_file_path)
        assert adj_path is not None

        # data_frame = pd.read_csv("../dataset/taobao_raw/{}.csv".format(dataset))
        dataset_path = os.path.join(data_path, "{}.csv".format(dataset))
        data_frame = pd.read_csv(dataset_path)

        index = "商品id" if index is None else index

        data_frame = data_frame[[index,]]
        node_id = data_frame[index].values.tolist()
        adj = pkl.load(open(adj_file_path, "rb"))

        adj = 1 / adj
        adj[np.isinf(adj)] = 0.0
        adj = adj / np.max(adj)
        adj = np.power(adj, inv_max_sim)

        sort_ind = np.argsort(adj, axis = 1)
        adj[np.less(sort_ind, adj.shape[0] - k_nearest)] = 0.0
    
    
        sort_ind = sort_ind[:, -k_nearest : ]
        rows = np.array([[x,] * k_nearest for x in range(adj.shape[0])]).flatten()
        cols = sort_ind.flatten()
        poss = [rows[x] * adj.shape[0] + cols[x] for x in range(len(rows))]
        vals = adj.flatten()[poss]
        indx = np.vstack((rows, cols))
        print("Index size, ", indx.shape)

        vals = np.ones_like(vals)
        return node_id, adj, indx, vals
    else:
        rows = np.array([])
        cols = np.array([])
        indx = np.vstack((rows, cols))
        vals = np.array([])
        print("Index size is 0")

        vals = np.ones_like(vals)
        return None, None, indx, vals
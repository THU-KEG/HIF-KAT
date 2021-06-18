"""
1. Convert the vector representation into numpy array, which can be fed into xgboost 
2. Using the tree structure of xgboost to calculate the loss for representation model
"""
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import random

from utils import pearsonCoef

class Interface():
    def __init__(
        self,
        attributes,
        index_path = "../dataset/taobao_raw",
        index_file = "mobile",
        train_rate = 0.6,
        val_rate = 0.16,
        test_rate = 0.24,
        perm = False,
        device = "cuda"
    ):
        self.attributes = attributes
        total_rate = train_rate + val_rate + test_rate
        train_rate = train_rate / total_rate
        val_rate = val_rate / total_rate
        test_rate = test_rate / total_rate
        index_path = os.path.join(index_path, "{}.pair.pkl".format(index_file))

        indices = pkl.load(open(index_path, "rb"))
        if perm == True:
            random.shuffle(indices)

        labels = [x[2] for x in indices]
        pairs = [[x[0], x[1]] for x in indices]
        
        # self.train_labels = np.array(labels[ : int(train_rate * len(labels))])
        # self.val_labels = np.array(labels[int(train_rate * len(labels)) : int((train_rate + val_rate) * len(labels))])
        # self.test_labels = np.array(labels[int((train_rate + val_rate) * len(labels)) : ])

        self.train_labels = np.array(labels[ : int(train_rate * len(labels))])
        self.test_labels = np.array(labels[int(train_rate * len(pairs)) : int((train_rate + test_rate) * len(pairs))])
        self.val_labels = np.array(labels[int((train_rate + test_rate) * len(pairs)) : ])
        

        self.__pos_train_labels_tensor = torch.FloatTensor(self.train_labels)
        self.__pos_val_labels_tensor = torch.FloatTensor(self.val_labels)
        self.__pos_test_labels_tensor = torch.FloatTensor(self.test_labels)

        self.__neg_train_labels_tensor = torch.FloatTensor(self.train_labels)
        self.__neg_train_labels_tensor[self.train_labels == 0] = 1
        self.__neg_train_labels_tensor[self.train_labels == 1] = 0
        self.__neg_val_labels_tensor = torch.FloatTensor(self.val_labels)
        self.__neg_val_labels_tensor[self.val_labels == 0] = 1
        self.__neg_val_labels_tensor[self.val_labels == 1] = 0
        self.__neg_test_labels_tensor = torch.FloatTensor(self.test_labels)
        self.__neg_test_labels_tensor[self.test_labels == 0] = 1
        self.__neg_test_labels_tensor[self.test_labels == 1] = 0


        # self.train_pairs = np.array(pairs[ : int(train_rate * len(pairs))])
        # self.val_pairs = np.array(pairs[int(train_rate * len(pairs)) : int((train_rate + val_rate) * len(pairs))])
        # self.test_pairs = np.array(pairs[int((train_rate + val_rate) * len(pairs)) : ])

        self.train_pairs = np.array(pairs[ : int(train_rate * len(pairs))])
        self.test_pairs = np.array(pairs[int(train_rate * len(pairs)) : int((train_rate + test_rate) * len(pairs))])
        self.val_pairs = np.array(pairs[int((train_rate + test_rate) * len(pairs)) : ])

        self.train_pairs = torch.LongTensor(self.train_pairs)
        self.val_pairs = torch.LongTensor(self.val_pairs)
        self.test_pairs = torch.LongTensor(self.test_pairs)

        if "cuda" in device:
            self.__pos_train_labels_tensor = self.__pos_train_labels_tensor.cuda()
            self.__pos_val_labels_tensor = self.__pos_val_labels_tensor.cuda()
            self.__pos_test_labels_tensor = self.__pos_test_labels_tensor.cuda()

            self.__neg_train_labels_tensor = self.__neg_train_labels_tensor.cuda()
            self.__neg_val_labels_tensor = self.__neg_val_labels_tensor.cuda()
            self.__neg_test_labels_tensor = self.__neg_test_labels_tensor.cuda()

            self.train_pairs = self.train_pairs.cuda()
            self.val_pairs = self.val_pairs.cuda()
            self.test_pairs = self.test_pairs.cuda()

        self.__pos_labels_tensor = {
            "train": self.__pos_train_labels_tensor,
            "val": self.__pos_val_labels_tensor,
            "test": self.__pos_test_labels_tensor
        }
        self.__neg_labels_tensor = {
            "train": self.__neg_train_labels_tensor,
            "val": self.__neg_val_labels_tensor,
            "test": self.__neg_test_labels_tensor
        }



    def convert(self, reprentations, mode = "train"):
        ### TODO Need to add Pearson Correlation ###
        ### DONE ###
        scores = []
        score_name = []

        for attribute in self.attributes:
            repre = reprentations[attribute]
            # lab = getattr(self, "{}_labels".format(mode))
            ind = getattr(self, "{}_pairs".format(mode))
            
            left_ind = ind[:,0].view(-1)
            right_ind = ind[:,1].view(-1)

            left = repre[left_ind]
            right = repre[right_ind]

            dist = F.pairwise_distance(left, right)
            cosine = torch.cosine_similarity(left, right, dim = 1)
            pearson = pearsonCoef(left, right)

            score_name.append("{}-dist".format(attribute))
            scores.append(dist.view(-1, 1))
            score_name.append("{}-cosine".format(attribute))
            scores.append(cosine.view(-1, 1))
            score_name.append("{}-pearson".format(attribute))
            scores.append(pearson.view(-1, 1))

        scores = torch.stack(scores, dim = 1).squeeze()
        flat_scores = scores.detach().cpu().numpy()
        return scores, flat_scores, score_name

    def geoLoss(self, attr_weight, scores, score_name, mode = "train"):
        assert mode in ["train", "val", "test"]
        loss = 0

        xxx = [(xx, attr_weight[xx]) for xx in attr_weight]
        xxx = sorted(xxx, key=lambda x: -x[1])
        xx = xxx[0][0]

        for attribute in attr_weight:
            idx = score_name.index(attribute + "-pearson")
            score = scores[:, idx].view(-1)

            pos_label = self.__pos_labels_tensor[mode].view(-1)
            neg_label = self.__neg_labels_tensor[mode].view(-1)

            pos_score = (-score) * pos_label
            neg_score = (score) * neg_label
            
            final_score = pos_label + neg_score
            attr_loss = torch.mean(final_score) * attr_weight[attribute]

            loss += attr_loss

        loss.register_hook(hook_fn)

        return loss

def hook_fn(grad):
    pkl.dump(grad, open("debug_geo.pkl", "wb+"))
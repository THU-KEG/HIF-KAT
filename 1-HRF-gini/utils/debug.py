import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class FetchTopConfidence():
    def __init__(self, itos):
        self.itos = itos

    def fetch_tensor(self, tensor, k= 8):
        tensor = tensor.view(-1).detach().cpu().numpy()

        index = [x for x in range(tensor.shape[0])]
        index = sorted(index, key = lambda x: -tensor[x])

        for i in range(k):
            ind = index[i]
            val = tensor[ind]
            tok = self.itos[ind]
            end = " " if i != k - 1 else "\n"
            print("%5s, %.5f" % (tok, val), end = end)

    def fetch_tok(self, tok):
        tok = tok.detach().cpu().numpy()

        for t in tok:
            s = self.itos[t]
            if s not in ["<s>", "</s>", "<pad>", "<unk>"]:
                print("%4s" % s, end=" ")
        print()
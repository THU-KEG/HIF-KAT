import torch
import torch.nn as nn
import torch.nn.functional as F

def pearsonCoef(x1, x2, clip = 1e-12):
    x1v = x1 - torch.mean(x1, 1).view(-1, 1)
    x2v = x2 - torch.mean(x2, 1).view(-1, 1)

    score = torch.sum(x1v * x2v, dim = 1).view(-1, 1) / (
                torch.sqrt(torch.sum(x1v ** 2, dim = 1).view(-1, 1) + clip) *
                torch.sqrt(torch.sum(x2v ** 2, dim = 1).view(-1, 1) + clip)
            )

    return score

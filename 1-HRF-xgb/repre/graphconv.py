import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv

class GraphConvLayer(nn.Module):
    def __init__(
        self,
        adj_index,
        adj_weight, 
        input_dim = 32,
        output_dim = 16,
        dropout = 0.4,
        activation = F.leaky_relu,
        device = "cuda",
        text_field = None,
        share = True,
        bias = False,
        residual = False
    ):
        assert text_field is not None
        super(GraphConvLayer, self).__init__()
        self.share = share
        self.residual = residual
        self.fields = text_field[:]
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.adj_index = adj_index
        self.adj_weight = adj_weight

        self.gcns = nn.ModuleDict()

        if residual:
            self.reslin = nn.ModuleDict()

        if share:
            gcn = GCNConv(input_dim, output_dim, cached = False, bias = bias, normalize = True, improved = True)
            if residual:
                lin = nn.Linear(input_dim, output_dim, bias = bias)
            for field in self.fields:
                self.gcns[field] = gcn
                if residual:
                    self.reslin[field] = lin
        else:
            for field in self.fields:
                self.gcns[field] = GCNConv(
                    input_dim, 
                    output_dim, 
                    cached = False, 
                    bias = bias, 
                    normalize = True, 
                    improved = True
                )
                if residual:
                    self.reslin[field] = nn.Linear(input_dim, output_dim, bias = bias)

        if "cuda" in device:
            self.cuda()

    def forward(self, batch):
        ret = dict()
        for field in self.fields:
            rep = batch[field]
            ### Add Dropout 2020-03-25 ### 
            # rep = self.dropout(rep)
            ###

            rep = self.gcns[field](rep, self.adj_index, self.adj_weight)
            if self.residual:
                rep += self.reslin[field](batch[field])
            rep = self.activation(rep)
            ret[field] = rep

        return ret
        # return {field: self.activation(self.gcns[field](batch[field], self.adj_index, self.adj_weight)) for field in self.fields}









class MLP(nn.Module):
    def __init__(
        self,
        input_dim = 32,
        output_dim = 16,
        dropout = 0.4,
        activation = F.leaky_relu,
        device = "cuda",
        text_field = None,
        bias = False
    ):
        assert text_field is not None
        super(MLP, self).__init__()
        self.fields = text_field[:]
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.mlps = nn.ModuleDict()



        for field in self.fields:
            self.mlps[field] = torch.nn.Linear(
                in_features = input_dim,
                out_features = output_dim,
                bias = bias
            )

        if "cuda" in device:
            self.cuda()

    def forward(self, batch):
        ret = dict()
        for field in self.fields:
            rep = batch[field]
            ### Add Dropout 2020-03-25 ### 
            # rep = self.dropout(rep)
            ###

            rep = self.mlps[field](rep)
            rep = self.activation(rep)
            ret[field] = rep

        return ret




class GraphAtteLayer(nn.Module):
    def __init__(
        self,
        adj_index,
        adj_weight, 
        input_dim = 32,
        output_dim = 16,
        head_number = 8,                    ### New
        dropout = 0.4,
        activation = F.leaky_relu,
        device = "cuda",
        text_field = None,
        share = True,
        bias = False,
        residual = False,
        concat = False
    ):
        assert text_field is not None
        super(GraphAtteLayer, self).__init__()
        self.share = share
        self.residual = residual
        self.fields = text_field[:]
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.adj_index = adj_index
        self.adj_weight = adj_weight

        self.gats = nn.ModuleDict()

        if share:
            pass

        else:
            for field in self.fields:
                self.gats[field] = GATConv(
                    input_dim,
                    output_dim,
                    head_number,
                    concat = concat,
                    dropout = dropout
                )
        
        if "cuda" in device:
            self.cuda()

    def forward(self, batch):
        ret = dict()
        for field in self.fields:
            rep = batch[field]
            rep = self.gats[field](rep, self.adj_index)
            rep = self.activation(rep)
            ret[field] = rep

        return ret








class HighWay(torch.nn.Module):
    def __init__(self, f_in, f_out, bias=True):
        super(HighWay, self).__init__()
        self.w = Parameter(torch.Tensor(f_in, f_out))
        nn.init.xavier_uniform_(self.w)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, ori_input, in_1, in_2):
        t = torch.mm(ori_input, self.w)
        if self.bias is not None:
            t = t + self.bias
        gate = torch.sigmoid(t)
        return gate * in_2 + (1.0 - gate) * in_1



class GraphConvHighWayLayer(nn.Module):
    def __init__(
        self,
        adj_index,
        adj_weight, 
        input_dim = 32,
        output_dim = 16,
        dropout = 0.4,
        activation = F.leaky_relu,
        device = "cuda",
        text_field = None,
        share = True,
        bias = False,
        residual = False
    ):
        assert text_field is not None
        super(GraphConvHighWayLayer, self).__init__()
        self.share = share
        self.residual = residual
        self.fields = text_field[:]
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.adj_index = adj_index
        self.adj_weight = adj_weight

        self.gcns = nn.ModuleDict()
        self.reslin = nn.ModuleDict()
        self.highway_net = nn.ModuleDict()


        for field in self.fields:
            self.gcns[field] = GCNConv(
                input_dim, 
                output_dim, 
                cached = True, 
                bias = bias, 
                normalize = True, 
                improved = True
            )
            self.reslin[field] = nn.Linear(input_dim, output_dim, bias = bias)
            self.highway_net[field] = HighWay(f_in = input_dim, f_out = output_dim, bias = bias)


        if "cuda" in device:
            self.cuda()

    def forward(self, batch):
        ret = dict()
        for field in self.fields:
            ori_rep = batch[field]
            ### Add Dropout 2020-03-25 ### 
            # rep = self.dropout(rep)
            ###
            gcn_rep = self.gcns[field](rep, self.adj_index, self.adj_weight)
            res_rep += self.reslin[field](ori_rep)
            
            rep = self.highway_net[field](ori_rep, res_rep, gcn_rep)
            rep = self.activation(rep)
            ret[field] = rep

        return ret
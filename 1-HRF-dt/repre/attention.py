import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class AttentionLayer(nn.Module):
    def __init__(
        self,
        input_dim = 64,
        output_dim = 32,
        q_dim = 16,
        k_dim = 16,
        self_drop_rate = 0.5,
        activation = F.leaky_relu,
        share = True,
        text_field = None,
        residual = True,
        device = "cuda",
    ):
        assert text_field is not None
        assert q_dim == k_dim
        self.residual = residual

        v_dim = int(output_dim / 2) if residual else int(output_dim)
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        super(AttentionLayer, self).__init__()
        self.self_drop_rate = self_drop_rate
        self.share = share
        self.fields = [x for x in text_field]
        self.dropout = nn.Dropout(self.self_drop_rate)
        self.activation = activation

        if share:
            self.wq = torch.nn.parameter.Parameter(torch.Tensor(input_dim, q_dim))
            nn.init.kaiming_uniform_(self.wq, a = math.sqrt(5))
            self.wk = torch.nn.parameter.Parameter(torch.Tensor(input_dim, k_dim))
            nn.init.kaiming_uniform_(self.wk, a = math.sqrt(5))
            self.wv = torch.nn.parameter.Parameter(torch.Tensor(input_dim, v_dim))
            nn.init.kaiming_uniform_(self.wv, a = math.sqrt(5))

        else:
            self.wq = nn.ParameterList()
            self.wk = nn.ParameterList()
            self.wv = nn.ParameterList()

            for field in self.fields:
                wq = torch.nn.parameter.Parameter(torch.Tensor(input_dim, q_dim))
                nn.init.kaiming_uniform_(wq, a = math.sqrt(5))
                self.wq.append(wq)
                wk = torch.nn.parameter.Parameter(torch.Tensor(input_dim, k_dim))
                nn.init.kaiming_uniform_(wk, a = math.sqrt(5))
                self.wk.append(wk)
                wv = torch.nn.parameter.Parameter(torch.Tensor(input_dim, v_dim))
                nn.init.kaiming_uniform_(wv, a = math.sqrt(5))
                self.wv.append(wv)

        self.mask = torch.ones((len(text_field), len(text_field)))
        ### The diagonal is a large negative number ###
        ### Multiply the mask matrix with any attention coefficents after softmax will output zero for diagonal elements ###
        neg = -1e12 * torch.eye(len(text_field))
        self.mask = self.mask + neg

        if "cuda" in device:
            self.cuda()
            self.mask = self.mask.cuda()
                

    def forward(self, batch):
        input_tensor = [batch[field] for field in self.fields]
        # (Attributes, Batch, Hidden)
        input_tensor = torch.stack(input_tensor)

        ### TODO optimize the calculation flow here ###
        ### Maybe we can move the stacking operation to the initial stage ###
        if not self.share:
            # (Attributes, Hidden, q/k/v dim)
            wqs = torch.stack(tuple(self.wq))
            wks = torch.stack(tuple(self.wk))
            wvs = torch.stack(tuple(self.wv))

            # (Attributes, Batch, q/k/v dim)
            query_tensor = torch.bmm(input_tensor, wqs)
            key_tensor = torch.bmm(input_tensor, wks)
            value_tensor = torch.bmm(input_tensor, wvs)
        else:
            query_tensor = torch.matmul(input_tensor, self.wq)
            key_tensor = torch.matmul(input_tensor, self.wk)
            value_tensor = torch.matmul(input_tensor, self.wv)

        # (Batch, Attributes, Que dim)
        query_tensor = query_tensor.permute(1, 0, 2)
        # (Batch, Key dim, Attributes)
        key_tensor = key_tensor.permute(1, 2, 0)
        # (Batch, Attributes, Val dim)
        value_tensor = value_tensor.permute(1, 0, 2)

        # (Batch, Attributes, Attributes)
        attention_logits = torch.bmm(query_tensor, key_tensor)
        # scaling as transformer does
        attention_logits = attention_logits / self.q_dim
        # Remove Self Attention
        attention_logits = attention_logits * self.mask
        attention_coef = F.softmax(attention_logits, dim = 2)
        
        # (Batch, Attributes, Val dim)
        output_tensor = torch.bmm(attention_coef, value_tensor)
        # (Attributes, Batch, Val dim)
        output_tensor = output_tensor.permute(1, 0, 2)
        value_tensor = value_tensor.permute(1, 0, 2)

        # Very Coarse Dropout
        # TODO Dropout the whole vector in the next step
        output_tensor = torch.cat((output_tensor, self.dropout(value_tensor)), dim = 2) if self.residual else output_tensor

        output_tensor = self.activation(output_tensor)

        ret = dict()
        for i in range(len(self.fields)):
            ret[self.fields[i]] = output_tensor[i]
        
        return ret

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_dim = 64,
        output_dim = 32,
        q_dim = 16,
        k_dim = 16,
        self_drop_rate = 0.5,
        activation = F.leaky_relu,
        share = True,
        text_field = None,
        residual = True,
        device = "cuda",
        ## New Arguments
        concat = False,
        head_number = 8
    ):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList()
        self.concat = concat
        self.fields = [x for x in text_field]

        for _ in range(head_number):
            head = AttentionLayer(
                input_dim = input_dim,
                output_dim = output_dim,
                q_dim = q_dim,
                k_dim = k_dim,
                self_drop_rate = self_drop_rate,
                activation = activation,
                share = share,
                text_field = text_field,
                device = device
            )
            self.heads.append(head)
        
    def forward(self, batch):
        batches = [head(batch) for head in self.heads]
        ret = dict()

        def aggregate_func(reps):
            torch.stack(reps, dim = 0)
            return torch.cat(reps, dim = -1) if self.concat else torch.mean(torch.stack(reps, dim = 0), dim = 0)

        for field in self.fields:
            field_rep = []
            for head in batches:
                field_rep.append(head[field])
            field_rep = aggregate_func(field_rep)
            ret[field] = field_rep
        
        return ret



class AttentionLayerAttendToTitle(nn.Module):
    def __init__(
        self
    ):
        super(AttentionLayerAttendToTitle, self).__init__()
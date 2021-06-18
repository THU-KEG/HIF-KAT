import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class LSTMSummarizor(nn.Module):
    def __init__(
        self, 
        embedding_dim = 200, 
        output_dim = 64, 
        layer_num = 2,
        drop_rate = 0.2,
        fields = None, 
        share = False, 
        bias = False,
        bidirectional = True, 
        linear = False,
        activation = F.leaky_relu,
        device = "cuda"
    ):
        assert fields is not None
        super(LSTMSummarizor, self).__init__()
        hidden_dim = int(output_dim / 2) if bidirectional else output_dim

        self.lstm = nn.ModuleDict()

        common_lstm_module = None
        if share:
            common_lstm_module = nn.LSTM(
                input_size = embedding_dim,
                hidden_size = hidden_dim,
                num_layers = layer_num,
                bias = bias,
                batch_first = True,
                dropout = drop_rate,
                bidirectional = bidirectional
            )
        
        for field in fields:
            if not share:
                self.lstm[field] = nn.LSTM(
                    input_size = embedding_dim,
                    hidden_size = hidden_dim,
                    num_layers = layer_num,
                    bias = bias,
                    batch_first = True,
                    dropout = drop_rate,
                    bidirectional = bidirectional
                )
            else:
                self.lstm[field] = common_lstm_module

        self.linear = nn.Linear(hidden_dim + hidden_dim * int(bidirectional), hidden_dim, bias = bias) if linear else None
        self.activation = activation

        if "cuda" in device:
            self.cuda()

    def forward(self, batch):
        assert set(batch.keys()) == set(self.lstm.keys())

        summarize_field = dict()
        for field in self.lstm:
            lstm = self.lstm[field]
            # input()
            out, (h, c) = lstm(batch[field])
            out = out[:, -1, :].squeeze()
            if self.linear is not None:
                out = self.linear(out)
                out = self.activation(out)
            summarize_field[field] = out
        return summarize_field

class AttentionSummarizor(nn.Module):
    def __init__(
        self, 
        embedding_dim = 200, 
        output_dim = 64, 
        layer_num = 2,
        drop_rate = 0.2,
        title = "商品标题", 
        title_seq_length = 64,
        other_seq_length = 32,
        training_fields = None,
        title_field = None,
        share = False, 
        bias = False,
        linear = False,
        activation = F.leaky_relu,
        device = "cuda"
    ):
        assert training_fields is not None
        assert title_field is not None
        fields = training_fields + [title_field]
        super(AttentionSummarizor, self).__init__()
        self.training_fields = training_fields
        self.title_field = title_field
        self.fields = fields
        
        self.share = share
        self.fields = fields
        v_dim = output_dim
        if share:
            self.wq = torch.nn.parameter.Parameter(torch.Tensor(embedding_dim, 1))
            nn.init.kaiming_uniform_(self.wq, a = math.sqrt(5))
            self.wv = torch.nn.parameter.Parameter(torch.Tensor(embedding_dim, v_dim))
            nn.init.kaiming_uniform_(self.wv, a = math.sqrt(5))

        else:
            self.wq = nn.ParameterDict()
            self.wv = nn.ParameterDict()

            for field in self.fields:
                wq = torch.nn.parameter.Parameter(torch.Tensor(embedding_dim, 1))
                nn.init.kaiming_uniform_(wq, a = math.sqrt(5))
                self.wq[field] = wq
                wv = torch.nn.parameter.Parameter(torch.Tensor(embedding_dim, v_dim))
                nn.init.kaiming_uniform_(wv, a = math.sqrt(5))
                self.wv[field] = wv

        # self.linear = nn.Linear(output_dim, output_dim, bias = bias) if linear else None
        self.activation = activation

        ### Positional Encoder
        self.title_position_encoding = torch.zeros(title_seq_length, embedding_dim)
        self.attri_position_encoding = torch.zeros(other_seq_length, embedding_dim)

        # (seq len, 1)
        title_positions = torch.arange(0, title_seq_length).unsqueeze(1)
        attri_positions = torch.arange(0, other_seq_length).unsqueeze(1)
        # in log space, (1, embedding)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.) / embedding_dim))

        # (seq len, embedding)
        self.title_position_encoding[:, 0::2] = torch.sin(title_positions * div_term)
        self.title_position_encoding[:, 1::2] = torch.cos(title_positions * div_term)

        self.attri_position_encoding[:, 0::2] = torch.sin(attri_positions * div_term)
        self.attri_position_encoding[:, 1::2] = torch.cos(attri_positions * div_term)
        ### Positional Encoder

        if "cuda" in device:
            self.cuda()
            # self.title_position_encoding = self.title_position_encoding.cuda()
            # self.attri_position_encoding = self.attri_position_encoding.cuda()

    def forward(self, batch):
        summ = dict()
        for field in self.fields:
            # (embedding, output_dim)
            wv = self.wv if self.share else self.wv[field]
            # (embedding, 1)
            wq = self.wq if self.share else self.wq[field]
            # (batch, words, embedding)
            input_tensor = batch[field]


            ### TODO Add Positional Embedding Here ###
            # input_tensor = input_tensor + self.title_position_encoding \
            #     if field == self.title_field \
            #     else input_tensor + self.attri_position_encoding
            ###                                    ###

            # (batch, words, output_dim)
            transformation_tensor = torch.matmul(input_tensor, wv)
            # (batch, words, 1)
            weight_logits = torch.matmul(input_tensor, wq)
            # (batch, 1, words)
            weights = F.softmax(weight_logits.permute(0, 2, 1), dim = -1)

            # (batch, 1, output_dim)
            summary = torch.bmm(weights, transformation_tensor)
            # (batch, output_dim)
            # summary = torch.squeeze(summary)  
            summary = summary.squeeze(1)
            summary = self.activation(summary)           
            summ[field] = summary
        return summ
            

class MultiHeadAttentionSummarizor(nn.Module):
    def __init__(
        self, 
        embedding_dim = 200, 
        output_dim = 100,

        head_num = 10,

        layer_num = 2,
        drop_rate = 0.2,
        title = "商品标题", 
        title_seq_length = 64,
        other_seq_length = 32,
        training_fields = None,
        title_field = None,
        share = False, 
        bias = False,
        linear = False,
        activation = F.leaky_relu,
        device = "cuda"
    ):
        output_dim = output_dim / head_num
        super(MultiHeadAttentionSummarizor, self).__init__()
        self.atte_head = nn.ModuleList()
        for _ in range(head_num):
            head = AttentionSummarizor(
                embedding_dim = embedding_dim,
                output_dim = int(output_dim),
                layer_num = layer_num,
                drop_rate = drop_rate,
                title = title,
                title_seq_length = title_seq_length,
                other_seq_length = other_seq_length,
                training_fields = training_fields,
                title_field = title_field,
                share = share,
                bias = bias,
                linear = linear,
                activation = activation,
                device = device
            )
            self.atte_head.append(head)
    
    def forward(self, batch):
        ret_list = [head(batch) for head in self.atte_head]
        attr_list = [x for x in ret_list[0]]
        ret = dict()
        
        for head in ret_list:
            for attr in attr_list:
                if attr not in ret:
                    ret[attr] = []
                ret[attr].append(head[attr])
        
        for attr in attr_list:
            rep = ret[attr]            
            # rep = torch.stack(rep, dim = 0)
            ret[attr] = torch.cat(rep, dim = -1)
        return ret

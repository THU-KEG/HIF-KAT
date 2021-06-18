import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import pickle as pkl

from torch_scatter import scatter

from .embedding import AttributeEmbedding
from .summarize import LSTMSummarizor, AttentionSummarizor, MultiHeadAttentionSummarizor
from .attention import AttentionLayer, MultiHeadAttention
from .graphconv import GraphConvLayer, GraphAtteLayer, MLP
from .ablations import AttentionAblation

class RepresentationModel(nn.Module):
    def __init__(
        self,
        training_fields,
        title_field,
        index_field,
        embedding_weights,
        embedding_trainable,
        vocab_size,
        title_seq_len,  # How many words (token) in product's title
        other_seq_len,  # How many words (token) in product's title apart from the title
        summarize_para_share = False,
        attention_para_share = False,
        graphconv_para_share = False,
        add_bias = True,
        gcn_drop_rate = 0.4,
        lstm_drop_rate = 0.4,
        mask_drop_rate = 0.2,
        activation = F.leaky_relu,

        summarize_dimension = [100, ],
        attention_dimension = [64, ],
        gcn_dimension = [32, 64],

        device = "cuda"
    ):
        super(RepresentationModel, self).__init__()
        self.index_field = index_field
        self.title_field = title_field
        self.training_fields = training_fields
        # assert node_id == prod_id

        summarize_dimension = iter(summarize_dimension)
        attention_dimension = iter(attention_dimension)
        gcn_dimension = iter(gcn_dimension)

        self.embedding_layer = AttributeEmbedding(
            embedding_matrix = embedding_weights,
            trainable = embedding_trainable,
            title = title_field,
            index = index_field,
            text_attributes = training_fields + [title_field, ],
            device = device
        )

        # Input: embedding size, 200 as default
        # Output: hidden size, 100 as default

        cur_dim = embedding_weights.size(1)
        next_dim = next(summarize_dimension)

        self.summarizer = MultiHeadAttentionSummarizor(
            embedding_dim = cur_dim,
            output_dim = next_dim,
            head_num = 4,
            layer_num = 2,
            drop_rate = lstm_drop_rate,
            title_seq_length = title_seq_len,
            other_seq_length = other_seq_len,
            training_fields = training_fields,
            title_field = title_field,
            share = summarize_para_share,
            bias = add_bias,
            linear = False,                             # Linear Layer Before output
            activation = activation,
            device = device
        )

        cur_dim = next_dim
        next_dim = next(attention_dimension)

        self.multihead_attention1 = MultiHeadAttention(
            input_dim = cur_dim,
            output_dim = int(next_dim / 16),
            q_dim = 16,
            k_dim = 16,
            self_drop_rate = mask_drop_rate,
            activation = activation,
            share = attention_para_share,
            text_field = training_fields + [title_field, ],
            residual = True,
            device = device,
            concat = True,
            head_number = 16
        )

        cur_dim = next_dim
        next_dim = next(gcn_dimension)

        self.gcn1 = MLP(
            input_dim = cur_dim,
            output_dim = next_dim,
            dropout = 0.0,
            activation = activation,
            device = device,
            text_field = training_fields + [title_field, ],
            bias = add_bias
        )

        cur_dim = next_dim
        next_dim = next(gcn_dimension)

        self.gcn2 = MLP(
            input_dim = cur_dim,
            output_dim = next_dim,
            dropout = 0.0,
            activation = lambda x:x,
            device = device,
            text_field = training_fields + [title_field, ],
            bias = add_bias
        )

        cur_dim = next_dim

        self.classifier = nn.ModuleDict()
        for f in training_fields + [title_field, ]:
            self.classifier[f] = nn.Linear(
                cur_dim,
                vocab_size,
                False
            )

        if "cuda" in device:
            self.classifier.cuda()

    def forward(self, inputs, target = None, masks = None, train = True):
        embedding   = self.embedding_layer(inputs)
        summarize   = self.summarizer(embedding)
        attention1  = self.multihead_attention1(summarize)
        gcn1        = self.gcn1(attention1)
        gcn2        = self.gcn2(gcn1)


        intermediate_representation = {
            "emb": embedding,
            "sum": summarize,
            "att": attention1,
            "gcn1": gcn1,
            "gcn2": gcn2
        } 

        if train:
            out = {x: self.classifier[x](gcn2[x]) for x in self.training_fields + [self.title_field, ]}
            loss_dict, loss = MLMLoss(self.training_fields + [self.title_field, ], out, target, masks)

            return out, loss_dict, loss, intermediate_representation
        else:
            return intermediate_representation


class PairMaker():
    def __init__(
        self,
        init_token,
        end_token,
        pad_token,
        batch_size,
        vocab_size,
        text_fields,
        obligate_fields = ["商品标题"],
        index_attribute = "index",
        embedding_dim = 200,
        sentences_len = 64,
        mask_rate = 0.1,
        device = "cuda"
    ):
        self.init_token = init_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.end_token = end_token

        self.obligate_fields = obligate_fields[:]
        self.text_fields = text_fields
        self.index_attribute = index_attribute
        self.mask_rate = mask_rate
        
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sentences_len = sentences_len

        self.not_miss_list = dict()

        self.device = device

    def _is_missing(self, x):
        return x[0] == self.init_token and x[1] == self.end_token

    def make_pairs(self, batch, mini_batch_num):
        batch_masks = dict()
        batch_input = dict()
        batch_output = dict()

        for field in self.text_fields:
            if field != self.index_attribute:
                sentences_len = getattr(batch, field).size(0)
                self.batch_size = getattr(batch, field).size(1)
                
                empty_token = (torch.ones(size = (sentences_len, )) * self.pad_token).long()
                empty_token[0] = self.init_token
                empty_token[1] = self.end_token
                if "cuda" in self.device:
                    empty_token = empty_token.cuda()

                target_tokens = getattr(batch, field)

                if field not in self.not_miss_list:
                    not_miss_list = [i for i in range(self.batch_size) if not self._is_missing(target_tokens[:, i])]
                    self.not_miss_list[field] = dict()
                    self.not_miss_list[field][mini_batch_num] = not_miss_list
                elif field in self.not_miss_list and mini_batch_num not in self.not_miss_list[field]:
                    not_miss_list = [i for i in range(self.batch_size) if not self._is_missing(target_tokens[:, i])]
                    self.not_miss_list[field][mini_batch_num] = not_miss_list
                else:
                    not_miss_list = self.not_miss_list[field][mini_batch_num]

                # not_miss_list = [i for i in range(self.batch_size) if not self._is_missing(target_tokens[:, i])]

                # self.not_miss_list[field] = not_miss_list

                random.shuffle(not_miss_list)

                input_tokens = target_tokens.clone().detach()

                ### Note that products' title should not be masked as it serves to recover other informations ###
                if field not in self.obligate_fields:
                    input_tokens[:, not_miss_list[:int(len(not_miss_list) * self.mask_rate)]] = empty_token.view((-1, 1))

                target_lens = (
                    (target_tokens != self.init_token) * \
                    (target_tokens != self.end_token) * \
                    (target_tokens != self.pad_token)
                )
                target_lens = target_lens.sum(0)

                inv_target_lens = 1. / (target_lens.float() + 1e-12)
                
                scatter_src = inv_target_lens.view((1, -1)).expand((sentences_len, -1))
                target_logits = scatter(
                    scatter_src, 
                    target_tokens, 
                    dim = 0, 
                    dim_size = self.vocab_size, 
                    reduce = "sum"
                )

                target_logits[self.init_token][:] = 0
                target_logits[self.end_token][:] = 0
                target_logits[self.pad_token][:] = 1
                target_logits[self.pad_token][not_miss_list] = 0

                batch_output[field] = target_logits
                batch_input[field] = input_tokens
                batch_masks[field] = torch.zeros(size=(self.batch_size, 1))
                
                mask_indx = not_miss_list
                batch_masks[field][mask_indx] = self.batch_size / int(len(mask_indx))

                if "cuda" in self.device:
                    batch_masks[field] = batch_masks[field].detach().cuda()
                    batch_input[field] = batch_input[field].detach().cuda()
                    batch_output[field] = batch_output[field].detach().cuda()

            else:
                batch_masks[field] = torch.zeros(size=(self.batch_size, 1))
                if field != self.index_attribute:
                    batch_input[field] = getattr(batch, field)

        return batch_masks, batch_input, batch_output

def MLMLoss(fields, batch, ground_truth, f_mask):
    loss_dict = dict()
    loss = 0

    for field in fields:
        pred = batch[field]
        grth = ground_truth[field].permute(1, 0)
        mask = f_mask[field]
        lgsftmx = F.log_softmax(pred, dim = 1)

        f_inst_loss = -torch.sum(
            lgsftmx * grth,
            dim = 1
        )

        f_loss = torch.mean(f_inst_loss * mask.view(-1))
        loss_dict[field] = f_loss
        loss += f_loss
    
    loss /= len(fields)


    return loss_dict, loss
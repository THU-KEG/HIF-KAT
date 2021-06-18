import torch
import torchtext
import gensim
import fasttext
import os
import pandas as pd
import datetime
import numpy as np
import copy
from collections import Counter, OrderedDict
from itertools import chain

class TencentEmbedding(torchtext.vocab.Vectors):
    def __init__(
        self,
        path = "~/.vector_cache/tencent",
        model = "Tencent_AILab_ChineseEmbedding"
    ):
        embedding_path = os.path.join(path, model + ".bin")
        embedding_path = os.path.expanduser(embedding_path)
        print(embedding_path)

        start = datetime.datetime.now()
        print("Loading Embedding")

        self.word_embedding = gensim.models.KeyedVectors.load(embedding_path, mmap="r")
        self.dim = int(self.word_embedding['.'].shape[0])

        self.word_embedding['<unk>'] = np.random.normal(size=(self.dim))
        # self.word_embedding['<pad>'] = np.random.normal(size=(self.dim))
        self.word_embedding['<pad>'] = np.zeros(shape=(self.dim))

        end = datetime.datetime.now()
        print("Loading Complete")
        print("Total Time Used:", (end - start).seconds, "Seconds")

    def __getitem__(self, word):
        emb = torch.FloatTensor(self.word_embedding[word])
        return emb

class GloveEmbedding(torchtext.vocab.Vectors):
    def __init__(
        self, 
        path = "~/.vector_cache/glove",
        model = "glove.uncase.wiki.300d",
    ):
        start = datetime.datetime.now()
        print("Loading Embedding")

        embedding_path = os.path.join(path, model + ".bin")
        embedding_path = os.path.expanduser(embedding_path)
        self.word_embedding = gensim.models.KeyedVectors.load(embedding_path, mmap="r")
        self.dim = int(self.word_embedding["a"].shape[0])

        self.word_embedding['<unk>'] = np.random.normal(size=(self.dim))
        self.word_embedding['<s>'] = np.random.normal(size=(self.dim))
        self.word_embedding['</s>'] = np.random.normal(size=(self.dim))
        self.word_embedding['<pad>'] = np.zeros(shape=(self.dim))
        
        end = datetime.datetime.now()
        print("Loading Complete")
        print("Total Time Used:", (end - start).seconds, "Seconds")

    def __getitem__(self, word):
        return torch.FloatTensor(self.word_embedding[word])

class FasttextEmbedding(torchtext.vocab.Vectors):
    def __init__(
        self,
        path = "~/.vector_cache/fasttext",
        model = "fasttext.case.en.300d"
    ):
        embedding_path = os.path.join(path, model + ".bin")
        embedding_path = os.path.expanduser(embedding_path)

        start = datetime.datetime.now()
        print("Loading Embedding")

        self.word_embedding = fasttext.load_model(embedding_path)
        self.dim = int(self.word_embedding['a'].shape[0])

        end = datetime.datetime.now()
        print("Loading Complete")
        print("Total Time Used:", (end - start).seconds, "Seconds")

    def __getitem__(self, word):
        return torch.FloatTensor(self.word_embedding[word])

class PretrainEmbeddingField(torchtext.data.Field):
    def __init__(
        self, 
        *args, 
        embedding_vocab = None,        
        **kwargs
    ):
        assert embedding_vocab is not None
        self.embedding_vocab = embedding_vocab
        super(PretrainEmbeddingField, self).__init__(*args, **kwargs)

    def build_vocab(self, *args, counter=None, embedding="tencent", **kwargs):
        # super(PretrainEmbeddingField, self).build_vocab(*args, vectors=self.embedding_vocab, **kwargs)
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        if counter is None:
            ### Original Part, To Calculate the Counter ###
            counter = Counter()
            sources = []
            for arg in args:
                if isinstance(arg, torchtext.data.Dataset) or \
                isinstance(arg, torchtext.data.TabularDataset):
                    sources += [getattr(arg, name) for name, field in
                                arg.fields.items() if field is self]
                else:
                    sources.append(arg)
            for data in sources:
                for x in data:
                    if not self.sequential:
                        x = [x]
                    try:
                        counter.update(x)
                    except TypeError:
                        counter.update(chain.from_iterable(x))

        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token] + kwargs.pop('specials', [])
            if tok is not None))

        ### Add vectors = self.embedding_vocab
        self.vocab = self.vocab_cls(counter, specials=specials, vectors=self.embedding_vocab, **kwargs)

class Dataset(torchtext.data.TabularDataset):
    """
    Dataset Attributes
        ~title_attribute: Name of title
        ~text_attributes
        ~index_attribute
        ~embedding_vector: TencentEmbedding object
        ~counter: Word frequency counter of sentences from all the columns
        ~vocab: 
        ~embedding_weight_matrix
    """
    def __init__(
        self, 
        data_path="../dataset/taobao_raw", 
        data_file="mobile.csv",
        omit_columns=["产品id", "index", "id"],
        title_column="商品标题",
        index_column="index",
        title_len = 64,
        attri_len = 32,

        init_token = "<s>",
        eos_token = "</s>",

        embedding_type = "tencent",          # Tencent, Glove, Fasttext
        embedding_path = None,          # .vector_cache
        embedding_model = None,         # wiki, (un)case, dimensions
    ):
        data_file_name = os.path.join(data_path, data_file)
        data_frame = pd.read_csv(data_file_name)
        # print(data_frame.columns)
        self.title_attribute = title_column
        self.index_attribute = index_column
        self.text_attributes = []

        assert embedding_type in ["tencent", "glove", "fasttext"]
        if embedding_type == "tencent":
            self.embedding_vector = TencentEmbedding(
                path = embedding_path,
                model = embedding_model
            )
        elif embedding_type == "glove":
            self.embedding_vector = GloveEmbedding(
                path = embedding_path,
                model = embedding_model
            )
        elif embedding_type == "fasttext":
            self.embedding_vector = FasttextEmbedding(
                path = embedding_path,
                model = embedding_model
            )

        ### Get The Counter Object for the Whole dataset ###
        whole_fields = []
        whole_text_field = PretrainEmbeddingField(
                    embedding_vocab = self.embedding_vector,
                    init_token = init_token,
                    eos_token = eos_token,
                    fix_length = 256,
                    lower = True
        )

        for x in data_frame.columns:
            if x in omit_columns:
                whole_fields.append((x, None))
            elif x == "产品id" or x == "商品id":
                whole_fields.append((x, None))
            else:
                whole_fields.append((x, whole_text_field))
                self.text_attributes.append(x)

        
        whole_dataset = torchtext.data.TabularDataset(
            path = data_file_name, 
            format = data_file_name.split(".")[-1],
            fields = whole_fields,
            skip_header = True
        )

        whole_text_field.build_vocab(whole_dataset)
        counter = copy.deepcopy(whole_text_field.vocab.freqs)
        self.counter = counter
        self.vocab = whole_text_field.vocab
        self.embedding_weight_matrix = whole_text_field.vocab.vectors
        ### set <pad> to zero vectors ###
        self.embedding_weight_matrix[self.vocab.stoi["<pad>"]] = torch.zeros(self.embedding_weight_matrix[0].shape)
        ###
        self.vocab_size = self.embedding_weight_matrix.size(0)

        self.init_token_num = self.vocab.stoi["<s>"]
        self.end_token_num = self.vocab.stoi["</s>"]
        self.pad_token_num = self.vocab.stoi["<pad>"]

        ### End of Counter ###

        fields = []
        for x in data_frame.columns:
            if x in omit_columns:
                fields.append((x, None))
            elif x == "产品id" or x == "商品id":
                # data_field = torchtext.data.Field(
                data_field = torchtext.data.Field(
                    sequential = False,
                    use_vocab = False,
                    dtype = torch.long
                )
                fields.append((x, data_field))
            else:
                if x == title_column:
                    fix_len = title_len
                else:
                    fix_len = attri_len
                data_field = PretrainEmbeddingField(
                    embedding_vocab = self.embedding_vector,
                    init_token = init_token,
                    eos_token = eos_token,
                    fix_length = fix_len,
                    lower = True
                )
                fields.append((x, data_field))

        super(Dataset, self).__init__(
            path = data_file_name, 
            format = data_file_name.split(".")[-1],
            fields = fields,
            skip_header = True
        )

        for x, f in fields:
            f.build_vocab(self, counter=counter) if f is not None and x not in ["产品id", "商品id"] else None
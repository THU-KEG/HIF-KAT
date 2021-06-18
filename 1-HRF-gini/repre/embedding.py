import torch
import torch.nn as nn
import torch.nn.functional as F

class AttributeEmbedding(nn.Module):
    def __init__(
        self, 
        embedding_matrix, 
        trainable = True, 
        title = "商品标题", 
        index = "商品id",
        text_attributes = None, # Attributes that need to be converted to vectors 
        device="cuda"
    ):
        assert text_attributes is not None
        super(AttributeEmbedding, self).__init__()
        
        self.vocab_size = int(embedding_matrix.size()[0])
        self.embedding_size = int(embedding_matrix.size()[1])

        self.title_attribute = title
        self.index_attribute = index
        self.text_attributes = text_attributes

        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size, 
            embedding_dim = self.embedding_size, 
            _weight = embedding_matrix
        )
        
        self.embedding.requires_grad_(trainable)

        if device == "cuda":
            self.embedding.cuda()

    def forward(self, batch):
        # (Batch, seq length, embedding dimension)
        embedded_fields = {x: self.embedding(batch[x]).permute(1, 0, 2) for x in self.text_attributes}

        return embedded_fields
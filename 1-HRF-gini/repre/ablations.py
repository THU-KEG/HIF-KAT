import torch 
import torch.nn as nn
import torch.nn.functional as F

class AttentionAblation(nn.Module):
    def __init__(
        self,
        output_dim,
        input_dim,
        drop_rate,
        activation,
        text_field
    ):
        super(AttentionAblation, self).__init__()
        self.model = nn.ModuleDict()
        for field in text_field:
            self.model[field] = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(input_dim, output_dim),
                nn.ReLU()
            )
        self.cuda()
    
    def forward(self, batch):        
        return {x: self.model[x](batch[x]) for x in batch}

### Ablation studies for GCN are conducted by setting Adj = I_N
import torch
import torch.nn as nn
import re
import esm
import math
import numpy as np
from Bio.Align import substitution_matrices
from lora import modify_with_lora 
from config import lora_config 

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class AtchleyFactorEmbedding(nn.Module):
    def __init__(self, target_dim=512):
        super(AtchleyFactorEmbedding, self).__init__()
        self.atchley_factors = {
            'A': [-0.591, -1.302, -0.733, 1.570, -0.146], 'C': [-1.343, 0.465, -0.862, -1.020, -0.255],
            'D': [1.050, 0.302, -3.656, -0.259, -3.242], 'E': [1.357, -1.453, 1.477, 0.113, -0.837],
            'F': [-1.006, -0.590, 1.891, -0.397, 0.412], 'G': [-0.384, 1.652, 1.330, 1.045, 2.064],
            'H': [0.336, -0.417, -1.673, -1.474, -0.078], 'I': [-1.239, -0.547, 2.131, 0.393, 0.816],
            'K': [1.831, -0.561, 0.533, -0.277, 1.648], 'L': [-1.019, -0.987, -1.505, 1.266, -0.912],
            'M': [-0.663, -1.524, 2.219, -1.005, 1.212], 'N': [0.945, 0.828, 1.299, -0.169, 0.933],
            'P': [0.189, 2.081, -1.628, 0.421, -1.392], 'Q': [0.931, -0.179, -3.005, -0.503, -1.853],
            'R': [1.538, -0.055, 1.502, 0.440, 2.897], 'S': [-0.228, 1.399, -4.760, 0.670, -2.647],
            'T': [-0.032, 0.326, 2.213, 0.908, 1.313], 'V': [-1.337, -0.279, -0.544, 1.242, -1.262],
            'W': [-0.595, 0.009, 0.672, -2.128, -0.184], 'Y': [0.260, 0.830, 3.097, -0.838, 1.512]
        }
        self.vector_size = 5
        self.expand_to_target_dim = nn.Sequential(nn.Linear(self.vector_size, target_dim), nn.ReLU(), nn.LayerNorm(target_dim))
        self.positional_encoding = SinusoidalPositionalEncoding(d_model=target_dim)

    def get_atchley_vector(self, aa):
        return self.atchley_factors.get(aa, np.zeros(self.vector_size))

    def forward(self, sequence, device):
        embeddings = [self.get_atchley_vector(aa) for aa in sequence if aa in self.atchley_factors]
        embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
        embeddings = self.expand_to_target_dim(embeddings).unsqueeze(0)
        embeddings = self.positional_encoding(embeddings)
        cls_placeholder = embeddings.mean(dim=1, keepdim=True)
        sep_placeholder = torch.zeros((1, 1, embeddings.shape[2]), device=embeddings.device)
        return torch.cat([cls_placeholder, embeddings, sep_placeholder], dim=1)


class Blosum50Embedding(nn.Module):
    def __init__(self, target_dim=512):
        super(Blosum50Embedding, self).__init__()
        self.BLOSUM50 = substitution_matrices.load('BLOSUM50')
        self.vector_size = len(self.BLOSUM50.alphabet)
        self.expand_to_target_dim = nn.Sequential(nn.Linear(self.vector_size, target_dim), nn.ReLU(), nn.LayerNorm(target_dim))
        self.positional_encoding = SinusoidalPositionalEncoding(d_model=target_dim)

    def get_blosum50_vector(self, aa):
        vector = np.zeros(self.vector_size)
        for i, other_aa in enumerate(self.BLOSUM50.alphabet):
            score = self.BLOSUM50.get((aa, other_aa), self.BLOSUM50.get((other_aa, aa), 0))
            vector[i] = score
        return vector

    def forward(self, sequence, device):
        embeddings = [self.get_blosum50_vector(aa) for aa in sequence if aa in self.BLOSUM50.alphabet]
        embeddings = np.array(embeddings)
        embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
        embeddings = self.expand_to_target_dim(embeddings).unsqueeze(0)
        embeddings = self.positional_encoding(embeddings)
        cls_placeholder = embeddings.mean(dim=1, keepdim=True)
        sep_placeholder = torch.zeros((1, 1, embeddings.shape[2]), device=embeddings.device)
        return torch.cat([cls_placeholder, embeddings, sep_placeholder], dim=1)

class ExtendedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.5):
        super(ExtendedMLP, self).__init__()
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TransformerSelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super(TransformerSelfAttentionLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim))
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_old = x
        x, _ = self.self_attention(x, x, x)
        x = self.dropout(x)
        x = x + x_old
        x = self.layer_norm1(x)
        x_old = x
        x = self.feed_forward(x)
        x = x + x_old
        x = self.layer_norm2(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, esm_model, mlp_config, attention_config):
        super(CombinedModel, self).__init__()
        self.esm_model = esm_model
        self.blosum50_model = Blosum50Embedding()
        self.atchley_model = AtchleyFactorEmbedding()
        
        num_layers = attention_config['num_layers']
        
        self.atchley_selfattention = nn.ModuleList(
            [TransformerSelfAttentionLayer(embed_dim=512, dropout=attention_config['dropout']) for _ in range(num_layers)]
        )
        self.blosum50_selfattention = nn.ModuleList(
            [TransformerSelfAttentionLayer(embed_dim=512, dropout=attention_config['dropout']) for _ in range(num_layers)]
        )
        self.combined_selfattention = nn.ModuleList(
            [TransformerSelfAttentionLayer(embed_dim=2304, dropout=attention_config['dropout']) for _ in range(num_layers)]
        )
        
        self.mlp_model = ExtendedMLP(
            input_dim=2304,
            hidden_dims=mlp_config['hidden_dims'],
            dropout=mlp_config['dropout']
        )
    
    def forward(self, batch_tokens, sequence_for_b_a, device):
        results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        esm_embedding = results["representations"][33]
        
        blosum50_embedding = self.blosum50_model(sequence_for_b_a, device)
        atchley_embedding = self.atchley_model(sequence_for_b_a, device)
        
        for layer in self.atchley_selfattention:
            atchley_embedding = layer(atchley_embedding)
        for layer in self.blosum50_selfattention:
            blosum50_embedding = layer(blosum50_embedding)
        
        combine_embedding = torch.cat([esm_embedding, blosum50_embedding, atchley_embedding], dim=-1)
        
        for layer in self.combined_selfattention:
            combine_embedding = layer(combine_embedding)
        
        return self.mlp_model(combine_embedding)

def build_model(mlp_config, attention_config):
    """Factory function to build the model."""
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = modify_with_lora(esm_model, lora_config)
    
    for name, param in esm_model.named_parameters():
        if not re.match(lora_config.trainable_param_names, name):
            param.requires_grad = False
            
    model = CombinedModel(esm_model, mlp_config, attention_config)
    return model, alphabet
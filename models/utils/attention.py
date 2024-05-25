import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)

    def forward(self, query, key, value, attn_mask=None):
        attn_output, attn_output_weights = self.attention(query, key, value, attn_mask=attn_mask)
        return attn_output, attn_output_weights

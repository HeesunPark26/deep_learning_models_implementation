import torch
import torch.nn as nn
import math

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc) # d_embed, d_model
        self.k_fc = copy.deepcopy(qkv_fc) # d_embed, d_model
        self.v_fc = copy.deepcopy(qkv_fc) # d_embed, d_model
        self.out_fc = out_fc # d_model, d_embed 
        
    def calculate_attention(self, query, key, value, mask):
        # query, key, value : batch_size, h, seq_len, d_k
        # mask : batch_size, 1, seq_len, seq_len
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-1, -2)) # batch_size, h, query_len, key_len
        attention_score = attention_score / math.sqrt(d_k) # scaling, # batch_size, h, query_len, key_len
        
        # masking
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9) # batch_size, h, query_len, key_len
        # softmax
        attention_prob = nn.functional.softmax(attention_score, dim=-1) # batch_size, h, query_len, key_len
        # value matmul
        attention_score = torch.matmul(attention_prob, value) # batch_size, h, seq_len, d_k
        
        return attention_score
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        def transform(x, fc):
            # x : batch, seq_len, d_embed
            # fc: d_embed , d_model
            out = fc(x) # batch, seq_len, d_model
            out = out.view(batch_size, -1, self.h, self.d_model//self.h) # batch, seq_len, h, d_k
            out = out.transpose(1, 2) # batch, h, seq_len, d_k
            return out
        
        query = self.transform(query, self.q_fc)
        key = self.transform(key, self.k_fc)
        value = self.transform(value, self.v_fc)
        
        out = self.calculate_attention(query, key, value, mask) # batch_size, h, seq_len, d_k 
        out = out.transpose(1, 2) # batch_size, seq_len, h, d_k
        out = out.contiguous().view(batch_size, -1, self.d_model) # batch_size, seq_len, d_model
        
        out = self.out_fc(out) # batch_size, seq_len, d_embed
        return out


class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, fc1, fc2):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1 # d_embed, d_ff
        self.act1 = nn.functional.ReLU()
        self.fc2 = fc2 # d_ff, d_embed
    def forward(self, x):
        
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        return out
      
class ResidualConnectionLayer(nn.Module):
    def __init__(self, d_embed, dropout=0):
        super(ResidualConnectionLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(d_embed)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, sub_layer):
        out = x
        out = self.layer_norm(out)
        out = self.sub_layer(out)
        out = self.dropout(out)

        out = out + x 
        return out
    

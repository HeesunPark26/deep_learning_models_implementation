import torch.nn as nn
import copy
class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff, residual_block):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals_1 = copy.deepcopy(residual_block)
        self.residuals_2 = copy.deepcopy(residual_block)
    def forward(self, src, src_mask):
        out = src
        out = self.residuals_1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residuals_2(out, self.position_ff)
        return out

class Encoder(nn.Module):
    def __init__(self, encoder_block, n_layer):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(n_layer)])
    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        return out

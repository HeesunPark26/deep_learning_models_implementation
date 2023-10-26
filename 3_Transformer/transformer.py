
import torch.nn as nn
import torch
import copy
import numpy as np
import math

class Transformer(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
    def encode(self, src, src_mask):
        out = self.encoder(self.src_embed(src), src_mask)
        return out
    
    def decode(self, tgt, encoder_out, tgt_mask):
        out = self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask)
        return out
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out) # n_batch x seq_len x le문(vocab)
        out = F.log_sofgmax(out, dim=-1) # 마지막 dimension인 len(vocab)에 대한 확률값을 구해야하기 때
        return out, decoder_out
    
    ## --  Pad mask code in pytorch
    # pad masking 을 생성하는 코드.
    def make_pad_mask(self, query, key, pad_idx=1):
        # embedding을 획득하기도 전, token sequence 상태로 input이 들어옴
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        # pad_idx: pad의 index. 얘와 일치하는 token들은 0, 그 외에는 1인 mask 생성

        # self attention의 경우에는 query와 key가 동일할 것
        # 그러나 서로 다른 문장 사이에 이루어지는 cross-attention의 경우 서로 다른 값이 될 수 있다.
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2) # (n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1) # (n_batch, 1, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3) # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len) # (n_batch, 1, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask
    
    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask
    
    def make_subsequent_mask(query, key):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        # np.tril을 이용해 lower triangle을 생성한다.
        # decoder의 mask는 subsequent masking이 적용되어야 하는데 동시에 encoder와 마찬가지로
        # pad masking이 적용되어야 한다
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8') # lower triangle without diagonal
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
        return mask

    def make_tgt_mask(self, tgt):
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        mask = pad_mask & seq_mask
        return mask
    
    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask

    
class Encoder(nn.Module):
    def __init__(self, encoder_block, n_layer): # n_layer: Encoder Block의 개수
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))
    
    def forward(self, src, src_mask):
        """
        Encoder block들을 순서대로 실행하면서, 
        이전 block의 output을 이후 block의 input으로 넣는다.
        """
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(2)]
    
    def forward(self, src, src_mask):
        out = src
        # 인자 1개만 받는게 아닌 경우 lambda를 통해 layer를 넘겨줄 수도 있음
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.redisuals[1](out, self.position_ff)
        return out
    

def calculate_attention(query, key, value, mask):
    # query, key, value: (n_batch, h, seq_len, d_k)
    # mask: (n_batch, 1, seq_len, seq_len)

    d_k = key.shape[-1]
    attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, h, seq_len, seq_len)
    attention_score = attention_score / math.sqrt(d_k)

    if mask is not None:
        attention_score = attention_score.masked_fill(mask==0, -1e9)
    attention_prob = F.softmax(attention_score, dim=-1)  # (n_batch, h, seq_len, seq_len)
    out = torch.matmul(attention_prob, value) # (n_batch, h, seq_len, d_k)
    return out


# Multi-Head Attention in Python
class MultiHeadAttentionLyaer(nn.Module):
    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttentionLyaer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.k_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.v_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.out_fc = out_fc # (d_model, d_embed)
    def forward(self, *args, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # 인자로 받은 query, key, value는 실제로 Q, K, V matrix가 아니라, 
        # input sentence embedding 이다.
        # 이를 3개의 FC layer에 넣어 Q, K, V 를 구함. 
        # # 별개의 인자로 받는 이유는 decoder에서 활용하기 위함.
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)
        def transform(x, fc): # (n_batch, seq_len, d_embed)
            # Q, K, V를 구하는 함수
            out = fc(x) # (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            # view 와 transpose로 dimension을 바꿔주는 이유
            # calculate_attention()이 input으로 n_batchx...xseq_lenxd_k이기 때문
            return out
        
        # multi-head attention layer 역시 shape에 멱등해야하므로 
        # output shape도 n_batch x seq_len x d_embed 여야 함. 
        # 이를 위해 h,와 seq_len의 순서를 바꾸고 다시 h와 d_k를 d_model로 결합
        # 마지막으로 fc layer를 거쳐 d_model을 d_embed로 변환

        query = transform(query, self.q_fc) # (n_batch, h, seq_len, d_k)
        key = transform(key, self.k_fc) # (n_batch, h, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, h, seq_len, d_k)

        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed)
        return out

## --- Position-wise Feed Forward Layer
# 단순하게 2개의 FC Layer를 갖는 Layer
# 각 FC Layer는 (d_embed x d_ff), (d_ff x d_embed)의 weight matrix를 갖는다.
# 즉, feedforward layer 역시 shape에 대해 멱등하다. 
# 다음 encoder block 에게 shape를 유지한 채 넘겨줘야 하기 때문
# multi-head attention layer의 output을 input으로 받아 연산을 수행, 다음 Encoder Block에게 output을 넘겨준다.
# fc layer의 output에 ReLU 적용
class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, fc1, fc2):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1 # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = fc2 # (d_ff, d_embed)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

## --- Residual Connection Layer
# multi-head attention layer와 position-wise feed-foward layer는 
# residual connection으로 둘러싸여 있다.
class ResidualConnectionLayer(nn.Module):
    def __init__(self):
        super(ResidualConnectionLayer, self).__init__()

    def forward(self, x, sub_layer):
        out = x
        out = sub_layer(out)
        out = out + x
        return out
    
##### Decoder
class Decoder(nn.Module):
    def __init__(self, decoder_block, n_layer):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])
    
    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            # tgt_mask: decoder의 input으로 주어지는 target sentence의 
            # pad masking과 subsequent masking (make_tgt_mask()로 생성된 것)
            # self-multi head attention에서 사용됨

            # src_tgt_mask는 self-multi-head attention layer에서 넘어온 query, 
            # encoder에서 넘어온 key, value 사이의 pad masking 
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        return out
class DecoderBlock(nn.Module):
    def __init__(self, self_attention, cross_attention, position_ff):
        super(DecoderBlock, self).__init__
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(3)]
    
    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residuals[1](out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        out = self.residuals[2](out, self.position_ff)
        return out
    

#### Embedding
class TransformerEmbedding(nn.Module):
    def __init__(self, token_embed, pos_embed):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)
    
    def forward(self, x):
        out = self.embedding(x)
        return out

class TokenEmbedding(nn.Module):
    # vocabulary 와 d_embed를 이용해 embedding을 생성한다. 
    # embedding에도 scaling을 적용한다는 점
    # vocab_size는 총 단어 갯수? 
    def __init__(self, d_embed, vocab_size):
        super(TokenEmbedding,self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed
    
    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_len=256, device=torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2)*-(math.log(10000.0)/d_embed))

        encoding[:,0::2] = torch.sin(position * div_term)
        encoding[:,1::2] = torch.cos(position * div_term)

        self.encoding = encoding.unsqueeze(0).to(device)

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out
    
### build model
def build_model(src_vocab_size, tgt_vocab_size, device=torch.device("cpu"), max_len=256, d_embed=512, n_layer=6, d_model=512, h=8, d_ff=2048):
    import copy
    copy = copy.deepcopy
    src_token_embed = TokenEmbedding(
        d_embed = d_embed,
        vocab_size = src_vocab_size
    )
    tgt_token_embed = TokenEmbedding(
        d_embed = d_embed,
        vocab_size = tgt_vocab_size
    )
    pos_embed = PositionalEncoding(
        d_embed = d_embed,
        max_len = max_len,
        device = device
    )
    src_embed = TransformerEmbedding(
        token_embed = src_token_embed,
        pos_embed = copy(pos_embed)
    )
    tgt_embed = TransformerEmbedding(
        token_embed = tgt_token_embed,
        pos_embed = copy(pos_embed)
    )
    attention = MultiHeadAttentionLyaer(
        d_model = d_model,
        h = h,
        qkv_fc = nn.Liner(d_embed, d_model),
        out_fc = nn.Liner(d_model, d_embed)
    )
    position_ff = PositionWiseFeedForwardLayer(
        fc1 = nn.Liner(d_embed, d_ff),
        fc2 = nn.Liner(d_ff, d_embed)
    )
    encoder_block = EncoderBlock(
        self_attention = copy(attention),
        position_ff = copy(position_ff)
    )
    decoder_block = DecoderBlock(
        self_attention = copy(attention),
        cross_attention = copy(attention),
        position_ff = copy(position_ff)
    )
    encoder = Encoder(
        encoder_block = encoder_block,
        n_layer = n_layer
    )
    decoder = Decoder(
        decoder_block = decoder_block,
        n_layer = n_layer)
    generator = nn.Liner(d_model, tgt_vocab_size)

    model = Transformer(
        src_embed = src_embed,
        tgt_embed = tgt_embed,
        encoder = encoder,
        decoder = decoder,
        generator = generator
    ).to(device)

    model.device = device

    return model


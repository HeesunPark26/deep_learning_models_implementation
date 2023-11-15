import math
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, d_embed, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed 
    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed) # why scaling? 이후 positional encoding의 영향력을 조금 줄이기 위함.
        return out
    
    ## nn.Embdding? 간단한 lookup table. input은 index들. output은 word embedding
    ## weight을 가지고 있음. learnable. N(0,1)로부터 initialize 된다.
    # 임베딩 층(embedding layer)을 만들어 훈련 데이터로부터 처음부터 임베딩 벡터를 학습하는 방법
    # https://wikidocs.net/64779
    # 인풋은 정수 인코딩이 되어 있어야 함.
    # 단어 -> 단어에 부여된 고유한 정수값 -> 임베딩 층 통과 -> 밀집 벡터
    # 입력 정수에 대해 밀집 벡터로 매핑. 이 벡터는 가중치가 학습되는 것과 같은 방식으로 학습된다.
    # 풀고자 하는 작업에 맞는 값으로 업데이트 된다.
    # 룩업테이블 - 특정 단어와 매핑되는 그 정수를 인덱스로 가지는 테이블. 여기서 인덱싱을 통해 임베딩 백터 값을 가져온다. '꺼내온다'
    # 단어 집합의 크기만큼의 행을 가지므로, 모든 단어는 고유한 임베딩벡터를 가진다.


class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_len=256, device=torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False # 학습되지 않는다!!!!! 
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2)*-(math.log(10000.0)/d_embed))

        encoding[:,0::2] = torch.sin(position * div_term)
        encoding[:,1::2] = torch.cos(position * div_term)

        self.encoding = encoding.unsqueeze(0).to(device)

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed # token embedding에 더해진다.
        return out

    # sin, cos 함수를 이용한 positional encoding. rnn이 없는 대신 위치 정보를 주기 위함.
    # 상대적인 순서를 부여한다. -1, 1 사이의 값으로 올 수 있어 영향력을 제한할 수 있다. 
    # 학습할 때보다 더 긴 문장이 와도 효과적으로 위치 정보를 부여할 수 있다는 강점이 있음.


class TransformerEmbedding(nn.Module):
    """ token embedding & positional encoding 수행"""
    def __init__(self, token_embed, pos_embed, dr_rate=0):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)
        self.dropout = nn.Dropout(p=dr_rate) # add dropout for regularization.


    def forward(self, x):
        out = x
        out = self.embedding(out)
        out = self.dropout(out)
        return out

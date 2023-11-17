# !pip install torchdata
# !pip install folium
# !pip install -U torchtext 
# !pip install -U spach
# https://velog.io/@nkw011/nlp-dataset-dataloader
# https://tutorials.pytorch.kr/beginner/translation_transformer.html

# import module and load multi30k
from torchtext.datasets import Multi30k

# load modules for tokenization

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import spacy
# !python -m spacy download en_core_web_sm
# !python -m spacy download de_core_news_sm

from torch.utils.data import DataLoader
import torchtext.transforms as T

class Multi30k_Dataset():
    def __init__(self, 
                 specials=['<unk>', '<pad>', '<sos>', '<eos>'],
                 vocab_min_freq=2,
                 max_seq_len=256):

        self.specials = {specials[idx]:idx for idx in range(len(specials))}
        
        self.vocab_min_freq = vocab_min_freq
        self.max_seq_len = 256
        # build dataset
        self.train, self.valid, self.test = self.build_dataset()
        
        # build tokenizer
        self.src_tokenizer, self.tgt_tokenizer = self.build_tokenizer()
        
        # build vocabulary
        self.src_vocab, self.tgt_vocab = self.build_vocab()
        
        # get token to id dict
        self.src_token2id = self.src_vocab.get_stoi()
        self.tgt_token2id = self.tgt_vocab.get_stoi()
    def build_dataset(self):
        train, valid, test = Multi30k(language_pair=('en', 'de'))
        # check the data
        for i, (eng, de) in enumerate(train):
            if i==1:break
            print("========================")
            print("Multi30k Dataset Loaded")
            print("========================")
            print("Example sentence")
            print(f"English: {eng}")
            print(f"das Deutsche: {de}")
            print("========================")
        return train, valid, test
    
    def build_tokenizer(self):
        en_tokenizer = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
        de_tokenizer = get_tokenizer(tokenizer='spacy', language='de_core_news_sm')
        return en_tokenizer, de_tokenizer
    
    def build_vocab(self):
        # build vocabulary from tokenized text
        src_vocab = build_vocab_from_iterator(
            map(self.src_tokenizer, [src for src, _ in self.train]), min_freq=self.vocab_min_freq, specials=self.specials.keys())
        tgt_vocab = build_vocab_from_iterator(
            map(self.tgt_tokenizer, [tgt for _, tgt in self.train]), min_freq=self.vocab_min_freq, specials=self.specials.keys())
        return src_vocab, tgt_vocab
    
    def transform(self, seq_token, token2id):
        unknown_id = token2id['<unk>']
        seq_int = [[token2id.get(token, unknown_id) for token in seq] for seq in seq_token]
        seq_int_proc = T.Sequential(
            T.Truncate(self.max_seq_len-2),
            T.AddToken(token=self.specials['<sos>'], begin=True),
            T.AddToken(token=self.specials['<eos>'], begin=False),
            T.ToTensor(padding_value=self.specials['<pad>']))(seq_int)
        return seq_int_proc    
    
    def collate_fn(self, pairs):        
        # tokenization
        src_seq = [self.src_tokenizer(src) for src, _ in pairs]
        tgt_seq = [self.tgt_tokenizer(tgt) for _, tgt in pairs]
        
        # token to integer
        batch_src = self.transform(src_seq, self.src_token2id)
        batch_tgt = self.transform(tgt_seq, self.tgt_token2id)
        return (batch_src, batch_tgt)
    
    
    def get_iter(self, **params):
        train_iter = DataLoader(self.train, collate_fn=self.collate_fn, batch_size=params["batch_size"])
        valid_iter = DataLoader(self.valid, collate_fn=self.collate_fn)
        test_iter = DataLoader(self.test, collate_fn=self.collate_fn)
        return train_iter, valid_iter, test_iter

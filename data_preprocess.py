import spacy
import numpy as np
from torch import nn
from torchtext.datasets import Multi30k
import torchtext.transforms as T
from torch.utils.data import DataLoader
from utils import vocab_utils, transform
from models.components import Transformer
import matplotlib.pyplot as plt


nlp_en = spacy.load("en_core_web_sm")
nlp_de = spacy.load("de_core_news_sm")

nlp_en_tokenize = lambda x: [i.text for i in nlp_en(x)]  # return Doc type.
nlp_de_tokenize = lambda x: [i.text for i in nlp_de(x)]  # return Doc type.

train_datapipe, valid_datapipe, test_datapipe = Multi30k(
    root='/transformer_pytorch/data/Multi30k/data', split=('train', 'valid', 'test'), language_pair=("en", "de")
)

max_seq_len = 64
batch_size = 32
unk_idx, pad_idx, bos_idx, eos_idx = 0, 1, 2, 3

vocab_en_path = '/transformer_pytorch/data/Multi30k/data/vocab/vocab_en.pickle'
vocab_de_path = '/transformer_pytorch/data/Multi30k/data/vocab/vocab_de.pickle'

vocab_en = vocab_utils.create_vocab(vocab_en_path, train_datapipe, nlp_en_tokenize, 'en')
vocab_de = vocab_utils.create_vocab(vocab_de_path, train_datapipe, nlp_de_tokenize, 'de')

en_transform = T.Sequential(
    transform.SpacyTokenize(nlp_en_tokenize),
    transform.IntegerEncoding(vocab_en),
    T.Truncate(max_seq_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
    T.ToTensor(padding_value=pad_idx),
    T.PadTransform(max_length=max_seq_len, pad_value=pad_idx),
)

de_transform = T.Sequential(
    transform.SpacyTokenize(nlp_de_tokenize),
    transform.IntegerEncoding(vocab_de),
    T.Truncate(max_seq_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
    T.ToTensor(padding_value=pad_idx),
    T.PadTransform(max_length=max_seq_len, pad_value=pad_idx),
)


def apply_transform(x):
    # x[0] : en, x[1]: de
    return en_transform(x[0]), de_transform(x[1])


train_datapipe = train_datapipe.map(apply_transform)
data_loader = DataLoader(train_datapipe, batch_size, num_workers=1, shuffle=True, drop_last=True)


transformer = Transformer(vocab_size=len(vocab_en), embedding_dim=512, n_head=6, head_dim=128)

for batch in data_loader:
    en_text, de_text = batch

    # padding_mask
    pad_mask = np.where(en_text == pad_idx)  #  (batch_idx, seq_idx)
    transformer(en_text, pad_mask)

import time
import spacy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
import torchtext.transforms as T
from utils import vocab_utils, transform
from models.transformer import Transformer


nlp_en = spacy.load("en_core_web_sm")
nlp_de = spacy.load("de_core_news_sm")

nlp_en_tokenize = lambda x: [i.text for i in nlp_en(x)]  # return Doc type.
nlp_de_tokenize = lambda x: [i.text for i in nlp_de(x)]  # return Doc type.

train_datapipe, valid_datapipe, test_datapipe = Multi30k(
    root='/transformer_pytorch/data/Multi30k/data', split=('train', 'valid', 'test'), language_pair=("en", "de")
)

max_seq_len = 64
batch_size = 256
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

transformer = Transformer(
    x_vocab_size=len(vocab_en),
    y_vocab_size=len(vocab_de),
    embedding_dim=256,
    n_head=4,
    head_dim=64,
    feed_forward_dim=512,
    n_encoder=3,
    n_decoder=3,
    drop_rate=0.1,
    pad_idx=pad_idx,
)

## training.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epoch = 20
lr = 1e-4
optimizer = torch.optim.Adam(transformer.parameters(), lr)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)  # padding index는 학습 x.

model = transformer.to(device)

model.train()
for i in range(epoch):
    print(f'######### {i+1} epoch #########')
    # trainging.
    for batch in data_loader:
        '''
        번역이기 때문에 y와 target은 서로 1 index 만큼 shift.

        Example)
        x = [I, liked, you, when, I, was, young]
        input_y =  [{sos}, 나는, 너를, 어렸을때, 좋아했다, {eos}] # for teacher forcing.
        target = [나는, 너를, 어렸을때, 좋아했다, {eos}, {padding}]
        '''
        en_text, de_text = batch

        x = en_text.to(device)
        y = de_text.to(device)

        # grad 초기화.
        optimizer.zero_grad()

        output = model(x, y[:, :-1])  # (batch, de_text:seq_len, tokens_len)
        output = output.permute(0, 2, 1)  # (batch, tokens_len, de_text:seq_len)

        # loss 계산 및 학습.
        loss = criterion(output, y[:, 1:])
        loss.backward()
        optimizer.step()

        print(loss.item())

    if i == epoch - 1:
        torch.save(model.state_dict(), f'transformer-{int(time.time())}.pth')

# TODO: mlflow 적용.
# TODO: validation BLEU score 계산.
# TODO: test 출력.

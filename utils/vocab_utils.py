import os
import pickle
from torchtext.vocab import build_vocab_from_iterator


def create_vocab(path, train_datapipe, tokenizer, language):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
    else:

        def yield_tokens(data_iter):
            for text in data_iter:
                yield tokenizer(text[0 if language == 'en' else 1])

        vocab = build_vocab_from_iterator(
            yield_tokens(train_datapipe), min_freq=2, specials=["<unk>", "<pad>", "<bos>", "<eos>"]
        )

        # 단어가 vocab에 없는 경우 출력할 default token.
        vocab.set_default_index(vocab["<unk>"])

        # save vocab.
        with open(path, 'wb') as f:
            pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

    return vocab


def batch_detokenize(x, vocab, pad_idx, bos_idx, eos_idx):
    sentences = []
    for tokens in x.detach().cpu().numpy().tolist():
        output_tokens = []
        for token in tokens:
            if token == bos_idx:
                pass
            elif token == eos_idx:
                break
            elif token == pad_idx:
                break
            else:
                output_tokens.append(vocab.get_itos()[token])

        sentences.append(output_tokens)
    return sentences

import torch


class SpacyTokenize(torch.nn.Module):
    """
    문장을 tokenize.
    """

    def __init__(self, tokenize) -> None:
        super().__init__()
        self.tokenize = tokenize

    def forward(self, x):
        return self.tokenize(x)


class IntegerEncoding(torch.nn.Module):
    def __init__(self, vocab) -> None:
        super().__init__()
        self.vocab = vocab

    def forward(self, x):
        return self.vocab.lookup_indices(x)

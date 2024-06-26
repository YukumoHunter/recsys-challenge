import torch
import torch.nn as nn

from pathlib import Path


class TitleEncoder(nn.Module):
    def __init__(self, word_vocab_path: Path):
        super(TitleEncoder, self).__init__()

        raise NotImplementedError("TitleEncoder is not implemented yet")

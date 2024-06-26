import torch
import torch.nn as nn

import numpy as np
from pathlib import Path

from ._modules import MultiHeadAttention, SelfAttention
from ..dataset import WordVocab


class TitleEncoder(nn.Module):
    def __init__(
        self,
        word_embedding_size: int,
        num_heads: int,
        head_size: int,
        word_vocab_path: Path,
        title_embedding_path: Path,
        pretrained_embedding_path: Path = None,
        dropout: float = 0.2,
    ):
        super(TitleEncoder, self).__init__()
        self.vocab = WordVocab.load_vocab(word_vocab_path)
        self.word_embedding = build_embedding_matrix(
            self.vocab,
            word_embedding_size,
            pretrained_embedding_path,
        )

        # article_id -> title token sequence
        self.title_embedding = torch.from_numpy(np.load(title_embedding_path)).long()

        self.mhs_attn = MultiHeadAttention(word_embedding_size, num_heads, head_size)

        mhs_output_size = num_heads * head_size
        self.word_self_attn = SelfAttention(mhs_output_size, mhs_output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seqs):
        titles = self.title_embedding[seqs]
        word_embeds = self.dropout(self.word_embedding(titles))

        h = self.dropout(self.mhs_attn(word_embeds, word_embeds, word_embeds))

        return self.word_self_attn(h)


def build_embedding_matrix(
    vocab: WordVocab, embedding_size: int, pretrained_embedding_path: Path = None
):
    num_embeddings = len(vocab)

    if pretrained_embedding_path is None:
        weights = np.load(pretrained_embedding_path)
        weights = torch.from_numpy(weights).float()

        assert list(weights.size()) == [
            num_embeddings,
            embedding_size,
        ], "Pretrained embedding size mismatch"

        print(f"Loaded pretrained embeddings from {pretrained_embedding_path}")
        return nn.Embedding.from_pretrained(weights, freeze=False)

    return nn.Embedding(num_embeddings, embedding_size)

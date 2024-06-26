import polars as pl
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

import torch

from recsys_challenge.dataset._vocab import WordVocab


def embed_tokens(x, model, tokenizer):
    tokens = torch.tensor([tokenizer.convert_tokens_to_ids(x)]).unsqueeze(1)
    embeddings = model.embeddings(tokens)
    return embeddings


def setup_word_embedder(embeddings="bert-multilingual-cased"):
    if embeddings == "bert-multilingual-cased":
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(
            "google-bert/bert-base-multilingual-cased"
        )
        model = AutoModel.from_pretrained("google-bert/bert-base-multilingual-cased")
        return lambda x: tokenizer.tokenize(
            x, add_special_tokens=False
        ), lambda x: embed_tokens(x, model, tokenizer)
    else:
        raise ValueError(f"Unknown embeddings: {embeddings}")


def build_vocab(values, output_path, lower=False, max_size=140_000, min_freq=1):
    vocab = WordVocab(values, max_size=max_size, min_freq=min_freq, lower=lower)
    vocab.save_vocab(output_path)
    return vocab


def build_word_embeddings(vocab, embedder, output_path, embedding_dim=768):
    weights_matrix = np.zeros((len(vocab), embedding_dim))
    for i, word in tqdm(
        enumerate(vocab.itos), total=len(vocab), desc="Building word embeddings"
    ):
        embedding = embedder(word).cpu().detach().numpy()
        weights_matrix[i] = embedding[0, 0, :].squeeze()

    np.save(output_path, weights_matrix)


def build_article_id_to_title(
    article_df: pl.DataFrame,
    article_id_vocab: WordVocab,
    word_vocab: WordVocab,
    output_path: Path,
    max_title_len: int = 20,
):
    article_to_title = np.zeros((len(article_id_vocab) + 1, max_title_len), dtype=int)
    article_to_title[0] = word_vocab.to_seq("<pad>", seq_len=max_title_len)
    for row in tqdm(
        article_df.iter_rows(named=True),
        desc="Building article id -> title mapping",
        total=len(article_df),
    ):
        article_id, title = row["article_id"], row["title"]
        # this needs to be a string instead of an integer, as the vocab uses strings
        article_index = article_id_vocab.stoi[f"{article_id}"]
        article_to_title[article_index], cur_len = word_vocab.to_seq(
            title, seq_len=max_title_len, with_len=True
        )

    np.save(output_path, article_to_title)
    print("Title embedding: ", article_to_title.shape)

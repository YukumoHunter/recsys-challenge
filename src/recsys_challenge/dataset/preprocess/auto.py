from pathlib import Path
import polars as pl

import argparse

from recsys_challenge.dataset._vocab import WordVocab

from recsys_challenge.utils._constants import (
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
)

from recsys_challenge.dataset.preprocess.vocab import (
    setup_word_embedder,
    build_vocab,
    build_word_embeddings,
    build_article_id_to_title,
)


from recsys_challenge.dataset.preprocess.neighbors import (
    build_one_hop_neighbors,
    build_two_hop_neighbors,
)

from recsys_challenge.dataset.preprocess.examples import load_hop_dict, build_examples


def main(args):
    split = args.split
    no_test = args.no_test

    PATH = Path(f"./data/{split}")
    TEST_PATH = Path("./data/testset")

    TRAIN_SPLIT = "train"
    VAL_SPLIT = "validation"
    TEST_SPLIT = "test"

    OUTPUT_PATH_VOCAB = Path("./data/vocab")
    OUTPUT_PATH_NEIGHBORS = Path("./data/neighbors")
    OUTPUT_PATH_EXAMPLES = Path("./data/examples")

    ## Generate vocab

    df_behaviors = pl.read_parquet(PATH / TRAIN_SPLIT / "behaviors.parquet")
    # df_history = pl.read_parquet(PATH / TRAIN_SPLIT / "history.parquet")
    df_articles = pl.read_parquet(PATH / "articles.parquet")

    tokenizer, word_embedder = setup_word_embedder()

    df_articles_tok = df_articles.with_columns(
        title_tokenized=pl.col(DEFAULT_TITLE_COL).map_elements(
            lambda x: " ".join(tokenizer(x)), return_dtype=pl.String
        )
    )

    _ = build_vocab(
        df_behaviors.get_column(DEFAULT_USER_COL),
        OUTPUT_PATH_VOCAB / "user_id_vocab.bin",
    )

    articles_vocab = build_vocab(
        df_articles_tok.get_column("article_id"),
        OUTPUT_PATH_VOCAB / "articles_id_vocab.bin",
    )

    word_vocab = build_vocab(
        df_articles_tok.get_column("title_tokenized"),
        OUTPUT_PATH_VOCAB / "word_vocab.bin",
    )

    build_word_embeddings(
        word_vocab,
        word_embedder,
        OUTPUT_PATH_VOCAB / "word_embeddings.npy",
    )

    build_article_id_to_title(
        df_articles_tok,
        articles_vocab,
        word_vocab,
        OUTPUT_PATH_VOCAB / "article_id_to_title.npy",
    )

    ## Generate neighbors

    for split in [TRAIN_SPLIT, TEST_SPLIT]:
        if split == TEST_SPLIT:
            df_behaviors = pl.read_parquet(TEST_PATH / split / "behaviors.parquet")
        else:
            df_behaviors = pl.read_parquet(PATH / split / "behaviors.parquet")

        vocab_articles = WordVocab.load_vocab(
            OUTPUT_PATH_VOCAB / "articles_id_vocab.bin"
        )
        vocab_user = WordVocab.load_vocab(OUTPUT_PATH_VOCAB / "user_id_vocab.bin")

        # Train one- and two-hops
        train_user_one_hop, train_article_one_hop = build_one_hop_neighbors(
            df_behaviors, vocab_user, vocab_articles, split, OUTPUT_PATH_NEIGHBORS
        )
        build_two_hop_neighbors(
            train_user_one_hop, train_article_one_hop, split, OUTPUT_PATH_NEIGHBORS
        )

    ## Generate examples
    if no_test:
        splits = [TRAIN_SPLIT, VAL_SPLIT]
    else:
        splits = [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT]

    for split in splits:
        if split == TEST_SPLIT:
            df_behaviors = pl.read_parquet(TEST_PATH / split / "behaviors.parquet")
        else:
            df_behaviors = pl.read_parquet(PATH / split / "behaviors.parquet")

        hops_split = "train" if split == VAL_SPLIT else split

        vocab_articles = WordVocab.load_vocab(
            OUTPUT_PATH_VOCAB / "articles_id_vocab.bin"
        )
        vocab_user = WordVocab.load_vocab(OUTPUT_PATH_VOCAB / "user_id_vocab.bin")

        user_one_hop = load_hop_dict(
            OUTPUT_PATH_NEIGHBORS / f"{hops_split}-user_one_hops.txt"
        )
        article_one_hop = load_hop_dict(
            OUTPUT_PATH_NEIGHBORS / f"{hops_split}-article_one_hops.txt"
        )

        user_two_hop = load_hop_dict(
            OUTPUT_PATH_NEIGHBORS / f"{hops_split}-user_two_hops.txt"
        )
        article_two_hop = load_hop_dict(
            OUTPUT_PATH_NEIGHBORS / f"{hops_split}-article_two_hops.txt"
        )

        match split:
            case "train":
                output_name = "training_examples.tsv"
            case "validation":
                output_name = "validation_examples.tsv"
            case "test":
                output_name = "test_examples.tsv"
            case _:
                raise ValueError(f"Unknown split {split}")

        build_examples(
            df_behaviors,
            vocab_user,
            vocab_articles,
            user_one_hop,
            article_one_hop,
            user_two_hop,
            article_two_hop,
            OUTPUT_PATH_EXAMPLES,
            output_name=output_name,
            test=(split != TRAIN_SPLIT),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="small")
    parser.add_argument("--no_test", action="store_true", default=False)
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        choices=["random", "read_time", "scroll_percentage"],
        default="random",
    )
    args = parser.parse_args()

    main(args)

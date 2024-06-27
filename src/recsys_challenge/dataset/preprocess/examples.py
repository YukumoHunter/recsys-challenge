import json
from typing import Dict

from pathlib import Path
from tqdm.auto import tqdm

import random

import polars as pl

from recsys_challenge.dataset._vocab import WordVocab


# split examples into first positive and negative samples
# and return a list with positive in index 0
def make_samples(row, newsid_vocab, test):
    samples = row["article_ids_inview"]

    if not test:
        positive = row["article_ids_clicked"]
        negative = [x for x in samples if x not in positive]

        samples = [positive[0]] + negative

    return [newsid_vocab.stoi.get(str(sample), 0) for sample in samples]


def build_examples(
    df: pl.DataFrame,
    user_vocab: WordVocab,
    newsid_vocab: WordVocab,
    user_one_hop: Dict,
    news_one_hop: Dict,
    user_two_hop: Dict,
    news_two_hop: Dict,
    output_path: Path,
    negative_sampling_ratio: int = 4,
    max_user_one_hop: int = 50,
    max_news_one_hop: int = 50,
    max_user_two_hop: int = 15,
    max_news_two_hop: int = 15,
    output_name: str = "training_examples.tsv",
    test: bool = False,
    seed: int = 7,
):
    random.seed(seed)

    def _get_neighbors(neighbor_dict, key, max_neighbor_num):
        neighbors = neighbor_dict.get(key, [])
        return neighbors[:max_neighbor_num]

    f_out = output_path / output_name
    fw = open(f_out, "w", encoding="utf-8")

    for row in tqdm(df.iter_rows(named=True), desc="Building examples", total=len(df)):
        uid = str(row["user_id"])
        user_index = user_vocab.stoi.get(uid, 0)

        hist_news = _get_neighbors(user_one_hop, user_index, max_user_one_hop)
        neighbor_users = _get_neighbors(user_two_hop, user_index, max_user_two_hop)

        neighbor_news = []
        y = 0

        target_news = make_samples(row, newsid_vocab, test)

        # not enough negative samples
        if len(target_news) < (negative_sampling_ratio + 1):
            continue

        target_news = target_news[: negative_sampling_ratio + 1]


        for news_index in target_news:
            # hist_users.append(_get_neighors(news_one_hop, news_index, cfg.max_news_one_hop))
            neighbor = _get_neighbors(news_two_hop, news_index, max_news_two_hop)
            neighbor_news.append(neighbor)

        if test:
            for i, (target, neighbor) in enumerate(zip(target_news, neighbor_news)):
                j = {
                    "user": user_index,
                    "hist_news": hist_news,
                    "neighbor_users": neighbor_users,
                    "target_news": target,
                    # "hist_users": hist_users,
                    "y": 1 if i == 0 else 0,
                    "neighbor_news": neighbor,
                    "impression_id": row["impression_id"],
                }
                fw.write(json.dumps(j) + "\n")
        else:

            j = {
                "user": user_index,
                "hist_news": hist_news,
                "neighbor_users": neighbor_users,
                "target_news": target_news,
                # "hist_users": hist_users,
                "y": y,
                "neighbor_news": neighbor_news,
                "impression_id": row["impression_id"],
            }
            fw.write(json.dumps(j) + "\n")


def load_hop_dict(fpath: str) -> Dict:
    lines = open(fpath, "r", encoding="utf-8").readlines()
    d = dict()
    error_line_count = 0
    for line in lines:
        row = line.strip().split("\t")
        if len(row) != 2:
            error_line_count += 1
            continue
        key, vals = row[:2]
        vals = [int(x) for x in vals.split(",")]
        d[int(key)] = vals
    print("{} error lines: {}".format(fpath, error_line_count))
    return d

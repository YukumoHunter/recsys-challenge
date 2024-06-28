import random
from typing import Dict, Tuple, Literal

import polars as pl
from pathlib import Path
from tqdm.auto import tqdm

from recsys_challenge.dataset._vocab import WordVocab


def build_two_hop_neighbors(
    user_one_hop: Dict,
    news_one_hop: Dict,
    part: str,
    output_path: Path,
    max_user_two_hop: int = 15,
    max_news_two_hop: int = 15,
    sampling_strategy: Literal["random", "read_time", "scroll_percentage"] = "random",
):
    user_dict = dict()
    news_dict = dict()
    for user, news_list in tqdm(user_one_hop.items(), desc="Building hop-2 user"):
        two_hop_users = []
        for news in news_list:
            two_hop_users += news_one_hop[news]
        if len(two_hop_users) > max_user_two_hop:
            if sampling_strategy == "random":
                two_hop_users = random.sample(two_hop_users, max_user_two_hop)
            else:
                two_hop_users = two_hop_users[-max_user_two_hop:]
        user_dict[user] = two_hop_users
    for news, user_list in tqdm(news_one_hop.items(), desc="Building hop-2 news"):
        two_hop_news = []
        for user in user_list:
            two_hop_news += user_one_hop[user]
        if len(two_hop_news) > max_news_two_hop:
            two_hop_news = random.sample(two_hop_news, max_news_two_hop)
        news_dict[news] = two_hop_news

    f_user = output_path / f"{part}-user_two_hops.txt"
    f_news = output_path / f"{part}-article_two_hops.txt"

    with open(f_user, "w", encoding="utf-8") as fw:
        for user, news_list in user_dict.items():
            news_list_str = ",".join([str(x) for x in news_list])
            fw.write("{}\t{}\n".format(user, news_list_str))
    with open(f_news, "w", encoding="utf-8") as fw:
        for news, user_list in news_dict.items():
            user_list_str = ",".join([str(x) for x in user_list])
            fw.write("{}\t{}\n".format(news, user_list_str))


def build_one_hop_neighbors(
    behavior_df: pl.DataFrame,
    user_vocab: WordVocab,
    newsid_vocab: WordVocab,
    part: str,
    output_path: Path,
    max_user_one_hop: int = 50,
    max_news_one_hop: int = 50,
    sampling_strategy: Literal["random", "read_time", "scroll_percentage"] = "random",
) -> Tuple[Dict, Dict]:
    behavior_df = behavior_df.fill_nan("")

    if sampling_strategy == "random":
        # random sampling, no need to sort
        pass
    else:
        # sort based on sampling strategy
        # generally, we want to ensure the HIGHEST values are at the bottom
        # so that when the list is truncated using max_user_one_hop, we keep the most important ones
        behavior_df = behavior_df.sort(sampling_strategy)

    user_dict = dict()
    news_dict = dict()
    for row in tqdm(
        behavior_df.iter_rows(named=True),
        # behavior_df[["user_id", "article_ids_inview"]].values,
        desc="Building Hop-1",
        total=len(behavior_df),
    ):
        uid, hist = row["user_id"], row["article_ids_inview"]

        uid = str(uid)
        if uid not in user_vocab.stoi:
            continue
        user_index = user_vocab.stoi[uid]

        if user_index not in user_dict:
            user_dict[user_index] = []

        for newsid in hist:
            newsid = str(newsid)
            if newsid not in newsid_vocab.stoi:
                continue
            news_index = newsid_vocab.stoi[newsid]
            if news_index not in news_dict:
                news_dict[news_index] = []
            # click_list.append([user_index, news_index])
            if len(user_dict[user_index]) < max_user_one_hop:
                user_dict[user_index].append(news_index)
            if len(news_dict[news_index]) < max_news_one_hop:
                news_dict[news_index].append(user_index)

    f_user = output_path / f"{part}-user_one_hops.txt"
    f_news = output_path / f"{part}-article_one_hops.txt"

    with open(f_user, "w", encoding="utf-8") as fw:
        for user, news_list in user_dict.items():
            news_list_str = ",".join([str(x) for x in news_list[:max_user_one_hop]])
            fw.write("{}\t{}\n".format(user, news_list_str))
    with open(f_news, "w", encoding="utf-8") as fw:
        for news, user_list in news_dict.items():
            user_list_str = ",".join([str(x) for x in user_list[:max_news_one_hop]])
            fw.write("{}\t{}\n".format(news, user_list_str))

    return user_dict, news_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._modules import SelfAttention


class GERLModel(nn.Module):
    def __init__(
        self,
        batch_size: int,
        neg_count: int,
        embedding_size: int,
        max_user_one_hop: int,
        max_user_two_hop: int,
        max_article_two_hop: int,
        id_embedding_size: int,
        user_count: int,
        article_count: int,
        dropout: float = 0.2,
    ):
        super(GERLModel, self).__init__()
        self.batch_size = batch_size
        self.neg_count = neg_count
        self.embedding_size = embedding_size

        self.max_user_one_hop = max_user_one_hop
        self.max_user_two_hop = max_user_two_hop
        self.max_article_two_hop = max_article_two_hop

        self.user_embedding = nn.Embedding(user_count, id_embedding_size)
        self.article_embedding = nn.Embedding(article_count, id_embedding_size)

        self.title_encoder = None  # todo; need title encoder

        self.user_two_hop_attn = SelfAttention(id_embedding_size, id_embedding_size)
        self.user_one_hop_attn = SelfAttention(id_embedding_size, id_embedding_size)

        self.article_two_hop_id_attn = SelfAttention(
            id_embedding_size, id_embedding_size
        )
        self.article_two_hop_title_attn = SelfAttention(
            id_embedding_size, id_embedding_size
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        user,
        hist_articles,
        neighbour_users,
        target_articles,
        neighbour_articles,
        num_target_news: int,
    ):
        user_embedding = self.dropout(self.user_embedding(user))
        neighbour_users_embedding = self.dropout(self.user_embedding(neighbour_users))
        neighbour_articles_embedding = self.dropout(
            self.article_embedding(neighbour_articles)
        )

        # users
        user_two_hop_repr = self.user_two_hop_attn(neighbour_users_embedding)
        hist_articles = hist_articles.view(-1)
        hist_articles_repr = self.title_encoder(hist_articles)
        hist_articles_repr = hist_articles_repr.view(
            -1, self.max_user_one_hop, self.embedding_size
        )
        user_one_hop_repr = self.user_one_hop_attn(hist_articles_repr)

        # articles
        target_articles = target_articles.view(-1)
        target_articles_repr = self.title_encoder(target_articles)
        target_articles_repr = target_articles_repr.view(
            -1, num_target_news, self.embedding_size
        )

        neighbour_articles_embedding = neighbour_articles_embedding.view(
            -1, self.max_article_two_hop, self.embedding_size
        )
        articles_two_hop_id_repr = self.article_two_hop_id_attn(
            neighbour_articles_embedding
        )
        articles_two_hop_id_repr = articles_two_hop_id_repr.view(
            -1, num_target_news, self.embedding_size
        )

        neighbour_articles = neighbour_articles.view(-1)
        neighbour_articles_repr = self.title_encoder(neighbour_articles)
        neighbour_articles_repr = neighbour_articles_repr.view(
            -1, self.max_article_two_hop, self.embedding_size
        )
        articles_two_hop_title_repr = self.article_two_hop_title_attn(
            neighbour_articles_repr
        )
        articles_two_hop_title_repr = articles_two_hop_title_repr.view(
            -1, num_target_news, self.embedding_size
        )

        # logits
        final_user_repr = (
            (user_one_hop_repr + user_embedding + user_two_hop_repr)
            .repeat(1, num_target_news)
            .view(-1, self.embedding_size)
        )

        final_target_articles_repr = (
            target_articles_repr
            + articles_two_hop_id_repr
            + articles_two_hop_title_repr
        ).view(-1, self.embedding_size)

        return torch.sum(final_user_repr * final_target_articles_repr, dim=-1).view(
            -1, num_target_news
        )

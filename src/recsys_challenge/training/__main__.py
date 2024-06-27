import torch
from torch.utils.data import DataLoader
import argparse

from tqdm import tqdm

from ..model import GERLModel
from ..dataset import TrainingDataset, ValidationDataset

import numpy as np
import random


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Using seed: {args.seed}")
    setup_seed(args.seed)

    print("Setting up training dataset...")
    train_dataset = setup_train_dataset(args)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    print("Setting up model...")
    model = setup_model(args, device)

    # todo: accumulate gradients and average over multiple devices
    steps_per_epoch = len(train_dataset)
    train_steps = args.epochs * steps_per_epoch
    print("> Total training steps:", train_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    pbar = tqdm(range(args.epochs), desc="Epoch")
    for epoch in pbar:
        train_metrics = train_single_epoch(model, train_loader, optimizer, device)
        # todo: validation
        pbar.write(
            f"[{epoch + 1}/{args.epochs}] Train Loss: {train_metrics['loss']:.4f}"
        )


def train_single_epoch(model, train_loader, optimizer, device):
    model.train()

    total_loss = []

    for batch in tqdm(train_loader, desc="Train", leave=False):
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model.forward_train(batch)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    return {"loss": sum(total_loss) / len(total_loss)}


def setup_train_dataset(args):
    train_dataset = TrainingDataset(
        args.max_user_one_hop,
        args.max_user_two_hop,
        args.max_article_two_hop,
        args.train_examples,
    )

    return train_dataset


def setup_model(args, device):
    model = GERLModel(
        batch_size=args.batch_size,
        neg_count=args.neg_count,
        max_user_one_hop=args.max_user_one_hop,
        max_user_two_hop=args.max_user_two_hop,
        max_article_two_hop=args.max_article_two_hop,
        id_embedding_size=args.id_embedding_size,
        word_embedding_size=args.word_embedding_size,
        user_count=args.user_count,
        article_count=args.article_count,
        word_vocab_path=args.word_vocab_path,
        title_embedding_path=args.title_embedding_path,
        num_heads=args.num_heads,
        head_size=args.head_size,
        pretrained_embedding_path=args.pretrained_embedding_path,
        dropout=args.dropout,
    ).to(device)

    model.title_encoder.title_embedding = model.title_encoder.title_embedding.to(device)

    return model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    params = parser.add_argument_group("training config")
    params.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    params.add_argument(
        "--lr",
        type=float,
        default=0.0001,
    )
    params.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    params.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )

    dataset_params = parser.add_argument_group("dataset config")
    dataset_params.add_argument(
        "--train-examples",
        type=str,
        default="data/examples/training_examples.tsv",
    )
    dataset_params.add_argument(
        "--validation-examples",
        type=str,
        default="data/examples/validation_examples.tsv",
    )
    dataset_params.add_argument(
        "--word-vocab-path",
        type=str,
        default="data/vocab/word_vocab.bin",
    )
    dataset_params.add_argument(
        "--title-embedding-path",
        type=str,
        default="data/vocab/article_id_to_title.npy",
    )
    dataset_params.add_argument(
        "--pretrained-embedding-path",
        type=str,
        default="data/vocab/word_embeddings.npy",
    )

    dataset_params.add_argument(
        "--user-count",
        type=int,
        default=50000,
    )
    dataset_params.add_argument(
        "--article-count",
        type=int,
        default=30000,
    )

    model_params = parser.add_argument_group("model config")
    model_params.add_argument(
        "--neg-count",
        type=int,
        default=4,
    )
    model_params.add_argument(
        "--max-user-one-hop",
        type=int,
        default=50,
    )
    model_params.add_argument(
        "--max-user-two-hop",
        type=int,
        default=15,
    )
    model_params.add_argument(
        "--max-article-two-hop",
        type=int,
        default=15,
    )
    model_params.add_argument(
        "--id-embedding-size",
        type=int,
        default=128,
    )
    model_params.add_argument(
        "--word-embedding-size",
        type=int,
        default=768,
    )
    model_params.add_argument(
        "--num-heads",
        type=int,
        default=8,
    )
    model_params.add_argument(
        "--head-size",
        type=int,
        default=16,
    )
    model_params.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()

import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from tqdm import tqdm

from ..model import GERLModel
from ..dataset import TrainingDataset, ValidationDataset

from ..evaluation import (
    MetricEvaluator,
    AucScore,
    MrrScore,
    NdcgScore,
    LogLossScore,
    RootMeanSquaredError,
    AccuracyScore,
    F1Score,
)

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

    print("Setting up validation dataset...")
    validation_dataset = setup_validation_dataset(args)
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    print("Setting up model...")
    model = setup_model(args, device)

    Path(".checkpoints").mkdir(exist_ok=True, parents=True)

    # todo: accumulate gradients and average over multiple devices
    steps_per_epoch = len(train_dataset)
    train_steps = args.epochs * steps_per_epoch
    print("> Total training steps:", train_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    pbar = tqdm(range(args.epochs), desc="Epoch")
    for epoch in pbar:
        train_metrics = train_single_epoch(model, train_loader, optimizer, device)
        dev_metrics = evaluate(model, validation_loader, device)

        # save model checkpoint
        torch.save(model.state_dict(), f".checkpoints/model_checkpoint_{epoch}.pt")

        pbar.write(
            f"[{epoch + 1}/{args.epochs}] Train Loss: {train_metrics['loss']:.4f}"
        )
        pbar.write(f"Dev Metrics: {dev_metrics}")


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


def evaluate(model, validation_loader, device):
    model.eval()

    metrics = {}

    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Dev", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model.forward_eval(batch)

            target = batch["y"]

            predictions = logits.cpu().numpy().tolist()
            targets = target.long().cpu().numpy().tolist()
            impression_ids = batch["impression_id"].cpu().numpy().tolist()

            all_labels, all_preds = group_labels(targets, predictions, impression_ids)

            metrics = (
                MetricEvaluator(
                    all_labels,
                    all_preds,
                    metric_functions=[
                        AucScore(),
                        MrrScore(),
                        NdcgScore(k=5),
                        NdcgScore(k=10),
                        LogLossScore(),
                        RootMeanSquaredError(),
                        AccuracyScore(threshold=0.5),
                        F1Score(threshold=0.5),
                    ],
                )
                .evaluate()
                .evaluations
            )

            for k, v in metrics.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

    for k, v in metrics.items():
        metrics[k] = sum(v) / len(v)

    return metrics


def group_labels(labels, preds, group_keys):
    """Devide labels and preds into several group according to values in group keys.
    Args:
        labels (list): ground truth label list.
        preds (list): prediction score list.
        group_keys (list): group key list.
    Returns:
        all_labels: labels after group.
        all_preds: preds after group.
    """

    all_keys = list(set(group_keys))
    group_labels = {k: [] for k in all_keys}
    group_preds = {k: [] for k in all_keys}

    for label, pred, key in zip(labels, preds, group_keys):
        group_labels[key].append(label)
        group_preds[key].append(pred)

    all_labels = []
    all_preds = []
    for k in all_keys:
        all_labels.append(group_labels[k])
        all_preds.append(group_preds[k])

    return all_labels, all_preds


def setup_train_dataset(args):
    train_dataset = TrainingDataset(
        args.max_user_one_hop,
        args.max_user_two_hop,
        args.max_article_two_hop,
        args.train_examples,
    )

    return train_dataset


def setup_validation_dataset(args):
    validation_dataset = ValidationDataset(
        args.max_user_one_hop,
        args.max_user_two_hop,
        args.max_article_two_hop,
        args.validation_examples,
    )

    return validation_dataset


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

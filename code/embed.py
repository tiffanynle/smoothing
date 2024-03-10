import argparse
import os

import torch
from datasets import DATASETS, get_dataset
from architectures import EMBEDDINGS, get_foundation_model
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="generate embeddings over a dataset")
parser.add_argument("dataset", type=str, choices=DATASETS)
parser.add_argument("embedding", type=str, choices=EMBEDDINGS)
parser.add_argument("outdir", type=str, help="folder to save embeddings")
parser.add_argument(
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--batch", default=256, type=int, metavar="N", help="batchsize (default: 256)"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for random number generator"
)
parser.add_argument(
    "--gpu", default=None, type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)
args = parser.parse_args()


def generate_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: str,
    split: str,
) -> dict[torch.Tensor, torch.Tensor]:
    embeddings = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.cuda()

            # retrieve embedding from model
            embed = model(inputs).cpu()

            # store embeddings and labels
            embeddings.append(embed)
            labels.append(targets)

    return {
        "embeddings": torch.cat(embeddings),
        "labels": torch.cat(labels),
        "dataset": dataset,
        "split": split,
    }


def save_embeddings(
    embeddings: dict[torch.Tensor, torch.Tensor],
    outdir: str,
    embedding: str,
    dataset: str,
    split: str,
) -> None:
    torch.save(embeddings, os.path.join(outdir, f"{embedding}_{dataset}_{split}.pt"))


def main():
    print(os.environ)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    train_dataset = get_dataset(
        args.dataset, "train", embed=True, embedding=args.embedding
    )
    test_dataset = get_dataset(
        args.dataset, "test", embed=True, embedding=args.embedding
    )
    pin_memory = args.dataset == "imagenet"
    train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    model = get_foundation_model(arch=args.embedding).cuda()

    train_embeds = generate_embeddings(model, train_loader, args.dataset, "train")
    test_embeds = generate_embeddings(model, test_loader, args.dataset, "test")

    save_embeddings(train_embeds, args.outdir, args.embedding, args.dataset, "train")
    save_embeddings(test_embeds, args.outdir, args.embedding, args.dataset, "test")


if __name__ == "__main__":
    main()

# evaluate a smoothed classifier on a dataset
import argparse
import datetime
import os
from time import time

import torch
from architectures import get_head
from core import Smooth
from torch.utils.data import TensorDataset
from train_utils import minmax_normalize

import setGPU
from datasets import DATASETS, EMBEDDINGS, get_num_classes, NormalizeLayer

parser = argparse.ArgumentParser(description="Certify many examples")
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument(
    "embeddir",
    type=str,
    help="folder where embeddings are stored",
)
parser.add_argument(
    "base_classifier", type=str, help="path to saved pytorch model of base classifier"
)
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument(
    "--split", choices=["train", "test"], default="test", help="train or test set"
)
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    embedding = checkpoint["embedding"]

    head = get_head(checkpoint["head"], checkpoint["backbone"], args.dataset)
    head = head.cuda()

    # load mean, sds for standardization after noising
    data_norm = torch.load(f"{args.embeddir}/{embedding}_{args.dataset}_norm.pt")
    means = data_norm["mean"]
    sds = data_norm["sd"]

    # wrap model and load state dict
    normalize_layer = NormalizeLayer(means=means, sds=sds)
    model = torch.nn.Sequential(normalize_layer, head)
    model.load_state_dict(checkpoint["state_dict"])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(model, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, "w")
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    data_norm = torch.load(f"{args.embeddir}/{embedding}_{args.dataset}_norm.pt")
    train_min = data_norm["min"]
    train_max = data_norm["max"]

    # load tensor dataset
    data = torch.load(f"{args.embeddir}/{embedding}_{args.dataset}_{args.split}.pt")

    inputs = minmax_normalize(data["inputs"], min=train_min, max=train_max)
    labels = data["labels"]

    dataset = TensorDataset(inputs, labels)

    for i in range(len(dataset)):
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(
            x, args.N0, args.N, args.alpha, args.batch
        )
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print(
            "{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed
            ),
            file=f,
            flush=True,
        )

    f.close()

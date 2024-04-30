import argparse
import datetime
import os
import time

import torch
from architectures import HEADS, get_head
from datasets import DATASETS, EMBEDDINGS, NormalizeLayer
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from train_utils import AverageMeter, accuracy, init_logfile, log, minmax_normalize

import wandb

parser = argparse.ArgumentParser("Training for Vision Transformer Head")
parser.add_argument("dataset", type=str, choices=DATASETS)
parser.add_argument("embedding", type=str, choices=EMBEDDINGS)
parser.add_argument("head", type=str, default=None, choices=HEADS)
parser.add_argument(
    "embeddir",
    type=str,
    help="folder where embeddings are stored",
)
parser.add_argument("outdir", type=str, help="folder to save model and training log)")
parser.add_argument(
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--batch", default=256, type=int, metavar="N", help="batchsize (default: 256)"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--lr_step_size",
    type=int,
    default=30,
    help="How often to decrease learning by gamma.",
)
parser.add_argument(
    "--gamma", type=float, default=0.1, help="LR is multiplied by gamma on schedule."
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--noise_sd",
    default=0.0,
    type=float,
    help="standard deviation of Gaussian noise for data augmentation",
)
parser.add_argument(
    "--gpu", default=None, type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)
parser.add_argument(
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument("--project", type=str, help="project name for WandB")
parser.add_argument("--entity", type=str, help="entity name for WandB")
args = parser.parse_args()


def main():
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    # load train, test datasets
    data_train = torch.load(f"{args.embeddir}/{args.embedding}_{args.dataset}_train.pt")
    data_test = torch.load(f"{args.embeddir}/{args.embedding}_{args.dataset}_test.pt")
    data_norm = torch.load(f"{args.embeddir}/{args.embedding}_{args.dataset}_norm.pt")

    # what backbone did we use?
    backbone = data_train["backbone"]

    # min-max normalization and clamping so inputs will fall between [0, 1]
    train_min = data_norm["min"]
    train_max = data_norm["max"]
    inputs_train = minmax_normalize(data_train["inputs"], min=train_min, max=train_max)
    inputs_test = minmax_normalize(data_test["inputs"], min=train_min, max=train_max)

    assert (
        inputs_train.max() <= 1 and inputs_train.min() >= 0
    ), "error with normalization of train tensors"
    assert (
        inputs_test.max() <= 1 and inputs_test.min() >= 0
    ), "error with normalization of test tensors"

    labels_train = data_train["labels"]
    labels_test = data_test["labels"]

    train_dataset = TensorDataset(inputs_train, labels_train)
    test_dataset = TensorDataset(inputs_test, labels_test)

    pin_memory = args.dataset == "imagenet"
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
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

    if args.head == "linear":
        head = get_head(args.head, backbone, args.dataset)
        head = DataParallel(head).cuda()

    means = data_norm["mean"]
    sds = data_norm["sd"]
    
    # wrap this in a Sequential so we get standardization after noise
    normalize_layer = NormalizeLayer(means=means, sds=sds)
    model = torch.nn.Sequential(normalize_layer, head)

    criterion = CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    logfilename = os.path.join(args.outdir, "log.txt")
    init_logfile(
        logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc"
    )

    wandb.init(
        project=args.project,
        entity=args.entity,
        name=f"train: {args.embedding} {args.head} noise {args.noise_sd}",
        config=vars(args),
    )

    for epoch in range(args.epochs):
        before = time.time()
        train_loss, train_acc = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            args.noise_sd,
        )
        scheduler.step()
        test_loss, test_acc = test(
            test_loader,
            model,
            criterion,
            args.noise_sd,
        )
        after = time.time()

        log(
            logfilename,
            "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch,
                str(datetime.timedelta(seconds=(after - before))),
                scheduler.get_last_lr()[0],
                train_loss,
                train_acc,
                test_loss,
                test_acc,
            ),
        )
        wandb.log(
            {
                "lr": scheduler.get_last_lr()[0],
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "embedding": args.embedding,
                "backbone": backbone,
                "head": args.head,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(args.outdir, "checkpoint.pth.tar"),
        )
    wandb.finish()


def train(
    loader: DataLoader,
    model: torch.nn.Module,
    criterion,
    optimizer: Optimizer,
    epoch: int,
    noise_sd: float,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()

        # augment inputs with noise
        # inputs = inputs + torch.randn_like(inputs, device="cuda") * noise_sd
        inputs = inputs + torch.randn_like(inputs) * noise_sd

        # compute output
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )

    return (losses.avg, top1.avg)


def test(
    loader: DataLoader,
    model: torch.nn.Module,
    criterion,
    noise_sd: float,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # inputs = inputs + torch.randn_like(inputs, device="cuda") * noise_sd

            inputs = inputs + torch.randn_like(inputs) * noise_sd

            # compute output
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        i,
                        len(loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )

        return (losses.avg, top1.avg)


if __name__ == "__main__":
    main()

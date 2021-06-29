# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch.distributions
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.language_bottleneck.mnist_classification.data import DoubleMnist
from egg.zoo.language_bottleneck.mnist_overfit.archs import Receiver, Sender
from egg.zoo.language_bottleneck.mnist_overfit.data import corrupt_labels_
from egg.zoo.language_bottleneck.relaxed_channel import AlwaysRelaxedWrapper


def diff_loss_symbol(
    _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
):
    loss = F.nll_loss(receiver_output, labels, reduction="none").mean()
    acc = (receiver_output.argmax(dim=1) == labels).float()
    return loss, {"acc": acc}


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender (default: 1)",
    )

    parser.add_argument(
        "--early_stopping_thr",
        type=float,
        default=1.0,
        help="Early stopping threshold on accuracy (default: 1.0)",
    )

    parser.add_argument("--deeper", type=int, default=0, help="Addition FC layer")

    parser.add_argument(
        "--deeper_alice", type=int, default=0, help="Addition FC layer goes to Alice"
    )

    parser.add_argument(
        "--p_corrupt",
        type=float,
        default=0,
        help="Probability of corrupting a label (default: 0.0)",
    )

    parser.add_argument(
        "--softmax_non_linearity",
        type=int,
        default=0,
        help="Disable GS training, treat channel as softmax non-linearity (default: 0)",
    )

    parser.add_argument(
        "--linear_channel",
        type=int,
        default=0,
        help="Disable GS training, treat channel as a linear connection (default: 0)",
    )
    parser.add_argument("--force_discrete", type=int, default=0, help="")

    args = core.init(parser, params)

    assert 0.0 <= args.p_corrupt <= 1.0
    return args


def main(params):
    opts = get_params(params)
    print(opts)

    kwargs = {"num_workers": 1, "pin_memory": True} if opts.cuda else {}
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, download=False, transform=transform
    )

    n_classes = 10

    corrupt_labels_(
        dataset=train_dataset, p_corrupt=opts.p_corrupt, seed=opts.random_seed + 1
    )
    label_mapping = torch.LongTensor([x % n_classes for x in range(100)])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opts.batch_size, shuffle=True, **kwargs
    )
    train_loader = DoubleMnist(train_loader, label_mapping)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16 * 1024, shuffle=False, **kwargs
    )
    test_loader = DoubleMnist(test_loader, label_mapping)

    deeper_alice = opts.deeper_alice == 1 and opts.deeper == 1
    deeper_bob = opts.deeper_alice != 1 and opts.deeper == 1

    sender = Sender(
        vocab_size=opts.vocab_size,
        deeper=deeper_alice,
        linear_channel=opts.linear_channel == 1,
        softmax_channel=opts.softmax_non_linearity == 1,
    )
    receiver = Receiver(
        vocab_size=opts.vocab_size, n_classes=n_classes, deeper=deeper_bob
    )

    if (
        opts.softmax_non_linearity != 1
        and opts.linear_channel != 1
        and opts.force_discrete != 1
    ):
        sender = AlwaysRelaxedWrapper(sender, temperature=opts.temperature)
    elif opts.force_discrete == 1:
        sender = core.GumbelSoftmaxWrapper(sender, temperature=opts.temperature)

    game = core.SymbolGameGS(sender, receiver, diff_loss_symbol)

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=[
            core.ConsoleLogger(as_json=True, print_train_loss=True),
            EarlyStopperAccuracy(opts.early_stopping_thr),
        ],
    )

    trainer.train(n_epochs=opts.n_epochs)
    core.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])

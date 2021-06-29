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
from egg.zoo.language_bottleneck.intervention import CallbackEvaluator
from egg.zoo.language_bottleneck.mnist_classification.archs import Receiver, Sender
from egg.zoo.language_bottleneck.mnist_classification.data import DoubleMnist


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
        "--sender_rows",
        type=int,
        default=28,
        help="Number of image rows revealed to Sender (default: 28)",
    )
    parser.add_argument(
        "--early_stopping_thr",
        type=float,
        default=0.98,
        help="Early stopping threshold on accuracy (defautl: 0.98)",
    )
    parser.add_argument("--n_labels", type=int, default=10)
    parser.add_argument("--n_hidden", type=int, default=0)

    args = core.init(parser, params=params)
    return args


def main(params):
    opts = get_params(params)
    print(opts)

    label_mapping = torch.LongTensor([x % opts.n_labels for x in range(100)])
    print("# label mapping", label_mapping.tolist())

    kwargs = {"num_workers": 1, "pin_memory": True} if opts.cuda else {}
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opts.batch_size, shuffle=True, **kwargs
    )
    train_loader = DoubleMnist(train_loader, label_mapping)

    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16 * 1024, shuffle=False, **kwargs
    )
    test_loader = DoubleMnist(test_loader, label_mapping)

    sender = Sender(vocab_size=opts.vocab_size)
    receiver = Receiver(
        vocab_size=opts.vocab_size, n_classes=opts.n_labels, n_hidden=opts.n_hidden
    )
    sender = core.GumbelSoftmaxWrapper(sender, temperature=opts.temperature)

    logging_strategy = core.LoggingStrategy(store_sender_input=False)
    game = core.SymbolGameGS(
        sender, receiver, diff_loss_symbol, logging_strategy=logging_strategy
    )

    optimizer = core.build_optimizer(game.parameters())

    intervention = CallbackEvaluator(
        test_loader,
        device=opts.device,
        loss=game.loss,
        is_gs=True,
        var_length=False,
        input_intervention=False,
    )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=[
            core.ConsoleLogger(as_json=True),
            EarlyStopperAccuracy(opts.early_stopping_thr),
            intervention,
        ],
    )

    trainer.train(n_epochs=opts.n_epochs)
    core.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])

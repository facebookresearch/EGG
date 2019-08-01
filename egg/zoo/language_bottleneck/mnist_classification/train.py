# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from torch.nn import functional as F
import torch.utils.data
from torchvision import datasets, transforms
import torch.distributions
import egg.core as core

from egg.zoo.language_bottleneck.mnist_classification.archs import Sender, Receiver
from egg.zoo.language_bottleneck.intervention import CallbackEvaluator
from egg.zoo.language_bottleneck.mnist_classification.data import TakeFirstLoader, SplitImages
from egg.core import EarlyStopperAccuracy


def diff_loss_symbol(_sender_input, _message, _receiver_input, receiver_output, labels):
    loss = F.nll_loss(receiver_output, labels, reduction='none').mean()
    acc = (receiver_output.argmax(dim=1) == labels).float().mean()
    return loss, {'acc': acc}


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="GS temperature for the sender (default: 1)")
    parser.add_argument('--sender_rows', type=int, default=28,
        help="Number of image rows revealed to Sender (default: 28)")
    parser.add_argument('--receiver_rows', type=int, default=28,
                        help="Number of image rows revealed to Receiver (default: 28)")
    parser.add_argument('--early_stopping_thr', type=float, default=0.98,
                        help="Early stopping threshold on accuracy (defautl: 0.98)")

    args = core.init(parser, params=params)
    return args


def main(params):
    opts = get_params(params)
    print(json.dumps(vars(opts)))

    kwargs = {'num_workers': 1, 'pin_memory': True} if opts.cuda else {}
    transform = transforms.ToTensor()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
           transform=transform),
           batch_size=opts.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
           batch_size=16 * 1024, shuffle=False, **kwargs)

    n_classes = 10

    binarize = False

    test_loader = SplitImages(TakeFirstLoader(test_loader, n=1), rows_receiver=opts.receiver_rows,
                              rows_sender=opts.sender_rows, binarize=binarize, receiver_bottom=True)
    train_loader = SplitImages(train_loader, rows_sender=opts.sender_rows, rows_receiver=opts.receiver_rows,
                               binarize=binarize, receiver_bottom=True)

    sender = Sender(vocab_size=opts.vocab_size)
    receiver = Receiver(vocab_size=opts.vocab_size, n_classes=n_classes)
    sender = core.GumbelSoftmaxWrapper(sender, temperature=opts.temperature)

    game = core.SymbolGameGS(sender, receiver, diff_loss_symbol)

    optimizer = core.build_optimizer(game.parameters())

    intervention = CallbackEvaluator(test_loader, device=opts.device, loss=game.loss,
                                     is_gs=True,
                                     var_length=False,
                                     input_intervention=True)

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=test_loader,
                           callbacks=[core.ConsoleLogger(as_json=True),
                                      EarlyStopperAccuracy(opts.early_stopping_thr),
                                      intervention])

    trainer.train(n_epochs=opts.n_epochs)
    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])


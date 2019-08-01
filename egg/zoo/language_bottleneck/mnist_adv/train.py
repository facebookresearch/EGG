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

from egg.zoo.language_bottleneck.mnist_adv.archs import Sender, Receiver
from egg.zoo.language_bottleneck.relaxed_channel import AlwaysRelaxedWrapper
from egg.core import EarlyStopperAccuracy


def diff_loss_symbol(_sender_input, _message, _receiver_input, receiver_output, labels):
    loss = F.nll_loss(receiver_output, labels, reduction='none').mean()
    acc = (receiver_output.argmax(dim=1) == labels).float().mean()
    return loss, {'acc': acc}


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="GS temperature for the sender (default: 1)")

    parser.add_argument('--early_stopping_thr', type=float, default=1.0,
                        help="Early stopping threshold on accuracy (default: 1.0)")

    parser.add_argument('--softmax_non_linearity', type=int, default=0,
                        help="Disable GS training, treat channel as softmax non-linearity (default: 0)")

    parser.add_argument('--linear_channel', type=int, default=0,
                        help="Disable GS training, treat channel as a linear connection (default: 0)")

    args = core.init(parser, params)

    assert not (args.softmax_non_linearity == 1 and args.linear_channel == 1)
    return args


def main(params):
    opts = get_params(params)
    print(json.dumps(vars(opts)))

    kwargs = {'num_workers': 1, 'pin_memory': True} if opts.cuda else {}
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST('./data', train=True, download=True,
                   transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=False,
                   transform=transform)
    n_classes = 10

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=opts.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=opts.batch_size, shuffle=False, **kwargs)

    sender = Sender(vocab_size=opts.vocab_size, linear_channel=opts.linear_channel == 1,
                    softmax_channel=opts.softmax_non_linearity)
    receiver = Receiver(vocab_size=opts.vocab_size, n_classes=n_classes)

    if opts.softmax_non_linearity == 0 and opts.linear_channel == 0:
        sender = AlwaysRelaxedWrapper(sender, temperature=opts.temperature)

    game = core.SymbolGameGS(sender, receiver, diff_loss_symbol)

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=test_loader,
                           callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True),
                                      EarlyStopperAccuracy(opts.early_stopping_thr)])

    trainer.train(n_epochs=opts.n_epochs)
    core.close()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

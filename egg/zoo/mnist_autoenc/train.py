# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import torch
import torch.distributions
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms

import egg.core as core


class Sender(nn.Module):
    def __init__(self, vocab_size):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, vocab_size)

    def forward(self, x, _aux_input):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # by EGG convention, in the case of 1-symbol communication
        # the agents return log-probs over the vocab
        logits = F.log_softmax(x, dim=1)
        return logits


class Receiver(nn.Module):
    def __init__(self):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(400, 784)

    def forward(self, x, _input, _aux_input):
        # Under GS-based optimization, the embedding layer of SymbolReceiverWrapper would be
        # essentially a linear layer. Since there is no point in having two linear layers
        # sequentially, we put a non-linearity
        x = F.leaky_relu(x)
        x = self.fc(x)
        return torch.sigmoid(x)


def loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    """
    The autoencoder's loss function; cross-entropy between the original and restored images.
    """
    loss = F.binary_cross_entropy(
        receiver_output, sender_input.view(-1, 784), reduction="none"
    ).mean(dim=1)
    return loss, {}


def main(params):
    # initialize the egg lib
    opts = core.init(params=params)
    # get pre-defined common line arguments (batch/vocab size, etc).
    # See egg/core/util.py for a list

    # prepare the dataset
    kwargs = {"num_workers": 1, "pin_memory": True} if opts.cuda else {}
    transform = transforms.ToTensor()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=opts.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=False, transform=transform),
        batch_size=opts.batch_size,
        shuffle=True,
        **kwargs
    )

    # initialize the agents and the game
    sender = Sender(opts.vocab_size)  # the "data" transform part of an agent
    sender = core.GumbelSoftmaxWrapper(
        sender, temperature=1.0
    )  # wrapping into a GS interface

    receiver = Receiver()
    receiver = core.SymbolReceiverWrapper(
        receiver, vocab_size=opts.vocab_size, agent_input_size=400
    )
    # setting up as a standard Sender/Receiver game with 1 symbol communication
    game = core.SymbolGameGS(sender, receiver, loss)
    # This callback would be called at the end of each epoch by the Trainer; it reduces the sampling
    # temperature used by the GS
    temperature_updater = core.TemperatureUpdater(
        agent=sender, decay=0.75, minimum=0.01
    )
    # get an optimizer that is set up by common command line parameters,
    # defaults to Adam
    optimizer = core.build_optimizer(game.parameters())

    # initialize and launch the trainer
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=[
            temperature_updater,
            core.ConsoleLogger(as_json=True, print_train_loss=True),
        ],
    )
    trainer.train(n_epochs=opts.n_epochs)

    core.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])

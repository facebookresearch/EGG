# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import datasets

import egg.core as core
from egg.zoo.simclr.games import ContinuousGame, SimCLRGameWrapper
from egg.zoo.simclr.models import get_resnet
from egg.zoo.simclr.nt_xent import NT_Xent
from egg.zoo.simclr.transformations import TransformsAugment, TransformsIdentity

class Sender(nn.Module):
    def __init__(self, projection_dim: int, output_dim: int):
        super(Sender, self).__init__()
        self.fc = nn.Linear(projection_dim, output_dim)

    def forward(self, x):
        x = self.fc(F.leaky_relu(x))
        return x


class Receiver(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, _input):
        x = self.fc(F.leaky_relu(x))
        return x

def get_loss(batch_size: int, temperature: float, device: torch.device):
    nt_xent_entropy = NT_Xent(batch_size, temperature, device)

    def nt_xent_loss(sender_input, _message, _receiver_input, receiver_output, _labels):
        loss, acc = nt_xent_entropy(receiver_output)
        return loss, {"acc": acc.unsqueeze(0)}

    return nt_xent_loss

def build_game(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, num_features = get_resnet(opts.model_name, pretrained=False)
    # initialize game
    loss = get_loss(opts.batch_size, opts.ntxent_tau, device)
    # setting up as a standard Sender/Receiver game
    sender = Sender(opts.projection_dim, opts.sender_output_size)
    receiver = Receiver(opts.sender_output_size, opts.receiver_output_size)

    if opts.game_channel == "continuous":
        print("Play game with continuous channel")
        game = ContinuousGame(sender, receiver, loss)
    elif opts.game_channel == "gs":
        print("Play game with GS channel")
        sender_rnn = core.RnnSenderGS(
            sender,
            opts.vocab_size,
            opts.sender_embedding,
            opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
            temperature=opts.tau,
        )
        receiver_rnn = core.RnnReceiverGS(
            receiver,
            opts.vocab_size,
            opts.receiver_embedding,
            opts.receiver_hidden,
            cell=opts.receiver_cell,
        )
        game = core.SenderReceiverRnnGS(sender_rnn, receiver_rnn, loss)
    else:
       raise NotImplementedError(f"{opts.game_channel} is currently not supported.")

    simclr_game = SimCLRGameWrapper(
        game,
        encoder,
        num_features,
        opts.projection_dim,
    )

    return simclr_game


def get_datasets(
    dataset_name: str,
    image_size: int,
    use_augmentations: bool = True,
    dataset_dir: str = "./data",
):
    print(
        f"Using dataset {dataset_name} with image size: {image_size}. "
        f"Applying augmentations: {use_augmentations}"
    )
    if use_augmentations:
        transformations = TransformsAugment(image_size)
    else:
        transformations = TransformsIdentity(image_size)

    # TODO: Add ImageNet
    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            dataset_dir,
            train=True,
            download=True,
            transform=transformations,
        )
        return train_dataset
    else:
        raise NotImplementedError(f"{dataset_name} is currently not supported.")


def main(params):
    # initialize the egg lib
    opts = get_opts(params=params)

    train_dataset = get_datasets(
        opts.dataset_name,
        opts.image_size,
        use_augmentations=opts.use_augmentations,
        dataset_dir=opts.dataset_dir,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8, # TODO: use parameter
    )

    simclr_game = build_game(opts)
    callbacks = [
        core.ConsoleLogger(as_json=True, print_train_loss=True),
        core.InteractionSaver(),
    ]
    optimizer = core.build_optimizer(simclr_game.parameters())

    trainer = core.Trainer(
        game=simclr_game,
        optimizer=optimizer,
        train_data=train_loader,
        callbacks=callbacks,
    )
    trainer.train(n_epochs=opts.n_epochs)
    core.close()


def get_opts(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="CIFAR10",
        help="Dataset name",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="Dataset download location",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        help="Model name for the encoder",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="Image size"
    )
    parser.add_argument(
        "--use_augmentations",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--ntxent_tau",
        type=float,
        default=0.1,
        help="Temperature for NT XEnt loss",
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=64,
        help="Projection head's dimension"
    )
    parser.add_argument(
        "--game_channel",
        type=str,
        default="continuous",
        choices=["continuous", "gs"],
        help="Type of channel used for the game setting",
    )
    # TODO: use group
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Sender GS temperature"
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=128,
        help="Sender embedding size in discrete communication setting"
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=128,
        help="Sender embedding size in discrete communication setting"
    )
    parser.add_argument(
        "--sender_output_size",
        type=int,
        default=128,
        help="Sender output size"
    )
    parser.add_argument(
        "--sender_cell",
        default="rnn",
        choices=["rnn", "lstm", "gru"],
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=128,
        help="Receiver embedding size in discrete communication setting"
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=128,
        help="Receiver hidden size in discrete communication setting"
    )
    parser.add_argument(
        "--receiver_output_size",
        type=int,
        default=128,
        help="Receiver output size"
    )
    parser.add_argument(
        "--receiver_cell",
        default="rnn",
        choices=["rnn", "lstm", "gru"],
    )

    opts = core.init(arg_parser=parser, params=params)

    return opts


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

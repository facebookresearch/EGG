# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from xmlrpc.client import Boolean

import pathlib

import egg.core as core
import torch
import json


def get_data_opts(parser):
    group = parser.add_argument_group("data")
    group.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="Dataset location",
    )

    group.add_argument(
        "--dataset_name",
        choices=[
            "cifar100",
            "imagenet",
            "gaussian_noise",
            "inaturalist",
            "imagenet_alive",
            "imagenet_ood",
            "places205",
            "imagenet_val",
        ],
        default="imagenet",
        help="Dataset used for training a model",
    )

    group.add_argument("--image_size", type=int, default=224, help="Image size")

    group.add_argument(
        "--num_workers", type=int, default=4, help="Workers used in the dataloader"
    )
    group.add_argument("--use_augmentations", action="store_true", default=False)
    group.add_argument(
        "--return_original_image",
        action="store_true",
        default=False,
        help="Dataloader will yield also the non-augmented version of the input images",
    )
    group.add_argument(
        "--test_time_augment",
        action="store_true",
        default=False,
        help="augmentations will be applied to the distractors, but not to the sender input",
    )


def get_gs_opts(parser):
    group = parser.add_argument_group("gumbel softmax")
    group.add_argument(
        "--gs_temperature",
        type=float,
        default=1.0,
        help="gs temperature used in the relaxation layer",
    )
    group.add_argument(
        "--gs_temperature_decay",
        type=float,
        default=1.0,
        help="gs temperature update_factor (default: 1.0)",
    )
    group.add_argument(
        "--train_gs_temperature",
        default=False,
        action="store_true",
        help="train gs temperature used in the relaxation layer",
    )
    group.add_argument(
        "--straight_through",
        default=False,
        action="store_true",
        help="use straight through gumbel softmax estimator",
    )
    group.add_argument(
        "--update_gs_temp_frequency",
        default=1,
        type=int,
        help="update gs temperature frequency (default: 1)",
    )
    group.add_argument(
        "--minimum_gs_temperature",
        default=1.0,
        type=float,
        help="minimum gs temperature when frequency update (default: 1.0)",
    )


def get_vision_module_opts(parser):
    group = parser.add_argument_group("vision module")
    group.add_argument(
        "--vision_model_name",
        type=str,
        default="",
        choices=[
            "resnet50",
            "resnet101",
            "resnet152",
            "vgg11",
            "inception",
            "vit",
            "swin",
            "dino",
            "twins_svt",
            "deit",
            "xcit",
            "resnext",
            "mobilenet",
            "densenet",
            "virtex",
            "fcn_resnet50",
        ],
        help="Model name for the encoder",
    )
    group.add_argument(
        "--retrain_vision",
        default=False,
        action="store_true",
        help="by default pretrained vision modules will be used, otherwise they will be trained from scratch",
    )
    # group.add_argument(
    #     "--vision_model_names",
    #     type=str,
    #     default="[]",
    #     help="Model names for the encoder of senders and receivers.",
    # )
    group.add_argument(
        "--vision_model_names_senders",
        type=str,
        default="[]",
        help="Model names for the encoder of senders.",
    )
    group.add_argument(
        "--vision_model_names_recvs",
        type=str,
        default="[]",
        help="Model names for the encoder of receivers.",
    )
    group.add_argument(
        "--use_different_architectures",
        default=True,
        action="store_false",  # what's this ? shouldn't it be the opposite
        help="Population game with different architectures.",
    )


def get_new_agents_opts(parser):
    group = parser.add_argument_group("language transmission")
    group.add_argument(
        "--base_checkpoint_path",
        type=str,
        default="",
        help="in the ease of transmission experiments, where to get the basic experiment weights",
    )
    group.add_argument(
        "--additional_receivers",
        type=str,
        default="[]",
        help="Model names for the encoders of receivers added to the experiment after training",
    )
    group.add_argument(
        "--additional_senders",
        type=str,
        default="[]",
        help="Model names for the encoders of senders added to the experiment after training to measure language transmission",
    )


def get_game_arch_opts(parser):
    group = parser.add_argument_group("game architecture")
    # group.add_argument(
    #     "--n_senders", type=int, default=1, help="Number of senders in the population"
    # )
    # group.add_argument(
    #     "--n_recvs", type=int, default=1, help="Number of receivers in the population"
    # )
    group.add_argument(
        "--recv_temperature",
        type=float,
        default=0.1,
        help="Temperature for similarity computation in the loss fn. Ignored when similarity is 'dot'",
    )
    group.add_argument(
        "--recv_hidden_dim",
        type=int,
        default=2048,
        help="Hidden dim of the non-linear projection of the distractors",
    )
    # Removed as it HAS to be the same as vocab_size so as to be able to apply cosine similarity
    # group.add_argument(
    #     "--recv_output_dim",
    #     type=int,
    #     default=2048,
    #     help="Output dim of the non-linear projection of the distractors, used to compare with msg embedding",
    # )
    group.add_argument(
        "--non_linearity",
        type=str,
        default=None,
        choices=["sigmoid", "softmax"],
        help="non_linearity for the continuous sender",
    )
    group.add_argument(
        "--augmentation_type",
        type=str,
        default=None,
        choices=["resize", "color_jitter", "grayscale", "gaussian_blur"],
        help="non_linearity for the continuous sender",
    )
    group.add_argument(
        "--com_channel",
        type=str,
        default="continuous",
        choices=["gs", "reinforce", "lstm", "continuous"],
        help="communication channel to use, the first three are discrete, the last one is continuous. rnn is multi-symbol",
    )

    group.add_argument(
        "--continuous_com",
        default=False,
        action="store_true",
        help="legacy : use continuous communication channel",
    )

    group.add_argument(
        "--noisy_channel",
        type=float,
        default=None,
        help="variance of gaussian noise to add to the communication channel. If none is given, then no noise will be added",
    )
    group.add_argument(
        "--keep_classification_layer",
        default=False,
        action="store_true",
        help="instead of the finale layer's representation, use the pretrained architecture's chosen class as image encoders",
    )
    group.add_argument(
        "--force_gumbel",
        default=False,
        action="store_true",
        help="force non_discretised gumbel messages for both training and testing (only for continuous) ",
    )
    group.add_argument(
        "--remove_auxlogits",
        default=False,
        action="store_true",
        help="forces the inception model to be lauded without the auxiliary loss channel. This is to handle retro_compatibility with past models trained this way.",
    )
    group.add_argument(
        "--block_com_layer",
        default=False,
        action="store_true",
        help="blocks the communication layer from being used in the continuous sender and receiver (TODO, discrete sender). Used as baseline.",
    )
    group.add_argument(
        "--aux_loss",
        default=None,
        type=str,
        choices=["random", "best", "best_averaged", "random_kl", "chosen"],
        help="auxiliary loss to reduce differences between sender messages on same input",
    )
    group.add_argument(
        "--aux_loss_weight",
        default=0.0,
        type=float,
        help="weight of the auxiliary loss (default: 0.0)",
    )
    group.add_argument(
        "--is_single_class_batch",
        default=False,
        action="store_true",
        help="if true, the batch will be composed of only one class, and the auxiliary loss will be computed on the same class",
    )
    group.add_argument(
        "--simplicial_temperature",
        default=1.0,
        type=float,
        help="when communicating with simplicial messages, the temperature to use to sample the simplex",
    )


def get_common_opts(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=10e-6,
        help="Weight decay used for SGD",
    )
    parser.add_argument(
        "--use_larc", action="store_true", default=False, help="Use LARC optimizer"
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )

    get_data_opts(parser)
    get_gs_opts(parser)
    get_vision_module_opts(parser)
    get_game_arch_opts(parser)
    get_new_agents_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts


def load(game, checkpoint):
    game.load_state_dict(checkpoint.model_state_dict)
    # game.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    # if checkpoint.optimizer_scheduler_state_dict:
    #     game.optimizer_scheduler.load_state_dict(
    #         checkpoint.optimizer_scheduler_state_dict
    #     )
    # game.start_epoch = checkpoint.epoch


def load_from_checkpoint(game, path):
    """
    Loads the game, agents, and optimizer state from a file
    :param path: Path to the file
    """
    print(f"# loading trainer state from {path}")
    checkpoint = torch.load(path)
    load(game, checkpoint)


def add_weight_decay(model, weight_decay=1e-5, skip_name=""):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or skip_name in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def path_to_parameters(path, type="wandb"):
    if type == "wandb":
        _path = pathlib.Path(path)
        return _path.parent / "wandb" / "latest-run" / "files" / "wandb-metadata.json"
    else:
        # legacy from when submitit was in use and the parameters were saved in the .out file
        old_game = pathlib.Path(path)
        job_number = str(old_game.parents[0].stem)
        all_out_files = [f for f in old_game.parents[1].glob(f"*{job_number}.out")]
        assert (
            len(all_out_files) == 1
        ), f"did not find one out file (missing or duplicates) : {all_out_files}"
        return all_out_files[0]


def metadata_opener(file, data_type: str, verbose=False):
    """
    data_type : str in {"wandb", "nest"}
    Mat : case match in python 3.10 will cover all of this in syntactic sugar
    """
    if (
        data_type == "wandb"
    ):  # TODO : using the yaml file instead would use load all params instead of non-defaults
        meta = json.load(file)
        return meta["args"]

    if data_type == "nest":
        # in nest, metadata are written as comments on the first line of the .out file
        # TODO: only parameters used in the sweep json file are available here.
        # All other parameters will be set to default values but will not appear here
        # A future version of this opener should take into account the Namespace object on the following line
        lines = file.readlines()
        file.seek(0)  # reset file
        for i in range(len(lines)):
            if lines[i][0] == "#":
                params = eval(lines[i][12:])  # Mat : injection liability
                return params
        if verbose:
            print("failed to find metadata in file")
        return []

    else:
        raise KeyError(
            f"{data_type} is not a valid type for data_type in metadata_opener"
        )

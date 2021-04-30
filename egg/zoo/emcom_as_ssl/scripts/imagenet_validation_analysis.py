# copyright (c) facebook, inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This script only works with a single gpu

import argparse

import torch
from torchvision import datasets

from egg.zoo.emcom_as_ssl.data import ImageTransformation
from egg.zoo.emcom_as_ssl.scripts.utils import (
    add_common_cli_args,
    evaluate,
    get_game,
    get_params,
    save_interaction
)


def get_dataloader(
    dataset_dir: str,
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    use_augmentations: bool = True,
):
    transformations = ImageTransformation(image_size, use_augmentations)

    dataset = datasets.ImageFolder(
        dataset_dir,
        transform=transformations
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def main():
    parser = argparse.ArgumentParser()
    add_common_cli_args(parser)
    cli_args = parser.parse_args()
    opts = get_params(
        simclr_sender=cli_args.simclr_sender,
        shared_vision=cli_args.shared_vision,
        loss_type=cli_args.loss_type
    )

    if cli_args.pdb:
        breakpoint()

    print(f"| Loading model from {cli_args.checkpoint_path} ...")
    game = get_game(opts, cli_args.checkpoint_path)
    print("| Model loaded.")

    o_test_path = (
        "/private/home/mbaroni/agentini/representation_learning/"
        "generalizaton_set_construction/80_generalization_data_set/"
    )
    i_test_path = "/datasets01/imagenet_full_size/061417/val"
    if cli_args.test_set == "o_test":
        dataset_dir = o_test_path
    elif cli_args.test_set == "i_test":
        dataset_dir = i_test_path
    else:
        raise NotImplementedError(f"Cannot recognize {cli_args.test_set} test_set")

    print(f"| Fetching data for {cli_args.test_set} test set from {dataset_dir}...")
    dataloader = get_dataloader(
        dataset_dir=dataset_dir,
        use_augmentations=cli_args.evaluate_with_augmentations
    )
    print("| Test data fetched.")

    print("| Starting evaluation ...")
    loss, soft_acc, game_acc, full_interaction = evaluate(game=game, data=dataloader)
    print(f"| Loss: {loss}, soft_accuracy (out of 100): {soft_acc * 100}, game_accuracy (out of 100): {game_acc * 100}")

    if cli_args.dump_interaction_folder:
        save_interaction(
            interaction=full_interaction,
            log_dir=cli_args.dump_interaction_folder,
            test_set=cli_args.test_set
        )
        print(f"| Interaction saved at {cli_args.dump_interaction_folder}")

    print("Finished evaluation.")


if __name__ == "__main__":
    main()

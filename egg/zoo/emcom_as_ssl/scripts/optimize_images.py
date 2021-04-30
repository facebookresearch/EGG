# copyright (c) facebook, inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse

import torch
import torch.nn as nn

from egg.core.interaction import Interaction
from egg.core.util import move_to
from egg.zoo.emcom_as_ssl.scripts.utils import add_common_cli_args
from egg.zoo.emcom_as_ssl.scripts.imagenet_validation_analysis import get_dataloader, save_interaction


def evaluate(
    game: nn.Module,
    data: torch.utils.data.DataLoader,
):
    if torch.cuda.is_available():
        game.cuda()
    game.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean_loss = 0.0
    interactions = []
    n_batches = 0
    soft_accuracy, game_accuracy = 0.0, 0.0
    with torch.no_grad():
        for batch in data:
            batch = move_to(batch, device)
            optimized_loss, interaction = game(*batch)

            interaction = interaction.to("cpu")
            interactions.append(interaction)

            mean_loss += optimized_loss
            soft_accuracy += interaction.aux['acc'].mean().item()
            game_accuracy += interaction.aux['game_acc'].mean().item()
            n_batches += 1
            if n_batches % 10 == 0:
                print(f"finished batch {n_batches}")

    print(f"processed {n_batches} in total")
    mean_loss /= n_batches
    soft_accuracy /= n_batches
    game_accuracy /= n_batches
    full_interaction = Interaction.from_iterable(interactions)

    return mean_loss, soft_accuracy, game_accuracy, full_interaction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_set",
        type=str,
        choices=["o_test", "i_test"],
        default="i_test",
        help="Choose which imagenet validation test to use, choices [i_test, o_test] (default: o_test)"
    )
    parser.add_argument(
        "--dump_interaction_folder",
        type=str,
        required=True,
        help="Path where the newly generated interaction will be saved"
    )
    add_common_cli_args(parser)
    cli_args = parser.parse_args()

    if cli_args.pdb:
        breakpoint()

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
    print(f"| Fetching data from {cli_args.test_set} test set from {dataset_dir}...")

    dataloader = get_dataloader(
        dataset_dir=dataset_dir,
        use_augmentations=cli_args.evaluate_with_augmentations
    )
    print("| Data fetched.")

    save_interaction(
        interaction=interaction,
        log_dir=cli_args.dump_interaction_folder,
        interaction_name=cli_args.interaction_name
    )
    print(f"| Interaction saved at {cli_args.dump_interaction_folder}")

    print("Finished evaluation.")


if __name__ == "__main__":
    main()

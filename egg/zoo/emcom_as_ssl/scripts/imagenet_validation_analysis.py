# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This script only works with a single gpu

import argparse

from egg.zoo.emcom_as_ssl.scripts.utils import (
    add_common_cli_args,
    evaluate,
    get_dataloader,
    get_game,
    get_params,
    save_interaction,
)


def main():
    parser = argparse.ArgumentParser()
    add_common_cli_args(parser)
    cli_args = parser.parse_args()
    opts = get_params(
        simclr_sender=cli_args.simclr_sender,
        shared_vision=cli_args.shared_vision,
        loss_type=cli_args.loss_type,
        discrete_evaluation_simclr=cli_args.discrete_evaluation_simclr
    )

    if cli_args.pdb:
        breakpoint()

    print(f"| Loading model from {cli_args.checkpoint_path} ...")
    game = get_game(opts, cli_args.checkpoint_path)
    print("| Model loaded.")

    print(f"| Fetching data from {cli_args.test_dataset_dir}...")
    dataloader = get_dataloader(
        dataset_dir=cli_args.test_dataset_dir,
        use_augmentations=cli_args.evaluate_with_augmentations,
    )
    print("| Test data fetched.")

    print("| Starting evaluation ...")
    loss, soft_acc, game_acc, full_interaction = evaluate(game=game, data=dataloader)
    print(f"| Loss: {loss}, soft_accuracy (out of 100): {soft_acc * 100}, game_accuracy (out of 100): {game_acc * 100}")

    if cli_args.dump_interaction_folder:
        save_interaction(
            interaction=full_interaction,
            log_dir=cli_args.dump_interaction_folder
        )
        print(f"| Interaction saved at {cli_args.dump_interaction_folder}")

    print("Finished evaluation.")


if __name__ == "__main__":
    main()

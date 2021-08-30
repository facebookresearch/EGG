# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

import torch
from egg.zoo.population_game.scripts.utils import (
    add_common_cli_args,
    add_reshaped_interaction_fields,
    evaluate,
    get_game,
    get_params,
    get_test_data,
    save_interaction,
)


def main(params):
    parser = argparse.ArgumentParser()
    add_common_cli_args(parser)
    cli_args = parser.parse_args()
    opts = get_params(
        n_senders=cli_args.n_senders,
        n_recvs=cli_args.n_recvs,
        vocab_size=cli_args.vocab_size,
        use_different_architectures=cli_args.use_different_architectures,
        vision_model_name=cli_args.vision_model_name,
        vision_model_names=cli_args.vision_model_names,
        vision_model_names_senders=cli_args.vision_model_names_senders,
        vision_model_names_recvs=cli_args.vision_model_names_recvs,
    )

    if cli_args.pdb:
        breakpoint()

    print(f"| Loading model from {cli_args.checkpoint_path} ...")
    game = get_game(opts, cli_args.checkpoint_path)
    print("| Model loaded")
    data = get_test_data(
        dataset_name=cli_args.dataset_name, batch_size=cli_args.batch_size
    )

    print("| Starting evaluation ...")
    loss, interaction = evaluate(
        game,
        data,
        torch.device("cuda"),
        cli_args.n_senders,
        cli_args.n_recvs,
    )
    print(
        f"| Loss: {loss}, accuracy across all agents (out of 100): {interaction.aux['acc'].mean().item() * 100}"
    )

    add_reshaped_interaction_fields(
        interaction=interaction,
        n_senders=cli_args.n_senders,
        n_recvs=cli_args.n_recvs,
        batch_size=cli_args.batch_size,
    )
    if cli_args.dump_interaction_folder:
        save_interaction(
            interaction=interaction, log_dir=cli_args.dump_interaction_folder
        )
        print(f"| Interaction saved at {cli_args.dump_interaction_folder}")


if __name__ == "__main__":
    main(sys.argv[1:])

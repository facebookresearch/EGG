# copyright (c) facebook, inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from typing import List, Tuple

import torch
import torch.nn as nn

from egg.zoo.emcom_as_ssl.scripts.utils import add_common_cli_args
from egg.zoo.emcom_as_ssl.scripts.imagenet_validation_analysis import (
    get_dataloader,
    get_params,
    get_game
)


def generate_token_histogram(
    game: nn.Module,
    data: torch.utils.data.DataLoader,
):
    if torch.cuda.is_available():
        game.cuda()
    game.eval()

    n_batches = 0
    token_histogram = []
    with torch.no_grad():
        for (x_i, x_j), labels in data:
            if torch.cuda.is_available():
                x_i = x_i.cuda()
                x_j = x_j.cuda()

            sender_encoded_input, receiver_encoded_input = game.vision_module(x_i, x_j)
            *rest, pre_gs = game.game.sender(sender_encoded_input)

            tokens = pre_gs.argmax(dim=-1).cpu().tolist()
            token_histogram.extend(tokens)

            n_batches += 1

            if n_batches % 10 == 0:
                print(f"finished batch {n_batches}")

            if len(token_histogram) > 50_000:
                break

    print(f"processed {n_batches} in total")

    return token_histogram


def optimize_image(
    game: nn.Module,
    token_histogram: List[Tuple[int, int]],
    dim_to_optimize: int
):

    histogram = {}
    for token in token_histogram:
        histogram[token] = 1 + histogram.get(token, 0)

    histogram = [(token, frq) for token, frq in histogram.items()]
    histogram.sort(key=lambda x: x[1], reverse=True)

    for p in game.parameters():
        p.requires_grad_(False)

    game.game.sender.gs_layer.train()

    random_image = torch.rand(3, 224, 224)
    random_image.uniform_().div_(100).unsqueeze_(0)
    random_image.requires_grad_(True)

    optimizer = torch.optim.SGD((random_image, ), lr=0.1)

    for i in range(1000):
        optimizer.zero_grad()
        sender_encoded_input, _ = game.vision_module(random_image, random_image)

        # TO CHANGE
        *rest, message, pre_gs = game.game.sender(sender_encoded_input)

        loss = -pre_gs[0, dim_to_optimize]  # (-2 * pre_gs[0, 1755] + pre_gs.sum())

        loss.backward()
        optimizer.step()
        if i > 0 and i % 100 == 0:
            print(loss.detach().item())

    return loss, random_image


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
    opts = get_params(
        simclr_sender=cli_args.simclr_sender,
        shared_vision=cli_args.shared_vision,
        loss_type=cli_args.loss_type
    )

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

    print(f"| Loading model from {cli_args.checkpoint_path} ...")
    game = get_game(opts, cli_args.checkpoint_path)
    print("| Model loaded.")

    token_histogram = generate_token_histogram(game, dataloader)

    dim_to_optimize = 1
    loss, random_image = optimize_image(game, token_histogram, dim_to_optimize)

    loss, optimized_image = random_image.detach().squeeze().cpu().transpose(2, 0)
    torch.save(optimized_image, f"/private/home/rdessi/optimized_images/optimized_img_dim_{dim_to_optimize}")

    print(f"| Loss {loss}")

    print("Finished evaluation.")


if __name__ == "__main__":
    main()

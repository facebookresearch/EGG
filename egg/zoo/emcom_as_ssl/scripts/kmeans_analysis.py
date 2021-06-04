# copyright (c) facebook, inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This script only works with a single gpu

import argparse

try:
    from sklearn.cluster import KMeans
except ImportError:
    print("Please install scikit-learn to perform k-means clustering.")
import torch
import torch.nn as nn

from egg.core.interaction import Interaction, LoggingStrategy
from egg.core.util import move_to
from egg.zoo.emcom_as_ssl.scripts.utils import (
    add_common_cli_args,
    evaluate,
    get_dataloader,
    get_game,
    get_params,
    save_interaction
)


def assign_kmeans_labels(interaction: Interaction, num_clusters=1000):
    resnet_output_sender = interaction.aux["resnet_output_sender"][:100000].cpu().numpy()
    print(f"assigning {num_clusters} clusters")
    k_means = KMeans(n_clusters=num_clusters, random_state=0).fit(resnet_output_sender)
    return k_means


def evaluate_test_set(
    game: nn.Module,
    data: torch.utils.data.DataLoader,
    k_means_clusters: KMeans,
    num_clusters: int
):
    if torch.cuda.is_available():
        game.cuda()
    game.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging_strategy = LoggingStrategy(False, False, True, True, True, False)

    mean_loss = 0.0
    interactions = []
    n_batches = 0
    soft_accuracy, game_accuracy = 0.0, 0.0
    for batch in data:
        batch = move_to(batch, device)
        (x_i, x_j), labels = batch
        if torch.cuda.is_available():
            x_i = x_i.cuda()
            x_j = x_j.cuda()

        with torch.no_grad():
            sender_encoded_input, receiver_encoded_input = game.vision_module(x_i, x_j)
            message, message_like, resnet_output_sender = game.game.sender(sender_encoded_input, sender=True)

            resnet_output_sender_to_predict = resnet_output_sender.cpu().numpy()
            k_means_labels = torch.from_numpy(
                k_means_clusters.predict(resnet_output_sender_to_predict)
            ).to(device=message_like.device, dtype=torch.int64)

            one_hot_k_means_labels = torch.zeros((message_like.size()[0], num_clusters), device=message_like.device)
            one_hot_k_means_labels.scatter_(1, k_means_labels.view(-1, 1), 1)

            receiver_output, resnet_output_recv = game.game.receiver(message, receiver_encoded_input)

            loss, aux_info = game.game.loss(
                sender_encoded_input, message, receiver_encoded_input, receiver_output, labels
            )

            if hasattr(game.game.sender, "temperature"):
                if isinstance(game.game.sender.temperature, torch.nn.Parameter):
                    temperature = game.game.sender.temperature.detach()
                else:
                    temperature = torch.Tensor([game.game.sender.temperature])
                aux_info["temperature"] = temperature

            aux_info["message_like"] = message_like
            aux_info["kmeans"] = one_hot_k_means_labels
            aux_info["resnet_output_sender"] = resnet_output_sender
            aux_info["resnet_output_recv"] = resnet_output_recv

            interaction = logging_strategy.filtered_interaction(
                sender_input=sender_encoded_input,
                receiver_input=receiver_encoded_input,
                labels=labels,
                receiver_output=receiver_output.detach(),
                message=message,
                message_length=torch.ones(message_like.shape[0]),
                aux=aux_info,
            )

            interaction = interaction.to("cpu")
            interactions.append(interaction)

            mean_loss += loss.mean()
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
    parser.add_argument("--num_clusters", type=int, default=1000)
    parser.add_argument("--train_dataset_dir", required=True)
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

    print(f"| Fetching train data from {cli_args.train_dataset_dir} to learn clusters...")
    train_dataloader = get_dataloader(
        dataset_dir=cli_args.train_dataset_dir,
        use_augmentations=cli_args.evaluate_with_augmentations,
    )
    print("| Fetched train data.")

    print(f"| Fetching test data from {cli_args.test_dataset_dir}...")
    test_dataloader = get_dataloader(
        dataset_dir=cli_args.test_dataset_dir,
        use_augmentations=cli_args.evaluate_with_augmentations,
    )
    print("| Fetched test data")

    print(f"| Loading model from {cli_args.checkpoint_path} ...")
    game = get_game(opts, cli_args.checkpoint_path)
    print("| Model loaded.")

    print("| Starting evaluation ...")
    _, _, _, interaction = evaluate(
        game=game,
        data=train_dataloader
    )
    print("| Finished processing train_data")

    print("| Clustering resnet outputs ...")
    k_means_clusters = assign_kmeans_labels(interaction, cli_args.num_clusters)
    print("| Done clustering resnet outputs")

    print("| Running evaluation on the test set ...")
    loss, soft_acc, game_acc, interaction = evaluate_test_set(
        game=game,
        data=test_dataloader,
        k_means_clusters=k_means_clusters,
        num_clusters=cli_args.num_clusters
    )
    print("| Done evaluation on the test set")

    print(f"| Loss: {loss}, soft_accuracy (out of 100): {soft_acc * 100}, game_accuracy (out of 100): {game_acc * 100}")

    if cli_args.dump_interaction_folder:
        print("| Saving interaction ...")
        save_interaction(
            interaction=interaction,
            log_dir=cli_args.dump_interaction_folder
        )
        print(f"| Interaction saved at {cli_args.dump_interaction_folder}")

    print("Finished evaluation.")


if __name__ == "__main__":
    main()

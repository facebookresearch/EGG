# Usage:

# python ./compute_rsa_on_resnet_outputs.py \
#  /private/home/rdessi/interactions_for_marco/latest_interaction_for_neurips_all_fields_valid

import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.stats import pearsonr
from sklearn import metrics

from egg.core.util import move_to
from egg.zoo.emcom_as_ssl.scripts.imagenet_validation_analysis import get_dataloader
from egg.zoo.emcom_as_ssl.scripts.utils import add_common_cli_args


def process_pretrained_data_with_resnet(
    data: torch.utils.data.DataLoader,
    process_recv: bool
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_resnet = torchvision.models.resnet50(pretrained=True)
    pretrained_resnet.fc = nn.Identity()
    for param in pretrained_resnet.parameters():
        param.requires_grad = False
    pretrained_resnet = pretrained_resnet.eval()
    if torch.cuda.is_available():
        pretrained_resnet.cuda()

    pretrained_resnet_output_sender = []
    pretrained_resnet_output_recv = []
    with torch.no_grad():
        for i, batch in enumerate(data):
            batch = move_to(batch, device)
            (x_i, x_j), _ = batch
            pretrained_resnet_output_sender.append(pretrained_resnet(x_i))
            if process_recv:
                pretrained_resnet_output_recv.append(pretrained_resnet(x_j))

            print(f"Processed batch {i+1}")

    pretrained_resnet_output_sender = torch.cat(pretrained_resnet_output_sender, dim=0)

    if pretrained_resnet_output_recv:
        pretrained_resnet_output_recv = torch.cat(pretrained_resnet_output_recv, dim=0)

    return pretrained_resnet_output_sender, pretrained_resnet_output_recv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interaction_path",
        type=str,
        help="Path to model checkpoint"
    )
    add_common_cli_args(parser)
    cli_args = parser.parse_args()

    if cli_args.pdb:
        breakpoint()

    print(f"| Loading interaction from {cli_args.interaction_path} ...")
    interaction = torch.load(cli_args.interaction_path)
    print("| Interaction loaded.")

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

    print("| Processing data with pretrained resnet")
    pt_resnet_output_sender, pt_resnet_output_recv = process_pretrained_data_with_resnet(
        data=dataloader,
        process_recv=(not cli_args.shared_vision)
    )

    print("| Done")

    sender_res = interaction.aux["resnet_output_sender"].numpy()
    print("| Computing similarity of sender")
    sender_sim = metrics.pairwise.cosine_similarity(sender_res)
    print("| Done")
    sender_upper_tri = sender_sim[np.triu_indices(sender_sim.shape[0], k=1)]

    pt_sender_sim = metrics.pairwise.cosine_similarity(pt_resnet_output_sender.cpu().numpy())
    print("| Computing similarity of pretrained sender")
    pt_sender_upper_tri = pt_sender_sim[np.triu_indices(pt_sender_sim.shape[0], k=1)]
    print("| Done")

    print(f"Senders Pearson correlation: {pearsonr(sender_upper_tri, pt_sender_upper_tri)[0]}")

    if not cli_args.shared_vision:
        receiver_res = interaction.aux["resnet_output_recv"].numpy()
        print("| Computing similarity of recv")
        receiver_sim = metrics.pairwise.cosine_similarity(receiver_res)
        print("| Done")
        recv_upper_tri = receiver_sim[np.triu_indices(receiver_sim.shape[0], k=1)]

        pt_recv_sim = metrics.pairwise.cosine_similarity(pt_resnet_output_recv.cpu().numpy())
        print("| Computing similarity of pretrained recv")
        pt_recv_upper_tri = pt_recv_sim[np.triu_indices(pt_recv_sim.shape[0], k=1)]
        print("| Done")

        print(f"Recvs Pearson correlation: {pearsonr(recv_upper_tri, pt_recv_upper_tri)[0]}")


if __name__ == "__main__":
    main()

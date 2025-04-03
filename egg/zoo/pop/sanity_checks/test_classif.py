import torch
from egg.core.util import move_to
from egg.zoo.pop.utils import (
    get_common_opts,
    metadata_opener,
    load_from_checkpoint,
    path_to_parameters,
)
from egg.zoo.pop.games import build_game
from egg.zoo.pop.data import get_dataloader
import argparse
import sys


# load models from given experiment
def load_models(model_path: str, metadata_path: str, device: str) -> list:
    """
    Load models from a given experiment.

    :param model_path: Path to the model.
    :param metadata_path: Path to the metadata.
    :param device: The device to use ('cuda' or 'cpu').
    :return: List of senders.
    """
    opts = None
    with open(metadata_path) as f:
        opts = get_common_opts(metadata_opener(f, data_type="wandb", verbose=True))

    pop_game = build_game(opts)
    load_from_checkpoint(pop_game, model_path)
    senders = pop_game.agents_loss_sampler.senders

    # make non-trainable + send to gpu if required
    for sender in senders:
        sender.eval()
        sender.to(device)
        for param in sender.parameters():
            param.requires_grad = False
    return senders


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim: int = 2, output_dim: int = 3):
        """
        Initialize the LinearClassifier.

        :param input_dim: Dimension of the input.
        :param output_dim: Dimension of the output.
        """
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LinearClassifier.

        :param x: Input tensor.
        :return: Output tensor.
        """
        x = self.linear(x)
        return x


def forward_backward(
    sender: torch.nn.Module,
    classifier: torch.nn.Module,
    input_images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
) -> tuple:
    """
    Perform a forward and backward pass.

    :param sender: The sender model.
    :param classifier: The classifier model.
    :param input_images: Input images tensor.
    :param labels: Labels tensor.
    :param optimizer: Optimizer.
    :param criterion: Loss function.
    :return: Tuple containing message, output, and loss.
    """
    message = sender(input_images)
    output = classifier(message)

    loss = criterion(output, labels.view(-1))
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    return message, output, loss


def train_epoch(
    sender: torch.nn.Module,
    classifier: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
) -> list:
    """
    Train the classifier for one epoch.

    :param sender: The sender model.
    :param classifier: The classifier model.
    :param dataloader: DataLoader for the training dataset.
    :param optimizer: Optimizer.
    :param criterion: Loss function.
    :param device: The device to use ('cuda' or 'cpu').
    :return: List of accuracies for each batch.
    """
    accs = []
    for batch_idx, batch in enumerate(dataloader):
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)

        _, output, loss = forward_backward(
            sender, classifier, images, labels, optimizer, criterion
        )

        acc = (output.argmax(dim=1) == labels).float().mean().to("cpu")
        accs.append(acc.item())

    return accs


def test_epoch(
    senders: list,
    classifier: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    device: str,
) -> list:
    """
    Test the classifier for one epoch.

    :param senders: List of sender models.
    :param classifier: The classifier model.
    :param val_dataloader: DataLoader for the validation dataset.
    :param device: The device to use ('cuda' or 'cpu').
    :return: List of accuracies for each sender.
    """
    sender_accs = []
    for sender in senders:
        accs = []
        for batch_idx, batch in enumerate(val_dataloader):
            images, labels, _ = batch
            images = images.to(device)
            labels = labels.to(device)

            message = sender(images)
            output = classifier(message)

            acc = (output.argmax(dim=1) == labels).float().mean()
            accs.append(acc.item())
    sender_accs.append(accs)
    return sender_accs


def test_classification(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> float:
    """
    Test the classification accuracy of a model on a test dataset.

    :param model: The model to test.
    :param test_loader: DataLoader for the test dataset.
    :param device: The device to use ('cuda' or 'cpu').
    :return: The classification accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


if __name__ == "__main__":
    # create classifiers & parametrise learning
    # classifiers and optimizers are on gpu if device is set to cuda
    # an additional classifier is created for the shuffled input, where the sender is randomly chosen to get input
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "-d",
        "--device",
        help="device to run this on",
        choices=["cpu", "cuda"],
        required=False,
        default="cuda",
    )
    parser.add_argument(
        "-D",
        "--dataset_name",
        help="which data to train classifier on",
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
        required=False,
        default="imagenet_ood",
    )

    parser.add_argument(
        "--dataset_dir",
        help="path to directory with dataset",
        type=str,
        required=False,
        default="/projects/colt/imagenet21k_resized/imagenet21k_train/",
    )
    parser.add_argument(
        "-s",
        "--selected_sender_idx",
        help="index in original training game of sender used to train classifier (classification is then tested on all senders available)",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "-m",
        "--model_path",
        help="path to model to load",
        type=str,
        required=False,
        default="/home/mmahaut/projects/exps/tmlr/vocab_64/646399/final.tar",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        help="path to directory to save checkpoints",
        type=str,
        required=False,
        default="/home/mmahaut/projects/exps/tmlr/classif",
    )
    args = parser.parse_args(sys.argv[1:])

    metadata_path = path_to_parameters(args.model_path, "wandb")
    senders = load_models(args.model_path, metadata_path, args.device)
    sender = senders[args.selected_sender_idx]
    classifier = LinearClassifier(64, 58).to(
        args.device
    )  # TODO : hard coded output dim to be replaced

    criterion = torch.nn.CrossEntropyLoss()

    val_dataloader, train_dataloader = get_dataloader(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        batch_size=128,
        image_size=384,
        num_workers=4,
        is_distributed=False,
        use_augmentations=False,
        return_original_image=False,
        seed=111,
        split_set=True,
        augmentation_type=None,
    )

    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    optimizer.state = move_to(optimizer.state, args.device)

    for epoch in range(10):
        train_acc = train_epoch(
            sender, classifier, train_dataloader, optimizer, criterion, args.device
        )
        test_accs = test_epoch(senders, classifier, val_dataloader, args.device)

        print(f"Epoch {epoch} train acc: {sum(train_acc)/len(train_acc)}")
        for i in range(len(test_accs)):
            print(
                f"Epoch {epoch} test acc from sender {args.selected_sender_idx} tested on {i}: {sum(test_accs[i])/len(test_accs[i])}"
            )

        # save models
        torch.save(
            classifier.state_dict(),
            f"{args.checkpoint_dir}/cl_s{args.selected_sender_idx}_{epoch}.tar",
        )

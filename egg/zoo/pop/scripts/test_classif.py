import torch
from egg.core.util import move_to
from egg.zoo.pop.utils import (
    get_common_opts,
    metadata_opener,
    load_from_checkpoint,
    path_to_parameters,
)
from egg.zoo.pop.games import build_game
import hub
from torchvision import transforms
from egg.zoo.pop.archs import initialize_vision_module
from egg.zoo.pop.data import get_dataloader
import argparse

# load models from given experiment
def load_models(model_path, metadata_path, device):
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


def get_archs(names):
    archs = []
    features = []
    for name in names:
        arch, n_features, _ = initialize_vision_module(
            name, pretrained=True, aux_logits=True
        )
        archs.append(arch)
        features.append(n_features)
    return archs, features


def load_places():
    # load data
    def collate_fn(batch):
        return (
            torch.stack([x["images"] for x in batch], dim=0),
            torch.stack([torch.Tensor(x["labels"]).long() for x in batch], dim=0),
            torch.stack([torch.Tensor(x["index"]) for x in batch], dim=0),
        )

    ds = hub.load("hub://activeloop/places205")
    size = 384
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1)),
            transforms.Resize(size=(size, size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataloader = ds.pytorch(
        num_workers=0,
        shuffle=True,
        batch_size=128,
        collate_fn=collate_fn,
        transform={"images": transformations, "labels": None, "index": None},
        pin_memory=True,
    )
    return dataloader


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim=2, output_dim=3):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x


def forward_backward(sender, classifier, input_images, labels, optimizer, criterion):
    message = sender(input_images)
    output = classifier(message)

    loss = criterion(output, labels.view(-1))
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    return message, output, loss


def train_epoch(sender, dataloader, optimizer, criterion, device):
    accs = []
    for batch_idx, batch in enumerate(dataloader):
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)

        _, output, loss = forward_backward(
            sender, optimizer, images, labels, optimizer, criterion
        )

        acc = (output.argmax(dim=1) == labels).float().mean()
        accs.append(acc.item())

    return accs


def test_epoch(senders, val_dataloader, device):
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
        default="/datasets/COLT/imagenet21k_resized/imagenet21k_train/",
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
        default="/homedtcl/mmahaut/projects/experiments/im1k_cont/211295/final.tar",
    )
    args = vars(parser.parse_args())

    metadata_path = path_to_parameters(args["model_path"], "wandb")
    senders = load_models(args.model_path, metadata_path, args.device)
    sender = senders[args.selected_sender_idx]
    classifier = LinearClassifier(16, 245).to(args.device)

    criterion = torch.nn.CrossEntropyLoss()

    val_dataloader, train_dataloader = get_dataloader(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        batch_size=256,
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
            sender, train_dataloader, optimizer, criterion, args.device
        )
        test_accs = test_epoch(senders, val_dataloader, args.device)

        print(f"Epoch {epoch} train acc: {sum(train_acc)/len(train_acc)}")
        for i in range(len(test_accs)):
            print(
                f"Epoch {epoch} test acc from sender {args.selected_sender_idx} tested on {i}: {sum(test_accs[i])/len(test_accs[i])}"
            )

        # save models
        for i, classifier in enumerate(classifier):
            # TODO: remove hardcoded path
            torch.save(
                classifier.state_dict(),
                f"../experiments/feature_classif/cl_s{args.selected_sender_idx}_{epoch}.tar",
            )

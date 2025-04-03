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
import glob
import numpy as np

def load_models(model_path: str, metadata_path: str, device: str) -> list:
    """
    Load models from the given paths and prepare them for evaluation.

    :param model_path: Path to the model checkpoint.
    :param metadata_path: Path to the metadata file.
    :param device: Device to use ('cuda' or 'cpu').
    :return: List of sender models.
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

        :param input_dim: Dimension of the input features.
        :param output_dim: Dimension of the output classes.
        """
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.

        :param x: Input tensor.
        :return: Output tensor.
        """
        x = self.linear(x)
        return x

def test_epoch(
    senders: list,
    classifier: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    device: str,
) -> list:
    """
    Evaluate the classifier for one epoch.

    :param senders: List of sender models.
    :param classifier: Classifier model.
    :param val_dataloader: DataLoader for validation data.
    :param device: Device to use ('cuda' or 'cpu').
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

def transfer_classification(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    """
    Perform transfer classification using the given model and dataset.

    :param model: The model to use for classification.
    :param dataset: The dataset to classify.
    :param batch_size: The batch size for the DataLoader.
    :param device: The device to use ('cuda' or 'cpu').
    :return: The classification accuracy.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

if __name__ == "__main__":
    # load data
    val_dataloader, train_dataloader = get_dataloader(
        dataset_dir="/projects/colt/imagenet21k_resized/imagenet21k_train/",
        dataset_name="imagenet_ood",
        batch_size=1,
        image_size=384,
        num_workers=4,
        is_distributed=False,
        use_augmentations=False,
        return_original_image=False,
        seed=111,
        split_set=True,
        augmentation_type=None,
    )

    # load senders
    model_path = "/home/mmahaut/projects/exps/tmlr/vocab_64/646399/final.tar"
    metadata_path = path_to_parameters(model_path, "wandb")
    senders = load_models(model_path, metadata_path, "cuda")

    # load classifiers
    classifier_paths = glob.glob("/home/mmahaut/projects/exps/tmlr/classif/cl_s*9.tar")
    print(len(classifier_paths), "classifiers")
    classifiers = []
    for f in classifier_paths:
        classifiers.append(LinearClassifier(64,58).to("cuda"))
        classifiers[-1].load_state_dict(torch.load(f))
    # test, get std
    cl_sacc = []
    for classifier in classifiers:
        sender_accs = test_epoch(senders, classifier, val_dataloader, "cuda")
        print(sender_accs, np.array(sender_accs).mean(), np.array([np.array(sender_accs[i]).mean() for i in range(len(sender_accs))]).std())
        cl_sacc.append(sender_accs)

    for i,val in enumerate([0,5,2,3,4,1]):
        print(classifier_paths[i])
        sender_accs = cl_sacc
        het = np.array([
                np.array(sender_accs[i]).mean()
                for i in range(len(sender_accs))
                if i != val 
            ])
        print(het.mean(), het.std())
        print(np.array(sender_accs[val]).mean())
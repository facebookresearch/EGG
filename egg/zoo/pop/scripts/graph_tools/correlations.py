# load a vision module
# look at representation of each image
# compare with representation of all other vision modules
# correlate with accuracy
import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import random
import glob
from PIL import Image
import hub
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional


def seed_all(seed):
    if not seed:
        seed = 111
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def get_model(name, pretrained, aux_logits=True):
    modules = {
        "resnet50": (torchvision.models.resnet50, {"pretrained": pretrained}),
        "resnet101": (torchvision.models.resnet101, {"pretrained": pretrained}),
        "resnet152": (torchvision.models.resnet152, {"pretrained": pretrained}),
        "inception": (
            torchvision.models.inception_v3,
            {"pretrained": pretrained, "aux_logits": aux_logits},
        ),
        "resnext": (torchvision.models.resnext50_32x4d, {"pretrained": pretrained}),
        "mobilenet": (
            torchvision.models.mobilenet_v3_large,
            {"pretrained": pretrained},
        ),
        "vgg11": (torchvision.models.vgg11, {"pretrained": pretrained}),
        "densenet": (torchvision.models.densenet161, {"pretrained": pretrained}),
        "vit": (
            timm.create_model,
            {"model_name": "vit_base_patch16_384", "pretrained": pretrained},
        ),
        "swin": (
            timm.create_model,
            {"model_name": "swin_base_patch4_window12_384", "pretrained": pretrained},
        ),
        "dino": (
            torch.hub.load,
            {
                "repo_or_dir": "facebookresearch/dino:main",
                "model": "dino_vits16",
                "verbose": False,
            },
        ),
        "twins_svt": (
            timm.create_model,
            {"model_name": "twins_svt_base", "pretrained": pretrained},
        ),
        "deit": (
            timm.create_model,
            {"model_name": "deit_base_patch16_384", "pretrained": pretrained},
        ),
        "xcit": (
            timm.create_model,
            {"model_name": "xcit_large_24_p8_384_dist", "pretrained": pretrained},
        ),
        "virtex": (
            torch.hub.load,
            {
                "repo_or_dir": "kdexd/virtex",
                "model": "resnet50",
                "pretrained": pretrained,
            },
        ),
        "fcn_resnet50": (
            torchvision.models.segmentation.fcn_resnet50,
            {
                "weights": "DEFAULT" if pretrained else None,
            },
        ),
    }
    if name not in modules:
        raise KeyError(f"{name} is not currently supported.")
    return modules[name][0](**modules[name][1])


def initialize_vision_module(
    name: str = "resnet50", pretrained: bool = False, aux_logits=True
):
    print("initialize module", name)
    model = get_model(name, pretrained, aux_logits)
    # TODO: instead of this I'd feel like using the dictionary structure further and including in_features
    if name in ["resnet50", "resnet101", "resnet152", "resnext"]:
        n_features = model.fc.in_features
        model.fc = nn.Identity()
    if name == "virtex":
        # virtex has a no pooling layer, so we add the one in the original resnet
        n_features = 2048
        model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    if name == "densenet":
        n_features = model.classifier.in_features
        model.classifier = nn.Identity()
    if name == "mobilenet":
        n_features = model.classifier[3].in_features
        model.classifier[3] = nn.Identity()
    elif name == "vgg11":
        n_features = model.classifier[6].in_features
        model.classifier[6] = nn.Identity()
    elif name == "inception":
        n_features = model.fc.in_features
        if model.AuxLogits is not None:
            model.AuxLogits.fc = nn.Identity()
        model.fc = nn.Identity()
    elif name in ["vit", "swin", "xcit", "twins_svt", "deit"]:
        n_features = model.head.in_features
        model.head = nn.Identity()
    elif name == "dino":
        n_features = 384  # ... could go and get that somehow instead of hardcoding ?
        # Dino is already chopped and does not require removal of classif layer
    if pretrained:
        # prevent training by removing gradients
        for param in model.parameters():
            param.requires_grad = False
        if name == "inception":
            model.aux_logits = False
        # prevent training dependant behaviors (dropout...)
        model = model.eval()
    return model, n_features, name


def get_dataloader(
    dataset_dir: str,
    dataset_name: str,
    batch_size: int = 32,
    image_size: int = 32,
    num_workers: int = 0,
    is_distributed: bool = False,
    use_augmentations: bool = True,
    return_original_image: bool = False,
    seed: int = 111,
    split_set: bool = True,
    augmentation_type=None,
    is_single_class_batch: bool = False,
):
    # Param : split_set : if true will return a training and testing set. Otherwise will load train set only.
    seed_all(
        seed
    )  # set the seed for all the random generators, for some reason the later ones weren't sufficient
    transformations = ImageTransformation(
        image_size,
        use_augmentations,
        return_original_image,
        dataset_name,
        None,
    )
    if dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root="./data", download=True, transform=transformations
        )
    elif dataset_name == "inaturalist":
        train_dataset = datasets.INaturalist(
            root="./data", download=False, transform=transformations
        )
    elif dataset_name == "gaussian_noise":
        # Note : augmentations on gaussian noise make little sense, transformations are ignored
        train_dataset = Gaussian_noise_dataset(
            n_images=204800,  # matching cifar100
            image_size=image_size,
            n_labels=100,  # matching cifar100, does not matter as random
            seed=seed,
        )
    elif dataset_name == "imagenet_alive":
        raise NotImplementedError
        # train_dataset = datasets.ImageFolder(dataset_dir, transform=transformations)
        # train_dataset = select_inanimate_idxs(train_dataset)
    elif dataset_name == "imagenet_ood":
        train_dataset = datasets.ImageFolder(dataset_dir, transform=transformations)
        train_dataset = select_ood_idxs(train_dataset)
    elif dataset_name == "places205":
        train_dataset = PlacesDataset(
            hub.load("hub://activeloop/places205"), transform=transformations
        )
    elif dataset_name == "imagenet_val":
        # TODO Matéo : correct this so that path is not hardcoded
        train_dataset = ImagenetValDataset(
            dataset_dir,
            annotations_file=Path(dataset_dir).parent
            / "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
            transform=transformations,
        )
    else:
        train_dataset = datasets.ImageFolder(dataset_dir, transform=transformations)
    train_sampler = None
    if is_single_class_batch:
        train_sampler = SingleClassDatasetSampler(train_dataset, batch_size=batch_size)
        # for now, cannot be distributed ! will be overriden !
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True, seed=seed
        )
        if is_single_class_batch:
            raise NotImplemented(
                "Cannot use distributed sampling with single class batch"
            )
    if split_set:
        test_dataset, train_dataset = torch.utils.data.random_split(
            train_dataset,
            [len(train_dataset) // 10, len(train_dataset) - (len(train_dataset) // 10)],
            torch.Generator().manual_seed(seed),
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn_imood
            if dataset_name == "imagenet_ood"
            else collate_fn,
            drop_last=True,
            pin_memory=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn_imood
            if dataset_name == "imagenet_ood"
            else collate_fn,
            drop_last=True,
            pin_memory=True,
        )
        return test_loader, train_loader
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn_imood
            if dataset_name == "imagenet_ood"
            else collate_fn,
            drop_last=True,
            pin_memory=True,
        )
    return train_loader


class ImageTransformation:
    """
    A stochastic data augmentation module that transforms any given data example
    randomly resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(
        self,
        size: int,
        augmentation: bool = False,
        return_original_image: bool = False,
        dataset_name: Optional[str] = None,
        test_attack=None,
    ):
        if augmentation:
            s = 1
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            transformations = [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
            ]
        else:
            transformations = [
                transforms.Resize(size=(size, size)),
            ]
            if test_attack is not None:
                transformations.append(test_attack)
        transformations.append(transforms.ToTensor())
        transformations.append(
            transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1))
        )
        if (
            dataset_name == "imagenet"
            or dataset_name == "imagenet_alive"
            or dataset_name == "imagenet_ood"
            or dataset_name == "imagenet_val"
        ):
            pass  # temporary, to remove
            transformations.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            )
        elif dataset_name in ["cifar100", "inaturalist", "places205"]:
            transformations.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )
        self.transform = transforms.Compose(transformations)
        self.test_attack = test_attack is None
        self.return_original_image = return_original_image
        if self.return_original_image or self.test_attack:
            self.original_image_transform = transforms.Compose(
                [
                    transforms.Resize(size=(size, size)),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1)),
                ]
            )

    def __call__(self, x):
        x_i, x_j = self.transform(x), self.transform(x)
        if self.return_original_image:
            return x_i, x_j, self.original_image_transform(x)
        if self.test_attack:
            return self.original_image_transform(x), x_j
        return x_i, x_j


class ImagenetValDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        with open(annotations_file) as f:
            self.img_labels = [int(line) for line in f.readlines()]
        self.transform = transform
        self.files = sorted(glob.glob(f"{img_dir}/*.JPEG"))
        print(len(self.files), flush=True)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = self.pil_loader(img_path)
        label = self.img_labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def pil_loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


def collate_fn(batch):
    return (
        torch.stack([x[0][0] for x in batch], dim=0),  # sender_input
        torch.cat(
            [torch.Tensor([x[1]]).long() for x in batch], dim=0
        ),  # labels (original classes, not used in emecom_game)
        torch.stack([x[0][1] for x in batch], dim=0),  # receiver_input
    )


if __name__ == "__main__":
    vision_modules = [
        "vgg11",
        "vit",
        "resnet152",
        "inception",
        "dino",
        "swin",
        "virtex",
    ]
    representation = {}
    dataset_dir = "/gpfs/projects/colt/ILSVRC2012/ILSVRC2012_img_val/"
    dataset_name = "imagenet_val"
    batch_size = 64
    image_size = 384
    num_workers = 0
    is_distributed = False
    use_augmentations = False
    return_original_image = False
    seed = 111
    split_set = True
    augmentation_type = None
    is_single_class_batch = False
    test_loader, train_loader = get_dataloader(
        dataset_dir,
        dataset_name,
        batch_size,
        image_size,
        num_workers,
        is_distributed,
        use_augmentations,
        return_original_image,
        seed,
        split_set,
        augmentation_type,
        is_single_class_batch,
    )
    for vm in vision_modules:
        model, n_features, name = initialize_vision_module(vm, True)
        model = model.cuda()
        representation[name] = []
        for i, b in enumerate(train_loader):
            if i > 10:
                break
            # print(len(b), b[0].shape, b[1].shape, b[2].shape)
            representation[name].append(model(b[0].cuda()).detach().cpu())
        representation[name] = torch.cat(representation[name])
        print(representation[name].shape)
    for vm in vision_modules:
        for vm2 in vision_modules:
            if vm == vm2:
                continue
            print(
                vm,
                vm2,
                torch.nn.functional.cosine_similarity(
                    representation[vm], representation[vm2], dim=1
                ).mean(),
            )

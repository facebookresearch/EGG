# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional
from pathlib import Path
import glob

from tqdm.auto import tqdm 
import hub
import torch
import random
from PIL import ImageFilter, Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import torchvision
import numpy as np
from torch.utils.data import Subset
from egg.zoo.pop.proximity_sampler import ProximitySampler
import warnings


# TODO: move table to its own file
imood_class_ids = np.array(
    [
        4491,
        4624,
        6114,
        6774,
        8786,
        9333,
        10518,
        10548,
        10731,
        10862,
        11361,
        12012,
        12069,
        12121,
        12491,
        12640,
        14127,
        14129,
        14131,
        14150,
        14172,
        14257,
        14280,
        14373,
        14559,
        15124,
        15214,
        15412,
        15669,
        16334,
        16514,
        17094,
        17299,
        17301,
        17350,
        17377,
        17384,
        17401,
        17594,
        17603,
        17611,
        18151,
        18808,
        18847,
        19100,
        19213,
        19481,
        19548,
        19578,
        19644,
        19705,
        20127,
        20356,
        21650,
        21662,
        21688,
        21741,
        21765,
    ]
)
class ClosestImagesSampler(torch.utils.data.sampler.Sampler):
    """
    Organizes the dataset into batches of images with similar representations using KMeans clustering.
    """
    def __init__(self, dataset, batch_size, model):
        self.n_clusters = len(dataset) // batch_size
        self.batch_size = batch_size
        self.labels = self._get_labels(dataset)
        self.num_samples = len(self.labels)
        self.clusters = self._get_clusters(model, dataset)
    
    def _get_labels(self, dataset):
        if isinstance(dataset, datasets.ImageFolder):
            return torch.Tensor([torch.Tensor(x[1]).int() for x in dataset.imgs])
        elif isinstance(dataset, ImagenetValDataset):
            return torch.Tensor(dataset.img_labels)
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        else:
            raise NotImplementedError("Dataset type not supported")
    
    def _get_clusters(self, model, dataset, batch_size=64, seed=42):
        # get image representations
        reps=[]
        for i in range(len(dataset)//batch_size):
            dl = torch.stack([dataset[i*batch_size+j][0][0] for j in range(batch_size)], dim=0)
            with torch.no_grad():
                rep = model(dl)
            reps.extend(rep)
        reps = torch.stack(reps, dim=0)
        # cluster the representations
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=seed).fit(reps)
        # to torch tensor
        return torch.tensor(kmeans.labels_)

    def __iter__(self):
        idxs = torch.tensor([])
        for i in range(self.n_clusters//self.batch_size):
            c_size=0
            while c_size<self.batch_size:
                cluster_id = random.choice(self.clusters)
                c_size=sum(self.clusters==cluster_id)
            idxs = torch.concat(
                [
                    idxs,
                    torch.multinomial(
                        (self.clusters == cluster_id).float(),
                        self.batch_size,
                        False,
                    ),
                ]
            )
        return (i for i in torch.tensor(idxs).int())



class SingleClassDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    Samples elements uniformly from a given class to create single-class batches.
    Arguments:
        num_samples: number of samples to draw
    """

    def __init__(
        self,
        dataset,
        batch_size,
        replacement=False,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.replacement = replacement
        self.batch_size = batch_size

        # distribution of classes in the dataset
        self.labels = self._get_labels(dataset)
        self.dataset=dataset
        self.len = len(dataset)
        # self.num_samples = (len(dataset) // self.batch_size) * self.batch_size

    def _get_labels(self, dataset):
        if isinstance(dataset, datasets.ImageFolder):
            return torch.tensor([x[1] for x in dataset.imgs], dtype=torch.int32)
        elif isinstance(dataset, ImagenetValDataset):
            return torch.Tensor(dataset.img_labels)
        elif isinstance(dataset, torch.utils.data.Subset):
            if isinstance(dataset.dataset, ImagenetValDataset):
                return torch.tensor([dataset.dataset.img_labels[x] for x in tqdm(dataset.indices, desc="Getting labels")])
            return torch.tensor([x[1] for x in tqdm(dataset)], dtype=torch.int32)
        else:
            raise NotImplementedError("Dataset type not supported")

    def __iter__(self):
        # randomly select a label
        idxs = torch.tensor([])
        # check there are enough images of at least one class
        for _ in range(self.len // self.batch_size):
            possible_idxs = []
            while len(possible_idxs) <= self.batch_size:
                label_id = random.choice(self.labels)
                possible_idxs = torch.where(self.labels == label_id)[0]
            _rand_idxs = torch.multinomial(
                torch.ones(len(possible_idxs)), self.batch_size, self.replacement
            )
            selected_idxs = possible_idxs[_rand_idxs]
            idxs = torch.concat(
                [
                    idxs,
                    selected_idxs,
                ]
            )
        return (i for i in torch.tensor(idxs).int())

    def __len__(self):
        return self.len


# separate imagenet into subsets such as animate and inanimate
# Image selection
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


def select_ood_idxs(dataset):
    """
    from all imagenet classes select those which are inanimate
    """
    img_ids = (
        (torch.tensor(dataset.targets)[..., None] == torch.tensor(imood_class_ids))
        .any(-1)
        .nonzero(as_tuple=True)[0]
    )
    return Subset(dataset, img_ids)


def get_augmentation(attck_name: str, size: int):
    s = 1
    if attck_name is None:
        return
    transformations = {
        "resize": transforms.RandomResizedCrop(size=size),
        "color_jitter": transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
        "grayscale": transforms.Grayscale(),
        "gaussian_blur": GaussianBlur([0.1, 10.0]),
    }
    return transformations[attck_name]


def collate_fn(batch):

    return (
        torch.stack([x[0][0] for x in batch], dim=0),  # sender_input
        torch.cat(
            [torch.Tensor([x[1]]).long() for x in batch], dim=0
        ),  # labels (original classes, not used in emecom_game)
        torch.stack([x[0][1] for x in batch], dim=0),  # receiver_input
        # {"path": [x[2] for x in batch]},  # path to image
    )


def collate_fn_imood(batch):

    return (
        torch.stack([x[0][0] for x in batch], dim=0),  # sender_input
        torch.cat(
            [
                torch.Tensor([np.where(imood_class_ids == x[1])[0][0]]).long()
                for x in batch
            ],
            dim=0,
        ),  # labels, corrected for out of domain selection
        torch.stack([x[0][1] for x in batch], dim=0),  # receiver_input
        # {"path": [x[2] for x in batch]},  # path to image
    )


class Gaussian_noise_dataset(torch.utils.data.Dataset):
    def __init__(self, n_images, image_size, n_labels, seed):
        self.n_images = n_images
        self.image_size = image_size
        self.n_labels = n_labels
        self.seed = seed

    def __getitem__(self, idx):
        if idx <= self.n_images:
            gaussian_noise = torch.randn(
                [3, self.image_size, self.image_size],
                generator=torch.Generator().manual_seed(idx),
            )  # ,torch.Generator().manual_seed(self.seed)
            label = 0  # empirical value, not used in com game
            return [gaussian_noise, gaussian_noise], label
        else:
            raise

    def __len__(self):
        return self.n_images


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
    similbatch_training: bool = False,
    shuffle: bool = True,
):
    """
    Returns a DataLoader for the specified dataset with optional augmentations and batching strategies.
    :param dataset_dir: Path to the dataset directory.
    :param dataset_name: Name of the dataset (e.g., 'cifar100', 'imagenet_val').
    :param batch_size: Number of samples per batch.
    :param image_size: Size of the images after resizing.
    """
    # Param : split_set : if true will return a training and testing set. Otherwise will load train set only.
    seed_all(
        seed
    )  # set the seed for all the random generators, for some reason the later ones weren't sufficient
    transformations = ImageTransformation(
        image_size,
        use_augmentations,
        return_original_image,
        dataset_name,
        get_augmentation(augmentation_type, image_size),
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
    elif dataset_name in ["imagenet_ood", "imagenet_train"]:
        train_dataset = datasets.ImageFolder(dataset_dir, transform=transformations)
        if dataset_name == "imagenet_ood":
            train_dataset = select_ood_idxs(train_dataset)
    elif dataset_name == "places205":
        train_dataset = PlacesDataset(
            hub.load("hub://activeloop/places205"), transform=transformations
        )
    elif dataset_name == "places365":
        train_dataset = datasets.Places365(
            root=dataset_dir, split="train-standard", transform=transformations
        )
    elif dataset_name == "ade20k":
        pass
    elif dataset_name in ["imagenet_val"]:
        # TODO Matéo : correct this so that path is not hardcoded
        train_dataset = ImagenetValDataset(
            dataset_dir,
            annotations_file=Path(dataset_dir).parent
            / "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
            transform=transformations,
        )
    else: # includes dataset_name == "celeba"
        train_dataset = datasets.ImageFolder(dataset_dir, transform=transformations)
    train_sampler = None
        # for now, cannot be distributed ! will be overriden !
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True, seed=seed
        )
        if is_single_class_batch or similbatch_training:
            raise NotImplemented(
                "Cannot use distributed sampling with single class batch or kmeans training."
            )

    if split_set:
        test_dataset, train_dataset = torch.utils.data.random_split(
            train_dataset,
            [len(train_dataset) // 10, len(train_dataset) - (len(train_dataset) // 10)],
            torch.Generator().manual_seed(seed),
        )
        if is_single_class_batch:
            train_sampler = SingleClassDatasetSampler(train_dataset, batch_size=batch_size)
            test_sampler = SingleClassDatasetSampler(test_dataset, batch_size=batch_size)
        if similbatch_training:
            # get indexes of test set
            assert dataset_name == "imagenet_val", "cosine_sim training only supported for imagenet_val dataset."
            train_sampler = ProximitySampler("./output/cos_sim_matrix.npy", batch_size, train_dataset.indices)
            test_sampler = ProximitySampler("./output/cos_sim_matrix.npy", batch_size, test_dataset.indices)
            if shuffle:
                warnings.warn("Shuffling is not supported with proximity sampler. Setting shuffle to False.")
            shuffle = False
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed),
            sampler=test_sampler,
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
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed),
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
        if is_single_class_batch:
            train_sampler = SingleClassDatasetSampler(train_dataset, batch_size=batch_size)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn_imood
            if dataset_name == "imagenet_ood"
            else collate_fn,
            drop_last=True,
            pin_memory=True,
        )
    return train_loader


class GaussianBlur:
    """Gaussian blur augmentation as in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class PlacesDataset(Dataset):
    # for palces205, which is not a folder of images but calls on an API to stream images in realtime
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.to_pil = transforms.ToPILImage()
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        dl_im = self.ds.images[idx]
        np_dl_im = dl_im.numpy()
        image = self.to_pil(np_dl_im)
        label = self.ds.labels[idx].numpy(fetch_chunks=True).astype(np.int32)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class ImagenetValDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        with open(annotations_file) as f:
            self.img_labels = [int(line) for line in f.readlines()]
        self.transform = transform
        self.files = sorted(glob.glob(f"{img_dir}/*.JPEG"))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = self.pil_loader(img_path)
        label = self.img_labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
        # return image, label

    def pil_loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


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
            transformations.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            )
        elif dataset_name in ["cifar100", "inaturalist", "places205", "places365"]:
            transformations.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )
        elif dataset_name == "celeba":
            transformations.append(transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3))

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
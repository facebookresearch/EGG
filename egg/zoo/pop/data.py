# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import Optional
from pathlib import Path
import glob

import hub
import torch
from PIL import ImageFilter, Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import numpy as np
from torch.utils.data import Subset

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


class SingleClassDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements uniformly from a given class
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
        self.num_samples = len(self.labels)

    def _get_labels(self, dataset):
        if isinstance(dataset, datasets.ImageFolder):
            return torch.Tensor([torch.Tensor(x[1]).int() for x in dataset.imgs])
        elif isinstance(dataset, ImagenetValDataset):
            return torch.Tensor(dataset.img_labels)
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        else:
            raise NotImplementedError("Dataset type not supported")

    def __iter__(self):
        # randomly select a label

        idxs = torch.tensor([])
        for _ in range(self.num_samples // self.batch_size):
            label_id = random.choice(self.labels)
            idxs = torch.concat(
                [
                    idxs,
                    torch.multinomial(
                        (self.labels == label_id).float(),
                        self.batch_size,
                        self.replacement,
                    ),
                ]
            )
        print(idxs, flush=True)
        return (i for i in idxs.int())

    def __len__(self):
        return self.num_samples


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


# def select_inanimate_idxs(dataset):
#     """
#         from all imagenet classes select those which are inanimate
#     """
#     animate_ids = {'n02113712', 'n02037110', 'n01877812', 'n02086079', 'n01847000', 'n02422106', 'n02123159', 'n02115641', 'n02091831', 'n02093859', 'n01871265', 'n02483708', 'n02086240', 'n02281406', 'n02106166', 'n02492035', 'n01443537', 'n01985128', 'n02097298', 'n02123394', 'n02093991', 'n02233338', 'n02100236', 'n02085782', 'n01592084', 'n02033041', 'n01667778', 'n02493509', 'n02102177', 'n01768244', 'n01675722', 'n01773549', 'n01806143', 'n02095889', 'n01530575', 'n01631663', 'n02110185', 'n02281787', 'n01440764', 'n01728572', 'n01855672', 'n02087394', 'n02236044', 'n02102040', 'n02137549', 'n02107683', 'n02174001', 'n01622779', 'n02325366', 'n02443484', 'n02105251', 'n01498041', 'n02101556', 'n02086646', 'n01694178', 'n02417914', 'n02098105', 'n02112137', 'n01775062', 'n02056570', 'n01514668', 'n02109961', 'n01695060', 'n02268443', 'n01692333', 'n02497673', 'n02089973', 'n02113624', 'n02116738', 'n01984695', 'n02326432', 'n01734418', 'n02169497', 'n01818515', 'n01819313', 'n02086910', 'n01632777', 'n01860187', 'n01630670', 'n02398521', 'n01665541', 'n02066245', 'n02097658', 'n02113799', 'n02268853', 'n01795545', 'n02090622', 'n01641577', 'n02111277', 'n01614925', 'n02490219', 'n02110341', 'n02099267', 'n01484850', 'n02489166', 'n02486261', 'n02107142', 'n02395406', 'n01664065', 'n02108089', 'n02112706', 'n01824575', 'n02504013', 'n01980166', 'n02114855', 'n01728920', 'n02097130', 'n02097209', 'n02096051', 'n02100877', 'n01558993', 'n01924916', 'n01776313', 'n01749939', 'n02099601', 'n02110958', 'n02130308', 'n01644373', 'n02125311', 'n01873310', 'n01491361', 'n02123045', 'n09835506', 'n02134084', 'n01798484', 'n02127052', 'n02028035', 'n01689811', 'n02085936', 'n01774384', 'n02114712', 'n02106030', 'n02328150', 'n02133161', 'n02009912', 'n02132136', 'n02134418', 'n02091635', 'n02509815', 'n10565667', 'n02403003', 'n02480495', 'n02514041', 'n02088466', 'n02504458', 'n02422699', 'n02051845', 'n02091467', 'n02114548', 'n02412080', 'n02085620', 'n01773797', 'n02361337', 'n02092002', 'n02102480', 'n02107312', 'n02363005', 'n01693334', 'n02113186', 'n02002556', 'n02101006', 'n02105412', 'n02641379', 'n02510455', 'n01855032', 'n01704323', 'n02110063', 'n01883070', 'n12985857', 'n01807496', 'n02445715', 'n02640242', 'n02397096', 'n02177972', 'n02396427', 'n02279972', 'n02280649', 'n02481823', 'n02101388', 'n13037406', 'n02093647', 'n02442845', 'n02087046', 'n02102318', 'n02111500', 'n02091244', 'n01930112', 'n01537544', 'n02219486', 'n01742172', 'n02488702', 'n02108422', 'n02124075', 'n01531178', 'n01632458', 'n02536864', 'n01843065', 'n02111129', 'n02017213', 'n01735189', 'n02106662', 'n01608432', 'n01990800', 'n02089867', 'n01514859', 'n02113023', 'n10148035', 'n02105505', 'n01629819', 'n01843383', 'n02391049', 'n02229544', 'n02117135', 'n01601694', 'n02108000', 'n01914609', 'n02011460', 'n02112018', 'n02441942', 'n02105162', 'n02096437', 'n02091032', 'n02088094', 'n01685808', 'n01744401', 'n02447366', 'n02077923', 'n02484975', 'n01644900', 'n01774750', 'n02437616', 'n02488291', 'n01978455', 'n02096585', 'n01833805', 'n02119789', 'n02109525', 'n01494475', 'n11939491', 'n02655020', 'n13040303', 'n01917289', 'n02099712', 'n01751748', 'n02112350', 'n02107574', 'n02104365', 'n01882714', 'n01784675', 'n02129165', 'n02606052', 'n01981276', 'n02231487', 'n02276258', 'n02487347', 'n02259212', 'n02099429', 'n02119022', 'n02009229', 'n01616318', 'n02110806', 'n02492660', 'n02423022', 'n02168699', 'n02027492', 'n02123597', 'n01496331', 'n02120079', 'n02444819', 'n02120505', 'n01829413', 'n01739381', 'n01978287', 'n02007558', 'n02138441', 'n01828970', 'n02109047', 'n02104029', 'n01669191', 'n02100735', 'n02321529', 'n02494079', 'n02389026', 'n13054560', 'n02099849', 'n02114367', 'n01797886', 'n01698640', 'n02058221', 'n02486410', 'n02256656', 'n01518878', 'n02483362', 'n01560419', 'n02107908', 'n02457408', 'n02607072', 'n01580077', 'n02025239', 'n01682714', 'n02071294', 'n01729977', 'n02493793', 'n02002724', 'n01796340', 'n02165105', 'n01817953', 'n01983481', 'n02090721', 'n01753488', 'n02408429', 'n02094114', 'n02095570', 'n02437312', 'n02106382', 'n02093256', 'n02190166', 'n02317335', 'n02128925', 'n02226429', 'n02093754', 'n02090379', 'n01667114', 'n02415577', 'n01945685', 'n02277742', 'n02480855', 'n01872401', 'n02094258', 'n01806567', 'n02018795', 'n13044778', 'n02356798', 'n01740131', 'n02526121', 'n01773157', 'n02643566', 'n02410509', 'n02113978', 'n02018207', 'n01748264', 'n02105855', 'n02454379', 'n01770081', 'n02105056', 'n02096294', 'n01968897', 'n02074367', 'n02364673', 'n02096177', 'n01755581', 'n13052670', 'n02129604', 'n02089078', 'n02106550', 'n02105641', 'n02128385', 'n02098286', 'n01688243', 'n02108551', 'n01770393', 'n12998815', 'n01582220', 'n01729322', 'n02264363', 'n02097474', 'n02346627', 'n01944390', 'n02110627', 'n02097047', 'n01955084', 'n01687978', 'n01950731', 'n02128757', 'n12057211', 'n02115913', 'n02088364', 'n02006656', 'n02443114', 'n02088238', 'n01756291', 'n01820546', 'n02091134', 'n02206856', 'n02167151', 'n01677366', 'n02102973', 'n02500267', 'n02319095', 'n02100583', 'n01737021', 'n01943899', 'n02098413', 'n02092339', 'n01986214', 'n01532829', 'n01910747', 'n02095314', 'n02111889', 'n02088632', 'n02165456', 'n02342885', 'n02108915', 'n02172182', 'n01697457', 'n02093428', 'n01534433', 'n02094433', 'n02012849', 'n02013706'}
#     idx = [i for i in range(len(dataset.imgs)) if (Path(dataset.imgs[:][i][0]).parent.stem in animate_ids)]
#     # build the appropriate subset
#     return Subset(dataset, idx)


def select_ood_idxs(dataset):
    """
    from all imagenet classes select those which are inanimate
    """
    # ood_ids = {'n04454908', 'n12267411', 'n12164363', 'n09270657', 'n12435777', 'n13154841', 'n12489815', 'n12461673', 'n05604434', 'n10020890', 'n12674895', 'n10296176', 'n11706761', 'n10347446', 'n12152532', 'n10382710', 'n08518171', 'n04250850', 'n10160280', 'n04961062', 'n05262698', 'n02772435', 'n09805151', 'n10530571', 'n11616486', 'n12275317', 'n12731401', 'n04981658', 'n07596160', 'n04298661', 'n11709674', 'n08524735', 'n09896685', 'n09495962', 'n03147280', 'n12401684', 'n12300840', 'n09666883', 'n08573842', 'n04257684', 'n12432574', 'n11620673', 'n09932508', 'n11615026', 'n14564779', 'n10721321', 'n02802544', 'n11508382', 'n12427184', 'n07601290', 'n12495670', 'n03087521', 'n10333838', 'n11928858', 'n12161577', 'n13908580', 'n12159804', 'n13869788', 'n10622053', 'n09403427', 'n11608250', 'n10123844', 'n05450617', 'n12651611', 'n10734963', 'n08616050', 'n12135898', 'n11703669', 'n03823111', 'n08521623', 'n10471640', 'n07643891', 'n03963645', 'n13865904', 'n10036929', 'n11524451', 'n09247410', 'n04330746', 'n03320262', 'n13881644'}
    # ood_ids = {'n04454908', 'n12267411', 'n12164363', 'n09270657', 'n12435777', 'n12489815', 'n12461673', 'n10020890', 'n11706761', 'n12152532', 'n10382710', 'n08518171', 'n04250850', 'n04961062', 'n02772435',  'n11616486', 'n12731401', 'n04981658', 'n04298661', 'n11709674', 'n08524735', 'n09896685', 'n03147280', 'n12401684', 'n12300840', 'n09666883', 'n08573842', 'n04257684', 'n11620673', 'n09932508', 'n11615026', 'n14564779', 'n10721321', 'n02802544', 'n11508382', 'n12427184', 'n07601290', 'n11928858', 'n13908580', 'n13869788', 'n09403427', 'n11608250', 'n10123844', 'n05450617', 'n12651611', 'n08616050', 'n11703669', 'n03823111', 'n08521623', 'n10471640', 'n07643891', 'n03963645', 'n13865904', 'n11524451', 'n09247410', 'n04330746', 'n03320262', 'n13881644'}

    # build the appropriate subset
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

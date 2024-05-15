from egg.zoo.pop.data import ImageTransformation, ImagenetValDataset, get_dataloader, ClosestImagesSampler, get_augmentation, imood_class_ids
import torch
from pathlib import Path
import numpy as np


if __name__ == "__main__":
    # make cosine similarity matrix of imagenet validation set
    image_size = 384
    batch_size = 64
    use_augmentations = False
    return_original_image = False
    dataset_name = "imagenet_val"
    # dataset_dir = "/projects/colt/imagenet21k_resized/imagenet21k_val/"
    dataset_dir = "/projects/colt/ILSVRC2012/ILSVRC2012_img_val"

    transformations = ImageTransformation(
        image_size,
        use_augmentations,
        return_original_image,
        dataset_name,
        get_augmentation(None, image_size),
    )
    train_dataset = ImagenetValDataset(
        dataset_dir,
        annotations_file=Path(dataset_dir).parent
        / "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
        transform=transformations,
    )

    cos_sim_matrix = np.zeros((len(train_dataset), len(train_dataset)))
    cosim_fn = torch.nn.CosineSimilarity(dim=0)
    for i in range(len(train_dataset)):
        for j in range(i, len(train_dataset)):
            _sim = cosim_fn(
                train_dataset[i][0][0].flatten(),
                train_dataset[j][0][0].flatten(),
            ).item()
            cos_sim_matrix[i, j] = _sim
            cos_sim_matrix[j, i] = _sim
    np.save("~/projects/cos_sim_matrix.npy", cos_sim_matrix)


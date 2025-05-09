import torch
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from egg.zoo.pop.data import ImagenetValDataset

if __name__ == "__main__":
    # make cosine similarity matrix of imagenet validation set
    image_size = 182
    batch_size = 500
    use_augmentations = False
    return_original_image = False
    dataset_name = "imagenet_val"
    dataset_dir = "./data/ILSVRC2012/ILSVRC2012_img_val"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformations = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImagenetValDataset(
        dataset_dir,
        annotations_file=Path(dataset_dir).parent
        / "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
        transform=transformations,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

    num_images = len(train_dataset)
    cos_sim_matrix = np.zeros((num_images, num_images))

    for i, (chunk_imgs_i, _) in enumerate(tqdm(train_dataloader, desc="Calculating cosine similarity", leave=True)):
        chunk_imgs_i = chunk_imgs_i.view(chunk_imgs_i.size(0), -1).to(device)
        if i == len(train_dataloader) - 1:
            chunk_imgs_i = torch.cat((chunk_imgs_i, torch.zeros((batch_size - chunk_imgs_i.size(0), chunk_imgs_i.size(1))).to(device)))

        for j, (chunk_imgs_j, _) in enumerate(tqdm(train_dataloader, desc=f"{i}th horizontal sweep", leave=False)):
            chunk_imgs_j = chunk_imgs_j.view(chunk_imgs_j.size(0), -1).to(device)
            # pad the last chunk
            if j == len(train_dataloader) - 1:
                chunk_imgs_j = torch.cat((chunk_imgs_j, torch.zeros((batch_size - chunk_imgs_j.size(0), chunk_imgs_j.size(1))).to(device)))

            # Calculate pairwise cosine similarity for the chunk
            chunk_cos_sim_matrix = torch.nn.functional.cosine_similarity(chunk_imgs_i, chunk_imgs_j, dim=1)

            # Assign the chunk's cosine similarity matrix to the corresponding indices in the full matrix
            cos_sim_matrix[i * batch_size: (i + 1) * batch_size, j * batch_size: (j + 1) * batch_size] = chunk_cos_sim_matrix[0].cpu().numpy()
    np.save("./output/cos_sim_matrix.npy", cos_sim_matrix)

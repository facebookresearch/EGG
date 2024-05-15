from egg.zoo.pop.data import ImageTransformation, ImagenetValDataset, get_dataloader, ClosestImagesSampler, get_augmentation, imood_class_ids
import torchvision
from pathlib import Path
from egg.zoo.pop.archs import initialize_vision_module
from torch import nn
import numpy as np
import torch
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
    )

if __name__ == "__main__":
    image_size = 384
    batch_size = 64
    use_augmentations = False
    return_original_image = False
    dataset_name = "imagenet_ood"
    augmentation_type = None
    # dataset_dir = "/projects/colt//ILSVRC2012/ILSVRC2012_img_val"
    dataset_dir = "/projects/colt/imagenet21k_resized/imagenet21k_val/"
    transformations = ImageTransformation(
        image_size,
        use_augmentations,
        return_original_image,
        dataset_name,
        get_augmentation(augmentation_type, image_size),
    )
    # train_dataset = ImagenetValDataset(
    #         dataset_dir,
    #         annotations_file=Path(dataset_dir).parent
    #         / "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
    #         transform=transformations,
    #     )
    te_dl, tr_dl = get_dataloader(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=0,
        is_distributed=False,
        seed=0,
        use_augmentations=use_augmentations,
        return_original_image=return_original_image,
        split_set=True,
        kmeans_training=False,
    )
    # colate fn set back to default
    tr_dl.collate_fn = collate_fn_imood
    te_dl.collate_fn = collate_fn_imood

    for _arch_name in ["vit", "inception", "resnet152", "vgg11", "dino", "swin", "virtex"]:
        # train a layer to recognise ood classes
        print(_arch_name, "len training set", len(tr_dl.dataset), "len test set", len(te_dl.dataset))
        model, n_features, name = initialize_vision_module(_arch_name, pretrained=True)
        model = model.cuda()
        model.eval()
        # add layer
        classifier = nn.Linear(n_features, len(imood_class_ids))
        classifier = classifier.cuda()
        # train
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(5):
            for i, (image, label) in enumerate(tr_dl):
                image = torch.tensor(image).cuda()
                label = torch.tensor(label).cuda()
                features = model(image)
                output = classifier(features)
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"epoch {epoch} tr_loss {loss.item()}, tr_acc {(output.argmax(1) == label).float().mean().item()}")
        # evaluate
        running_acc = 0
        for i, (image, label) in enumerate(te_dl):
            image = image.cuda()
            label = label.cuda()
            features = model(image)
            output = classifier(features)
            predicted = output.argmax(1)
            running_acc = running_acc / (i + 1) * i + (predicted == label).float().mean().item() / (i + 1)

        print("Accuracy of the network on the 10000 test images: %d %%" % (100 * running_acc))

        




    # dataloader = get_dataloader(
    #     dataset_dir=dataset_dir,
    #     dataset_name=dataset_name,
    #     image_size=image_size,
    #     batch_size=batch_size,
    #     num_workers=0,
    #     is_distributed=False,
    #     seed=0,
    #     use_augmentations=use_augmentations,
    #     return_original_image=return_original_image,
    #     split_set=True,
    #     kmeans_training=True,
    # )
    for i, (image, label) in enumerate(dataloader):
        print(i, image.shape, label.shape)

from egg.zoo.pop.data import ImageTransformation, ImagenetValDataset, get_dataloader, ClosestImagesSampler, get_augmentation, imood_class_ids
import torch
from pathlib import Path
import numpy as np
# kmeans clustering
from sklearn.cluster import KMeans
from archs import initialize_vision_module
from tqdm import tqdm
from joblib import dump

class ProximitySampler:
    # samples the batch_size closest images to the current image using the cosine similarity matrix
    def __init__(self, cos_sim_matrix, batch_size):
        self.cos_sim_matrix = np.load(cos_sim_matrix)
        self.batch_size = batch_size

    def __iter__(self):
        idxs = torch.tensor([])
        for i in range(len(self.cos_sim_matrix)):
            idxs = torch.cat(
                (
                    idxs,
                    torch.tensor(
                        [torch.argsort(self.cos_sim_matrix[i], descending=True)[1 : self.batch_size + 1]]
                    ),
                )
            )
        return iter(idxs)


if __name__ == "__main__":
    # make cosine similarity matrix of imagenet validation set
    seed = 111
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
    test_dataset, train_dataset = torch.utils.data.random_split(
            train_dataset,
            [len(train_dataset) // 10, len(train_dataset) - (len(train_dataset) // 10)],
            torch.Generator().manual_seed(seed),
        )

    models = ["vit","inception","resnet152","vgg11","dino","swin","virtex"]
    # every model pair
    k_means = {}
    accs = {}
    for m in models:
        # train kmeans on representations of each model
        print(m)
        model, _, _= initialize_vision_module(m, pretrained=True)
        model.to("cuda")
        model.eval()
        # get representations
        train_representations = []
        for idxs in tqdm(range(0, len(train_dataset), batch_size)):
            x = torch.stack([train_dataset[i][0][0] for i in range(idxs, min(idxs + batch_size, len(train_dataset)))]).to("cuda")
            with torch.no_grad():
                train_representations.append(model(x).cpu().numpy())
        train_representations = np.concatenate(train_representations, axis=0)
        # train kmeans
        _kmeans = KMeans(n_clusters=1000, random_state=seed).fit(train_representations)
        k_means[m] = _kmeans
        # test alignment with every other model
        accs[m] = {}
        test_representations_1 = []
        for idxs in tqdm(range(0, len(test_dataset), batch_size)):
            x = torch.stack([test_dataset[i][0][0] for i in range(idxs, min(idxs + batch_size, len(test_dataset)))]).to("cuda")
            with torch.no_grad():
                test_representations_1.append(model(x).cpu().numpy())
        test_representations_1 = np.concatenate(test_representations_1, axis=0)

        for m2 in k_means.keys():
            print(m2)
            accs[m][m2] = 0
            # use train_representations to align labels, by maximising the number of images in the same cluster
            labels1 = k_means[m].predict(train_representations)
            m2_train_representations = []
            model, _, _ = initialize_vision_module(m2, pretrained=True)
            model.to("cuda")
            for idxs in tqdm(range(0, len(train_dataset), batch_size)):
                x = torch.stack([train_dataset[i][0][0] for i in range(idxs, min(idxs + batch_size, len(train_dataset)))]).to("cuda")
                with torch.no_grad():
                    m2_train_representations.append(model(x).cpu().numpy())
            m2_train_representations = np.concatenate(m2_train_representations, axis=0)
            labels2 = k_means[m2].predict(m2_train_representations)
            # align labels
            m2_to_m1 = {}
            for i in range(1000):
                m2_to_m1[i] = np.argmax(np.bincount(labels1[labels2 == i]))

            # test on test set
            test_representations_2 = []
            for idxs in tqdm(range(0, len(test_dataset), batch_size)):
                x = torch.stack([test_dataset[i][0][0] for i in range(idxs, min(idxs + batch_size, len(test_dataset)))]).to("cuda")
                with torch.no_grad():
                    test_representations_2.append(model(x).cpu().numpy())
            test_representations_2 = np.concatenate(test_representations_2, axis=0)

            labels1 = k_means[m].predict(test_representations_1)
            labels2 = k_means[m2].predict(test_representations_2)
            # align labels
            labels2 = np.array([m2_to_m1[l] for l in labels2])
            _acc = (labels1 == labels2).mean()
            print(_acc)
            accs[m][m2] = _acc
    
            # build a game
    for _k in k_means.keys():
        print(_k)
        save_dir = Path("/home/mmahaut/projects/exps/tmlr/kmeans")
        save_dir.mkdir(exist_ok=True, parents=True)
        dump(k_means[_k], save_dir / f"{_k}_kmeans.joblib")
    print(accs)




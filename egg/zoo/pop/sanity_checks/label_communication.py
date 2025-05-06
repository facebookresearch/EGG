from egg.zoo.pop.archs import get_model
from egg.zoo.pop.data import get_dataloader
import torch
import tqdm

def main():
    n_collected_images=128
    batch_size=64
    device= "cuda" if torch.cuda.is_available() else "cpu"
    model_names=['vit', 'inception', 'resnet152', 'vgg11', 'swin']
    # for a given 10 000 images, get the outputs of n models 
    # (we remove dino which is unsupervised, vit_clip which is not a classification model, and virtex which is trained on MS COCO)
    # we store the outputs in a dict with the model name as key
    outputs = {}

    # to keep everything comparable, we use the same dataloader as in other experiments, ignoring labels and receiver inputs
    dataset_names = ["cifar100", "places205", "imagenet_ood", "imagenet_val", "single_class", "places365", "gaussian_noise", "celeba"]
    data_paths = [None, None, "/projects/colt/imagenet21k_resized/imagenet21k_train/", "/projects/colt/ILSVRC2012/ILSVRC2012_img_val/", "/projects/colt/ILSVRC2012/ILSVRC2012_img_val/", "/home/mmahaut/projects/emecom/data", None, "/projects/colt/celebA"]
    _sc = False
    for model_name in model_names:
        outputs[model_name] = {}
        model = get_model(model_name, pretrained=True).to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        for dataset_name, dataset_path in zip(dataset_names, data_paths):
            outputs[model_name][dataset_name] = []
            if dataset_name == "imagenet_val":
                # no need for train data, but we still want to exclude it to be coherent with previous experiments
                _, dl= get_dataloader(dataset_path, dataset_name, batch_size=batch_size, shuffle=True, seed=111, split_set=True, image_size=384, num_workers=0)
            else:
                # deal with the single class case which is not a dataset per se, but a parameter which forces the use of a sampler
                if dataset_name == "single_class":
                    # continue
                    dataset_name = "imagenet_val"
                    _sc = True
                    # when using the single class sampler, we do not shuffle the data
                dl = get_dataloader(dataset_path, dataset_name, batch_size=batch_size, shuffle=not _sc, seed=111, split_set=False, image_size=384, num_workers=0, is_single_class_batch=_sc)
                if _sc:
                    dataset_name = "single_class"
                    _sc = False
            for batch in tqdm.tqdm(dl, desc=f"Processing {dataset_name} with {model_name}", total=n_collected_images//batch_size-1):
                input = batch[0].to(device)
                output = model(input).argmax(dim=1)
                outputs[model_name][dataset_name].extend(output.cpu().detach().numpy())
                if len(outputs[model_name][dataset_name]) >= n_collected_images:
                    break

    # save the outputs
    outpath = f"./output/label_communication_outputs"
    torch.save(outputs, outpath)
    # check that for the same images, the outputs are the same
    for dataset in dataset_names:
        # matrix of model agreement
        agreement = torch.eye(len(model_names))
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:
                    agreement[i, j] = (torch.tensor(outputs[model1][dataset])==torch.tensor(outputs[model2][dataset])).float().mean()
        print(f"Agreement matrix for {dataset}")
        print(agreement)
    
    # simulated communication
    accs = {}
    for dataset in dataset_names:
        accs[dataset] = []
        for model_name1 in model_names:
            for model_name2 in model_names:
                if model_name1 != model_name2:
                    # for each model, we check if the label is sufficient to identify the image
                    # we use the outputs of the other models as the messages
                    scores=[]
                    for i in range(0, len(outputs[model_name1][dataset]), batch_size):
                        score = 0
                        msgs1 = torch.tensor(outputs[model_name1][dataset][i:i+batch_size])
                        msgs2 = torch.tensor(outputs[model_name2][dataset][i:i+batch_size])
                        mask = msgs1 == msgs2
                        score += (torch.rand((~mask).sum()) < 1/batch_size).float().sum()
                        # count labels which are used multiple times
                        msgs1 = msgs1[mask]
                        # for each non-unique label
                        output, counts = torch.unique(msgs1, return_counts=True)
                        score += (counts==1).sum()
                        for c in counts[counts!=1]:
                            score += (torch.rand(c) < 1/c).float().sum()
                        scores.append(score/batch_size)

                    print(f"Score for {model_name1} and {model_name2} on {dataset}")
                    print(sum(scores)/len(scores))
                    accs[dataset].append(sum(scores)/len(scores))
    print(accs)

if __name__ == "__main__":
    main()
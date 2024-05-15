# Imports
import pandas as pd
import torch
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns

# used in a later part of the script :
import numpy as np
import scipy


from egg.zoo.pop.data import get_dataloader
from egg.zoo.pop.utils import get_common_opts, metadata_opener, path_to_parameters
from egg.zoo.pop.scripts.analysis_tools.analysis import interaction_to_dataframe, name_to_idx, extract_name
from egg.core.batch import Batch
from egg.zoo.pop.games import build_game
from egg.zoo.pop.utils import load_from_checkpoint

import random

from egg.zoo.pop.archs import initialize_vision_module
import torch
import torch.nn.functional as F

class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim=2, output_dim=3):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x

def main(model_name,model_name2):
    device = "cuda"
    test_loader, train_loader = get_dataloader(
        dataset_dir="/gpfs/projects/colt/ILSVRC2012/ILSVRC2012_img_val/",
        dataset_name="imagenet_val",
        image_size=384,
        batch_size=1,
        num_workers=4,
        is_distributed=False,
        seed=111,  # same as hardcoded version used in experiments
        use_augmentations=False,
        return_original_image=False,
        split_set=True,
        augmentation_type=None,
        is_single_class_batch=False,
    )

    #train linear classifier on top of vision module to predict the output of another vision module
#optimizer

    model,out_dim,_=initialize_vision_module(model_name)
    model2,out_dim2,_=initialize_vision_module(model_name2)
    model.to("cuda")
    model2.to("cuda")
    lc1 = LinearClassifier(out_dim, out_dim2).to("cuda")
    lc2 = LinearClassifier(out_dim2, out_dim).to("cuda")
    optimizer = torch.optim.Adam(lc1.parameters(), lr=0.001).to("cuda")
    optimizer2 = torch.optim.Adam(lc2.parameters(), lr=0.001).to("cuda")

    for i,b in enumerate(train_loader):
        # b[0] is the image
        im=b[0].to("cuda")
        o1 = model(im)
        o2 = model2(im)
        

        optimizer.zero_grad()
        optimizer2.zero_grad()
        # deal with inception output
        if model_name == "inception":
            o1 = o1.logits
        if model_name2 == "inception":
            o2 = o2.logits
        loss1 = F.cross_entropy(lc1(o1), o2.detach())
        loss2 = F.cross_entropy(lc2(o2), o1.detach())
        loss1.backward()
        loss2.backward()
        print(loss1.item(),loss2.item())
        optimizer.step()
        optimizer2.step()
    correct = 0
    correct2 = 0
    total = 0
    for b in test_loader:
        im=b[0].to("cuda")
        o1 = model(im)
        o2 = model2(im)
        if model_name == "inception":
            o1 = o1.logits
        if model_name2 == "inception":
            o2 = o2.logits
        # use distance as a proxy for accuracy
        correct += F.pairwise_distance(lc1(o1), o2).mean()
        correct2 += F.pairwise_distance(lc2(o2), o1).mean()
        total += 1
    print(model_name,model_name2,correct/total,correct2/total)

def sweep():
    import os
    models = ['vgg11','vit','resnet152','inception','dino','swin','virtex']
    # sbatch file to run this on a cluster job with GPU for all model pairs
    default_dir = "/gpfs/home/mmahaut/projects/experiments/representation_correlation/"
    for i,m1 in enumerate(models):
        for j,m2 in enumerate(models):
            if i > j:
                command = f"python /gpfs/home/mmahaut/projects/EGG/egg/zoo/pop/scripts/analysis_tools/representation_correlation.py --model_name {m1} --model_name2 {m2}"
                sbatch = f"""#!/bin/bash
#SBATCH --job-name=rep_corr_{m1}_{m2}
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --nice=0
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G
#SBATCH --output=/home/mmahaut/projects/experiments/representation_correlation/repr_corr_{m1}_{m2}.out
#SBATCH --error=/home/mmahaut/projects/experiments/representation_correlation/repr_corr_{m1}_{m2}.err
source ~/.bashrc
conda activate omelette

{command}
    """
                with open(f"{default_dir}rep_corr_{m1}_{m2}.sbatch","w") as f:
                    f.write(sbatch)
                os.system(f"sbatch {default_dir}rep_corr_{m1}_{m2}.sbatch")
            
if __name__ == "__main__":
    import argparse
    # get both models
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vit")
    parser.add_argument("--model_name2", type=str, default="vit")
    args = parser.parse_args()
    main(args.model_name,args.model_name2)


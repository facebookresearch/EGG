#!/bin/bash
#SBATCH --job-name=job
#SBATCH --partition=alien
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=8G
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

python -m egg.zoo.pop.train --n_epochs=25 --batch_size=64 --lr=0.0001 --continuous_com --non_linearity="sigmoid" --vocab_size=16 --random_seed=111 --recv_hidden_dim=2048 --dataset_name="cifar100" --gs_temperature=5.0 --keep_classification_layer --vision_model_names_senders="['vgg11','vit','resnet152', 'inception']" --vision_model_names_recvs="['vgg11','vit','resnet152', 'inception']" --image_size=384 
echo "done"
egg/zoo/pop/scripts/sweeper.py
egg/zoo/pop/sweeps/continuous_het_pop_classif.json
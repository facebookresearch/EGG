`simclr` is a reimplementation of *"A Simple Framework for Contrastive Learning of Visual Representations"* [link](https://arxiv.org/pdf/2002.05709.pdf)

Example of minimum working command that launches a 2-gpu training with cifar10

```bash
python -m torch.distributed.launch --use_env --nproc_per_node=2 egg/zoo/simclr/train.py --batch_size=64 --dataset_name="cifar10" --dataset_dir="./cifar10" --image_size=32
```

In `sweeps` there's a configuration that tries to reproduce the setup of the paper with a batch size of 2048 using 16 GPUs
In can be launched calling nest nest from the root directory of EGG with:
```bash
python egg/nest/nest.py --game egg.zoo.simclr.train --sweep egg/zoo/simclr/sweeps/simclr.json --checkpoint_dir="replicate_simclr" --nodes=2 --tasks=8
```

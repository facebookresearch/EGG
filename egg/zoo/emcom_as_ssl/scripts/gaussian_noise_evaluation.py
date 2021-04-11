# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import pathlib

import torch
from torchvision import transforms

from egg.core.interaction import Interaction
from egg.core.util import move_to


def noise_eval(
    game,
    validation_data,
    is_distributed,
    device,
):
    mean_loss = 0.0
    interactions = []
    n_batches = 0
    game.eval()
    with torch.no_grad():
        for batch in validation_data:
            batch = move_to(batch, device)
            optimized_loss, interaction = game(*batch)

            if is_distributed:
                interaction = Interaction.gather_distributed_interactions(interaction)

            interaction = interaction.to("cpu")
            mean_loss += optimized_loss

            interactions.append(interaction)
            n_batches += 1

    mean_loss /= n_batches
    full_interaction = Interaction.from_iterable(interactions)

    return mean_loss.item(), full_interaction


def dump_noise_interaction(logs: Interaction, dump_dir: str):
    dump_dir = pathlib.Path(dump_dir) / "gaussian_noise"
    dump_dir.mkdir(exist_ok=True, parents=True)
    torch.save(logs, dump_dir / "interaction_noise")


def perform_gaussian_noise_evaluation(
    game,
    batch_size,
    distributed_context,
    seed,
    device,
    checkpoint_dir
):

    noise_loader = get_random_noise_dataloader(
        batch_size=batch_size,
        is_distributed=distributed_context.is_distributed,
        seed=seed
    )

    loss, noise_interaction = noise_eval(
        game=game,
        validation_data=noise_loader,
        is_distributed=distributed_context.is_distributed,
        device=device
    )

    if distributed_context.is_leader:
        dump = dict(val_gaussian_loss=loss, val_gaussian_acc=noise_interaction.aux["acc"].mean().item())
        output_message = json.dumps(dump)
        print(output_message, flush=True)

        dump_noise_interaction(noise_interaction, checkpoint_dir)


def get_random_noise_dataloader(
    batch_size: int = 128,
    is_distributed: bool = False,
    seed: int = 111
):
    validation_dataset = GaussianNoiseDataset(size=49152, image_size=224)

    validation_sampler = None
    if is_distributed:
        validation_sampler = torch.utils.data.distributed.DistributedSampler(
            validation_dataset,
            shuffle=False,
            drop_last=True,
            seed=seed
        )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=(validation_sampler is None),
        sampler=validation_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    return validation_loader


class GaussianNoiseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            size: int = 49152,
            image_size: int = 224
    ):
        self.data = [torch.randn(3, image_size, image_size) for _ in range(size)]
        self.transformation = TransformsAugmentNoise(image_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transformation(self.data[index]), torch.Tensor([1])


class TransformsAugmentNoise:
    def __init__(self, size):
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        transformations = [
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        self.transform = transforms.Compose(transformations)

    def __call__(self, x):
        x_i, x_j = self.transform(x), self.transform(x)
        return x_i, x_j, x_i

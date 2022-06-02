from egg.zoo.pop.games import build_game
from egg.zoo.pop.utils import load_from_checkpoint

# from egg.core.callbacks import InteractionSaver
import sys
import pathlib
import torch
from egg.core.batch import Batch
from egg.core.interaction import Interaction
from egg.zoo.pop.data import get_dataloader
from egg.zoo.pop.utils import get_common_opts


def main(params):
    torch.autograd.set_detect_anomaly(True)
    opts = get_common_opts(params)
    build_and_test_game(opts, exp_name=None, dump_dir=opts.checkpoint_dir)


def path_to_parameters():
    # WIP... this is mainly convenience

    # open the yaml file, look into it and get ALL the parameters.
    # input them in the opts, instead of rebuilding the game (beware, if its a second game this might be tricky and require a bit of hacking)
    pass


# Taken from core.trainers.py and modified
def eval(game, data=None):
    interactions = []
    n_batches = 0
    validation_data = data
    with torch.no_grad():
        for batch in validation_data:
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to("cuda")
            _, interaction = game(*batch)
            interaction = interaction.to("cpu")
            game.to("cpu")
            interactions.append(interaction)
            n_batches += 1

    full_interaction = Interaction.from_iterable(interactions)

    return full_interaction


# Taken from core.callbacks.InteractionSaver and modified
def dump_interactions(
    logs: Interaction,
    exp_name: str = "interaction_file",
    dump_dir: str = "./interactions",
):
    dump_dir = pathlib.Path(dump_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)
    torch.save(logs, dump_dir / exp_name)


# buids a game using the usual pop parameters, perfroms evaluations, saves interactions
def build_and_test_game(opts, exp_name, dump_dir):
    """
    From an existing game run some tests
    """

    pop_game = build_game(opts)
    load_from_checkpoint(pop_game, opts.base_checkpoint_path)
    # make everything go to evaluation mode (non-trainable, no training behaviour of any layers)
    for param in pop_game.parameters():
        param.requires_grad = False
    pop_game.eval()

    # get validation data
    val_loader, _ = get_dataloader(
        dataset_dir=opts.dataset_dir,
        dataset_name=opts.dataset_name,
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=111,  # same as hardcoded version used in experiments
        use_augmentations=opts.use_augmentations,
        return_original_image=opts.return_original_image,
        split_set=True,
    )
    # Instead of letting the population game use the agent sampler to select a different pair for every batch
    # We choose the pair and evaluate it on all batches
    interactions = []

    for (
        sender_idx,
        recv_idx,
        loss_idx,
    ) in pop_game.agents_loss_sampler.available_indexes:
        # run inference
        # I feel like this is sort of evil and if it was not python I would definently not get away with this sort of meddling with inner parameters from outside
        sender = pop_game.agents_loss_sampler.senders[sender_idx]
        receiver = pop_game.agents_loss_sampler.receivers[recv_idx]
        loss = pop_game.agents_loss_sampler.losses[loss_idx]
        interactions.append(eval(pop_game.game(sender, receiver, loss), val_loader))

    # save data
    dump_interactions(
        Interaction.from_iterable(interactions),
        exp_name if exp_name is not None else "interactions",
        dump_dir,
    )


# if __name__ == "__main__":
#     torch.autograd.set_detect_anomaly(True)
#     # quick temporary hack : write exp name and dump dir before all the options
#     opts = get_common_opts(params=sys.argv[3:])
#     build_and_test_game(opts, exp_name=sys.argv[1], dump_dir=sys.argv[2])

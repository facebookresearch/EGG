import torch
from egg.zoo.pop.games import build_game
from egg.zoo.pop.utils import load_from_checkpoint, get_common_opts, metadata_opener, path_to_parameters
from egg.zoo.pop.data import get_dataloader
from egg.core.batch import Batch
from egg.core.interaction import Interaction
from tqdm.auto import tqdm
import pathlib
import time
import warnings

def main(params):
    """
    starting point if script is executed from submitit or slurm with normal EGG parameters
    can load from a model path, finding in the metadata all the required parameters instead of requiring them to be rewritten
    """
    torch.autograd.set_detect_anomaly(True)
    _path = ""
    for param in params:
        if "base_checkpoint_path" in param:
            _path = param.rpartition("=")[2]
            print(_path)
    # assert _path != "", "--base_checkpoint_path must be defined"
    if _path == "":
        # only warn
        # raise Warning("No base_checkpoint_path found, attempting to run com extraction without a trained game")
        opts = get_common_opts(params)
    else:
        _path = path_to_parameters(_path)
        with open(_path) as f:
            opts = get_common_opts(metadata_opener(f, data_type="wandb", verbose=True) + params)
        print(opts)
    exp_name = (
        str(opts.dataset_name)
        + str(opts.com_channel)
        + str(opts.vocab_size)
        + (str(opts.noisy_channel) if opts.noisy_channel != None else "")
        + str(opts.augmentation_type)
        + str(opts.vision_model_names_senders)
        + str(opts.vision_model_names_recvs)
        + (str(opts.force_rank) if opts.force_rank is not None else "")
        + ".pth"
    )
    build_and_test_game(opts, exp_name=exp_name, dump_dir=opts.checkpoint_dir, force_rank=opts.force_rank)


def eval(
    sender,
    receiver,
    loss,
    game,
    validation_data=None,
    aux_input=None,
    gs=True,
    batch_size=64,
    device="cuda",
):
    """
    Taken from core.trainers.py and modified (removed loss logging and multi-gpu support)
    runs each batch as a forward pass through the game, returns the interactions that occured
    """
    interactions = []
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(validation_data):
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            aux_input["batch_number"] = (
                torch.arange(0, batch_size, 1, dtype=torch.int32)
                + batch_size * n_batches
            )

            batch = batch.to(device)

            _, interaction, _ = game(
                sender,
                receiver,
                loss,
                batch[0],
                batch[1],
                batch[2],
                aux_input,
            )
            interaction = interaction.to("cpu")
            if gs:
                interaction.message = interaction.message.argmax(dim=-1)
            interactions.append(interaction)
            n_batches += 1


    full_interaction = Interaction.from_iterable(interactions)
    return full_interaction


# Taken from core.callbacks.InteractionSaver and modified
def dump_interactions(
    logs: Interaction,
    exp_name: str = "%j_interactions",
    dump_dir: str = "./interactions",
):
    """
    Used to save interactions in a specified directory
    """
    dump_dir = pathlib.Path(dump_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)
    for i in range(100):
        try:
            torch.save(logs, dump_dir / exp_name)
            break
        except RuntimeError as e:
            print(f"Error saving interactions: {e}")
            time.sleep(5)
            print("Trying again")
    torch.save(logs, dump_dir / exp_name)


# buids a game using the usual pop parameters, perfroms evaluations, saves interactions
def build_and_test_game(opts, exp_name, dump_dir, device="cuda", force_rank=None):
    """
    From an existing game save interactions of each possible agent pair
    Each agent pairs plays on the whole validation set
    """
    pop_game = build_game(opts).to(device)
    if opts.base_checkpoint_path != "":
        load_from_checkpoint(pop_game, opts.base_checkpoint_path)
    # make everything go to evaluation mode (non-trainable, no training behaviour of any layers)
    for param in pop_game.parameters():
        param.requires_grad = False
    pop_game.train(False)
    pop_game.training=False

    # Instead of letting the population game use the agent sampler to select a different pair for every batch
    # We choose the pair and evaluate it on all batches
    interactions = []
    if force_rank is not None:
        print(f"Force rank {force_rank} out of {len(pop_game.agents_loss_sampler.available_indexes)}")
        pop_game.agents_loss_sampler.available_indexes = [pop_game.agents_loss_sampler.available_indexes[force_rank]]
    for (
        sender_idx,
        recv_idx,
        loss_idx,
    ) in pop_game.agents_loss_sampler.available_indexes:
        # run inference
        sender = pop_game.agents_loss_sampler.senders[sender_idx]
        receiver = pop_game.agents_loss_sampler.receivers[recv_idx]
        loss = pop_game.agents_loss_sampler.losses[loss_idx]
        aux_input = {
            "sender_idx": torch.Tensor([sender_idx] * opts.batch_size).int(),
            "recv_idx": torch.Tensor([recv_idx] * opts.batch_size).int(),
            "loss_idx": torch.Tensor([loss_idx] * opts.batch_size).int(),
        }
        # get validation data every time to reset seed (there might be a better and faster way to do this)
        # Beware ! Using all data, both test and train !
        if opts.dataset_name in ["imagenet_ood","places205","cifar100","celeba"] and opts.split_dataset:
            warnings.warn(f"You are splitting the {opts.dataset_name} which is only used at test time by using the default split_dataset=True option. This is not recommended, you could run this test on the entire dataset. Running anyway.")
        if not opts.split_dataset:
            test_loader = get_dataloader(
                dataset_dir=opts.dataset_dir,
                dataset_name=opts.dataset_name,
                image_size=opts.image_size,
                batch_size=opts.batch_size,
                num_workers=opts.num_workers,
                is_distributed=opts.distributed_context.is_distributed,
                seed=111,  # same as hardcoded version used in experiments
                use_augmentations=opts.use_augmentations,
                return_original_image=opts.return_original_image,
                split_set=False,
                augmentation_type=opts.augmentation_type,
                is_single_class_batch=opts.is_single_class_batch,
                shuffle=opts.shuffle,
            )
        else:
            test_loader, train_loader = get_dataloader(
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
                augmentation_type=opts.augmentation_type,
                is_single_class_batch=opts.is_single_class_batch,
                shuffle=opts.shuffle,
            )
        # run evaluation, collect resulting interactions
        interactions.append(
            eval(
                sender,
                receiver,
                loss,
                pop_game.game,
                train_loader if opts.is_single_class_batch or opts.extract_train_com else test_loader, # only train for sc
                aux_input,
                opts.com_channel == "gs",
                opts.batch_size,
                device,
            )
        )

        # save data
        dump_interactions(
            Interaction.from_iterable(interactions),
            exp_name if exp_name is not None else "%j_interactions",
            dump_dir,
        )

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])

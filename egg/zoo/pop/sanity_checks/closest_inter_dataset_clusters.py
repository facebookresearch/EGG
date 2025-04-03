import glob
import torch
from tqdm import tqdm
from egg.core.batch import Batch
from egg.zoo.pop.games import build_game
from egg.zoo.pop.utils import load_from_checkpoint
from egg.zoo.pop.data import get_dataloader
from pathlib import Path


def get_sender(game_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class DummyDistributedContext:
        def __init__(self):
            self.is_distributed = False
    class DummyOpts:
    # dummy opts variable:
        def __init__(self, game_path):
            self.base_checkpoint_path = game_path
            self.vocab_size = 64
            self.random_seed = 111
            self.vision_model_names_senders = "['vgg11']"
            self.vision_model_names_recvs = "['vit']"
            self.keep_classification_layer = False
            self.retrain_vision = False
            self.remove_auxlogits = True
            self.non_linearity = "sigmoid"
            self.max_len = 1
            self.block_com_layer=False
            self.dataset_name = "cifar100"
            self.force_gumbel = False
            self.gs_temperature=0
            self.com_channel = "continuous"
            self.recv_hidden_dim=2048
            self.recv_temperature=0.1
            self.noisy_channel=False
            self.aux_loss=None
            self.aux_loss_weight=0
            self.distributed_context=DummyDistributedContext()

    opts = DummyOpts(game_path)
    pop_game = build_game(opts).to(device)
    if opts.base_checkpoint_path != "":
        load_from_checkpoint(pop_game, game_path)
    for param in pop_game.parameters():
        param.requires_grad = False
    pop_game.train(False)
    pop_game.training=False

    sender = pop_game.agents_loss_sampler.senders[0]
    return sender

def get_distances(dataset_name, n_examples):
    #Todo: use dataset_path instead of hardcoded path
    if dataset_name == "celeba":
        dataset_dir = "/projects/colt/celebA/"
    else:
        dataset_dir = "/projects/colt/ILSVRC2012/ILSVRC2012_img_val/"

    _, t_loader = get_dataloader(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        image_size=384,
        batch_size=64,
        num_workers=4,
        is_distributed=False,
        seed=111,  # same as hardcoded version used in experiments
        use_augmentations=False,
        return_original_image=False,
        split_set=True,
        augmentation_type=None
    )

    sender=get_sender(game_path)
    sae_reps = []
    labels = []
    for batch in tqdm(t_loader):
        if not isinstance(batch, Batch):
            batch = Batch(*batch)
        labels.extend(batch[1])
        batch = batch.to(device)
        message = sender(batch[2].to(device))
        sae_reps.extend(sae.encoder(message).cpu())
        if len(sae_reps) > 10000:
            break
    # find the 10 closest image to each one hot representation
    n_dims = sae_reps[0].shape[0]
    c_distances = []
    for i in range(n_dims):
        # get the one hot representation
        one_hot = torch.zeros(n_dims)
        one_hot[i] = 1
        one_hot = one_hot
        # get the 10 closest images by ordering
        distances = torch.nn.functional.cross_entropy(torch.stack(sae_reps), one_hot.unsqueeze(0).repeat(len(sae_reps), 1), reduction='none').cpu()
        closest_indices = torch.argsort(distances)[:n_examples]
        c_distances.append(distances[closest_indices].mean())
    return torch.tensor(c_distances).argsort()[:60], torch.tensor(c_distances).sort()[0][:60]

if __name__ == "__main__":
    # Parameters
    file = glob.glob("./output/v64_com/imagenet*None*")[0]
    game_path = "./output/vocab_64/590150/final.tar" # vgg --> vit (hard coded in dummy opts)
    base_save_path = "./output/images/"
    Path(base_save_path).mkdir(parents=True, exist_ok=True)
    reps = torch.load(file).message
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_path = "./output/sae.pth"
    n_examples = 10
    # load the SAE
    sae = torch.load(sae_path).to(device)
    # pass the representations through the SAE
    reps = sae.encoder(reps.to(device)).cpu()
    # get the SAE representations for all images
    c_distances = get_distances("celeba", n_examples)
    im_distances = get_distances("imagenet_val", n_examples)
    pl_distances = get_distances("places205", n_examples)

    c_filtered= [im.item() for im in c_distances[0] if im not in im_distances[0][:10] and im not in pl_distances[0][:10]]
    im_filtered= [im.item() for im in im_distances[0] if im not in c_distances[0] and im not in pl_distances[0]]
    pl_filtered= [im.item() for im in pl_distances[0] if im not in c_distances[0][:10] and im not in im_distances[0][:10]]

    cd_filtered = [c_distances[1][i].item() for i, im in enumerate(c_distances[0]) if im not in im_distances[0][:10] and im not in pl_distances[0][:10]]
    imd_filtered = [im_distances[1][i].item() for i, im in enumerate(im_distances[0]) if im not in c_distances[0][:10] and im not in pl_distances[0]]
    pld_filtered = [pl_distances[1][i].item() for i, im in enumerate(pl_distances[0]) if im not in c_distances[0][:10] and im not in im_distances[0][:10]]

    print("celeba", c_filtered, "distances", cd_filtered)
    print("imagenet", im_filtered, "distances", imd_filtered)
    print("places", pl_filtered, "distances", pld_filtered)

# celeba [tensor(525), tensor(306), tensor(934), tensor(710), tensor(555), tensor(566), tensor(829), tensor(935), tensor(774), tensor(608), tensor(45), tensor(905), tensor(187)] distances [tensor(5.9596), tensor(6.0129), tensor(6.0416), tensor(6.0554), tensor(6.0566), tensor(6.0884), tensor(6.0944), tensor(6.0970), tensor(6.1315), tensor(6.1650), tensor(6.1890), tensor(6.2048), tensor(6.2051)]
# imagenet [tensor(245), tensor(742), tensor(436), tensor(75), tensor(745), tensor(919)] distances [tensor(5.5492), tensor(5.6153), tensor(5.6210), tensor(5.6666), tensor(5.7147), tensor(5.7286)]
# places [tensor(792), tensor(147), tensor(746), tensor(995), tensor(804), tensor(8), tensor(970)] distances [tensor(5.7769), tensor(5.7924), tensor(5.8155), tensor(5.8175), tensor(5.8235), tensor(5.8279), tensor(5.8932)]



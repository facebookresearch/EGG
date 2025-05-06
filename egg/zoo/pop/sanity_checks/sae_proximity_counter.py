import torch
from tqdm import tqdm
from egg.zoo.pop.games import build_game
from egg.zoo.pop.utils import load_from_checkpoint
from egg.zoo.pop.data import get_dataloader
from pathlib import Path

def get_sender(game_path, agent_idx=0):
    class DummyDistributedContext:
        def __init__(self):
            self.is_distributed = False
    class DummyOpts:
    # dummy opts variable:
        def __init__(self, game_path):
            self.base_checkpoint_path = game_path
            self.vocab_size = 64
            self.random_seed = 111
            self.vision_model_names_senders = "['vit', 'inception', 'resnet152', 'vgg11', 'dino', 'swin', 'virtex']"
            self.vision_model_names_recvs = "['vit', 'inception', 'resnet152', 'vgg11', 'dino', 'swin', 'virtex']"
            self.remove_auxlogits=False
            self.keep_classification_layer = False
            self.retrain_vision = False
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pop_game = build_game(opts).to(device)
    if opts.base_checkpoint_path != "":
        load_from_checkpoint(pop_game, game_path)
    # make everything go to evaluation mode (non-trainable, no training behaviour of any layers)
    for param in pop_game.parameters():
        param.requires_grad = False
    pop_game.train(False)
    pop_game.training=False

    sender = pop_game.agents_loss_sampler.senders[agent_idx]
    return sender

def get_ims(dataset_name, n_examples, keep_dims):
    v_loader, t_loader = get_dataloader(
        dataset_dir="/projects/colt/celebA/" if dataset_name == "celeba" else "/projects/colt/ILSVRC2012/ILSVRC2012_img_val/",
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
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) if dataset_name == "imagenet_val" else torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) if dataset_name == "imagenet_val" else torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

    sender=get_sender(game_path, agent_idx=3)
    ims = []
    sae_reps = []
    for batch_idx, batch in tqdm(enumerate(t_loader)):
        if not isinstance(batch, Batch):
            batch = Batch(*batch)
        ims.extend(batch[2])
        batch = batch.to(device)
        message = sender(batch[2].to(device))
        sae_reps.extend(sae.encoder(message).cpu())
        if len(sae_reps) > 10000:
            break
    # find the 10 closest image to each one hot representation
    ims = torch.stack(ims)
    n_dims = sae_reps[0].shape[0]
    seen_indices = set()
    c_dists = []
    for i in keep_dims:
        # get the one hot representation
        one_hot = torch.zeros(n_dims)
        one_hot[i] = 1
        one_hot = one_hot
        # get the 10 closest images by ordering
        distances = torch.nn.functional.cross_entropy(torch.stack(sae_reps), one_hot.unsqueeze(0).repeat(len(sae_reps), 1), reduction='none').cpu()
        closest_indices = torch.argsort(distances)[:n_examples]
        c_dists.append(distances[closest_indices])
    return c_dists

if __name__ == "__main__":
    file = glob.glob("./output/training_inter/imagenet*None*['vit', 'inception*")[0]
    game_path = "./output/vocab_64/646399/final.tar" # pop game
    base_save_path = "./output/images_pop/"
    # 100 random dimensions
    keep_dims = [np.random.randint(0, 1024) for _ in range(100)]
    Path(base_save_path).mkdir(parents=True, exist_ok=True)
    reps = torch.load(file).message
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # params 
    sae_path = "/home/mmahaut/projects/exps/tmlr/sae_pop.pth"
    # load the SAE
    sae = torch.load(sae_path).to(device)
    # pass the representations through the SAE
    reps = sae.encoder(reps.to(device)).cpu()

    # get the sae representations for all images
    c_dist = get_ims("celeba", 1, keep_dims)
    p_dists = get_ims("places205", 1, keep_dims)
    im_dists = get_ims("imagenet_val", 1000, keep_dims)
    print(len(im_dists), len(c_dist), len(p_dists), im_dists[0].shape, c_dist[0].shape, p_dists[0].shape)

    # how many images im_ims are closer to a feature than c_ims or p_ims

    im_dists = torch.stack(im_dists)
    av_closer_c_im = []
    for i, dist in enumerate(c_dist):
        av_closer_c_im.append(sum([1 for d in im_dists[i] if d < dist]))
    av_closer_c_im = np.mean(av_closer_c_im)

    print(f"Average number of ImageNet images closer to the closest CelebA image: {av_closer_c_im}")

    av_closer_p_im = []
    for i, dist in enumerate(p_dists):
        av_closer_p_im.append(sum([1 for d in im_dists[i] if d < dist]))
    av_closer_p_im = np.mean(av_closer_p_im)
    print(f"Average number of ImageNet images closer to the closest Places205 image: {av_closer_p_im}")
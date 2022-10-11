import torch 
from egg.core.util import move_to
from egg.zoo.pop.utils import get_common_opts, metadata_opener, load_from_checkpoint
from egg.zoo.pop.games import build_game
import hub
from torchvision import transforms
from egg.zoo.pop.archs import initialize_vision_module
from egg.zoo.pop.data import get_dataloader

# load models from given experiment
def load_models(model_path, metadata_path):
    opts = None
    with open(metadata_path) as f:
        opts = get_common_opts(metadata_opener(f, data_type="wandb", verbose=True))

    pop_game = build_game(opts)
    load_from_checkpoint(pop_game, model_path)
    senders = pop_game.agents_loss_sampler.senders

    # make non-trainable
    for sender in senders:
        sender.eval()
        for param in sender.parameters():
                param.requires_grad = False
    return senders

def get_archs(names):
    archs = []
    features = []
    for name in names:
        arch, n_features, _ = initialize_vision_module(name, pretrained=True, aux_logits=True)
        archs.append(arch)
        features.append(n_features)
    return archs, features

def load_places():
    # load data
    def collate_fn(batch):
        return (
            torch.stack([x["images"] for x in batch], dim=0),
            torch.stack([torch.Tensor(x["labels"]).long() for x in batch], dim=0),
            torch.stack([torch.Tensor(x["index"]) for x in batch], dim=0),
        )
    ds = hub.load("hub://activeloop/places205")
    size = 384
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1)),
        transforms.Resize(size=(size, size)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataloader = ds.pytorch(num_workers = 0, shuffle = True, batch_size= 128, collate_fn=collate_fn, transform={'images':transformations,'labels':None, 'index':None}, pin_memory=True)
    return dataloader

class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim=2, output_dim=3):
    super(LinearClassifier, self).__init__()
    self.linear = torch.nn.Linear(input_dim, output_dim)
    
  def forward(self, x):
    x = self.linear(x)
    return x

def forward_backward(model_idx, input_images, labels, optimizers, criterion):
    # everyone goes to selected device    
    message = senders[model_idx](input_images)
    output = classifiers[model_idx](message)

    loss = criterion(output, labels.view(-1))
    loss.backward()

    optimizers[model_idx].step()
    optimizers[model_idx].zero_grad()
    return message, output, loss


def train_epoch(senders, dataloader, optimizers, criterion, device):
    for batch_idx, batch in enumerate(dataloader):
        _rand_sender = torch.randint(0, len(senders), (1,)).item() # chosing the shuffled input for shuffled-classifier
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        
        for i in range(len(senders) - 1):
            _, output, loss = forward_backward(i, images, labels, optimizers, criterion)
            acc = (output.argmax(dim=1) == labels).float().mean()
            
            if batch_idx % 100 == 0 :
                print(f"{epoch}-{batch_idx} : acc_{i} : {acc}", flush=True)
            if i == _rand_sender:
                _, output, loss = forward_backward(-1, images, labels, optimizers, criterion)
                if batch_idx % 100 == 0 :
                    print(f"{epoch}-{batch_idx} : acc_shuffled : {acc}", flush=True)

if __name__ == "__main__":
    # create classifiers & parametrise learning
    # classifiers and optimizers are on gpu if device is set to cuda
    # an additional classifier is created for the shuffled input, where the sender is randomly chosen to get input
    device="cuda"
    is_baseline = False # set to True to train baseline classifiers
    dataset_name = "imagenet_ood" # "places205"
    dataset_dir =  "/datasets/COLT/imagenet21k_resized/imagenet21k_train/"

    if is_baseline:
        names = ['vgg11','vit','resnet152', 'inception','dino','swin']
        senders, n_features= get_archs(names)
        classifiers = [LinearClassifier(n_features[i], 245).to(device) for i in range(len(senders))] 
        # classifiers.append(LinearClassifier(16, 245).to(device)) can't do that because of different representation lengths
    else:
        model_path = "/homedtcl/mmahaut/projects/experiments/cont_fuller_pop/199721/final.tar"
        metadata_path = "/homedtcl/mmahaut/projects/experiments/cont_fuller_pop/199721/wandb/latest-run/files/wandb-metadata.json"
        senders = load_models(model_path, metadata_path)
        classifiers = [LinearClassifier(16, 245).to(device) for _ in range(len(senders))] 
        classifiers.append(LinearClassifier(16, 245).to(device))

    for sender in senders:
        sender.to(device)
    criterion = torch.nn.CrossEntropyLoss()


    val_dataloader, train_dataloader = get_dataloader(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        batch_size=128,
        image_size=384,
        num_workers=0,
        is_distributed=False,
        use_augmentations=False,
        return_original_image=False,
        seed=111,
        split_set=True,
        augmentation_type=None
    )

    optimizers = []
    for classifier in classifiers:
        opt = torch.optim.Adam(classifier.parameters(), lr=0.01)
        opt.state = move_to(opt.state, device)
        optimizers.append(opt)
    

    for epoch in range(10):
        train_epoch(senders, train_dataloader, optimizers, criterion, device)

    # save models
    for i, classifier in enumerate(classifiers):
        # TODO: remove hardcoded path
        torch.save(classifier.state_dict(), f"../experiments/feature_classif/{dataset_name}_{i}.pth")
    
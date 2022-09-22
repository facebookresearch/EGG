import torch 
from egg.zoo.pop.utils import get_common_opts, metadata_opener, load_from_checkpoint
from egg.zoo.pop.games import build_game
import hub
from torchvision import transforms

#from egg.zoo.pop.data import ImageTransformation
import numpy as np

model_path = "../../../../../experiments/cont_fuller_pop/199711/final.tar"
metadata_path = "../../../../../experiments/cont_fuller_pop/199711/wandb/latest-run/files/wandb-metadata.json"

opts = None
with open(metadata_path) as f:
    opts = get_common_opts(metadata_opener(f, data_type="wandb", verbose=True))

pop_game = build_game(opts)
load_from_checkpoint(pop_game, model_path)
senders = pop_game.agents_loss_sampler.senders
for sender in senders:
    sender.eval()
    # TODO : remove gradients

ds = hub.load("hub://activeloop/places205")
size = 384
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1)),
    transforms.Resize(size=(size, size)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def collate_fn(batch):
    return (
        torch.stack([x["images"] for x in batch], dim=0),
        torch.stack([torch.Tensor(x["labels"]).long() for x in batch], dim=0),
        torch.stack([torch.Tensor(x["index"]) for x in batch], dim=0),
    )
dataloader = ds.pytorch(num_workers = 0, shuffle = True, batch_size= 128, collate_fn=collate_fn, transform={'images':transformations,'labels':None, 'index':None})

class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim=2, output_dim=3):
    super(LinearClassifier, self).__init__()
    self.linear = torch.nn.Linear(input_dim, output_dim)

  def forward(self, x):
    x = self.linear(x)
    return x

criterion = torch.nn.CrossEntropyLoss()

classifiers = [LinearClassifier(16, 245).to("cuda")] * len(senders) # keeping all classifiers on cpu
classifiers.append(LinearClassifier(16, 245).to("cuda")) # <-shuffled-classifier : input a random message from one of the senders

optimizers = [torch.optim.Adam(classifier.parameters(), lr=0.01) for classifier in classifiers]
device="cuda"

def forward_backward(model_idx, input_images, labels):
    # everyone goes to selected device
    senders[model_idx].to(device)
    
    message = senders[model_idx](input_images)
    output = classifiers[model_idx](message)

    loss = criterion(output, labels.view(-1))
    loss.backward()

    optimizers[i].step()
    optimizers[i].zero_grad()
    
    senders[model_idx].to("cpu")

    return message, output, loss
for epoch in range(10):
    for batch_idx, batch in enumerate(dataloader):
        _rand_sender = torch.randint(0, len(senders), (1,)).item() # chosing the shuffled input for shuffled-classifier
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        
        for i in range(len(senders) - 1):
            _, output, loss = forward_backward(i, images, labels)
            if batch_idx % 1000 == 0 :
                print(f"acc_{i} : ", (labels.to(device) == output.argmax(dim=1)).float().mean())
            if i == _rand_sender:
                _, output, loss = forward_backward(-1, images, labels)
                if batch_idx % 1000 == 0 :
                    print("acc_shuffled : ", (labels == output.argmax(dim=1)).float().mean())
    
    

import torch
import torchvision
# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import timm
from egg.zoo.pop.data import get_dataloader
from pathlib import Path
def initialize_vision_module(name: str = "resnet50", pretrained: bool = False, aux_logits=True):
    print("initialize module", name)
    # TODO : Matéo this could use some lazyloading instead of loading them all even if they're not being used
    # It also reloads all of them every time we pick one !
    modules = {
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
        "inception": torchvision.models.inception_v3(
            pretrained=pretrained, aux_logits=aux_logits
        ),
        "vgg11": torchvision.models.vgg11(pretrained=pretrained),
        "vit": timm.create_model("vit_base_patch16_384", pretrained=pretrained),
        "swin":timm.create_model("swin_base_patch4_window12_384", pretrained=pretrained),
        "dino":torch.hub.load('facebookresearch/dino:main', 'dino_vits16',verbose=False)
    }

    return modules[name]

def train_one_epoch(epoch_index, training_loader, model, optimizer, loss_fn,device="cuda"):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels, _ = data
        labels = labels.to(device)
        inputs = inputs.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f"debug {outputs.argmax(1).shape} {labels.shape}")
            print('  batch {} loss: {} acc : {}'.format(i + 1, last_loss, (outputs.argmax(1)==labels).sum().item()/labels.size(0)))

            # tb_x = epoch_index * len(training_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


if __name__ == "__main__":
    import argparse
    import sys
    # chose model to train
    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument(
        "--model",
        type=str,   
        default="vit",
        help="Name of model to train",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/homedtcl/mmahaut/projects/experiments",
        help="Directory to save checkpoints",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training",
    )

    opts = parser.parse_args(sys.argv[1:])
    validation_loader, training_loader = get_dataloader(dataset_dir="/datasets/COLT/imagenet21k_resized" ,dataset_name="cifar100", batch_size=1, num_workers=4, seed=111, image_size=384)

    model = initialize_vision_module(name=opts.model, pretrained=False).to(opts.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, training_loader, model, optimizer, loss_fn, opts.device)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader.to(opts.device)):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        print('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = Path(opts.checkpoint_dir) / 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1



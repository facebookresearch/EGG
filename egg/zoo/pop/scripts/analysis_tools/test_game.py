# a test to surviving noise
# whole game scale : take the image, add some gaussian noise, measure performance. Increase gaussian noice, rince, and repeat
# a test to surviving adversarial attacks
# a test to surviving data augmentation
# (also... the out of scope data test ? is there new non gaussian stuff it could be good at ?)
# a test to playing with discretised messages post-training

# all of this on het vs hom languages

# all of these tests require a game to be launched as a test ground


# imports
from typing import Optional
import timm
# import numpy as np
import torch
import torch.nn as nn
import torchvision
from egg.core.interaction import LoggingStrategy


def initialize_classifiers(name: str = "resnet50", pretrained: bool = False):
    print("initialize module", name)
    modules = {
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
        "inception": torchvision.models.inception_v3(
            pretrained=pretrained, aux_logits=not pretrained
        ),
        "vgg11": torchvision.models.vgg11(pretrained=pretrained),
        "vit": timm.create_model("vit_base_patch16_384", pretrained=pretrained),
    }
    if name not in modules:
        raise KeyError(f"{name} is not currently supported.")

    model = modules[name]

    if name in ["resnet50", "resnet101", "resnet152"]:
        n_features = model.fc.out_features
        # model.fc = nn.Identity()

    elif name == "vgg11":
        n_features = model.classifier[6].out_features
        # model.classifier[6] = nn.Identity()

    elif name == "inception":
        n_features = model.fc.out_features
        # if model.AuxLogits is not None:
        #     model.AuxLogits.fc = nn.Identity()
        # model.fc = nn.Identity()

    else:  # vit
        n_features = model.head.out_features
        # model.head = nn.Identity()

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        if name == "inception":
            model.aux_logits = False
        # TODO : verify that this is not mistakenly turned back on
        model = (
            model.eval()
        )  # Mat : --> dropout blocked, as well as all other training dependant behaviors

    return model, n_features, name

def add_noise(message, variance=0.1,device="cuda"):
    noisy_message = message + (variance**0.5)*torch.randn(message.shape).to(device)
    return noisy_message

class Game(nn.Module):
    def __init__(
        self,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super(Game, self).__init__()

        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )
    def forward(
        self,
        sender,
        receiver,
        loss,
        sender_input,
        labels,
        receiver_input=None,
        aux_input=None,
    ):
        # if not self.training:
        # sender.to("cuda")  # Mat !! TODO : change this to common opts device
        # receiver.to("cuda")
        # sender_input = sender_input.to("cuda")
        # receiver_input = receiver_input.to("cuda")

        message = sender(sender_input, aux_input)
        # here add noise to the message, discrete or continuous
        receiver_output = receiver(add_noise(message), receiver_input, aux_input)

        loss, aux_info = loss(
            sender_input,
            message,
            receiver_input,
            receiver_output,
            labels,
            aux_input,
        )
        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output,
            message=message.detach(),
            message_length=torch.ones(message[0].size(0)),
            aux=aux_info,
        )
        return loss.mean(), interaction

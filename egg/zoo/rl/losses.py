# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import wandb


def contrastive_loss(queries, keys, temperature=0.1):
    b, device = queries.shape[0], queries.device
    logits = queries @ keys.t()
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits /= temperature
    return F.cross_entropy(logits, torch.arange(b, device=device))


def nt_xent_loss(queries, keys, temperature=0.1):
    b, device = queries.shape[0], queries.device

    n = b * 2
    projs = torch.cat((queries, keys))
    logits = projs @ projs.t()

    mask = torch.eye(n, device=device).bool()
    logits = logits[~mask].reshape(n, n - 1)
    logits /= temperature

    labels = torch.cat(((torch.arange(b, device=device) + b - 1), torch.arange(b, device=device)), dim=0)
    loss = F.cross_entropy(logits, labels, reduction='sum')
    loss /= 2 * (b - 1)
    return loss


def discriminative_loss(_sender_input, _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == _labels).detach().float()
    loss = F.cross_entropy(receiver_output, _labels, reduction="none")

    '''
    msg = "-".join([str(x) for x in _message[0].tolist()])
    tgt = str(_labels[0].item())
    example_images = []
    example_images.append(
        wandb.Image(
            _receiver_input[0][0].cpu(),
            caption=f"| Message: {msg}, target is in position {tgt}"
        )
    )
    wandb.log({"Examples": example_images, "Test Accuracy": 100. * acc.cpu() / den})
    '''
    if False:
        wandb.log({"Loss": torch.mean(loss.cpu()).item(), "Accuracy:": torch.mean(acc.cpu()).item()})

    return loss, {'acc': acc}

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Tuple

import torch


def get_loss() -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, Dict[str, Any]],
]:
    return Loss()


class Loss:
    def __init__(self):
        pass

    def __call__(
        self,
        sender_input: torch.Tensor,
        message,
        receiver_input: torch.Tensor,
        receiver_output: torch.Tensor,
        labels: torch.Tensor,
        aux_inpute: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        loss, acc = None, None
        return loss, {"acc": acc}

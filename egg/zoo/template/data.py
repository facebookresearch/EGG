# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Optional, Tuple

import torch


def get_dataloader() -> Iterable[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    "Returning an iterator for tuple(sender_input, labels, receiver_input)."
    pass

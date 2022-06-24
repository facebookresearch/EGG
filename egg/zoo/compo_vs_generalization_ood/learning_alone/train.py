# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json

import torch

import egg.zoo.compo_vs_generalization_ood.archs
from egg import core
from egg.zoo.compo_vs_generalization_ood.learning_alone import data


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_attributes", type=int)
    parser.add_argument("--n_values", type=int)
    parser.add_argument("--stats_freq", type=int, default=0)
    parser.add_argument("--archpart", type=str)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--sender_emb", type=int, default=10)
    parser.add_argument("--receiver_emb", type=int, default=10)
    parser.add_argument("--model", type=str)
    args = core.init(arg_parser=parser, params=params)
    return args


def get_n_correct_ignore_tail(preds, trues):
    """
    uses EOS_SYMBOL=0 to detect the final position of each message and ignores the rest of the
    sequence; used for evaluating encoder (i.e. producing longer, variable-length messages from
    shorter input)
    """
    mask = torch.cat(
        [
            torch.ones(trues.size(0), 1, dtype=torch.bool, device=trues.device),
            trues > 0,
        ],
        -1,
    )[:, :-1]
    return ((((preds == trues) * mask).sum(dim=-1) == mask.sum(-1))).sum().item()


def get_n_correct_exactmatch(preds, trues):
    return ((preds == trues).sum(-1) == preds.size(1)).sum().item()


def repackage(orig_model_class, opts):
    """
    changing the interface of the models (i.e. reformatting the input/output)

    `orig_model_class` works with the format used in the original experiments
    (compo_vs_generalization/train.py)
    i.e, concatenation of one-hot encodings (e.g. [0, 0, 1, 0, ..., 1, 0, 0, 0, ...]).
    The "learning alone" data are in the form of vector of indices (e.g. [2, 0])
    """
    if opts.archpart == "receiver":

        class Receiver(orig_model_class):
            """
            model output: [0, 0, 1, 0, ..., 1, 0, 0, 0, ...] -> [2, 0]
            """

            def forward(self, message, input=None, aux_input=None, lengths=None):
                message.masked_fill_(message == data.PAD_TOKEN, data.EOS_TOKEN)
                per_step_logits, _, _ = super(Receiver, self).forward(
                    message, input, aux_input, lengths
                )
                return (
                    per_step_logits.view(-1, opts.n_attributes, opts.n_values),
                    None,
                    None,
                )

        model = Receiver(opts)
        get_n_correct = get_n_correct_exactmatch
    else:
        if opts.model == "OrigSenderDeterministic":
            opts_dict = vars(opts)
            opts_dict["vocab_size"] = opts.vocab_size + 1

        class Sender(orig_model_class):
            """
            model input: [2, 0] -> [0, 0, 1, 0, ..., 1, 0, 0, 0, ...]
            """

            def forward(self, x, aux_input=None):
                x = torch.nn.functional.one_hot(x, opts.n_values).view(
                    -1, opts.n_attributes * (opts.n_values)
                )
                if opts.model == "OrigSenderDeterministic":
                    return super(Sender, self).forward(x.float())
                return super(Sender, self).forward(x, deterministic=True)

        model = Sender(opts)
        get_n_correct = get_n_correct_ignore_tail
    return model, get_n_correct


def main(params):
    opts = get_params(params)
    print(opts)

    device = opts.device

    dataset = data.get_data(opts)

    model_cls = getattr(egg.zoo.compo_vs_generalization_ood.archs, opts.model)
    model, get_n_correct = repackage(model_cls, opts)

    opt = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data.PAD_TOKEN, reduction="none")

    def evaluate(dataset):
        with torch.no_grad():
            n, loss, correct = 0, 0, 0
            for b in dataset:
                x, y = b
                bs = x.size(0)
                logits, _, _ = model(x)
                preds = logits.argmax(-1)
                correct += get_n_correct(preds, y)
                loss += loss_fn(logits.transpose(2, 1), y).sum().item()
                n += bs

        return loss / (n * opts.n_attributes), correct / n

    model.to(device)
    for epoch in range(opts.n_epochs):
        model.train()
        acc, n = 0, 0
        for b in dataset["train"]:
            opt.zero_grad()
            x, y = b
            logits, _, _ = model(x)
            loss = loss_fn(logits.transpose(2, 1), y).mean()
            loss.backward()
            opt.step()

            with torch.no_grad():
                preds = logits.argmax(-1)
                acc += get_n_correct(preds, y)
                n += x.size(0)

        model.eval()
        print(json.dumps(dict(acc=acc / n, loss=loss.mean().item(), epoch=epoch)))
        res = {
            "generalization hold out": {},
            "uniform holdout": {},
        }
        loss, acc = evaluate(dataset["test_unif"])
        res["uniform holdout"]["loss"] = loss
        res["uniform holdout"]["acc"] = acc
        loss, acc = evaluate(dataset["test_ood"])
        res["generalization hold out"]["loss"] = loss
        res["generalization hold out"]["acc"] = acc
        print(json.dumps(res))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])

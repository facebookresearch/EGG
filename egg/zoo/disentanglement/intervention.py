import torch
from egg.zoo.language_bottleneck.intervention import mutual_info, entropy
from egg.zoo.disentanglement.archs import LinearReceiver
import egg.core as core
import json

def ask_sender(n_attributes, n_values, dataset, sender, device):
    attributes = []
    strings = []
    meanings = []

    for i in range(len(dataset)):
        meaning = dataset[i]

        attribute = meaning.view(n_attributes, n_values).argmax(dim=-1)
        attributes.append(attribute)
        meanings.append(meaning.to(device))

        with torch.no_grad():
            string, *other = sender(meaning.unsqueeze(0).to(device))
        strings.append(string.squeeze(0))

    attributes = torch.stack(attributes, dim=0)
    strings = torch.stack(strings, dim=0)
    meanings = torch.stack(meanings, dim=0)

    return attributes, strings, meanings


def get_linearity_score(n_attributes, n_values,  dataset, sender, device, vocab_size, f_loss):
    _attributes, strings, meanings = ask_sender(n_attributes, n_values, dataset, sender, device)

    linear_receiver = LinearReceiver(n_attributes * n_values, vocab_size, strings.size(1)).to(device)
    optimizer = torch.optim.LBFGS(linear_receiver.parameters())

    def closure(verbose=False):
        optimizer.zero_grad()

        predicted, *rest = linear_receiver(strings)
        loss, rest = f_loss(meanings, None, None, predicted, None)
        loss = loss.mean()
        loss.backward()
        if not verbose:
            return loss
        else:
            return loss, rest
      

    for _ in range(5):
        optimizer.step(closure)

    _, rest = closure(verbose=True)
    return rest['acc'].item()


def information_gap_representation(meanings, representations):
    gaps = torch.zeros(representations.size(1))
    non_constant_positions = 0.0

    for j in range(representations.size(1)):
        symbol_mi = []
        h_j = None
        for i in range(meanings.size(1)):
            x, y = meanings[:, i], representations[:, j]
            info =  mutual_info(x, y)
            symbol_mi.append(info)
            
            if h_j is None:
                h_j = entropy(y)

        symbol_mi.sort(reverse=True)

        if h_j > 0.0:
            gaps[j] = (symbol_mi[0] - symbol_mi[1]) / h_j
            non_constant_positions += 1

    score = gaps.sum() / non_constant_positions
    return score.item()

def information_gap_position(n_attributes, n_values, dataset, sender, device):
    attributes, strings, _meanings = ask_sender(n_attributes, n_values, dataset, sender, device)
    return information_gap_representation(attributes, strings)


def information_gap_vocab(n_attributes, n_values,  dataset, sender, device, vocab_size):
    attributes, strings, _meanings = ask_sender(n_attributes, n_values, dataset, sender, device)

    histograms = []
    for i in range(strings.size(0)):
        representation = torch.zeros(vocab_size)
        for v in range(vocab_size):
            representation[v] = strings[i, :].eq(v).sum()
        histograms.append(representation)

    histograms = torch.stack(histograms, dim=0)
    return information_gap_representation(attributes, histograms[:, 1:])



class Evaluator(core.Callback):
    def __init__(self, dataset, device, n_attributes, n_values, vocab_size):
        self.dataset = dataset
        self.device = device
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.epoch = 0
        self.vocab_size = vocab_size

    def on_epoch_end(self, *stuff):
        game = self.trainer.game
        game.eval()

        positional_disent = information_gap_position(self.n_attributes, self.n_values, self.dataset, game.sender, self.device)
        bos_disent = information_gap_vocab(self.n_attributes, self.n_values, self.dataset, game.sender, self.device, self.vocab_size)
        linearity = get_linearity_score(self.n_attributes, self.n_values, self.dataset, game.sender, self.device, self.vocab_size, game.loss)

        output = dict(epoch=self.epoch, 
                            positional_disent=positional_disent, 
                            bag_of_symbol_disent=bos_disent,
                            linearity=linearity)

        output_json = json.dumps(output)
        print(output_json, flush=True)

        game.train()
        self.epoch += 1

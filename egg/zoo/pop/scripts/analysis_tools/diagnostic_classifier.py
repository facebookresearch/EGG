# from matplotlib import testing
from tabnanny import verbose
import torch
from pathlib import Path

def run_dc(interaction_file, base_path="/Users/u203445/Documents/projects/EGG/interactions",seed=111,device="cuda",n_samples=None,verbose=True, n_epochs=10000,lr=0.01):
    """
    Run the diagnostic classifier on the given interaction file.
    """
    inter = torch.load(Path(base_path) / interaction_file) 
    if n_samples > len(inter.message) or n_samples is None:
        n_samples = len(inter.message)
        if verbose and n_samples is not None: 
            print(f"Warning: n_samples is larger than the number of messages in the interaction file. Setting n_samples to {n_samples}")
    random_indexes = torch.randperm(len(inter.message))[:n_samples]
    data = torch.cat([inter.message,inter.labels[:, None]],dim = 1)[random_indexes]

    testing_data, training_data = torch.utils.data.random_split(
            data,
            [len(data) // 10, len(data) - (len(data) // 10)],
            torch.Generator().manual_seed(seed),
        )
    # from there train to classify back the original classes
    model = train_diagnostic_classifier(train_messages=training_data[:][:,:16], train_labels=training_data[:][:,16:].long(), device = device, n_epochs=n_epochs,lr=lr)
    print(test_diagnostic_classifier(model,test_messages=testing_data[:][:,:16], test_labels=testing_data[:][:,16:].long(), device = device,n_epochs=n_epochs,lr=lr))


class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim=2, output_dim=3):
    super(LinearClassifier, self).__init__()
    self.linear = torch.nn.Linear(input_dim, output_dim)

  def forward(self, x):
    x = self.linear(x)
    return x

def train_diagnostic_classifier(train_messages, train_labels,n_epochs=10000,lr=0.01, device="cuda"):
    # a linear neural network. Takes a message as input, and has to learn to find the cifar class in there.
    model = LinearClassifier(len(train_messages[0]),100).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_loss=[]
    for _ in range(n_epochs):
        output = model(train_messages.to(device))
        loss = criterion(output, train_labels.view(-1).to(device))
        all_loss.append(loss.item())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    return model

def test_diagnostic_classifier(model, test_messages, test_labels, device="cuda"):
    """
    Test the model on the given test data. returns the accuracy.
    """
    assert len(test_messages) == len(test_labels), f"We need one label for every message instead got {len(test_messages)} messages and {len(test_labels)} labels"
    predictions = model.to(device)(test_messages.to(device))
    return (predictions.argmax(1)==test_labels.to(device)).sum()/len(test_messages)

if __name__ == "__main__":
    import sys
    run_dc(sys.argv[1:])
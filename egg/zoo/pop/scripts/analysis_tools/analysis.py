import pandas as pd
import torch

def interaction_to_dataframe(interaction):
    """
    Function to turn the Interaction file into a pandas DataFrame which is covered with syntaxic sugar so easy to use
    """
    df = pd.DataFrame()
    # skipped as is empty in the emecom_pop case
    # df["sender_input"] = interaction.sender_input
    # df["receiver_input"] = interaction.receiver_input
    df["labels"] = interaction.labels
    for key in interaction.aux_input:
        if key == "receiver_message_embedding": # in continuous format message and receiver embedding are the same
            for dim, value in enumerate(interaction.message.T):
                df[f"dim_{dim}"] = value
        else:
            df[key] = interaction.aux_input[key]
    df["receiver_output"] = [i.argmax().item() for i in interaction.receiver_output]

    return df

class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim=2, output_dim=3):
    super(LinearClassifier, self).__init__()
    self.linear = torch.nn.Linear(input_dim, output_dim)

  def forward(self, x):
    x = self.linear(x)
    return x

def train_diagnostic_classifier(train_messages, train_labels):
    # a linear neural network. Takes a message as input, and has to learn to find the cifar class in there.
    model = LinearClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Adam no ?
    all_loss=[]
    for epoch in range(10000):
        output = model(train_messages)

        loss = criterion(output, train_labels.view(-1))
        all_loss.append(loss.item())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    return model

def test_diagnostic_classifier(model, test_messages, test_labels):
    assert len(test_messages) == len(test_labels), f"We need one label for every message instead got {len(test_messages)} messages and {len(test_labels)} labels"
    predictions = model(test_messages)
    return sum(predictions==test_labels)/len(test_messages)



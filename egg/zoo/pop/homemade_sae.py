# imports
import torch
import torch.nn as nn
import glob
from sklearn.model_selection import train_test_split
import wandb

# Create a basic SAE architecture
class SAE(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(SAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def sae_loss(x, y, l1_coef=0.01):
    """
    loss function for the SAE, we use MSE and the l1 regularization
    """
    return nn.MSELoss()(x, y) + l1_coef * torch.mean(torch.abs(x))

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="sae-project")

    # params
    l1_coef = 0.01
    expansion_factor = 16
    epochs = 1000
    learning_rate = 0.001
    batch_size = 64

    # Log hyperparameters
    wandb.config.update({
        "l1_coef": l1_coef,
        "expansion_factor": expansion_factor,
        "epochs": epochs,
        "learning_rate": learning_rate
    })

    # load all training representation from communication logs. We hardcode the path to one log we test on
    interaction_file = glob.glob("./data/training_inter/imagenet*None*['vit', 'inception*")[0]

    print(f"Loading representations from {interaction_file}")
    all_data=torch.load(interaction_file)
    reps = all_data.message
    sender = all_data.aux_input["sender_idx"]
    # sender 3 is VGG, which we use as an example
    # get indices of the sender
    sender_indices = [i for i, s in enumerate(sender) if s == 3]
    reps = reps[sender_indices]

    # Split the data into training, validation, and testing sets
    train_reps, test_reps = train_test_split(reps, test_size=0.2, random_state=42)
    train_reps, val_reps = train_test_split(train_reps, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    # have an SAE train to reconstruct the input
    input_size = train_reps.shape[1]
    hidden_size = input_size * expansion_factor
    print(f"Input representation size: {input_size}, expansion factor: {expansion_factor}, Hidden SAE size: {hidden_size}")
    sae = SAE(input_size, hidden_size)
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    # Check if GPU is available and move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae.to(device)
    # Training loop
    # Create DataLoader for batching
    train_loader = torch.utils.data.DataLoader(train_reps, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_reps, batch_size=batch_size, shuffle=False)
    for epoch in range(epochs):
        sae.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = sae(batch)
            loss = sae_loss(output, batch, l1_coef)
            loss.backward()
            optimizer.step()

        # Validation
        sae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_output = sae(batch)
                val_loss += sae_loss(val_output, batch, l1_coef).item()
        val_loss /= len(val_loader)

        # Log metrics
        wandb.log({
            "epoch": epoch,
            "training_loss": loss.item(),
            "validation_loss": val_loss
        })

        print(f"Epoch {epoch}, Training loss: {loss.item()}, Validation loss: {val_loss}")

    # Testing
    sae.eval()
    with torch.no_grad():
        test_output = sae(test_reps.to(device)).cpu()
        test_loss = sae_loss(test_output, test_reps, l1_coef)
    print(f"Test loss: {test_loss.item()}")

    # Log test loss
    wandb.log({"test_loss": test_loss.item()})

    # Save the SAE
    torch.save(sae, "./output/sae_pop.pth")
    wandb.save("./output/sae_pop.pth")
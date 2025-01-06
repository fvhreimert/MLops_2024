import torch
import typer
from data import corrupt_mnist
from model import FredNet
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt

app = typer.Typer()

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# PLOT

def plot_training_statistics(train_loss, train_accuracy, save_path: str):
    """Create and save a publication-quality plot of training statistics."""
    epochs = range(1, len(train_loss) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(epochs, train_loss, label="Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, train_accuracy, label="Accuracy", color="orange", linewidth=2)
    ax2.set_ylabel("Accuracy", fontsize=14, color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    fig.tight_layout()
    ax1.legend(loc="upper left", fontsize=12)
    ax2.legend(loc="upper right", fontsize=12)

    plt.title("Training Statistics", fontsize=16)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Training plot saved to {save_path}")


@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10):
    """Train a model on MNIST and save training statistics."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = FredNet().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = []
    train_accuracy = []

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for img, target in tqdm(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += (y_pred.argmax(dim=1) == target).float().mean().item()

        avg_loss = running_loss / len(train_dataloader)
        avg_accuracy = running_accuracy / len(train_dataloader)

        train_loss.append(avg_loss)
        train_accuracy.append(avg_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    print("Training complete.")
    torch.save(model.state_dict(), "model.pth")
    plot_training_statistics(train_loss, train_accuracy, save_path="training_statistics.png")

@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(f"Loading model from {model_checkpoint}...")

    # Load the model
    model = FredNet().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model.eval()

    # Load the test dataset
    _, test_set = corrupt_mnist()

    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Metrics
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for img, target in tqdm(test_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            total_loss += loss.item()

            correct_predictions += (y_pred.argmax(dim=1) == target).sum().item()
            total_predictions += target.size(0)

    avg_loss = total_loss / len(test_dataloader)
    accuracy = correct_predictions / total_predictions

    print(f"Evaluation complete: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")



if __name__ == "__main__":
    app()

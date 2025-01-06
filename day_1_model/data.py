from __future__ import annotations

import matplotlib.pyplot as plt  # only needed for plotting
import torch
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting
import random
import os


DATA_PATH = "/Users/frederikreimert/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Kandidat_DTU/2024E/MLops/dtu_mlops/corruptmnist_v1"

test_path = "/test_images.pt"
test_target_path = "/test_target.pt"

train_paths_list = ["/train_images_0.pt",
                    "/train_images_1.pt",
                    "/train_images_2.pt",
                    "/train_images_3.pt",
                    "/train_images_4.pt",
                    "/train_images_5.pt"]

train_target_list = ["/train_target_0.pt",
                     "/train_target_1.pt",
                     "/train_target_2.pt",
                     "/train_target_3.pt",
                     "/train_target_4.pt",
                     "/train_target_5.pt"]

# 0 must not be rotated
# 1 must be rotated by -45 degrees
# 2 must be rotated by -40 degrees
# 3 must be rotated by 45 degrees
# 4 must not be rotated
# 5 must not be rotated


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test dataloaders for corrupt MNIST."""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(DATA_PATH + train_paths_list[i]))
        train_target.append(torch.load(DATA_PATH + train_target_list[i]))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(DATA_PATH + test_path)
    test_target: torch.Tensor = torch.load(DATA_PATH + test_target_path)  # Updated name

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


def show_random_images_grid(dataset: torch.utils.data.Dataset, grid_size: int = 10) -> None:
    """Display a grid of random images and their labels from a dataset."""
    images, labels = dataset.tensors  # Unpack images and labels from the TensorDataset
    total_images = grid_size * grid_size  # Total number of images to display
    indices = random.sample(range(len(images)), total_images)  # Randomly select indices
    
    selected_images = images[indices]
    selected_labels = labels[indices]

    # Plot the selected images and their labels in a grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        ax.imshow(selected_images[i].squeeze(), cmap="gray")
        ax.set_title(f"{selected_labels[i].item()}", fontsize=20)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_set, test_set = corrupt_mnist()
    print(f"Size of training set: {len(train_set)}")
    print(f"Size of test set: {len(test_set)}")
    print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
    print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
    show_random_images_grid(train_set)
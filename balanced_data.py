from torch.utils.data import Dataset, DataLoader
import random
from collections import Counter
import numpy as np

class BalancedDataset(Dataset):
    def __init__(self, original_dataset):
        """
        A wrapper around a dataset to balance classes by subsampling over-represented classes.

        Args:
            original_dataset (Dataset): The original dataset to balance.
        """
        self.original_dataset = original_dataset

        # Extract labels from the dataset
        self.labels = []
        for i in range(len(original_dataset)):
            _, label = original_dataset[i]              #Iterating over all the samples in the original dataset to extract the labels

            # Ensure label is an integer for consistency
            if isinstance(label, int):
                self.labels.append(label)
            else:
                self.labels.append(label.item())  # Convert to integer if necessary

        # Count how many samples belong to each class
        label_counts = Counter(self.labels)

        # Find the size of the smallest class
        min_class_size = min(label_counts.values())

        # Create a balanced set of indices
        self.balanced_indices = []
        for label, count in label_counts.items():
            # Get all indices for the current label
            label_indices = [i for i, lbl in enumerate(self.labels) if lbl == label]

            # Subsample to the size of the smallest class
            self.balanced_indices.extend(
                np.random.choice(label_indices, min_class_size, replace=False)
            )

        # Shuffle the balanced indices
        np.random.seed(42)                          # Set a random seed for reproducibility
        np.random.shuffle(self.balanced_indices)

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        # Map the index to the balanced indices
        original_idx = self.balanced_indices[idx]
        return self.original_dataset[original_idx]
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
            _, label = original_dataset[i]

            # Ensure label is an integer
            if isinstance(label, int):
                self.labels.append(label)
            else:
                self.labels.append(label.item())  # Convert to integer if necessary

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

# class BalancedDataset(Dataset):
#     def __init__(self, original_dataset):
#         """
#         Balances the dataset by subsampling over-represented classes.
        
#         Args:
#             original_dataset: PyTorch Dataset object to balance.
#         """
#         self.original_dataset = original_dataset
#         self.balanced_indices = self._balance_classes()

#     def _balance_classes(self):
#         """
#         Subsamples over-represented classes to match the size of the smallest class.
#         """
#         # Group indices by class
#         class_indices = {}
#         for idx in range(len(self.original_dataset)):
#             _, label = self.original_dataset[idx]
            
#             if label not in class_indices:
#                 class_indices[label] = []
#             class_indices[label].append(idx)

#         # Find the size of the smallest class
#         min_class_size = min(len(indices) for indices in class_indices.values())

#         # Subsample each class to the minimum size
#         balanced_indices = []
#         for indices in class_indices.values():
#             balanced_indices.extend(random.sample(indices, min_class_size))

#         return balanced_indices

#     def __len__(self):
#         return len(self.balanced_indices)

#     def __getitem__(self, index):
#         balanced_index = self.balanced_indices[index]
#         return self.original_dataset[balanced_index]
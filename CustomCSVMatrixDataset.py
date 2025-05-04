import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# Define a custom normalization function
def z_score_normalize_columns(data, input_shape=(28, 28), mean_values=None, std_values=None):
    # Reshape the data to the desired input shape
    reshaped_data = data.reshape(-1, *input_shape)
    if mean_values is None or std_values is None:
    # Calculate the mean and standard deviation for each feature across all time steps
    # Calculate the minimum and maximum values for each column
        mean_values = reshaped_data.mean(axis=(0, 1))
        std_values = reshaped_data.std(axis=(0, 1))
        std_values[std_values == 0] = 1  # Avoid division by zero
    # Ensure min_values and range_values have the correct shape for broadcasting
        mean_values = mean_values[np.newaxis, np.newaxis, :]
        std_values = std_values[np.newaxis, np.newaxis, :]
    normalized_data = (reshaped_data - mean_values) / std_values

    return normalized_data, mean_values, std_values

# Define a custom normalization function
def normalize_columns(data, input_shape=(28, 28),min_values=None, max_values=None):
    # Reshape the data to the desired input shape
    reshaped_data = data.reshape(-1, *input_shape)
    if min_values is None or max_values is None:
        # Calculate the minimum and maximum values for each column
        min_values = reshaped_data.min(axis=(0, 1))
        max_values = reshaped_data.max(axis=(0, 1))

        # Print the minimum and maximum values
        print("Minimum values per column:\n", min_values)
        print("Maximum values per column:\n", max_values)

        # Ensure that there are no zero denominators
        range_values = max_values - min_values
        range_values[range_values == 0] = 1  # Avoid division by zero

    # Ensure min_values and range_values have the correct shape for broadcasting
    min_values = min_values[np.newaxis, np.newaxis, :]
    range_values = range_values[np.newaxis, np.newaxis, :]

    # Perform column-wise normalization
    normalized_data = (reshaped_data - min_values) / range_values

    return normalized_data, min_values, max_values

class CustomCSVMatrixDataset(Dataset):
    def __init__(self, data_path, labels_path, input_shape=(28, 28), transform=None, mean_values=None, std_values=None):
        self.data_path = data_path
        self.labels_path = labels_path
        self.input_shape = input_shape
        self.transform = transform

        # 读取数据
        self.data = pd.read_csv(data_path).values
        self.labels = pd.read_csv(labels_path).values.flatten()

        self.data, self.mean_values, self.std_values = z_score_normalize_columns(self.data, self.input_shape, mean_values, std_values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取数据和标签
        sample = self.data[idx]
        label = self.labels[idx]

        # Convert to tensor
        sample = torch.from_numpy(sample).float()
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return sample, label

class CustomCSVMatrixDataset1D(Dataset):
    def __init__(self, data_path, labels_path, input_shape=(28, 28), transform=None):
        self.data_path = data_path
        self.labels_path = labels_path
        self.input_shape = input_shape
        self.transform = transform

        # 读取数据
        self.data = pd.read_csv(data_path).values
        self.labels = pd.read_csv(labels_path).values.flatten()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取数据和标签
        sample = self.data[idx]
        label = self.labels[idx]

        # Reshape the data to the desired input shape
        sample = sample.reshape(self.input_shape)

        # Convert to tensor
        sample = torch.from_numpy(sample).float()
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return sample, label
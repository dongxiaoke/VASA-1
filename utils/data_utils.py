import os
import torch
from torch.utils.data import Dataset

class PreprocessedDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx])
        if self.transform:
            data = self.transform(data)
        return data

def save_tensor(tensor, file_path):
    """
    Save a tensor to a specified file path.
    """
    torch.save(tensor, file_path)
    print(f"Tensor saved to {file_path}")

def load_tensor(file_path):
    """
    Load a tensor from a specified file path.
    """
    tensor = torch.load(file_path)
    print(f"Tensor loaded from {file_path}")
    return tensor

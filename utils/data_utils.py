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

class PreprocessedVideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        self.video_dir = video_dir
        self.video_files = [os.path.join(video_dir, dir) for dir in os.listdir(video_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        frame_files = sorted(os.listdir(self.video_files[idx]), key=lambda x: int(x.split('_')[1].split('.')[0]))
        frames = [os.path.join(self.video_files[idx], frame_file) for frame_file in frame_files]
        return frames

def pad_collate(batch):
    max_len = max([item.size(0) for item in batch])
    batch_size, _, height, width = batch[0].size()
    padded_batch = torch.zeros(batch_size, max_len, 3, height, width)

    for i, video in enumerate(batch):
        padded_batch[i, :video.size(0), :, :, :] = video

    return padded_batch

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

def load_frame(frame_path, transform=None):
    frame = torch.load(frame_path)
    if transform:
        frame = transform(frame)
    return frame

def frame_collate(batch):
    return batch

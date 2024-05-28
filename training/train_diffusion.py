import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model
from models.transformers.diffusion_transformer import DiffusionTransformer
from utils.model_utils import save_model, load_model
from utils.data_utils import PreprocessedDataset, PreprocessedVideoDataset
import yaml

# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset for loading preprocessed data
class AudioVideoDataset(Dataset):
    def __init__(self, audio_dir, video_dir, transform=None):
        self.audio_files = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir)]
        self.video_dirs = [os.path.join(video_dir, dir) for dir in os.listdir(video_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio = torch.load(self.audio_files[idx])
        video_frames = []
        frame_files = sorted(os.listdir(self.video_dirs[idx]), key=lambda x: int(x.split('_')[1].split('.')[0]))
        for frame_file in frame_files:
            frame = torch.load(os.path.join(self.video_dirs[idx], frame_file))
            if self.transform:
                frame = self.transform(frame)
            video_frames.append(frame)
        video = torch.stack(video_frames)
        return audio, video

# Loss function
class DiffusionLoss(nn.Module):
    def __init__(self):
        super(DiffusionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        return self.mse_loss(input, target)

def train_diffusion(audio_dir, video_dir, model_save_path, embedding_dim, nhead, num_encoder_layers, dropout, learning_rate, batch_size, num_epochs, save_interval):
    # Load preprocessed data
    dataset = AudioVideoDataset(audio_dir, video_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    diffusion_transformer = DiffusionTransformer(embedding_dim, nhead, num_encoder_layers, dropout).to(device)

    # Optimizers
    optimizer = optim.Adam(diffusion_transformer.parameters(), lr=learning_rate)

    # Loss function
    criterion = DiffusionLoss()

    # Training loop
    for epoch in range(num_epochs):
        diffusion_transformer.train()
        total_loss = 0.0

        for audio, video in dataloader:
            audio = audio.to(device)
            video = video.to(device)

            # Extract audio features using Wav2Vec2
            with torch.no_grad():
                audio_features = wav2vec2_model(audio).last_hidden_state

            # Generate motion latent codes using the diffusion transformer
            gaze_direction = torch.randn(audio_features.size(0), audio_features.size(1), 2).to(device)
            head_distance = torch.randn(audio_features.size(0), audio_features.size(1), 1).to(device)
            emotion_offset = torch.randn(audio_features.size(0), audio_features.size(1), 1).to(device)

            motion_latent_codes = diffusion_transformer(audio_features, gaze_direction, head_distance, emotion_offset)

            # Compute loss
            loss = criterion(motion_latent_codes, video)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Save model checkpoint
        if (epoch + 1) % save_interval == 0:
            save_model(diffusion_transformer, os.path.join(model_save_path, f'diffusion_transformer_epoch_{epoch+1}.pth'))

    print("Training completed.")
    # Save final model
    save_model(diffusion_transformer, os.path.join(model_save_path, 'diffusion_transformer_final.pth'))

# Example usage
if __name__ == "__main__":
    # Load configuration
    with open('training/configs/config_diffusion.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Extract configuration parameters
    embedding_dim = config['model']['embedding_dim']
    nhead = config['model']['nhead']
    num_encoder_layers = config['model']['num_encoder_layers']
    dropout = config['model']['dropout']
    learning_rate = config['training']['learning_rate']
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    save_interval = config['training']['save_interval']
    audio_dir = config['data']['audio_dir']
    video_dir = config['data']['video_dir']
    model_save_path = config['output']['model_save_path']

    train_diffusion(audio_dir, video_dir, model_save_path, embedding_dim, nhead, num_encoder_layers, dropout, learning_rate, batch_size, num_epochs, save_interval)

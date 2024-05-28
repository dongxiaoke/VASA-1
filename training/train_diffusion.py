import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Model
from models.transformers.diffusion_transformer import DiffusionTransformer
from utils.data_utils import PreprocessedDataset, PreprocessedVideoDataset, frame_collate
import torch.nn.functional as F
import yaml

# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

def frame_collate(batch):
    audios = [item[0] for item in batch]
    videos = [item[1] for item in batch]
    max_len = max([video.size(0) for video in videos])
    batch_size = len(videos)
    channels, height, width = videos[0].size(1), videos[0].size(2), videos[0].size(3)
    padded_videos = torch.zeros(batch_size, max_len, channels, height, width)

    for i, video in enumerate(videos):
        padded_videos[i, :video.size(0), :, :, :] = video

    return torch.stack(audios), padded_videos

# Training function
def train_diffusion(audio_dir, video_dir, model_save_path, log_dir, embedding_dim, nhead, num_encoder_layers, dropout, learning_rate, batch_size, num_epochs, save_interval):
    # Load preprocessed data
    dataset = AudioVideoDataset(audio_dir, video_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=frame_collate)
    print("Dataset loaded successfully.")

    # Initialize models
    wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    diffusion_transformer = DiffusionTransformer(embedding_dim, nhead, num_encoder_layers, dropout).to(device)
    print("Models initialized.")

    # Optimizer
    optimizer = optim.Adam(diffusion_transformer.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.MSELoss()

    # Create directories if they do not exist
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Training loop
    print("Starting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        diffusion_transformer.train()
        total_loss = 0.0

        for batch_idx, (audio, video) in enumerate(dataloader):
            audio = audio.squeeze(0).to(device)  # Remove the extra batch dimension
            video = video.to(device)
            print(f"Audio shape: {audio.shape}, Video shape: {video.shape}")

            # Extract audio features using Wav2Vec2
            with torch.no_grad():
                audio_features = wav2vec2_model(audio).last_hidden_state
            print(f"Audio features shape: {audio_features.shape}")

            # Generate motion latent codes using the diffusion transformer
            gaze_direction = torch.randn(audio_features.size(0), audio_features.size(1), 2).to(device)
            head_distance = torch.randn(audio_features.size(0), audio_features.size(1), 1).to(device)
            emotion_offset = torch.randn(audio_features.size(0), audio_features.size(1), 1).to(device)

            motion_latent_codes = diffusion_transformer(audio_features, gaze_direction, head_distance, emotion_offset)
            print(f"Motion latent codes shape: {motion_latent_codes.shape}")

            # Resize the target video frames to match the output dimensions (temporal dimension)
            resized_video = F.interpolate(video.permute(0, 2, 1, 3, 4), size=(motion_latent_codes.size(1), 512, 512)).permute(0, 2, 1, 3, 4)
            print(f"Resized video shape: {resized_video.shape}")

            # Compute loss
            loss = criterion(motion_latent_codes, resized_video)
            print(f"Loss: {loss.item()}")

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

        # Save loss to log file
        with open(os.path.join(log_dir, 'training_log.txt'), 'a') as log_file:
            log_file.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}\n')

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")
    print(f'Final Average Loss: {total_loss / (num_epochs * len(dataloader)):.4f}')

    # Save final model
    save_model(diffusion_transformer, os.path.join(model_save_path, 'diffusion_transformer_final.pth'))
    print("Final model saved.")

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
    learning_rate = float(config['training']['learning_rate'])
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    save_interval = config['training']['save_interval']
    audio_dir = config['data']['audio_dir']
    video_dir = config['data']['video_dir']
    model_save_path = config['output']['model_save_path']
    log_dir = config['output']['log_dir']

    train_diffusion(audio_dir, video_dir, model_save_path, log_dir, embedding_dim, nhead, num_encoder_layers, dropout, learning_rate, batch_size, num_epochs, save_interval)
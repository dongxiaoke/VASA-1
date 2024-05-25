import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from models.encoders.face_encoder import FaceEncoder
from models.decoders.face_decoder import FaceDecoder
from models.transformers.diffusion_transformer import DiffusionTransformer
from training.train_utils import load_model
from sklearn.metrics import mean_squared_error

# Custom Dataset for loading preprocessed data
class EvaluationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = torch.load(self.image_files[idx])
        if self.transform:
            image = self.transform(image)
        return image

# Evaluation function
def evaluate_model_performance(image_dir, model_paths, device='cpu'):
    # Load preprocessed data
    dataset = EvaluationDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load models
    face_encoder = FaceEncoder().to(device)
    load_model(face_encoder, model_paths['face_encoder'], device)
    
    face_decoder = FaceDecoder().to(device)
    load_model(face_decoder, model_paths['face_decoder'], device)
    
    diffusion_transformer = DiffusionTransformer().to(device)
    load_model(diffusion_transformer, model_paths['diffusion_transformer'], device)

    # Initialize loss function
    criterion = nn.MSELoss()

    # Evaluation loop
    total_loss = 0.0
    for image in dataloader:
        image = image.to(device)

        # Forward pass
        with torch.no_grad():
            identity_features, appearance_features = face_encoder(image)
            motion_latent_codes = torch.zeros_like(appearance_features)  # Use zero motion latent codes for evaluation
            reconstructed_image = face_decoder(appearance_features, identity_features, motion_latent_codes)
        
        # Compute loss
        loss = criterion(reconstructed_image, image)
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Evaluation Loss: {avg_loss:.4f}')

    return avg_loss

# Example usage
if __name__ == "__main__":
    image_dir = "path/to/preprocessed/image_data"
    model_paths = {
        'face_encoder': 'path/to/saved/face_encoder.pth',
        'face_decoder': 'path/to/saved/face_decoder.pth',
        'diffusion_transformer': 'path/to/saved/diffusion_transformer.pth'
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_model_performance(image_dir, model_paths, device)

import os
import numpy as np
import torch
import torchaudio
import cv2
from torchvision import transforms
from torchaudio.transforms import Resample

# Constants
SAMPLE_RATE = 16000
AUDIO_FEATURE_DIM = 768  # Assuming Wav2Vec2 hidden size is 768
IMAGE_SIZE = 256  # Assuming input image size is 256x256

# Define image transformations
image_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_data(data_dir):
    """
    Load datasets from the specified directory.
    """
    audio_files = []
    image_files = []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
            elif file.endswith(('.jpg', '.png')):
                image_files.append(os.path.join(root, file))
    
    return audio_files, image_files

def preprocess_audio(audio_file):
    """
    Preprocess audio file: load, resample, normalize, and extract features.
    """
    waveform, original_sample_rate = torchaudio.load(audio_file)
    
    # Resample if necessary
    if original_sample_rate != SAMPLE_RATE:
        resampler = Resample(original_sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Normalize waveform
    waveform = (waveform - waveform.mean()) / waveform.std()
    
    return waveform

def preprocess_image(image_file):
    """
    Preprocess image file: load, resize, and normalize.
    """
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image_transforms(image)
    
    return image

def preprocess_data(audio_files, image_files):
    """
    Preprocess both audio and image files.
    """
    audio_data = [preprocess_audio(audio_file) for audio_file in audio_files]
    image_data = [preprocess_image(image_file) for image_file in image_files]
    
    return audio_data, image_data

def save_preprocessed_data(audio_data, image_data, output_dir):
    """
    Save preprocessed data to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save audio data
    audio_output_dir = os.path.join(output_dir, 'audio')
    os.makedirs(audio_output_dir, exist_ok=True)
    for i, audio in enumerate(audio_data):
        torch.save(audio, os.path.join(audio_output_dir, f'audio_{i}.pt'))
    
    # Save image data
    image_output_dir = os.path.join(output_dir, 'image')
    os.makedirs(image_output_dir, exist_ok=True)
    for i, image in enumerate(image_data):
        torch.save(image, os.path.join(image_output_dir, f'image_{i}.pt'))

# Testing the preprocessing
if __name__ == "__main__":
    # Example usage
    data_dir = "data/raw"
    output_dir = "data/processed"
    
    audio_files, image_files = load_data(data_dir)
    audio_data, image_data = preprocess_data(audio_files, image_files)
    save_preprocessed_data(audio_data, image_data, output_dir)
    
    print(f"Preprocessed {len(audio_data)} audio files and {len(image_data)} image files.")

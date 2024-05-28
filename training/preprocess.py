import os
import cv2
import torch
import torchaudio
from torchvision import transforms
from torchaudio.transforms import Resample
from PIL import Image

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
    video_files = []
    audio_files = []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
            elif file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f'Found {len(video_files)} video files: {video_files}')
    print(f'Found {len(audio_files)} audio files: {audio_files}')
    
    return video_files, audio_files

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

def extract_frames(video_file):
    """
    Extract frames from video file.
    """
    cap = cv2.VideoCapture(video_file)
    frames = []
    success, frame = cap.read()
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)  # Convert to PIL image
        frame = image_transforms(frame)
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    
    return frames

def preprocess_data(video_files, audio_files):
    """
    Preprocess both video and audio files.
    """
    audio_data = [preprocess_audio(audio_file) for audio_file in audio_files]
    video_data = [extract_frames(video_file) for video_file in video_files]
    
    return audio_data, video_data

def save_preprocessed_data(audio_data, video_data, output_dir):
    """
    Save preprocessed data to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save audio data
    audio_output_dir = os.path.join(output_dir, 'audio')
    os.makedirs(audio_output_dir, exist_ok=True)
    for i, audio in enumerate(audio_data):
        torch.save(audio, os.path.join(audio_output_dir, f'audio_{i}.pt'))
    
    # Save video data
    video_output_dir = os.path.join(output_dir, 'video')
    os.makedirs(video_output_dir, exist_ok=True)
    for i, frames in enumerate(video_data):
        video_dir = os.path.join(video_output_dir, f'video_{i}')
        os.makedirs(video_dir, exist_ok=True)
        for j, frame in enumerate(frames):
            torch.save(frame, os.path.join(video_dir, f'frame_{j}.pt'))

# Testing the preprocessing
if __name__ == "__main__":
    # Example usage
    data_dir = "data/raw"
    output_dir = "data/processed"
    
    video_files, audio_files = load_data(data_dir)
    audio_data, video_data = preprocess_data(video_files, audio_files)
    save_preprocessed_data(audio_data, video_data, output_dir)
    
    print(f"Preprocessed {len(audio_data)} audio files and {len(video_data)} video files.")

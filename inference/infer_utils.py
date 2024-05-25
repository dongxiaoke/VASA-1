import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torchvision import transforms
from PIL import Image
import cv2

# Load a saved model
def load_model(model, file_path, device='cpu'):
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {file_path}")

# Extract audio features using Wav2Vec2
def extract_audio_features(audio_path, model_name="facebook/wav2vec2-base-960h"):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    
    waveform, sample_rate = torchaudio.load(audio_path)
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        features = model(**inputs).last_hidden_state
    return features

# Preprocess an input image
def preprocess_image(image_path, resolution):
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

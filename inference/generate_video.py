import os
import yaml
import torch
import cv2
import numpy as np
from torchvision.utils import save_image
from models.encoders.face_encoder import FaceEncoder
from models.decoders.face_decoder import FaceDecoder
from models.transformers.diffusion_transformer import DiffusionTransformer
from inference.infer_utils import load_model, extract_audio_features, preprocess_image
from control_signals import generate_gaze_direction, generate_head_distance, generate_emotion_offset

def generate_talking_face(config_path):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract configuration parameters
    face_encoder_path = config['model']['face_encoder_path']
    face_decoder_path = config['model']['face_decoder_path']
    diffusion_transformer_path = config['model']['diffusion_transformer_path']
    wav2vec2_model_name = config['model']['wav2vec2_model_name']
    
    input_image_path = config['data']['input_image_path']
    input_audio_path = config['data']['input_audio_path']
    output_video_path = config['data']['output_video_path']
    
    frame_rate = config['video']['frame_rate']
    resolution = config['video']['resolution']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    face_encoder = FaceEncoder().to(device)
    load_model(face_encoder, face_encoder_path, device)
    
    face_decoder = FaceDecoder().to(device)
    load_model(face_decoder, face_decoder_path, device)
    
    diffusion_transformer = DiffusionTransformer().to(device)
    load_model(diffusion_transformer, diffusion_transformer_path, device)
    
    # Preprocess inputs
    input_image = preprocess_image(input_image_path, resolution).to(device)
    audio_features = extract_audio_features(input_audio_path, wav2vec2_model_name).to(device)
    
    # Generate control signals
    batch_size = audio_features.size(0)
    seq_len = audio_features.size(1)
    gaze_direction = generate_gaze_direction(batch_size, seq_len).to(device)
    head_distance = generate_head_distance(batch_size, seq_len).to(device)
    emotion_offset = generate_emotion_offset(batch_size, seq_len).to(device)
    
    # Generate motion latent codes
    with torch.no_grad():
        identity_features, appearance_features = face_encoder(input_image)
        motion_latent_codes = diffusion_transformer(audio_features, gaze_direction, head_distance, emotion_offset)
        generated_frames = face_decoder(appearance_features, identity_features, motion_latent_codes)
    
    # Save video
    save_video(generated_frames, output_video_path, frame_rate, resolution)

def save_video(frames, output_path, frame_rate, resolution):
    frames = frames.cpu().numpy().transpose(0, 2, 3, 1)
    frames = (frames * 255).astype(np.uint8)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (resolution[1], resolution[0]))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Video saved to {output_path}")

# Example usage
if __name__ == "__main__":
    config_path = "inference/configs/config_infer.yaml"
    generate_talking_face(config_path)

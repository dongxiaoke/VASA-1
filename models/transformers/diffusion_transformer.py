import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionTransformer(nn.Module):
    def __init__(self, embedding_dim=512, nhead=8, num_encoder_layers=6, dropout=0.1):
        super(DiffusionTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward=2048, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Linear layers to map from input features to the embedding dimension
        self.audio_linear = nn.Linear(768, embedding_dim)  # Assuming Wav2Vec2 hidden size is 768
        self.gaze_linear = nn.Linear(2, embedding_dim)     # Gaze direction (theta, phi)
        self.distance_linear = nn.Linear(1, embedding_dim) # Head-to-camera distance
        self.emotion_linear = nn.Linear(1, embedding_dim)  # Emotion offset
        
        # Output layer to generate motion latent codes
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, audio_features, gaze_direction, head_distance, emotion_offset):
        # Map input features to the embedding dimension
        audio_emb = self.audio_linear(audio_features)
        gaze_emb = self.gaze_linear(gaze_direction)
        distance_emb = self.distance_linear(head_distance)
        emotion_emb = self.emotion_linear(emotion_offset)
        
        # Concatenate all embeddings
        x = audio_emb + gaze_emb + distance_emb + emotion_emb  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply the transformer encoder
        x = self.transformer_encoder(x)
        
        # Generate the motion latent codes
        motion_latent_codes = self.output_layer(x)
        
        return motion_latent_codes

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embedding_dim)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Testing the transformer
if __name__ == "__main__":
    # Create dummy input tensors
    batch_size = 1
    seq_len = 100  # Example sequence length
    embedding_dim = 512
    
    dummy_audio = torch.randn(batch_size, seq_len, 768)
    dummy_gaze = torch.randn(batch_size, seq_len, 2)
    dummy_distance = torch.randn(batch_size, seq_len, 1)
    dummy_emotion = torch.randn(batch_size, seq_len, 1)
    
    # Instantiate the model
    model = DiffusionTransformer(embedding_dim=embedding_dim)
    
    # Perform a forward pass
    motion_latent_codes = model(dummy_audio, dummy_gaze, dummy_distance, dummy_emotion)
    
    print(f"Motion Latent Codes: {motion_latent_codes.shape}")

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h", embedding_dim=768):
        super(AudioEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Load Wav2Vec2 pretrained model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        
        # Additional linear layer to match the embedding dimension if necessary
        if self.model.config.hidden_size != embedding_dim:
            self.fc = nn.Linear(self.model.config.hidden_size, embedding_dim)
        else:
            self.fc = None

    def forward(self, audio, sampling_rate=16000):
        # Process the audio input
        inputs = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values  # Extract the input values tensor
        
        # Get the hidden states from Wav2Vec2 model
        with torch.no_grad():
            outputs = self.model(input_values)
        hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        
        # Apply the additional linear layer if necessary
        if self.fc is not None:
            hidden_states = self.fc(hidden_states)  # Shape: (batch_size, seq_len, embedding_dim)
        
        return hidden_states

# Testing the encoder
if __name__ == "__main__":
    # Create a dummy input audio tensor (batch_size, seq_len)
    dummy_audio = torch.randn(1, 16000)  # Assuming 1-second audio with 16kHz sampling rate
    
    # Instantiate the model
    model = AudioEncoder()
    
    # Perform a forward pass
    audio_features = model(dummy_audio.squeeze())  # Ensure the input is 2D
    
    print(f"Audio Features: {audio_features.shape}")

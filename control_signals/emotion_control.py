import torch

def generate_emotion_offset(batch_size, seq_len, emotion_range=(-1, 1)):
    """
    Generate emotion offset control signals.
    
    Parameters:
        batch_size (int): Number of samples in a batch.
        seq_len (int): Sequence length for each sample.
        emotion_range (tuple): Range of values for emotion offset (default: (-1, 1)).
        
    Returns:
        torch.Tensor: Emotion offset control signals of shape (batch_size, seq_len, 1).
    """
    emotion_offset = torch.FloatTensor(batch_size, seq_len, 1).uniform_(*emotion_range)
    return emotion_offset

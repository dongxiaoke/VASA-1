import torch

def generate_gaze_direction(batch_size, seq_len, gaze_range=(-1, 1)):
    """
    Generate gaze direction control signals.
    
    Parameters:
        batch_size (int): Number of samples in a batch.
        seq_len (int): Sequence length for each sample.
        gaze_range (tuple): Range of values for gaze direction (default: (-1, 1)).
        
    Returns:
        torch.Tensor: Gaze direction control signals of shape (batch_size, seq_len, 2).
    """
    gaze_direction = torch.FloatTensor(batch_size, seq_len, 2).uniform_(*gaze_range)
    return gaze_direction

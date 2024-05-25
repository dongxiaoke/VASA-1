import torch

def generate_head_distance(batch_size, seq_len, distance_range=(0.8, 1.2)):
    """
    Generate head-to-camera distance control signals.
    
    Parameters:
        batch_size (int): Number of samples in a batch.
        seq_len (int): Sequence length for each sample.
        distance_range (tuple): Range of values for head-to-camera distance (default: (0.8, 1.2)).
        
    Returns:
        torch.Tensor: Head-to-camera distance control signals of shape (batch_size, seq_len, 1).
    """
    head_distance = torch.FloatTensor(batch_size, seq_len, 1).uniform_(*distance_range)
    return head_distance

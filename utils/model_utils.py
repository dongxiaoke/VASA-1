import torch

def save_model(model, file_path):
    """
    Save the PyTorch model to the specified file path.
    """
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def load_model(model, file_path, device='cpu'):
    """
    Load the PyTorch model from the specified file path.
    """
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {file_path}")

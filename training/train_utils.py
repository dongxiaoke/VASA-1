import os
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
    print(f"Model loaded from {file_path}")

def compute_loss(predictions, targets, loss_fn):
    """
    Compute the loss between predictions and targets using the specified loss function.
    """
    return loss_fn(predictions, targets)

def evaluate_model(model, dataloader, loss_fn, device='cpu'):
    """
    Evaluate the model performance on the validation set.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = compute_loss(outputs, targets, loss_fn)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def setup_optimizer(model, learning_rate):
    """
    Setup the optimizer for training.
    """
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def setup_scheduler(optimizer, step_size, gamma=0.1):
    """
    Setup the learning rate scheduler.
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

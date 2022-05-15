import torch


@torch.no_grad()
def comp_loss_accuracy(model, criterion, data_loader, device):
    """Compute loss and accuracy on dataset."""
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    running_counter = 0
    for x_data, y_data in data_loader:
        inputs, labels = x_data.to(device), y_data.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels).item()
        pred = (torch.argmax(outputs, dim=1) == labels).float().sum().item()
        running_loss += loss
        running_accuracy += pred
        running_counter += labels.size(0)
    return running_loss / running_counter, running_accuracy / running_counter

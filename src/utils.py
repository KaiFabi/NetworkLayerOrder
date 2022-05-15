import torch


@torch.no_grad()
def comp_test_stats(model, criterion, test_loader, device):
    """

    Args:
        model: PyTorch model.
        criterion: Loss.
        test_loader: Dataloader.
        device: Accelerator.

    Returns:
        Loss and accuracy.

    """
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    running_counter = 0
    for x_data, y_data in test_loader:
        inputs, labels = x_data.to(device), y_data.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels).item()
        pred = (torch.argmax(outputs, dim=1) == labels).float().sum().item()
        running_loss += loss
        running_accuracy += pred
        running_counter += labels.size(0)
    return running_loss / running_counter, running_accuracy / running_counter


def remove_layer_config(layer_configs: list, pattern: list) -> list:
    """Remove layer configuration according to provided pattern.

    Args:
        layer_configs:
        pattern:

    Returns:
        Cleaned list of layer configurations.

    """
    layer_configs_tmp = []

    for cfg in layer_configs:
        keep = True
        for layer1, layer2 in zip(cfg[:-1], cfg[1:]):
            if [layer1, layer2] == pattern:
                keep = False
                break

        if keep:
            layer_configs_tmp.append(cfg)

    return layer_configs_tmp

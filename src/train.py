import os
import datetime
import time

import torch
import torch.optim as optim

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .stats import comp_loss_accuracy


def train(model: torch.nn.Module, dataloader: tuple, config: dict) -> None:
    """

    Args:
        model: PyTorch model.
        dataloader: Tuple holding training and test dataloader.
        config: Dictionary holding configuration for training.

    """
    uid = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')
    name = config["name"]
    print(f"{uid = }")
    print(f"{name = }")

    log_dir = os.path.join(config["runs_dir"], f"{uid}_{name}")
    writer = SummaryWriter(log_dir=log_dir)
    run_training(model=model, dataloader=dataloader, writer=writer, config=config)
    writer.close()


def run_training(model, dataloader, writer, config: dict) -> None:
    """Main training logic.

    Trains passed model with data coming from dataloader.

    Args:
        model: PyTorch model.
        dataloader: Training and test data loader.
        writer: Tensorboard writer instance.
        config: Dictionary holding configuration for training.

    """
    device = config["device"]
    save_train_stats_every_n_epochs = config["save_train_stats_every_n_epochs"]
    save_test_stats_every_n_epochs = config["save_test_stats_every_n_epochs"]

    n_epochs = config["n_epochs"]
    weight_decay = config["weight_decay"]
    ini_lr = config["ini_lr"]
    max_lr = config["max_lr"]
    min_lr = config["min_lr"]

    trainloader, testloader = dataloader

    div_factor = max_lr / ini_lr
    final_div_factor = ini_lr / min_lr

    optimizer = optim.Adam(model.parameters(), lr=ini_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=n_epochs,
        steps_per_epoch=len(trainloader),
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):

        writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], global_step=epoch)
        running_loss = 0.0
        running_accuracy = 0.0
        running_counter = 0

        model.train()
        t0 = time.time()

        for x_data, y_data in trainloader:

            # Get the inputs; data is a list of [inputs, lables]
            inputs, labels = x_data.to(device), y_data.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + gradient descent
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keeping track of some statistics
            running_loss += loss.item()
            running_accuracy += (torch.argmax(outputs, dim=1) == labels).float().sum()
            running_counter += labels.size(0)

        writer.add_scalar("time_per_epoch", time.time() - t0, epoch)

        # scheduler.step()
        running_loss = running_loss / running_counter
        running_accuracy = running_accuracy / running_counter

        if (epoch % save_train_stats_every_n_epochs == 0) or (epoch + 1 == n_epochs):
            writer.add_scalar("train_loss", running_loss, epoch)
            writer.add_scalar("train_accuracy", running_accuracy, epoch)

        if (epoch % save_test_stats_every_n_epochs == 0) or (epoch + 1 == n_epochs):
            test_loss, test_accuracy = comp_loss_accuracy(model=model,
                                                          criterion=criterion,
                                                          data_loader=testloader,
                                                          device=device)
            writer.add_scalar("test_loss", test_loss, epoch)
            writer.add_scalar("test_accuracy", test_accuracy, epoch)

        print(f"{epoch:04d} {running_loss:.5f} {running_accuracy:.4f}")

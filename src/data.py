import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloader(config: dict) -> tuple[DataLoader, DataLoader]:

    data_dir = config["data_dir"]
    dataset = config["dataset"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    if dataset == "cifar10":

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=30),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform_train
        )

        testset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform_test
        )

        config["input_shape"] = (3, 32, 32)
        config["n_classes"] = 10

    elif dataset == "fmnist":

        # Fashion-MNIST
        avg = (0.2859,)
        std = (0.3530,)

        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=30),
                transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(avg, std)
            ]
        )

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(avg, std)]
        )

        trainset = torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train
        )

        testset = torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform_test
        )

        config["input_shape"] = (1, 28, 28)
        config["n_classes"] = 10

    elif dataset == "mnist":

        # Fashion-MNIST
        avg = (0.1307,)
        std = (0.3081,)

        transform_train = transforms.Compose(
            [
                transforms.RandomRotation(degrees=20),
                transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(avg, std)
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(avg, std)
            ]
        )

        trainset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train
        )

        testset = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform_test
        )

        config["input_shape"] = (1, 28, 28)
        config["n_classes"] = 10

    else:
        raise NotImplementedError(f"Dataloader for {dataset} not implemented.")

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )

    testloader = DataLoader(
        testset,
        batch_size=2 * batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    return trainloader, testloader

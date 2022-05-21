from math import prod

import torch
import torch.nn as nn


class DenseLayerOrder(nn.Module):
    """Network to test best order of layers.

    Fully connected neural network with N blocks consisting of
    the following operations in defined order:

        Linear
        BatchNorm1d
        ReLU

    """

    def __init__(self, layer_config: list, config: dict):
        super().__init__()

        self.layer_config = layer_config
        self.n_dims_in = prod(config["input_shape"])
        self.n_dims_out = config["n_classes"]
        self.n_dims_hidden = 1024
        self.n_blocks = 8

        self.classifier = self.make_classifier(layer_config=layer_config, n_blocks=self.n_blocks)

    def make_classifier(self, layer_config: list, n_blocks: int):
        layers = []

        # Dense input
        layers += [
            torch.nn.Linear(in_features=self.n_dims_in, out_features=self.n_dims_hidden),
            torch.nn.BatchNorm1d(num_features=self.n_dims_hidden)
        ]

        # Dense hidden
        for i in range(n_blocks):
            for layer in layer_config:
                if layer is torch.nn.BatchNorm1d:
                    layers.append(nn.BatchNorm1d(num_features=self.n_dims_hidden))
                elif layer is nn.Dropout:
                    layers.append(nn.Dropout(p=self.dropout_rate))
                elif layer is nn.Linear:
                    layers.append(nn.Linear(in_features=self.n_dims_hidden,
                                            out_features=self.n_dims_hidden))
                elif layer is nn.ReLU:
                    layers.append(nn.ReLU())

        # Dense output
        layers.append(torch.nn.Linear(self.n_dims_hidden, self.n_dims_out))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.n_dims_in)
        x = self.classifier(x)
        return x


class ConvLayerOrder(nn.Module):
    """Fully convolutional neural network to test best layer configuration.

    Fully connected neural network with N blocks consisting of
    the following operations in defined order:

        Conv2d
        BatchNorm2d
        ReLU

    """
    def __init__(self, layer_config: list, config: dict):
        super().__init__()

        self.layer_config = layer_config

        self.n_dims_in = config["input_shape"]
        self.n_channels_in = self.n_dims_in[0]
        self.n_channels_hidden = 32
        self.n_channels_out = 16
        self.n_dims_out = 10
        self.n_blocks = 8

        self.features = self._feature_extractor(layer_config=layer_config, n_blocks=self.n_blocks)
        self.classifier = nn.Linear(self.n_channels_out*(self.n_dims_in[-1]//2)**2, self.n_dims_out)

        self._weights_init()

    def _feature_extractor(self, layer_config: list, n_blocks: int):
        layers = list()

        # Conv network input
        layers += [
            nn.Conv2d(in_channels=self.n_channels_in,
                      out_channels=self.n_channels_hidden,
                      kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(num_features=self.n_channels_hidden)
        ]

        # Conv network hidden
        for i in range(n_blocks):
            for layer in layer_config:
                if layer is torch.nn.BatchNorm2d:
                    layers.append(nn.BatchNorm2d(num_features=self.n_channels_hidden))
                elif layer is nn.Dropout:
                    layers.append(nn.Dropout(p=self.dropout_rate))
                elif layer is nn.Conv2d:
                    layers.append(nn.Conv2d(in_channels=self.n_channels_hidden,
                                            out_channels=self.n_channels_hidden,
                                            kernel_size=(3, 3), padding="same"))
                elif layer is nn.ReLU:
                    layers.append(nn.ReLU())

        # Conv network output
        layers += [
            nn.Conv2d(in_channels=self.n_channels_hidden,
                      out_channels=self.n_channels_out,
                      kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(num_features=self.n_channels_out)
        ]

        return nn.Sequential(*layers)

    def _weights_init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data, nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.classifier(x)
        return x

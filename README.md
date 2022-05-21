# NeuralNetworkLayerOrder

Experiments to evaluate the effect of different orders of layers in neural networks.


## Multilayer Perceptron 


### Architecture 

Isotropic fully connected neural network consisting of 8 blocks with 1024 hidden neurons.

Each block was composed of the following layers:

- BatchNorm1d
- Linear 
- ReLU


### Experimental Setup

- Adam optimizer
- Epochs: 100
- One cycle learning rate: 
  - initial_lr: 1e-6
  - max_lr: 1e-3
  - min_lr: 1e-7
- Batch size: 1024


### Results


#### Cifar10

Accuracy for different layer configurations averaged over 10 runs.

| Layer configuration           | Test accuracy    |
|-------------------------------|------------------|
| Linear -> BatchNorm1d -> ReLU | 0.5818 +- 0.0023 |
| BatchNorm1d -> Linear -> ReLU | 0.5790 +- 0.0020 |
| BatchNorm1d -> ReLU -> Linear | 0.5787 +- 0.0025 |
| Linear -> ReLU -> BatchNorm1d | 0.5714 +- 0.0028 |
| ReLU -> BatchNorm1d -> Linear | 0.5687 +- 0.0032 |
| ReLU -> Linear -> BatchNorm1d | 0.5683 +- 0.0031 |

| Layer configuration           | Test accuracy    |
|-------------------------------|------------------|
| ReLU -> BatchNorm1d -> Linear | 0.9796 +- 0.0011 |
| Linear -> ReLU -> BatchNorm1d | 0.9780 +- 0.0007 |
| BatchNorm1d -> Linear -> ReLU | 0.9769 +- 0.0010 |
| ReLU -> Linear -> BatchNorm1d | 0.9699 +- 0.0009 |
| BatchNorm1d -> ReLU -> Linear | 0.9677 +- 0.0010 |
| Linear -> BatchNorm1d -> ReLU | 0.9670 +- 0.0011 |

![](docs/assets/results/cifar10/mlp/test_accuracy.png)
![](docs/assets/results/cifar10/mlp/train_accuracy.png)


## Convolutional Neural Network 


### Architecture 

Isotropic convolutional neural network consisting of 4 blocks with shape `channels x height x width`. 
Number of channels were set to 32. Height and width were equal to the input of the selected dataset.

Each block was composed of the following layers:

- BatchNorm2d
- Conv2d 
- ReLU


### Experimental Setup

- Adam optimizer
- Epochs: 100
- One cycle learning rate: 
  - initial_lr: 1e-5
  - max_lr: 4e-3
  - min_lr: 1e-7
- Batch size: 1024


### Results


#### Cifar10

Accuracy for different layer configurations averaged over 12 runs.

| Layer configuration           | Test accuracy  |
|-------------------------------|----------------|
| BatchNorm2d -> Conv2d -> ReLU | 0.793 +- 0.004 |
| Conv2d -> ReLU -> BatchNorm2d | 0.793 +- 0.003 |
| Conv2d -> BatchNorm2d -> ReLU | 0.790 +- 0.004 |
| ReLU -> BatchNorm2d -> Conv2d | 0.785 +- 0.003 |
| ReLU -> Conv2d -> BatchNorm2d | 0.780 +- 0.004 |
| BatchNorm2d -> ReLU -> Conv2d | 0.779 +- 0.002 |

| Layer configuration           | Train accuracy |
|-------------------------------|----------------|
| Conv2d -> ReLU -> BatchNorm2d | 0.767 +- 0.002 |
| BatchNorm2d -> Conv2d -> ReLU | 0.766 +- 0.002 |
| Conv2d -> BatchNorm2d -> ReLU | 0.762 +- 0.004 |
| ReLU -> BatchNorm2d -> Conv2d | 0.757 +- 0.003 |
| ReLU -> Conv2d -> BatchNorm2d | 0.748 +- 0.003 |
| BatchNorm2d -> ReLU -> Conv2d | 0.747 +- 0.002 |

![](docs/assets/results/cifar10/cnn/test_accuracy.png)
![](docs/assets/results/cifar10/cnn/train_accuracy.png)

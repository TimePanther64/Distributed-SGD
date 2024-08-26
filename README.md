# Distributed-SGD
Simulates distributed learning with ResNet18 on CIFAR-10 using TensorFlow. Features model averaging across workers, Dirichlet-based data splitting, and training loss visualization. It includes ResNet and LeNet-5 models and supporting datasets like MNIST and CIFAR-100 for flexible deep-learning experimentation.

# Detailed Description
This project implements a distributed learning approach using TensorFlow, where the ResNet18 model is trained across multiple simulated workers using the CIFAR-10 dataset. The training process explores the impact of varying worker numbers and data distribution ratios on model performance and training efficiency.

#### Key Features:
- **Distributed Learning Simulation**: The project simulates a distributed learning environment with varying numbers of workers and data portions using the Dirichlet distribution.
- **Model Averaging**: After each global epoch, model weights from individual workers are averaged to create a new global model.
- **Training Visualization**: Training loss across epochs is visualized to analyze the impact of different configurations on model convergence.
- **Support for Multiple Datasets**: The framework supports several datasets, including MNIST, Fashion MNIST, CIFAR-10, and CIFAR-100.

#### `models.py` - Model Architecture and Utility Functions
- **ResNet Implementations**: Defines ResNet architectures (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152) with customizable block structures for flexible deep learning model experimentation.
- **LeNet-5 Model**: Includes a basic implementation of the LeNet-5 architecture for simpler tasks or comparisons.
- **Model Averaging**: Provides a function to average the weights of multiple models, which is crucial for aggregating results from different workers in the distributed learning setup.

#### `utils.py` - Data Handling and Preprocessing
- **Dataset Loading**: Functions to load and preprocess popular datasets like MNIST, Fashion MNIST, CIFAR-10, and CIFAR-100, with support for normalization and one-hot encoding.
- **Dirichlet Data Splitting**: Implements a Dirichlet distribution-based method to split datasets among workers, simulating non-i.i.d. data distribution scenarios commonly encountered in federated and distributed learning contexts.

## Devito4PyTorch

Devito4PyTorch integrates [Devito](https://www.devitoproject.org/) into Pytorch, allowing for exploiting highly optimized finite difference kernels with neural networks. Devito4PyTorch is an extension of Devito, a symbolic finite-difference domain specific language that provides a high-level interface to the definition of partial differential equations (PDE), such as wave equation. During backpropagation, Devito4PyTorch calls Devito's adjoint PDE solvers, thus making it possible to backpropagate efficiently through the composition of PDE solvers and neural networks.

## Prerequisites

This code has been tested using [PyTorch-1.5.0](https://github.com/pytorch/pytorch/releases/tag/v1.5.0).

Follow the steps below to install the necessary libraries:

```bash
git clone https://github.com/devitocodes/devito.git
cd devito
conda env create -f environment-dev.yml
source activate devito
pip install -e .
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch #If your system has GPU
```

## Installation

```bash
pip install devito4pytorch
```


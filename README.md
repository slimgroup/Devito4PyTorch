[![CI Status](https://github.com/slimgroup/Devito4PyTorch/workflows/CI-tests/badge.svg)](https://github.com/slimgroup/Devito4PyTorch/actions?query=workflow%3ACI-tests)
[![Code Coverage](https://codecov.io/gh/slimgroup/Devito4PyTorch/branch/master/graph/badge.svg)](https://codecov.io/gh/slimgroup/Devito4PyTorch)

## Devito4PyTorch

Devito4PyTorch integrates [Devito](https://www.devitoproject.org/) into PyTorch via defining highly optimized finite difference kernels as PyTorch "layers". Devito4PyTorch is an extension of Devito, a symbolic finite-difference domain specific language that provides a high-level interface to the definition of partial differential equations (PDE), such as wave equation. During backpropagation, Devito4PyTorch calls Devito's adjoint PDE solvers, thus making it possible to backpropagate efficiently through the composition of PDE solvers and neural networks.

## Installation

You can install the package with `pip` via

```bash
pip install git+https://github.com/slimgroup/Devito4PyTorch
```

Or if you want a developper version, you can clone and install the package as

```bash
git clone https://github.com/slimgroup/Devito4PyTorch
cd Devito4PyTorch
pip install -e .
```

# GPU requirement

If you wish to run experiments using a GPU, you will need to install `cudatoolkit`. To do so, you can use conda as:

```bash
conda install cudatoolkit=10.1 -c pytorch
```

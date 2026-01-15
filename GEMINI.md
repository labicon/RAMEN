# RAMEN: Real-time Asynchronous Multi-agent Neural Implicit Mapping

## Project Overview

This project implements and evaluates RAMEN (Real-time Asynchronous Multi-agent Neural Implicit Mapping), a novel approach for real-time, decentralized, and collaborative 3D scene reconstruction using multiple agents. The system is built on a neural implicit representation of the scene geometry and appearance, allowing for high-quality, memory-efficient mapping.

The core of the project is a distributed optimization framework that enables multiple agents to collaboratively build a consistent and accurate 3D map of their environment. Each agent maintains its own local map and communicates with its neighbors to exchange information and update its map in a decentralized manner. The project supports several distributed optimization algorithms, including AUQ-CADMM, CADMM (DiNNO), MACIM, DSGD, and DSGT.

The codebase is written in Python and leverages several popular libraries for deep learning, 3D geometry processing, and scientific computing, including:

*   **PyTorch:** For building and training the neural implicit models.
*   **tiny-cuda-nn:** For high-performance neural network training on NVIDIA GPUs.
*   **PyTorch3D:** For 3D data processing and transformations.
*   **Open3D:** For 3D data visualization and processing.
*   **NumPy:** For numerical operations.
*   **NetworkX:** For representing and manipulating the communication graph between agents.

The project is structured to support experiments on various datasets, such as Replica and ScanNet, and includes scripts for downloading and processing these datasets. The configuration of the experiments is managed through YAML files, allowing for easy customization of the system parameters, such as the number of agents, the communication topology, and the choice of the distributed optimization algorithm.

## Building and Running

### 1. Installation

First, clone the repository and create the conda environment:

```shell
git clone https://github.com/labicon/RAMEN.git
cd RAMEN
conda create -n RAMEN python=3.10
conda activate RAMEN
```

Install PyTorch and other Python packages:

```shell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install PyMCubes==0.1.6 open3d==0.19.0 trimesh==4.5.3 opencv-python==4.11.0.86 matplotlib==3.10.0 pyyaml==6.0.2
```

Build and install `tiny-cuda-nn` and `pytorch3d` from source:

```shell
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -U iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### 2. Dataset

Download the Replica dataset:

```shell
bash scripts/download_replica.sh
```

### 3. Running the System

To run the system, you need to provide a configuration file. The configuration files are located in the `configs` directory. For example, to run the system on the `office1` scene from the Replica dataset, use the following command:

```shell
python main.py --config configs/Replica/office1.yaml
```

### 4. Visualization and Evaluation

To visualize the reconstructed mesh, you can use the `visualizer.py` script:

```shell
python visualizer.py --config ./configs/Replica/office1.yaml 
```

For quantitative evaluation, you can use the `analysis.ipynb` notebook.

## Development Conventions

*   **Configuration:** The project uses YAML files for configuration. The configuration files are organized in the `configs` directory and can inherit from each other. The `config.py` module provides helper functions for loading and merging configurations.
*   **Modularity:** The code is organized into several modules, each with a specific responsibility. For example, the `model` module contains the neural network models, the `datasets` module contains the dataset loading and processing code, and the `optimization` module contains the distributed optimization algorithms.
*   **Class-based Structure:** The core logic for a single agent is encapsulated in the `Mapping` class in `main.py`. This class manages the agent's state, including its local map, its pose, and its communication with other agents.
*   **Command-line Interface:** The main entry point of the application is `main.py`, which takes a configuration file as a command-line argument.
*   **Jupyter Notebooks for Analysis:** The `analysis.ipynb` notebook is used for quantitative evaluation of the results. This is a good practice for separating the core application logic from the analysis and visualization code.
*   **Shell Scripts for Automation:** The `scripts` directory contains shell scripts for automating tasks such as downloading datasets.
*   **Distributed Algorithms:** The project supports several distributed optimization algorithms, which can be selected through the configuration file. The implementation of these algorithms is located in the `global_BA` method of the `Mapping` class in `main.py`.

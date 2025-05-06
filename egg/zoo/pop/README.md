# Population-Based Communication Experiments

This repository contains tools and experiments for studying communication in populations of agents. It builds upon the `encom_as_ssl` game and extends it to support populations of agents with diverse architectures.

## Summary of the Paper

This repository is based on the paper [**"Referential communication in heterogeneous communities of pre-trained visual deep networks"**](https://arxiv.org/abs/2302.08913). The paper investigates how populations of agents with heterogeneous architectures can develop emergent communication protocols. Key contributions of the paper include:

- **Robustness to Architectural Diversity**: Demonstrates that emergent communication is robust to differences in agent architectures, such as ResNet, ViT, and Swin.
- **Communication Space Analysis**: Introduces methods to analyze the communication space, including the SAEs, and perturbation analysis
- **Generalisation to new referents**
- **Generalisation to new agents**

The experiments in this repository replicate and extend the findings of the paper, providing tools to analyze communication in diverse populations.

## Features
- **Population-based communication experiments**: Study emergent communication in populations of agents with varying architectures.
- **Support for multiple architectures**: Includes models like ResNet, ViT, Swin, and more.
- **Communication space analysis**: Tools to evaluate message alignment, proximity, and efficiency in the latent space.
- **Tools for measuring individual agent performance**: Evaluate accuracy, learning speed, and communication consistency.
- **Compatibility with distributed training setups**: Scalable experiments for large populations.

## Recent Changes
- Added auxiliary loss options for better control over agent training.
- Introduced tools for analyzing inter-agent communication and the communication space.
- Improved dataset handling and augmentation options.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/facebookresearch/EGG.git
   cd EGG/egg/zoo/pop
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run experiments:
   - Use train.py or seq_train.py for training a population of agents.
     ```bash
     python train.py --params_path <path_to_params.json>
     ```
   - To extract communication logs, this allows the construction of visualisations:
     ```bash
     python extract_com.py --base_checkpoint_path <path_to_checkpoint>
     ```
   - Use scripts in sanity_checks/ for evaluating communication efficiency, alignment, and proximity.
   - Use sweeper.py to automate hyperparameter sweeps.

## Code Overview
### Core Modules
* games.py:

Defines the main game logic, including the population game and agent sampling strategies.
Implements loss functions and communication channel wrappers.
* archs.py:

Provides implementations for various agent architectures (e.g., ResNet, ViT).
Includes wrappers for continuous communication and KMeans-based communication.
* data.py:

Handles dataset loading, augmentation, and batching.
Implements custom datasets like GaussianNoiseDataset and ImagenetValDataset.
* utils.py:

Contains utility functions for parsing options, loading checkpoints, and managing metadata.
* game_callbacks.py:

Implements callbacks for logging, tracking best statistics, and integrating with WandB.

### Sanity Checks
* sae_proximity_counter.py:

Analyzes the proximity of representations in the latent space using a Simplicial Autoencoder (SAE).
* label_communication.py:

Evaluates label agreement and communication efficiency across different models and datasets.
* cluster_translator.py:

Aligns clusters across models using KMeans and evaluates alignment accuracy.
* transfer_classif.py:

Tests transfer classification by training linear classifiers on representations from different agents.
* test_classif.py:

Trains and evaluates classifiers on representations from a single sender or multiple senders.

### Training and Sweeping
* train.py:

Implements the main training loop for population-based communication experiments.
Supports adding new agents to an existing population.
* seq_train.py:

Implements sequential training, where agents are added one by one to the population.
* sweeper.py:

Automates hyperparameter sweeps by generating and submitting SLURM jobs.
* sequential_queue_tool.py:

Generates sequential job scripts for running experiments with different parameter combinations.

## Analysis and Evaluation
* extract_com.py:

Extracts communication logs and interactions for analysis.
* build_cosine_data.py:

Computes cosine similarity matrices for datasets to analyze inter-dataset relationships.
* sanity_checks/closest_inter_dataset_clusters.py:

Identifies the closest clusters across datasets using representations from trained agents.

## Contributing
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.

For questions or issues, please contact the maintainers.

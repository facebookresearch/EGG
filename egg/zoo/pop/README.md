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
   - To train a population of agents:
     ```bash
     python train.py --params_path <path_to_params.json>
     ```
   - To extract communication logs:
     ```bash
     python extract_com.py --base_checkpoint_path <path_to_checkpoint>
     ```

4. Analyze results:
   - Use the provided scripts in `sanity_checks/` to evaluate communication efficiency, alignment, and proximity in the latent space.

## Contributing
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.

For questions or issues, please contact the maintainers.

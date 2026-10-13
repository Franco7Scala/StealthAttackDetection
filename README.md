# A Semi-Supervised Few-Shot Framework for Detecting Stealth Malicious Network Traffic

[![Paper](https://img.shields.io/badge/Paper-Machine_Learning-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and resources for the **DDM (Data Driven Monitor)** framework presented in the research paper "A Semi-Supervised Few-Shot Framework for Detecting Stealth Malicious Network Traffic". Our proposal addresses the challenge of detecting advanced, stealthy network threats (such as covert data exfiltration, C&C beaconing, and slow DoS attacks) in scenarios with extreme label scarcity by combining semi-supervised representation learning on benign traffic with a few-shot adaptation module.

## Key Features

- **Semi-Supervised Pre-training:** Exploits an Adversarial Representation Network (ARN) trained exclusively on normal, benign network traffic to automatically extract robust, attack-agnostic, and anomaly-sensitive features;
- **Few-Shot Adaptation:** Adapts the learned representations to specific stealth attack classes using a lightweight classification head and an extremely limited budget of labeled malicious instances (e.g., b=5);
- **Dual-Layer Architecture:** Fuses VAE-based latent embeddings, adversarial discriminator features, and original input features via a skip connection to preserve both high-level semantics and low-level traffic indicators;
- **Robustness to Extreme Imbalance:** Employs batch-level resampling and selective Gaussian noise data augmentation to mitigate the scarcity of malicious network samples without overfitting.

## Repository Structure

- **`src/`**: Core source code directory.
  * `main.py`: Main entry point to train and evaluate the complete DDM framework via few-shot adaptation;
  * `main_arn.py`: Script dedicated to the pre-training phase of the semi-supervised Adversarial Representation Network (ARN);
  * `model/predictive_model_noise.py`: Contains the implementation of the overall dual-layer neural architecture combining the pre-trained backbone and the classification head;
  * `arn/`: Implements the ARN core components, including the VAE and discriminator (`model.py`), adversarial training routines (`trainer.py`), custom loss functions (`loss.py`), and configurations;
  * `dataset/`: Utilities and specific loaders for the three evaluated threat scenarios (`_cobalt_loader.py` for C&C beaconing, `_covert_loader.py` for covert channels, `_slowdos_loader.py` for slow DoS);
  * `support/`: Helper scripts containing argument parsers (`arguments.py`), custom loss implementations (`cost_sensitive.py`, `focal_loss.py`), and generic utilities (`utils.py`).

- **`requirements.txt`**: List of project dependencies.

## Requirements

To run the code, ensure you have the libraries specified in `requirements.txt` installed. The main dependencies include:

- Python (>=3.8)
- PyTorch
- NumPy
- Scikit-learn
- Pandas

## Citation

```bibtex
Coming soon...

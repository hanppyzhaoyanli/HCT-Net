# HCT-Net: A hybrid CNN-Transformer network for multi-class cervical cell classification

HCT-Net is a hybird deep learning model for automated classification of cervical cell images, combining the strengths of CNNs and Transformers to achieve superior performance in cervical cancer screening.

## Key Features

- **Hybrid Architecture**: Combines CNN's local feature extraction with Transformer's global context modeling
- **Checkpoint System**: Automatic saving and resuming of training progress
- **Early Stopping**: Prevents overfitting and optimizes training time
- **Comprehensive Metrics**: Tracks accuracy, precision, recall, F1-score, and more

## Installation

```bash
git clone https://github.com/hannpyzhaoyanli/HCT-Net.git
cd HCT-Net
pip install -r requirements.txt

##  Directory Structure

HCT-Net/
├── checkpoint/          # Training checkpoints and best models
├── data/                 # Dataset directory
├── docs/                 # Documentation and visualizations
├── src/
│   ├── hctnet_model.py   # Model architecture
│   ├── dataset.py        # Data loading and augmentation
│   ├── loss.py           # Joint loss function
│   ├── train.py          # Training script
│   └── test.py           # Evaluation script
├── requirements.txt      # Python dependencies
└── README.md             # This file

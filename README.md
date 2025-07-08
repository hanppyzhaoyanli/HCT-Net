# HCT-Net: A hybrid CNN-Transformer network for multi-class cervical cell classification

HCT-Net is a hybird deep learning model for automated classification of cervical cell images, combining the strengths of CNNs and Transformers to achieve superior performance in cervical cancer screening.

## Key Features

- **Hybrid Architecture**: Combines CNN's local feature extraction with Transformer's global context modeling
- **Checkpoint System**: Automatic saving and resuming of training progress
- **Early Stopping**: Prevents overfitting and optimizes training time
- **Comprehensive Metrics**: Tracks accuracy, precision, recall, F1-score, and more

## Requirements
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
pandas==2.0.2
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
Pillow==9.5.0
tqdm==4.65.0
thop==0.1.1.post2209072238
scikit-image==0.20.0
opencv-python==4.7.0.72
tabulate==0.9.0

## Model Weights Download
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILE_ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILE_ID" -O model_weights.pth && rm -rf /tmp/cookies.txt



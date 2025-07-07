HCT-Net: Hybrid CNN-Transformer for Cervical Cell Classification
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/badge/Python-3.8%2B-blue
https://img.shields.io/badge/PyTorch-2.0%2B-orange

HCT-Net is a state-of-the-art deep learning model for automated classification of cervical cell images, combining the strengths of CNNs and Transformers to achieve superior performance in cervical cancer screening.

Table of Contents
Key Features
Installation
Usage
Results
Citation
License
Key Features
​​Hybrid Architecture​​: Combines CNN's local feature extraction with Transformer's global context modeling
​​High Performance​​: Achieves >99% accuracy on benchmark datasets
​​Checkpoint System​​: Automatic saving and resuming of training progress
​​Early Stopping​​: Prevents overfitting and optimizes training time
​​Comprehensive Metrics​​: Tracks accuracy, precision, recall, F1-score, and more
Installation
Clone the repository:
git clone https://github.com/yourusername/HCT-Net.git
cd HCT-Net
Install dependencies:
pip install -r requirements.txt
Usage
Training
python train.py --dataset SIPaKMeD --epochs 65
Evaluation
python test.py --weights checkpoint/best_model.pth --dataset SIPaKMeD
Results
Dataset	Accuracy	Precision	Recall	F1-Score
SIPaKMeD	99.26%	99.50%	99.26%	99.34%
Herlev	98.93%	99.17%	99.24%	99.20%
Mendeley LBC	99.48%	99.80%	99.24%	99.51%
Citation
If you use HCT-Net in your research, please cite our paper:

@article{hctnet2025,
  title={HCT-Net: A Hybrid CNN-Transformer Network for Cervical Cell Classification},
  author={Author, A. and Coauthor, B.},
  journal={Medical Image Analysis},
  volume={XX},
  pages={XXX--XXX},
  year={2025},
  publisher={Elsevier}
}
License
This project is licensed under the MIT License - see the LICENSE file for details.


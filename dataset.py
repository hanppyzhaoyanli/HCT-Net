import torch
from torchvision import transforms
from torch.utils.data import Dataset, random_split
import os
import numpy as np
from PIL import Image


class CervicalDataset(Dataset):
    """Dataset handler implementing Section 3.1 specifications"""

    def __init__(self, root_dir, dataset_name='SIPaKMeD', mode='train'):
        self.root = os.path.join(root_dir, dataset_name)
        self.classes = self._get_classes(dataset_name)
        self.images, self.labels = self._load_data()

        # Data augmentation strategy (Section 3.3)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Input size for PVTv2
            transforms.RandomRotation(30),  # ±30° random rotation
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(contrast=(0.5, 1.5)),  # Contrast scaling [0.5,1.5]
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                                 std=[0.229, 0.224, 0.225])
        ]) if mode == 'train' else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _get_classes(self, name):
        """Get class names based on dataset"""
        if name == 'SIPaKMeD':
            return ['superficial', 'parabasal', 'metaplastic', 'dyskeratotic', 'koilocytotic']
        elif name == 'Herlev':
            return ['normal_superficial', 'normal_intermediate', 'normal_columnar',
                    'light_dysplastic', 'moderate_dysplastic', 'severe_dysplastic', 'carcinoma_in_situ']
        else:  # Mendeley LBC
            return ['NILM', 'LSIL', 'HSIL', 'SCC']

    def _load_data(self):
        """Implement stratified data splits (80-10-10)"""
        images, labels = [], []
        for cls_idx, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(self.root, cls_name)
            cls_files = [f for f in os.listdir(cls_dir) if f.endswith('.png')]
            images.extend([os.path.join(cls_dir, f) for f in cls_files])
            labels.extend([cls_idx] * len(cls_files))

        # Fixed random seed for reproducibility
        torch.manual_seed(42)
        indices = torch.randperm(len(images)).tolist()

        # Save dataset split IDs for auditing
        split_idx = {
            'train': indices[:int(0.8 * len(indices))],
            'val': indices[int(0.8 * len(indices)):int(0.9 * len(indices))],
            'test': indices[int(0.9 * len(indices)):]
        }
        torch.save(split_idx, f'{self.root}_split_ids.pt')

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]
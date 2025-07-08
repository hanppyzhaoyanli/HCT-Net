from hctnet_model import HCTNet
from dataset import CervicalDataset
from loss import JointLoss
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import time
import os
from thop import profile
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import gdown  # Added for downloading weights

# Function to download pretrained weights
def download_pretrained_weights():
    # Google Drive file ID
    file_id = '1DLmlOO7LbL1-Zf4f5g-qvkyxaFZwlZeG'
    output_path = 'checkpoint/pretrained_weights.pth'
    
    # Return if weights already exist
    if os.path.exists(output_path):
        print("Pretrained weights already exist")
        return output_path
    
    print("Downloading pretrained weights...")
    
    try:
        # Construct download URL
        url = f'https://drive.google.com/uc?id={file_id}'
        
        # Download file
        gdown.download(url, output_path, quiet=False)
        
        print("Pretrained weights downloaded successfully!")
        return output_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None

# Create checkpoint directory
os.makedirs('checkpoint', exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True

# Initialize dataset (SIPaKMeD)
train_set = CervicalDataset(root_dir='./data', dataset_name='SIPaKMeD', mode='train')
val_set = CervicalDataset(root_dir='./data', dataset_name='SIPaKMeD', mode='val')

# Calculate class weights for imbalance handling
class_counts = torch.bincount(torch.tensor([label for _, label in train_set]))
class_weights = (1. / class_counts.float()).to('cuda')

# Initialize model
model = HCTNet(num_classes=len(train_set.classes)).to('cuda')

# Download and load pretrained weights
weights_path = download_pretrained_weights()
if weights_path:
    try:
        model.load_state_dict(torch.load(weights_path))
        print("Successfully loaded pretrained weights")
    except Exception as e:
        print(f"Failed to load pretrained weights: {e}")
        print("Using random initialization")
else:
    print("No pretrained weights found, using random initialization")

# Initialize optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-6, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)

# Joint loss function with γ=2
criterion = JointLoss(class_weights=class_weights, gamma=2.0)

# Calculate model parameters and FLOPs
input_sample = torch.randn(1, 3, 224, 224).to('cuda')
flops, params = profile(model, inputs=(input_sample,))
print(f"Model Parameters: {params / 1e6:.2f}M | FLOPs: {flops / 1e9:.2f}G")

# Training metrics storage
train_history = {'loss': [], 'acc': [], 'f1': []}
val_history = {'loss': [], 'acc': [], 'f1': []}
best_val_f1 = 0.0
epochs_no_improve = 0
patience = 10  # Early stopping patience
start_epoch = 0

# Check for existing checkpoints to resume training
checkpoint_files = [f for f in os.listdir('checkpoint') if f.startswith('checkpoint_epoch_')]
if checkpoint_files:
    # Find latest checkpoint
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
    latest_epoch = max(epochs)
    checkpoint_path = f'checkpoint/checkpoint_epoch_{latest_epoch}.pth'

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_f1 = checkpoint['best_val_f1']
    epochs_no_improve = checkpoint['epochs_no_improve']

    # Load history if available
    if 'history' in checkpoint:
        train_history = checkpoint['history']['train']
        val_history = checkpoint['history']['val']

    print(f"Resuming training from epoch {start_epoch}")

# Training loop (65 epochs for SIPaKMeD)
for epoch in range(start_epoch, 65):
    model.train()
    epoch_loss, epoch_preds, epoch_labels = [], [], []
    start_time = time.time()

    for images, labels in DataLoader(train_set, batch_size=8, shuffle=True):
        images, labels = images.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Collect batch metrics
        epoch_loss.append(loss.item())
        epoch_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        epoch_labels.extend(labels.cpu().numpy())

    # Calculate epoch metrics
    epoch_acc = accuracy_score(epoch_labels, epoch_preds)
    epoch_f1 = f1_score(epoch_labels, epoch_preds, average='macro')
    train_history['loss'].append(np.mean(epoch_loss))
    train_history['acc'].append(epoch_acc)
    train_history['f1'].append(epoch_f1)

    # Validation phase
    model.eval()
    val_loss, val_preds, val_labels = [], [], []
    with torch.no_grad():
        for images, labels in DataLoader(val_set, batch_size=8):
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss.append(loss.item())
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    # Calculate validation metrics
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='macro')

    val_history['loss'].append(np.mean(val_loss))
    val_history['acc'].append(val_acc)
    val_history['f1'].append(val_f1)

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'checkpoint/best_model.pth')
        print(f"New best model saved with F1: {val_f1:.4f}")
    else:
        epochs_no_improve += 1

    # Print epoch summary
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch + 1}/{65} | Time: {epoch_time:.1f}s")
    print(f"Train Loss: {train_history['loss'][-1]:.4f} | Acc: {epoch_acc:.4f} | F1: {epoch_f1:.4f}")
    print(f"Val Loss: {val_history['loss'][-1]:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
    print(f"No improvement in {epochs_no_improve}/{patience} epochs")

    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_f1': best_val_f1,
        'epochs_no_improve': epochs_no_improve,
        'history': {
            'train': train_history,
            'val': val_history
        }
    }
    torch.save(checkpoint, f'checkpoint/checkpoint_epoch_{epoch}.pth')

    # Early stopping check
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {patience} epochs without improvement")
        break

    scheduler.step()

# Save final model
torch.save(model.state_dict(), 'checkpoint/final_model.pth')

# Save training history
history_df = pd.DataFrame({
    'epoch': range(1, len(train_history['loss']) + 1),
    'train_loss': train_history['loss'],
    'train_acc': train_history['acc'],
    'train_f1': train_history['f1'],
    'val_loss': val_history['loss'],
    'val_acc': val_history['acc'],
    'val_f1': val_history['f1']
})
history_df.to_csv('checkpoint/training_history.csv', index=False)

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_df['epoch'], history_df['train_loss'], label='Train')
plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_df['epoch'], history_df['train_f1'], label='Train F1')
plt.plot(history_df['epoch'], history_df['val_f1'], label='Validation F1')
plt.title('F1-Score Curve')
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.legend()
plt.savefig('checkpoint/training_curves.png', dpi=300)

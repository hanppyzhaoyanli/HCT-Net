from hctnet_model import HCTNet
from dataset import CervicalDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             roc_auc_score, classification_report)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from thop import profile

# Load best model
model = HCTNet(num_classes=5).to('cuda')
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Calculate FLOPs and parameters
input_sample = torch.randn(1, 3, 224, 224).to('cuda')
flops, params = profile(model, inputs=(input_sample,))
print(f"Model Parameters: {params / 1e6:.2f}M")
print(f"FLOPs: {flops / 1e9:.2f}G")

# Load test set
test_set = CervicalDataset(root_dir='./data', dataset_name='SIPaKMeD', mode='test')
test_loader = DataLoader(test_set, batch_size=8)

# Inference speed test
start_time = time.time()
with torch.no_grad():
    for images, _ in test_loader:
        _ = model(images.to('cuda'))
inference_time = time.time() - start_time
print(f"Inference Speed: {len(test_set) / inference_time:.2f} images/sec")

# Perform testing
all_preds, all_labels, all_probs = [], [], []
test_times = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to('cuda')
        start = time.time()
        outputs = model(images)
        test_times.append(time.time() - start)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

# Calculate metrics
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
conf_mat = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds, target_names=test_set.classes)

print(f"Test Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
print("Classification Report:\n", class_report)
print(f"Avg Inference Time: {np.mean(test_times) * 1000:.2f}ms/image")

# Save comprehensive results
results_df = pd.DataFrame({
    'image_path': [test_set.images[i] for i in range(len(test_set))],
    'true_label': all_labels,
    'pred_label': all_preds,
    **{f'prob_{cls}': np.array(all_probs)[:, i] for i, cls in enumerate(test_set.classes)}
})
results_df.to_csv('test_results.csv', index=False)

# Confusion matrix visualization
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_set.classes,
            yticklabels=test_set.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300)

# ROC curve and AUC
plt.figure(figsize=(10, 8))
for i, cls in enumerate(test_set.classes):
    fpr, tpr, _ = roc_curve(all_labels, np.array(all_probs)[:, i], pos_label=i)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.savefig('roc_curves.png', dpi=300)

# Per-class metrics
class_metrics = []
for i, cls in enumerate(test_set.classes):
    cls_precision = precision_score(all_labels, all_preds, average=None)[i]
    cls_recall = recall_score(all_labels, all_preds, average=None)[i]
    cls_f1 = f1_score(all_labels, all_preds, average=None)[i]
    class_metrics.append({
        'Class': cls,
        'Precision': cls_precision,
        'Recall': cls_recall,
        'F1-Score': cls_f1,
        'Support': np.sum(np.array(all_labels) == i)
    })

metrics_df = pd.DataFrame(class_metrics)
metrics_df.to_csv('class_metrics.csv', index=False)
print(metrics_df)

# Generate LaTeX table for paper
latex_table = metrics_df.to_latex(index=False, float_format="%.4f",
                                  caption="Per-class performance metrics",
                                  label="tab:class_metrics")
with open('class_metrics.tex', 'w') as f:
    f.write(latex_table)
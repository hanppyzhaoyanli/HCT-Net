import torch
import torch.nn as nn


class JointLoss(nn.Module):
    """WCE + Focal Loss implementation from Section 3.3"""

    def __init__(self, class_weights=None, gamma=2.0, alpha=0.25, smoothing=0.1):
        super().__init__()
        self.class_weights = class_weights  # For class imbalance
        self.gamma = gamma  # Focal parameter γ=2
        self.alpha = alpha
        self.smoothing = smoothing  # Label smoothing

    def forward(self, preds, targets):
        # Weighted Cross Entropy (WCE)
        ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.smoothing
        )(preds, targets)

        # Focal Loss component
        log_softmax = torch.log_softmax(preds, dim=-1)
        pt = torch.exp(log_softmax)
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * log_softmax

        return ce_loss + focal_loss.mean()
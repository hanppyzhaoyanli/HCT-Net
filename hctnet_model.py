import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth


class PVTv2Encoder(nn.Module):
    """PVTv2 backbone encoder as described in Section 2.1.1"""

    def __init__(self, embed_dims=[64, 128, 320, 512], depths=[3, 4, 18, 3]):
        super().__init__()
        # Implement 4-stage pyramid structure (F1-F4)
        self.stages = nn.ModuleList([
            self._build_stage(embed_dims[i], depths[i])
            for i in range(4)
        ])
        self.sr_ratios = [8, 4, 2, 1]  # Spatial reduction ratios

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features  # [F1, F2, F3, F4]


class HMSA(nn.Module):
    """Hybrid Multi-dimensional Self-Attention module from Section 2.1.2"""

    def __init__(self, dim, window_size=7, num_heads=4):
        super().__init__()
        # SW-MSA branch (Eq.4-6)
        self.sw_msa = nn.MultiheadAttention(dim, num_heads)

        # Spatial Attention branch (Eq.7)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )

        # Channel Attention branch (Eq.8)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        # Dynamic branch weighting (Eq.9)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Process through three parallel branches
        sw_out = self.sw_msa(x)
        sa_out = self.spatial_att(x)
        ca_out = self.channel_att(x)

        # Feature fusion (Eq.9)
        fused = sw_out + sa_out + ca_out
        fused = self.fc2(self.relu(self.fc1(fused)))
        return x + fused  # Residual connection


class MFF(nn.Module):
    """Multi-Scale Feature Fusion module from Section 2.1.3"""

    def __init__(self, channels=[64, 128, 320, 512]):
        super().__init__()
        # Four-branch processing (Eq.10)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=0.5), nn.Conv2d(channels[0], 64, 1)),
            nn.Conv2d(channels[1], 128, 1),
            nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(channels[2], 320, 1)),
            nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(channels[3], 512, 1))
        ])
        self.fusion_conv = nn.Conv2d(sum(channels), 512, 1)

    def forward(self, features):
        # Multi-scale feature interaction (Eq.11-14)
        branch_outs = [branch(feat) for branch, feat in zip(self.branches, features)]
        fused = torch.cat(branch_outs, dim=1)
        return self.fusion_conv(fused)


class HCTNet(nn.Module):
    """Complete HCT-Net architecture from Figure 2"""

    def __init__(self, num_classes=5):
        super().__init__()
        self.encoder = PVTv2Encoder()

        # HMSA modules for F1-F3
        self.hmsa1 = HMSA(64)
        self.hmsa2 = HMSA(128)
        self.hmsa3 = HMSA(320)

        self.mff = MFF()
        self.mcp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(512),  # Batch normalization for stain normalization
            nn.Linear(512, num_classes)
        )
        # Stochastic depth regularization (Section 3.3)
        self.stochastic_depth = StochasticDepth(p=0.2, mode='batch')

    def forward(self, x):
        f1, f2, f3, f4 = self.encoder(x)

        # HMSA optimized features
        f1 = self.stochastic_depth(self.hmsa1(f1))
        f2 = self.stochastic_depth(self.hmsa2(f2))
        f3 = self.stochastic_depth(self.hmsa3(f3))

        # Multi-scale fusion
        fused = self.mff([f1, f2, f3, f4])

        # Classification prediction
        return self.mcp(fused)
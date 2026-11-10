"""Image encoders. Common interface:

    out = encoder(images)     # dict
    out["feat"]      pooled backbone feature
    out["feat_bn"]   feature after the BatchNorm neck (classifier input, test feature)
    out["proj"]      feature in the CLIP text space, used by L_align
    encoder.feat_dim / encoder.proj_dim
"""
from __future__ import annotations

import copy

import torch
import torchvision
from torch import nn


class ResNet50Encoder(nn.Module):
    """ImageNet ResNet-50 as in DACS (reid/models/resnet.py): stride of the last stage
    set to 1, global average pooling, BatchNorm without affine parameters. The linear
    map into the text space is the only addition; it exists so that a 2048-d CNN feature
    can be compared with CLIP ViT-B/16 anchors."""

    def __init__(self, cfg, proj_dim: int):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if cfg.pretrained == "imagenet" else None
        resnet = torchvision.models.resnet50(weights=weights)
        if cfg.last_stride == 1:
            resnet.layer4[0].conv2.stride = (1, 1)
            resnet.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                  resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = resnet.fc.in_features
        self.proj_dim = proj_dim
        self.feat_bn = nn.BatchNorm1d(self.feat_dim, affine=False)
        self.text_proj = nn.Linear(self.feat_dim, proj_dim, bias=False)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.gap(self.base(x)).flatten(1)
        feat_bn = self.feat_bn(feat)
        return {"feat": feat, "feat_bn": feat_bn, "proj": self.text_proj(feat_bn)}


class ClipViTEncoder(nn.Module):
    """CLIP ViT image tower. `feat` is the class token after the final LayerNorm,
    `proj` is CLIP's own projection of it into the joint space."""

    def __init__(self, clip_model):
        super().__init__()
        self.visual = copy.deepcopy(clip_model.visual)
        proj = self.visual.proj
        self.visual.proj = None  # the tower now returns the un-projected class token
        self.proj = nn.Parameter(proj.detach().clone())
        self.feat_dim, self.proj_dim = proj.shape
        self.feat_bn = nn.BatchNorm1d(self.feat_dim)
        self.feat_bn.bias.requires_grad_(False)  # BNNeck convention (CLIP-ReID)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.visual(x)
        return {"feat": feat, "feat_bn": self.feat_bn(feat), "proj": feat @ self.proj}

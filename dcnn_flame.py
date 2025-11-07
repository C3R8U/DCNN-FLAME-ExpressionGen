#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCNN-FLAME (reference) – Minimal PyTorch implementation
Author: DCNN-FLAME Authors (2025)
License: BSD-3-Clause

This file defines:
- Backbone CNN
- Dual-branch heads: Expression classifier & AU detector
- Identity encoder
- FLAME-like parameter regressor (proxy)
- Multi-scale fusion block
- Style / perceptual loss (VGG19 features)
- MMD loss for human->animation domain adaptation
- Utility metrics (SSIM), LPIPS-lite (VGG-based)
"""
from __future__ import annotations
import math
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import vgg19, VGG19_Weights
except Exception:
    vgg19 = None
    VGG19_Weights = None


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, k, s, p, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class MultiScaleFusion(nn.Module):
    def __init__(self, c_small: int, c_mid: int, c_large: int, out_c: int):
        super().__init__()
        self.proj_s = nn.Conv2d(c_small, out_c, 1)
        self.proj_m = nn.Conv2d(c_mid, out_c, 1)
        self.proj_l = nn.Conv2d(c_large, out_c, 1)
        self.alpha = nn.Parameter(torch.zeros(3))

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        s, m, l = xs
        _, _, H, W = l.shape
        s = F.interpolate(s, size=(H, W), mode='bilinear', align_corners=False)
        m = F.interpolate(m, size=(H, W), mode='bilinear', align_corners=False)
        s = self.proj_s(s); m = self.proj_m(m); l = self.proj_l(l)
        w = torch.softmax(self.alpha, dim=0)
        return w[0]*s + w[1]*m + w[2]*l


class Backbone(nn.Module):
    def __init__(self, in_c=3, base=32):
        super().__init__()
        self.stem = ConvBNReLU(in_c, base, 3, 1, 1)
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(base, base, 3, 1, 1),
            DepthwiseSeparableConv(base, base, 3, 1, 1),
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(base, base*2, 3, 2, 1),
            DepthwiseSeparableConv(base*2, base*2),
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(base*2, base*4, 3, 2, 1),
            DepthwiseSeparableConv(base*4, base*4),
        )

    def forward(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f1, f2, f3


class IdentityEncoder(nn.Module):
    def __init__(self, in_c=128, dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_c, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return F.normalize(self.fc(x), dim=1)


class ExpressionHead(nn.Module):
    def __init__(self, in_c=128, n_classes=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_c, n_classes)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.fc(x)


class AUDetector(nn.Module):
    def __init__(self, in_c=128, n_aus=17):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_c, n_aus)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.fc(x)


class FLAMEParamRegressor(nn.Module):
    def __init__(self, in_c=128, pose_dim=6, expr_dim=50):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_pose = nn.Linear(in_c, pose_dim)
        self.fc_expr = nn.Linear(in_c, expr_dim)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return {"pose": self.fc_pose(x), "expr": self.fc_expr(x)}


# ----- perceptual / style features -----
class VGGPerceptual(nn.Module):
    def __init__(self, layers=(2, 7, 12, 21, 30)):
        super().__init__()
        if vgg19 is None:
            raise ImportError("torchvision is required for VGGPerceptual.")
        weights = VGG19_Weights.DEFAULT
        vgg = vgg19(weights=weights).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.slices = nn.ModuleList()
        last = 0
        for l in layers:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(last, l)]))
            last = l
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats = []
        cur = x
        for sl in self.slices:
            cur = sl(cur)
            feats.append(cur)
        return feats


def perceptual_loss(vgg: VGGPerceptual, x: torch.Tensor, y: torch.Tensor, weights=None) -> torch.Tensor:
    fx = vgg(x); fy = vgg(y)
    if weights is None:
        weights = [1.0, 1.0, 0.5, 0.25, 0.25]
    loss = 0.0
    for w, a, b in zip(weights, fx, fy):
        loss = loss + w * F.l1_loss(a, b)
    return loss


def lpips_lite(vgg: VGGPerceptual, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    fx = vgg(x); fy = vgg(y)
    d = 0.0
    for a, b in zip(fx, fy):
        d = d + F.mse_loss(F.normalize(a, dim=1), F.normalize(b, dim=1))
    return d / len(fx)


def ssim(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x*x, 3, 1, 1) - mu_x*mu_x
    sigma_y = F.avg_pool2d(y*y, 3, 1, 1) - mu_y*mu_y
    sigma_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))
    return ssim_map.mean()


def compute_mmd(src_feats: torch.Tensor, tgt_feats: torch.Tensor, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = src_feats.size(0)
    m = tgt_feats.size(0)
    total = torch.cat([src_feats, tgt_feats], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n + m) ** 2
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernels = [torch.exp(-L2_distance / b) for b in bandwidth_list]
    kernel_val = sum(kernels)

    XX = kernel_val[:n, :n]
    YY = kernel_val[n:, n:]
    XY = kernel_val[:n, n:]
    YX = kernel_val[n:, :n]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


class DCNNFLAME(nn.Module):
    def __init__(self,
                 n_expr_classes: int = 8,
                 n_aus: int = 17,
                 backbone_base: int = 32,
                 style_loss_weight: float = 1.0,
                 expr_loss_weight: float = 1.0,
                 au_loss_weight: float = 1.0,
                 geom_loss_weight: float = 1.0):
        super().__init__()
        self.backbone = Backbone(in_c=3, base=backbone_base)
        c1, c2, c3 = backbone_base, backbone_base*2, backbone_base*4
        self.fuse = MultiScaleFusion(c1, c2, c3, out_c=backbone_base*4)
        self.identity = IdentityEncoder(in_c=backbone_base*4, dim=128)
        self.expr_head = ExpressionHead(in_c=backbone_base*4, n_classes=n_expr_classes)
        self.au_head = AUDetector(in_c=backbone_base*4, n_aus=n_aus)
        self.param_reg = FLAMEParamRegressor(in_c=backbone_base*4, pose_dim=6, expr_dim=50)

        self.w_style = style_loss_weight
        self.w_expr = expr_loss_weight
        self.w_au = au_loss_weight
        self.w_geom = geom_loss_weight

        self.vgg = None
        if vgg19 is not None:
            try:
                self.vgg = VGGPerceptual()
            except Exception:
                self.vgg = None

    def forward(self, x):
        f1, f2, f3 = self.backbone(x)
        f = self.fuse([f1, f2, f3])
        id_emb = self.identity(f)
        expr_logits = self.expr_head(f)
        au_logits = self.au_head(f)
        params = self.param_reg(f)
        return {
            "features": f,
            "id": id_emb,
            "expr_logits": expr_logits,
            "au_logits": au_logits,
            "pose": params["pose"],
            "expr_code": params["expr"],
        }

    def compute_losses(self, inputs, outputs, targets):
        loss_dict = {}

        if "expr_labels" in targets:
            loss_expr = F.cross_entropy(outputs["expr_logits"], targets["expr_labels"])
            loss_dict["loss_expr"] = self.w_expr * loss_expr

        if "au_labels" in targets:
            au_logits = outputs["au_logits"]
            au_labels = targets["au_labels"]
            loss_au = F.binary_cross_entropy_with_logits(au_logits, au_labels)
            loss_dict["loss_au"] = self.w_au * loss_au

        if "pose_gt" in targets:
            loss_pose = F.mse_loss(outputs["pose"], targets["pose_gt"])
            loss_dict["loss_pose"] = self.w_geom * loss_pose
        if "expr_code_gt" in targets:
            loss_exprcode = F.mse_loss(outputs["expr_code"], targets["expr_code_gt"])
            loss_dict["loss_exprcode"] = self.w_geom * loss_exprcode

        if self.vgg is not None and ("recon" in outputs):
            loss_style = perceptual_loss(self.vgg, outputs["recon"], inputs["image"])
            loss_dict["loss_style"] = self.w_style * loss_style

        if "src_features" in inputs and "tgt_features" in inputs:
            loss_mmd = compute_mmd(inputs["src_features"], inputs["tgt_features"])
            loss_dict["loss_mmd"] = 0.1 * loss_mmd

        loss_total = sum(loss_dict.values()) if len(loss_dict) > 0 else torch.tensor(0.0, device=outputs["features"].device)
        loss_dict["loss_total"] = loss_total
        return loss_dict

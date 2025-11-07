#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_flame.py – Minimal training/inference script for DCNN-FLAME (reference)
"""
from __future__ import annotations
import os
import time
import math
import json
import random
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import read_image

from dcnn_flame import DCNNFLAME

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageFolderWithLabels(Dataset):
    def __init__(self, root: str, image_size: int = 256, n_expr: int = 8, n_aus: int = 17):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.lbl_dir = self.root / "labels"
        self.paths = sorted([p for p in self.img_dir.glob("*") if p.suffix.lower() in [".png",".jpg",".jpeg"]])
        self.n_expr = n_expr
        self.n_aus = n_aus
        self.tf = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize((image_size, image_size)),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = read_image(str(p))
        if img.size(0) == 1:
            img = img.repeat(3,1,1)
        img = self.tf(img/255.0 if img.max()>1 else img)

        expr_label = torch.tensor(-1, dtype=torch.long)
        au = torch.zeros(self.n_aus, dtype=torch.float32)

        j = self.lbl_dir / (p.stem + ".json")
        if j.exists():
            try:
                obj = json.loads(j.read_text())
                if "expr_label" in obj:
                    expr_label = torch.tensor(int(obj["expr_label"]), dtype=torch.long)
                if "au" in obj and isinstance(obj["au"], list):
                    arr = torch.tensor(obj["au"], dtype=torch.float32)
                    if arr.numel() == self.n_aus:
                        au = arr
            except Exception:
                pass

        return {"image": img, "expr_label": expr_label, "au": au, "path": str(p)}


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels >= 0
    if mask.sum() == 0:
        return float('nan')
    preds = logits.argmax(dim=1)
    acc = (preds[mask] == labels[mask]).float().mean().item()
    return acc

def f1_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    mask = labels >= 0
    if mask.sum() == 0:
        return float('nan')
    preds = logits.argmax(dim=1)
    f1 = 0.0
    for c in range(num_classes):
        tp = ((preds==c) & (labels==c) & mask).sum().item()
        fp = ((preds==c) & (labels!=c) & mask).sum().item()
        fn = ((preds!=c) & (labels==c) & mask).sum().item()
        if tp+fp==0 or tp+fn==0:
            f1_c = 0.0
        else:
            prec = tp/(tp+fp+1e-8)
            rec  = tp/(tp+fn+1e-8)
            f1_c = 2*prec*rec/(prec+rec+1e-8)
        f1 += f1_c
    return f1/num_classes


def train_one_epoch(model, loader, opt, device):
    model.train()
    t0 = time.time()
    total = 0.0
    n = 0
    for batch in loader:
        img = batch["image"].to(device)
        expr_label = batch["expr_label"].to(device)
        au = batch["au"].to(device)

        out = model(img)
        targets = {}
        if (expr_label>=0).any():
            targets["expr_labels"] = expr_label
        if au is not None:
            targets["au_labels"] = au

        losses = model.compute_losses({"image": img}, out, targets)
        loss = losses["loss_total"]

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()
        n += 1
    return total/max(n,1), (time.time()-t0)*1000.0/max(n,1)


@torch.no_grad()
def evaluate(model, loader, device, n_expr):
    model.eval()
    accs, f1s = [], []
    for batch in loader:
        img = batch["image"].to(device)
        expr_label = batch["expr_label"].to(device)
        out = model(img)
        acc = accuracy_from_logits(out["expr_logits"], expr_label)
        f1  = f1_from_logits(out["expr_logits"], expr_label, n_expr)
        if not math.isnan(acc): accs.append(acc)
        if not math.isnan(f1):  f1s.append(f1)
    return (sum(accs)/max(len(accs),1), sum(f1s)/max(len(f1s),1))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--val", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--expr_classes", type=int, default=8)
    parser.add_argument("--aus", type=int, default=17)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    train_ds = ImageFolderWithLabels(args.data, n_expr=args.expr_classes, n_aus=args.aus)
    val_root = args.val if args.val is not None else args.data
    val_ds = ImageFolderWithLabels(val_root, n_expr=args.expr_classes, n_aus=args.aus)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = DCNNFLAME(n_expr_classes=args.expr_classes, n_aus=args.aus).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    for epoch in range(1, args.epochs+1):
        loss, ms_per_it = train_one_epoch(model, train_loader, opt, args.device)
        acc, f1 = evaluate(model, val_loader, args.device, args.expr_classes)
        print(f"[Epoch {epoch:02d}] loss={loss:.4f}  val_acc={acc:.3f}  val_f1={f1:.3f}  iter_time_ms={ms_per_it:.1f}")

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/dcnn_flame_best.pt")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params/1e6:.2f} M")

if __name__ == "__main__":
    main()

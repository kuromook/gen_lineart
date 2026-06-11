import cv2
import numpy as np
import torch
import torch.nn.functional as F


def canny_edges(x, low_th=50, high_th=150):
    """(B,1,H,W) tensor [0,1] → Canny edge map (B,1,H,W) tensor"""
    x_np = (x.detach().cpu().numpy() * 255).astype(np.uint8)
    edges = []
    for i in range(x_np.shape[0]):
        edge = cv2.Canny(x_np[i, 0], low_th, high_th).astype(np.float32) / 255.0
        edges.append(edge)
    edges = np.stack(edges, axis=0)
    return torch.from_numpy(edges).unsqueeze(1).to(x.device)


def edge_loss(pred, target):
    return F.l1_loss(canny_edges(pred), canny_edges(target))


def tolerant_f1_loss(pred, target, tolerance_px=2):
    """Differentiable soft F1 that allows small local line displacement."""
    kernel_size = tolerance_px * 2 + 1
    target_near = F.max_pool2d(target, kernel_size, stride=1, padding=tolerance_px)
    pred_near = F.max_pool2d(pred, kernel_size, stride=1, padding=tolerance_px)

    precision = (pred * target_near).sum(dim=(1, 2, 3)) / (
        pred.sum(dim=(1, 2, 3)) + 1e-6
    )
    recall = (target * pred_near).sum(dim=(1, 2, 3)) / (
        target.sum(dim=(1, 2, 3)) + 1e-6
    )
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    return 1.0 - f1.mean()


def ink_loss(pred, target):
    """Penalize excess or missing total ink independently of line position."""
    return F.l1_loss(
        pred.mean(dim=(1, 2, 3)),
        target.mean(dim=(1, 2, 3)),
    )

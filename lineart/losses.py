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


def _soft_erode(x):
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x):
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x, iterations=10):
    """Differentiable morphological skeletonization (Shit et al., clDice,
    https://arxiv.org/abs/2003.07311). `iterations` must be >= the largest
    stroke radius in pixels expected in `x`, or thick strokes won't fully
    erode down to a 1px-wide centerline."""
    x1 = _soft_open(x)
    skel = F.relu(x - x1)
    for _ in range(iterations):
        x = _soft_erode(x)
        x1 = _soft_open(x)
        delta = F.relu(x - x1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(pred, target, iterations=10, smooth=1e-6):
    """1 - clDice: penalizes broken/disconnected predicted strokes (low
    topology sensitivity, GT centerline not covered by predicted ink) and
    predicted centerline branches landing outside the GT mask (low topology
    precision), independent of raw pixel overlap. `pred`/`target` are
    probability-space (post-sigmoid) tensors in [0, 1], same convention as
    the other loss functions in this module."""
    skel_pred = soft_skeletonize(pred, iterations)
    skel_target = soft_skeletonize(target, iterations)
    t_prec = (skel_pred * target).sum(dim=(1, 2, 3)) / (skel_pred.sum(dim=(1, 2, 3)) + smooth)
    t_sens = (skel_target * pred).sum(dim=(1, 2, 3)) / (skel_target.sum(dim=(1, 2, 3)) + smooth)
    cldice = 2.0 * t_prec * t_sens / (t_prec + t_sens + smooth)
    return 1.0 - cldice.mean()

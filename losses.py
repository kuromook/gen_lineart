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

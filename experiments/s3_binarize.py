"""登録 S3 の下絵の側の二値化の腕(doc/work_log.md 2026-11-27「登録 S3」)。線画の側は S1 の binarize のまま。"""
import cv2
import numpy as np

from s1_cleanup_delta import SPECK, WHITE, binarize

T1_LEVEL = 235          # T1: 明るさ < 235 を墨
T2_BLOCK, T2_C = 51, 10  # T2: 局所の平均の閾(窓 51px、C 10)かつ明るさ < 243


def despeck(b):
    b = b.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    small = np.where(st[:, cv2.CC_STAT_AREA] < SPECK)[0]
    small = small[small > 0]
    if len(small):
        b[np.isin(lab, small)] = 0
    return b.astype(bool)


def binarize_sketch(path, arm):
    if arm == "T0":
        return binarize(path)[0]
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arm == "T1":
        return despeck(g < T1_LEVEL)
    if arm == "T2":
        a = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, T2_BLOCK, T2_C) > 0
        return despeck(a & (g < WHITE))
    raise ValueError(arm)

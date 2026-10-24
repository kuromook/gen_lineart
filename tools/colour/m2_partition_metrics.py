"""M2: 面の分割(ラベル地図)に対する region-F1 と、その器具確認。

借りる側の事情: Track J/L の `region_metrics.match` は **二値マスクの連結成分**を面にする。
カラー原稿の面の分割は**余白以外の全部を覆う分割**なので、二値にすると面同士が接して
1つの塊になり、連結成分は面を返さない。ゆえにラベル地図に対する照合が必要である。

規則は借りたものと**同じにする**(8近傍、min_px、IoU 降順の貪欲1対1、調和平均):
    region      := ラベル地図のラベル値 > 0 の画素集合。面積 >= min_px のものだけ
    match       := IoU >= iou_thr、IoU 降順で貪欲に1対1
    region-F1   := matched/n_pred と matched/n_gt の調和平均

**min_px = 500 と IoU = 0.5 は Track J/L がモノクロ合成の都合で決めた値**であり、
この木が引き継ぐ値ではない。既定値としては置くが、カラーの面の面積分布を見てから
登録し直す(感度の格子は毎回印字する)。

器具確認(値は走らせる前に書いた):
    IC-M1  借りた二値の照合と一致する: 二値マスクの連結成分からラベル地図を作れば、
           `region_metrics.match` と n_pred / n_gt / matched / region_f1 が完全一致
    IC-M2  ラベル値は結果に影響しない: ラベルを置換しても同じ値
    IC-M3  自分自身を予測にすると 1.0、背景だけの予測は 0.0
    IC-M4  既知のずれ: 面を既知の画素数だけ平行移動したときの IoU を解析値と照合
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from region_metrics import match as binary_match  # noqa: E402  借りる(読み取り専用)

MIN_REGION_PX = 500   # Track J/L の値。この木では既定値であって継承ではない
MATCH_IOU = 0.5
SENS_MIN = (0, 100, 500, 2000)
SENS_IOU = (0.25, 0.5, 0.75)


def areas(lab, min_px=MIN_REGION_PX):
    """ラベル地図から面の一覧を返す。ラベル 0 は背景として捨てる。"""
    ids, counts = np.unique(lab, return_counts=True)
    return [(int(i), int(c)) for i, c in zip(ids, counts)
            if i != 0 and c >= min_px]


def match_from_counts(pair_counts, parea, garea, iou_thr=MATCH_IOU):
    """貪欲1対1照合の**本体**。借りた規則はここ1か所にしかない。

    `pair_counts`: {(pred ラベル, gt ラベル): 交差画素数}。床で落とした
    ラベルを含んでいてもよい(`parea`/`garea` に無い対は捨てる)。
    `parea` / `garea`: {ラベル: 面積}。**床で濾した後のもの**。

    ラベル地図を2枚メモリに置けない大きさ(原寸 46〜64M画素)でも同じ規則で
    採点できるように切り出した。`match_partition` はこれを呼ぶだけなので、
    IC-M1〜IC-M4 はこの関数を点検している。
    """
    np_, ng = len(parea), len(garea)
    if np_ == 0 and ng == 0:
        return {"n_pred": 0, "n_gt": 0, "matched": 0, "precision": 1.0,
                "recall": 1.0, "region_f1": 1.0}
    cand = []
    for (pi, gi), inter in pair_counts.items():
        pi, gi, inter = int(pi), int(gi), int(inter)
        if pi not in parea or gi not in garea:
            continue
        union = parea[pi] + garea[gi] - inter
        iou = inter / union if union else 0.0
        if iou >= iou_thr:
            cand.append((iou, pi, gi))
    cand.sort(reverse=True)
    up, ug, matched = set(), set(), 0
    for _, pi, gi in cand:
        if pi in up or gi in ug:
            continue
        up.add(pi)
        ug.add(gi)
        matched += 1
    prec = matched / np_ if np_ else (1.0 if ng == 0 else 0.0)
    rec = matched / ng if ng else (1.0 if np_ == 0 else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n_pred": np_, "n_gt": ng, "matched": matched,
            "precision": prec, "recall": rec, "region_f1": f1}


def pair_counts(pred, gt):
    """ラベル地図2枚から {(pred, gt): 交差画素数} を作る。"""
    out = {}
    both = (pred > 0) & (gt > 0)
    if both.any():
        pairs, inters = np.unique(np.stack([pred[both], gt[both]]), axis=1,
                                  return_counts=True)
        for (pi, gi), inter in zip(pairs.T, inters):
            out[(int(pi), int(gi))] = int(inter)
    return out


def match_partition(pred, gt, min_px=MIN_REGION_PX, iou_thr=MATCH_IOU):
    """ラベル地図どうしの貪欲1対1照合。借りた二値版と同じ規則。"""
    parea, garea = dict(areas(pred, min_px)), dict(areas(gt, min_px))
    pc = pair_counts(pred, gt) if (parea and garea) else {}
    return match_from_counts(pc, parea, garea, iou_thr)


def score(pred, gt, min_px=MIN_REGION_PX, iou_thr=MATCH_IOU):
    """主値と感度の格子。格子は印字専用で、判定には使わない。"""
    row = dict(match_partition(pred, gt, min_px, iou_thr))
    row["pred_area_frac"] = round(float((pred > 0).mean()), 6)
    row["gt_area_frac"] = round(float((gt > 0).mean()), 6)
    for m in SENS_MIN:
        for t in SENS_IOU:
            row[f"rf1_m{m}_t{t}"] = round(
                match_partition(pred, gt, m, t)["region_f1"], 4)
    return row


def label_of(mask, min_px=0):
    """二値マスクの8連結成分をラベル地図にする(IC-M1 のため)。"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(lab)
    nxt = 1
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            out[lab == i] = nxt
            nxt += 1
    return out


# ---------------------------------------------------------------- 器具確認

def _disc(shape, cy, cx, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def ic_m1(verbose=True):
    """借りた二値の照合と完全一致すること。IC5 と同じ配置 + 乱数の塊。"""
    cases = []
    for d in (16, 30):
        shape = (400, 500)
        gt = _disc(shape, 150, 120, 40) | _disc(shape, 150, 340, 40)
        pr = _disc(shape, 150, 120, 40) | _disc(shape, 150, 340 + d, 40)
        cases.append((f'IC5 d={d}', pr, gt))
    rng = np.random.default_rng(0)
    for k in range(6):
        shape = (300, 300)
        a = np.zeros(shape, bool)
        b = np.zeros(shape, bool)
        for _ in range(5):
            cy, cx, r = rng.integers(40, 260, 2).tolist() + [int(rng.integers(15, 45))]
            a |= _disc(shape, cy, cx, r)
            b |= _disc(shape, cy + int(rng.integers(-12, 13)),
                       cx + int(rng.integers(-12, 13)), r)
        cases.append((f'rand {k}', a, b))
    ok = True
    for name, pr, gt in cases:
        want = binary_match(pr, gt)
        got = match_partition(label_of(pr, MIN_REGION_PX),
                              label_of(gt, MIN_REGION_PX))
        good = all(abs(got[k] - want[k]) < 1e-12 for k in
                   ("n_pred", "n_gt", "matched", "region_f1"))
        ok &= good
        if verbose and not good:
            print(f'  IC-M1 MISMATCH {name}: borrowed {want} / partition {got}')
    print(f'IC-M1  借りた二値版と一致: {len(cases)}/{len(cases)} ケース'
          f' -> {"PASS" if ok else "FAIL"}')
    return ok


def ic_m2(verbose=True):
    """ラベル値の置換で値が変わらないこと。"""
    rng = np.random.default_rng(1)
    gt = np.zeros((300, 300), np.int32)
    gt[:150, :150] = 1
    gt[:150, 150:] = 2
    gt[150:, :] = 3
    pred = gt.copy()
    # 3番の面だけ IoU を 0.5 未満に落とす(1.0 では置換の検査にならない)。
    # gt の3番は 150x300 = 45,000px、予測を 260 行目からにすると
    # 40x300 = 12,000px で IoU 0.2667 -> 照合しない。期待 matched 2/3、F1 0.6667
    pred[150:260, :] = 0
    perm = {1: 77, 2: 5, 3: 1234}
    pr2 = np.zeros_like(pred)
    for a, b in perm.items():
        pr2[pred == a] = b
    gt2 = np.zeros_like(gt)
    for a, b in {1: 9, 2: 8, 3: 7}.items():
        gt2[gt == a] = b
    r1 = match_partition(pred, gt)
    r2 = match_partition(pr2, gt2)
    good = (r1 == r2 and r1['matched'] == 2 and abs(r1['region_f1'] - 2/3) < 1e-9)
    print(f'IC-M2  ラベル置換で不変: matched {r1["matched"]} (期待 2/3) '
          f'region-F1 {r1["region_f1"]:.4f} / {r2["region_f1"]:.4f} '
          f'(期待 0.6667) -> {"PASS" if good else "FAIL"}')
    return good


def ic_m3():
    """自分自身で 1.0、背景だけで 0.0。"""
    gt = np.zeros((300, 300), np.int32)
    gt[:150, :150] = 1
    gt[:150, 150:] = 2
    gt[150:, :] = 3
    a = match_partition(gt, gt)
    b = match_partition(np.zeros_like(gt), gt)
    good = (a["region_f1"] == 1.0 and a["matched"] == 3 and b["region_f1"] == 0.0)
    print(f'IC-M3  自分自身 {a["region_f1"]:.4f} (matched {a["matched"]}/3) / '
          f'背景だけ {b["region_f1"]:.4f} -> {"PASS" if good else "FAIL"}')
    return good


def ic_m4():
    """既知のずれ: 1辺 L の正方形を dx 画素ずらすと IoU = (L-dx)/(L+dx)。
    L = 100, dx = 33 -> IoU 0.5038 (>= 0.5 で照合する)
    L = 100, dx = 34 -> IoU 0.4925 (照合しない)"""
    expect = {33: (1, 0.5038), 34: (0, 0.4925)}
    ok = True
    for dx, (exp_m, exp_iou) in expect.items():
        gt = np.zeros((400, 400), np.int32)
        gt[100:200, 100:200] = 1
        pr = np.zeros_like(gt)
        pr[100:200, 100 + dx:200 + dx] = 1
        got_iou = (100 - dx) / (100 + dx)
        r = match_partition(pr, gt)
        good = r["matched"] == exp_m and abs(got_iou - exp_iou) < 1e-3
        ok &= good
        print(f'IC-M4  ずれ {dx}px: matched {r["matched"]} (期待 {exp_m}) '
              f'IoU {got_iou:.4f} (解析 {exp_iou}) -> {"PASS" if good else "FAIL"}')
    return ok


if __name__ == '__main__':
    ok = ic_m1() and ic_m2() and ic_m3() and ic_m4()
    print('M2 器具確認:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)

#!/usr/bin/env python
"""診断 H14(2026-10-17 事前登録): H11 の T2 の注意の重みを図にする。眉・耳を外したグループと、眉を当てたグループ。学習用だけ。

出力: index.html、図(png)、stats.json
"""
import argparse, html, json, sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_server as G                                               # noqa: E402
import h7_stroke_parts as H                                            # noqa: E402
import h10_scale_pose as H10                                           # noqa: E402
import h11_set_model as M                                              # noqa: E402

PX = 560
COL = {"eye": (40, 40, 220), "brow": (0, 140, 255), "nose": (40, 160, 40), "mouth": (200, 80, 20), "ear": (170, 40, 160),
       "other_face": (160, 160, 0), "none": (170, 170, 170)}
JA = {"eye": "目", "brow": "眉", "nose": "鼻", "mouth": "口", "ear": "耳", "other_face": "顔のほか", "none": "なし"}


def forward_attn(net, x):
    """層ごとに注意の重み(4 頭の平均)を取り出しながら計算し直す。返る: logit (n, C)、[層ごとの (n, n)]"""
    h = net.inp(x); A = []
    for L in net.enc.layers:
        a, w = L.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
        h = L.norm1(h + a); h = L.norm2(h + L._ff_block(h)); A.append(w[0].cpu().numpy())
    return net.out(h)[0], A


def recovered(f, pred):
    comps = H.components(f, pred, H10.TAU); out = {}
    for g in sorted(set(f["gid"][f["gid"] >= 0].tolist())):
        Gs = set(np.flatnonzero(f["gid"] == g).tolist()); c = int(f["y"][f["gid"] == g][0])
        out[g] = any(pc == c and len(Gs & Ps) >= H.TH * len(Gs) and len(Gs & Ps) >= H.TH * len(Ps) for pc, Ps in comps)
    return out


def draw(f, bg, colors, widths, title):
    q = G.S["queue"][f["order"]]; wx, wy, side = G.window(q); k = PX / side
    img = cv2.addWeighted(cv2.resize(bg, (PX, PX), interpolation=cv2.INTER_AREA), 0.25, np.full((PX, PX, 3), 255, np.uint8), 0.75, 0)
    order = np.argsort(widths)
    for i in order:
        p = ((f["pts"][i] - [wx, wy]) * k).round().astype(np.int32)
        cv2.polylines(img, [p], False, tuple(int(c) for c in colors[i]), int(widths[i]), cv2.LINE_AA)
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def heat(v):
    v = v / max(v.max(), 1e-9)
    return [(int(220 * (1 - t)), int(220 * (1 - t)), int(220 + 35 * t)) if t > 0 else (220, 220, 220) for t in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h14_attention_20261017")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    faces = H.load_faces(H10.MAX_ORDER, H10.N_LINES)
    fold, _ = H.folds_of(faces)
    R = {}; max_diff = 0.0
    for k in sorted(set(fold.tolist())):
        tr = np.flatnonzero(fold != k); te = np.flatnonzero(fold == k)
        _, net, mu, sd = M.train_eval(faces, tr, te, True, 0 * 10 + k, dev)          # H11 の T2、乱数の組 0 と同じ呼び方
        with torch.no_grad():
            for i in te:
                x = torch.tensor(((M.feats(faces[i], True) - mu) / sd)[None], dtype=torch.float32, device=dev)
                ref = net(x, torch.zeros((1, x.shape[1]), dtype=torch.bool, device=dev))[0]
                lg, A = forward_attn(net, x)
                max_diff = max(max_diff, float((lg - ref).abs().max()))
                R[i] = (torch.softmax(lg, 1).cpu().numpy(), A)
        print("fold", k, "done", flush=True)
    chk = dict(max_logit_diff=max_diff, passed=max_diff <= 1e-4)
    print(chk, flush=True)
    if not chk["passed"]:
        json.dump(chk, open(out / "check.json", "w"), indent=1); print("道具確認: 不合格"); return

    BROW, EAR = H.CLASSES.index("brow"), H.CLASSES.index("ear")
    targets = {"brow_miss": [], "ear_miss": [], "brow_hit": []}
    all_rec = {}
    for i in sorted(R, key=lambda i: faces[i]["order"]):
        f = faces[i]; pred = R[i][0].argmax(1); rec = recovered(f, pred); all_rec[i] = rec
        for g, ok in rec.items():
            c = int(f["y"][f["gid"] == g][0])
            key = {(BROW, False): "brow_miss", (EAR, False): "ear_miss", (BROW, True): "brow_hit"}.get((c, ok))
            if key:
                targets[key].append((i, g))
    lim = {"brow_miss": 12, "ear_miss": 12, "brow_hit": 6}

    # 参考の数: 外した眉・耳の線の予測の内訳と、対象の線からの最終層の注意が向く正解のパーツ(全グループで)
    stats = {}
    for key in targets:
        conf, att = Counter(), Counter()
        for i, g in targets[key]:
            f = faces[i]; m = np.flatnonzero(f["gid"] == g); pred = R[i][0].argmax(1)
            conf.update(H.CLASSES[c] for c in pred[m])
            w = R[i][1][-1][m].mean(0)
            for c in range(len(H.CLASSES)):
                att[H.CLASSES[c]] += float(w[f["y"] == c].sum())
        tot = sum(att.values())
        stats[key] = dict(n_groups=len(targets[key]), predicted_as=dict(conf), attention_to_true_class={c: round(v / tot, 3) for c, v in att.items()})
    stats["check"] = chk
    json.dump(stats, open(out / "stats.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats, ensure_ascii=False))

    sections = []
    titles = {"brow_miss": "眉を外したグループ", "ear_miss": "耳を外したグループ", "brow_hit": "対照: 眉を当てたグループ"}
    for key in ("brow_miss", "ear_miss", "brow_hit"):
        cards = []
        for i, g in targets[key][:lim[key]]:
            f = faces[i]; prob, A = R[i]; pred = prob.argmax(1); m = f["gid"] == g
            bg = cv2.imdecode(np.frombuffer(G.face_png(f["order"]), np.uint8), cv2.IMREAD_COLOR)
            wd = np.where(m, 5, 2)
            ytrue = [COL[H.CLASSES[c]] for c in f["y"]]; ypred = [COL[H.CLASSES[c]] for c in pred]
            hl, h1 = A[-1][np.flatnonzero(m)].mean(0), A[0][np.flatnonzero(m)].mean(0)
            hl, h1 = np.where(m, 0, hl), np.where(m, 0, h1)          # 色の濃さは対象の外の線の中で比べる
            panels = [draw(f, bg, ytrue, wd, "a: truth"), draw(f, bg, ypred, wd, "b: predicted"),
                      draw(f, bg, [(0, 0, 0) if t else c for t, c in zip(m, heat(hl))], np.where(m, 5, 2 + 3 * hl / max(hl.max(), 1e-9)).round(), "c: attention, layer 3"),
                      draw(f, bg, [(0, 0, 0) if t else c for t, c in zip(m, heat(h1))], np.where(m, 5, 2 + 3 * h1 / max(h1.max(), 1e-9)).round(), "d: attention, layer 1")]
            name = f"{key}_o{f['order']}_g{g}.png"
            cv2.imwrite(str(out / name), np.hstack(panels))
            pc = Counter(JA[H.CLASSES[c]] for c in pred[m])
            pose = f["pose"] or {}
            cards.append(f"<figure><img src='{name}' loading='lazy'><figcaption>order {f['order']}・グループ {g}・線 {int(m.sum())} 本・"
                         f"向き {html.escape(str(pose.get('yaw')))}/{html.escape(str(pose.get('dir')))}/{html.escape(str(pose.get('pitch')))}・"
                         f"予測: {html.escape('、'.join(f'{k} {v}' for k, v in pc.items()))}</figcaption></figure>")
        sections.append(f"<h2>{titles[key]}({min(lim[key], len(targets[key]))} / {len(targets[key])} 個)</h2>" + "".join(cards))
    legend = "".join(f"<span style='color:rgb{tuple(reversed(c))}'>■ {JA[k]}</span> " for k, c in COL.items())
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H14 注意の図</title><style>body{{font-family:sans-serif;margin:16px;background:#fff;color:#111}}img{{max-width:100%;border:1px solid #ccc}}
figure{{margin:0 0 28px}}figcaption{{font-size:14px}}</style></head><body><h1>H14: 網は何を見て答えたか</h1>
<p>a 正解のパーツ、b 網の予測、c・d 黒い太い線(対象のグループ)から見た、ほかの線への注意(赤いほど強く見ている。最終層 / 1 層目)。色: {legend}</p>
{''.join(sections)}</body></html>"""
    open(out / "index.html", "w").write(page)
    print("->", out)


if __name__ == "__main__":
    main()

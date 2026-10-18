#!/usr/bin/env python
"""K11 器具確認(登録案 D の第 1 段。**これが無ければ D の主値を読まない**)。

問い: `ink_mask(thr=128, grow=200)` の `grow` だけで墨になった画素を、
**全トークンのうち何割が 1 つ以上通るか。**

9 割なら述語は飽和していて何も分けられない(この問いの「全部塗る」に相当する腕)。
**主値(印のあるトークンとの対比)はここでは計算しない。** 帯は基盤が飽和の数を見てから出す。

grow だけの画素 = `ink_mask(128, 200)` では墨だが `gray < 128` では墨でない画素。
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import k2_box_b_gray as K


def work(t):
    pid, source, name = t
    g = cv2.imread(str(K.DATASET / source / "line" / name), 0)
    if g is None:
        return pid, None
    Q = K.pipeline(g)
    grow_only = Q["ink"].astype(bool) & ~(g < K.THR)
    per_tok = defaultdict(int)
    tok_px = defaultdict(int)
    for i, (rr, cc) in enumerate(Q["spans"]):
        t_ = Q["span_tok"].get(i, -1)
        if t_ < 0 or len(rr) == 0:
            continue
        per_tok[t_] += int(grow_only[rr, cc].sum())
        tok_px[t_] += len(rr)
    return pid, dict(n_tok=len(tok_px),
                     n_tok_touch=sum(1 for k_, v in per_tok.items() if v > 0),
                     grow_px=int(grow_only.sum()), ink_px=int(Q["ink"].sum()),
                     per_tok={int(k_): (int(per_tok[k_]), int(tok_px[k_])) for k_ in tok_px})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--k4", default="results/k4_main_leg_20261017/per_stroke.csv")
    ap.add_argument("--out", default="results/k11_grow_saturation_20261018")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

    rs = list(csv.DictReader(open(ROOT / a.k4)))
    pans = sorted({int(r["panel"]) for r in rs})
    print(f"母集団: トークン {len(rs)} / コマ {len(pans)}(K4 主脚と同じ集合)", flush=True)
    sys.path.insert(0, str(K.TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(Path(a.labels))
    P = G.S["panels"]
    tasks = [(p, P[p]["source"], P[p]["name"]) for p in pans]
    # D の主値の母集団は K4 と同じ (コマ, sid)。sid は K4 の `out` の添字で、
    # K4 と pipeline は gr.items() を同じ順に回し同じ条件で飛ばすので span_tok と一致する。
    k4set = {}
    for r in rs:
        k4set.setdefault(int(r["panel"]), set()).add(int(r["sid"]))
    res = {}
    with Pool(a.workers) as pool:
        for pid, d in pool.imap_unordered(work, tasks, chunksize=2):
            if d:
                res[pid] = d
    ntok = sum(d["n_tok"] for d in res.values())
    ntouch = sum(d["n_tok_touch"] for d in res.values())
    # K4 と同じ母集団に絞る(これが D の主値が乗る集合)
    k4_n = k4_touch = k4_miss = 0
    for pid, d in res.items():
        for sid in k4set.get(pid, ()):
            if sid not in d["per_tok"]:
                k4_miss += 1
                continue
            k4_n += 1
            k4_touch += int(d["per_tok"][sid][0] > 0)
    frac_px = [d["grow_px"] / max(1, d["ink_px"]) for d in res.values()]
    # トークンごとの「grow だけの画素」の本数の分布
    cnt = Counter()
    share = []
    for d in res.values():
        for k_, (gp, tp) in d["per_tok"].items():
            cnt[min(gp, 10)] += 1
            if tp:
                share.append(gp / tp)
    share = np.array(share)
    summary = dict(
        registered=dict(question="全トークンのうち grow だけの画素を 1 つ以上通る割合",
                        saturated_if=">= 0.90 なら述語は飽和。帯は基盤が出す",
                        note="D の主値(印との対比)はここでは計算しない"),
        n_panels=len(res), n_tokens=ntok, n_tokens_touching_grow=ntouch,
        saturation_all_tokens=round(ntouch / max(1, ntok), 4),
        k4_population=dict(n=k4_n, n_touching_grow=k4_touch,
                           saturation=round(k4_touch / max(1, k4_n), 4),
                           sid_not_found=k4_miss,
                           note="D の主値が乗る集合(K4 主脚と同じ コマ,sid)。"
                                "sid_not_found が 0 でなければ番号の対応が崩れている"),
        grow_px_share_of_ink=dict(median=round(float(np.median(frac_px)), 4),
                                  q25=round(float(np.percentile(frac_px, 25)), 4),
                                  q75=round(float(np.percentile(frac_px, 75)), 4)),
        per_token_grow_px_hist={str(k_): int(v) for k_, v in sorted(cnt.items())},
        per_token_grow_share=dict(median=round(float(np.median(share)), 4),
                                  q75=round(float(np.percentile(share, 75)), 4),
                                  q95=round(float(np.percentile(share, 95)), 4)))
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    json.dump({str(k_): dict(n_tok=v["n_tok"], n_tok_touch=v["n_tok_touch"],
                             grow_px=v["grow_px"], ink_px=v["ink_px"]) for k_, v in res.items()},
              open(out / "per_panel.json", "w"), indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

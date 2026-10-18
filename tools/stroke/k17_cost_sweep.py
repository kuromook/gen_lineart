#!/usr/bin/env python
"""K17 登録 E: 畳み込み閾値を動かす**費用**の掃引(2026-10-18 事前登録)。

**設計の探索であって検定ではない。**材料は train。Track F は封存、変種は K の tree。

腕: 係数 {0=何も畳まない, 0.75, 1.0, 1.25, 1.5(現行), 2.0, 3.0, inf=全部畳む}
    × 幅の門 {なし, 2.7px(線 1 本の尺度), 4.0px(副)}
    両端の自明な腕は必須。**1 回の掃引に同梱し、2 周目を回さない。**

直った = broken の印があるトークンがその腕で**分裂した**(2 値)
壊れた = 印の無いトークンが**分裂した**(作者は道具を持っていて使わなかった)
**主値 = 壊れた / 直った、棒 < 1.0。**
費用の内訳: 分裂した印無しトークンのうち、**1 つの部位群に収まっていたもの**(強い費用)と
            群に入っていないもの(弱い証拠)。**主値には入れない。**
副: 破片数の分布 / 基準から状態が変わった橋のうち局所インク幅 > 8px の割合
    (ベタの軸を飲んでいないかの検査。K4a の 92.7% が基準)
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
TRACK_H = Path("/home/sh1/deepl/lineart-face-words")
DATASET = Path("/home/sh1/deepl/lineart/dataset")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/sh1/deepl/lineart-stroke-grammar/tools/stroke")

THR, GROW, MAX_BEND = 128, 200, 35.0
COEFS = [None, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, float("inf")]
GATES = [None, 2.7, 4.0]
BASE = (1.5, None)
WIDE = 8.0


def arms():
    out = []
    for c in COEFS:
        for g in (GATES if c is not None else [None]):
            out.append((c, g))
    return out


def name(c, g):
    cs = "0" if c is None else ("inf" if c == float("inf") else f"{c:g}")
    return f"coef={cs},gate={'none' if g is None else g:g}" if g is not None else f"coef={cs},gate=none"


EXAMINED = {}                      # コマ -> 作者が見たトークンの sid(K4 と同じ集合)


def work(pid_src_name):
    pid, source, nm = pid_src_name
    from skimage.morphology import skeletonize
    from stroke_path import chain, span_paths
    import strokes_variant as SV
    g = cv2.imread(str(DATASET / source / "line" / nm), 0)
    if g is None:
        return pid, None
    ink = SV.ink_mask(g, thr=THR, grow=GROW)
    iw = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5) * 2.0
    sk = skeletonize(ink)
    res = {}
    base_tok, base_spans = None, None
    for c, gt in arms():
        spans, sos, keep, info, extra = SV.strokes_from_skeleton(
            sk, iw, max_bend=MAX_BEND, collapse_coef=c, width_gate=gt)
        br = {int(s): float(w) for lst in extra["merge_bridges"].values() for s, _n, w, _t in lst}
        if (c, gt) == BASE:
            gr = defaultdict(list)
            for i in range(len(spans)):
                gr[int(sos[i])].append(i)
            tok, nt = {}, 0
            for k, mem in gr.items():
                if not keep[k]:
                    continue
                path, _gp = chain(span_paths(spans, mem))
                if len(path) < 2:
                    continue
                for i in mem:
                    tok[i] = nt
                nt += 1
            base_tok = tok
            base_spans = defaultdict(list)
            for i, t in tok.items():
                base_spans[t].append(i)
        res[(c, gt)] = (np.asarray(sos), br)
    if base_tok is None:
        return pid, None
    ex = EXAMINED.get(pid)
    out = {}
    for key, (sos, br) in res.items():
        frag = {}
        for t, mem in base_spans.items():
            if ex is not None and t not in ex:
                continue               # 作者が見ていないトークンは費用の分母に入れない
            frag[t] = len({int(sos[i]) for i in mem})
        bb = res[BASE][1]
        changed = (set(br) ^ set(bb))
        wide = [bb.get(s, br.get(s)) for s in changed]
        out[key] = dict(frag=frag, n_changed=len(changed),
                        n_changed_wide=int(sum(1 for w in wide if w is not None and w > WIDE)))
    return pid, dict(arms=out, n_tok=len(base_spans))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(TRACK_H / "labels/state.json"))
    ap.add_argument("--groups", default=str(TRACK_H / "labels/face_groups.jsonl"))
    ap.add_argument("--out", default="results/k17_cost_sweep_20261018")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

    last = {}
    for l in open(a.groups):
        r = json.loads(l); last[r["order"]] = r
    tr = [r for r in last.values() if r.get("usage") == "train" and r.get("status") == "done"]
    pop = [r for r in tr if "broken" in r]
    excl = [r for r in tr if "broken" not in r]
    sys.path.insert(0, str(TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(Path(a.labels))
    P = G.S["panels"]
    n_excl_tok = sum(int(P[r["panel"]]["n"]) for r in excl)
    print(f"train/done {len(tr)} 顔: 母集団 {len(pop)} / **外す(broken の欄なし) {len(excl)} 顔、"
          f"そのコマのトークン {n_excl_tok}**", flush=True)
    pids = sorted({r["panel"] for r in pop})
    # **作者が見たトークンだけを費用の分母にする。**K4 と同じ (コマ, sid) の集合。
    ex = defaultdict(set)
    for r in csv.DictReader(open(ROOT / "results/k4_main_leg_20261017/per_stroke.csv")):
        ex[int(r["panel"])].add(int(r["sid"]))
    EXAMINED.update({k: v for k, v in ex.items()})
    print(f"コマ {len(pids)} / 腕 {len(arms())} / 作者が見たトークン "
          f"{sum(len(v) for v in ex.values())}(全トークンではない)", flush=True)
    res = {}
    json.dump({str(k): sorted(v) for k, v in ex.items()}, open(out / "examined.json", "w"))
    with Pool(a.workers) as pool:
        for i, (pid, d) in enumerate(pool.imap_unordered(
                work, [(p, P[p]["source"], P[p]["name"]) for p in pids], chunksize=1), 1):
            if d: res[pid] = d
            if i % 20 == 0 or i == len(pids): print(f"  {i}/{len(pids)}", flush=True)

    # 顔ごとに sid の印と部位群を引く
    marked, ingrp = defaultdict(set), defaultdict(set)
    for r in pop:
        p = r["panel"]
        for s in set(r.get("broken") or []):
            marked[p].add(int(s))
        for gg in r.get("groups", []):
            for s in gg["strokes"]:
                ingrp[p].add(int(s))
    rows = []
    for c, gt in arms():
        fixed = brk = brk_g = brk_ng = 0
        fr_fixed, fr_brk, chg, chg_w = Counter(), Counter(), 0, 0
        for pid, d in res.items():
            arm = d["arms"][(c, gt)]
            chg += arm["n_changed"]; chg_w += arm["n_changed_wide"]
            mk = marked.get(pid, set()); ig = ingrp.get(pid, set())
            for t, n in arm["frag"].items():
                if t in mk:
                    if n > 1: fixed += 1; fr_fixed[min(n, 7)] += 1
                else:
                    if n > 1:
                        brk += 1; fr_brk[min(n, 7)] += 1
                        if t in ig: brk_g += 1
                        else: brk_ng += 1
        rows.append(dict(arm=name(c, gt), coef=("0" if c is None else ("inf" if c == float("inf") else c)),
                         gate=("none" if gt is None else gt),
                         fixed=fixed, broken=brk,
                         ratio=round(brk / fixed, 4) if fixed else None,
                         broken_in_group=brk_g, broken_not_in_group=brk_ng,
                         n_changed_bridges=chg,
                         wide_share=round(chg_w / chg, 4) if chg else None,
                         frag_fixed={str(k): v for k, v in sorted(fr_fixed.items())},
                         frag_broken={str(k): v for k, v in sorted(fr_brk.items())}))
    n_tok = sum(len(d["arms"][BASE]["frag"]) for d in res.values())
    n_mk = sum(len(marked.get(p, set())) for p in res)
    summary = dict(registered=dict(primary="壊れた/直った、棒 < 1.0", material="train",
                                   note="設計の探索であって検定ではない。選んだ係数・門は設計の選択で"
                                        "あって確認値ではない(確認には新しい印=K3 の枠が要る)",
                                   arms=[name(c, g) for c, g in arms()], wide_px=WIDE),
                   excluded=dict(n_faces=len(excl), n_tokens_in_their_panels=n_excl_tok,
                                 reason="broken の欄が無い顔は『印が無い』ではなく『印が使えなかった』"),
                   population=dict(n_faces=len(pop), n_panels=len(res), n_tokens=n_tok, n_marked=n_mk),
                   arms=rows)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    with open(out / "arms.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if not k.startswith("frag")],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "arms"}, ensure_ascii=False, indent=1))
    print(f'{"腕":<22}{"直った":>7}{"壊れた":>8}{"比":>8}{"群内":>7}{"幅>8":>8}')
    for r in rows:
        print(f'{r["arm"]:<22}{r["fixed"]:>7}{r["broken"]:>8}'
              f'{(str(r["ratio"]) if r["ratio"] is not None else "-"):>8}{r["broken_in_group"]:>7}'
              f'{(str(r["wide_share"]) if r["wide_share"] is not None else "-"):>8}')


if __name__ == "__main__":
    main()

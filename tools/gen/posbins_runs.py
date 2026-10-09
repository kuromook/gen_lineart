#!/usr/bin/env python
"""位置ビンの歪み: 原corpus と補正corpus の対で 10 本を走らせる(2026-10-09 事前登録)。

K1c(原corpus の cloze 条件なしが記録値 6.314605 を ±0.030 で再現)を最初に実行し、
落ちたら以降は一切走らせない。Track F のファイルは読むだけ(train_cloze.py は
tools/trackf/ へ変更なしでコピーしたものを使い、出力も本track側へ書く)。
"""
import json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TRACKF = Path("/home/sh1/deepl/lineart-stroke-grammar")
OUT = ROOT / "results/posbins_20261009"
PY = sys.executable
CORP = {"orig": str(TRACKF / "results/grammar_corpus_20260920/corpus.npz"),
        "fixed": str(OUT / "corpus_fixed.npz")}
LAB = str(TRACKF / "results/panel_composition_20260921/labels_k12.npy")
TAG = str(TRACKF / "results/panel_composition_20260921/tags_top40.npy")
REC_CLOZE = 6.314605334367116
TOL = 0.030


def run(cmd, log):
    t0 = time.time()
    with open(log, "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return r.returncode, time.time() - t0


def cloze(arm, cond):
    o = OUT / f"cloze_{arm}_{cond}"
    cmd = [PY, str(HERE.parent / "trackf/train_cloze.py"), "--data", CORP[arm], "--out", str(o)]
    if cond == "tags":
        cmd += ["--tagfeats", TAG]
    elif cond == "type":
        cmd += ["--labels", LAB]
    rc, dt = run(cmd, OUT / f"cloze_{arm}_{cond}.log")
    ev = json.load(open(o / "eval.json")) if (o / "eval.json").exists() else None
    return rc, dt, ev


SHIM = """
import sys
sys.path.insert(0, {gen!r})
sys.path.insert(0, {trackf!r})
import common, {mod} as M
common.CORPUS = M.CORPUS = {corpus!r}
sys.argv = ['x', '--out', {out!r}]
M.main()
"""


def ar(arm, mod):
    o = OUT / f"{mod}_{arm}"
    code = SHIM.format(gen=str(HERE), trackf=str(HERE.parent / "trackf"),
                       mod=mod, corpus=CORP[arm], out=str(o))
    rc, dt = run([PY, "-c", code], OUT / f"{mod}_{arm}.log")
    ev = json.load(open(o / "eval.json")) if (o / "eval.json").exists() else None
    return rc, dt, ev


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    # --- K1c: 原corpus の cloze 条件なしが記録値を再現するか ---
    print("K1c: cloze orig/none ...", flush=True)
    rc, dt, ev = cloze("orig", "none")
    if rc != 0 or ev is None:
        print("K1c 実行失敗 rc", rc, flush=True); return 1
    d = ev["pos_bits"] - REC_CLOZE
    ok = abs(d) <= TOL
    res["K1c"] = dict(passed=bool(ok), pos_bits=ev["pos_bits"], recorded=REC_CLOZE,
                      diff=round(d, 5), tol=TOL, word_bits=ev["word_bits"],
                      best_epoch=ev["best_epoch"], sec=round(dt, 1))
    print(f"K1c: pos {ev['pos_bits']:.6f} 記録 {REC_CLOZE:.6f} 差 {d:+.5f} -> "
          f"{'合格' if ok else '不合格'}  ({dt:.0f}s)", flush=True)
    json.dump(res, open(OUT / "runs.json", "w"), ensure_ascii=False, indent=1)
    if not ok:
        print("主測定に進まず停止。", flush=True); return 1

    res["runs"] = {"cloze_orig_none": dict(pos_bits=ev["pos_bits"], word_bits=ev["word_bits"],
                                           best_epoch=ev["best_epoch"], sec=round(dt, 1))}
    todo = [("cloze", a, c) for a in ("orig", "fixed") for c in ("none", "tags", "type")
            if not (a == "orig" and c == "none")]
    todo += [(m, a, None) for a in ("orig", "fixed") for m in ("train_2stage", "train_set")]
    for kind, arm, cond in todo:
        name = f"cloze_{arm}_{cond}" if kind == "cloze" else f"{kind}_{arm}"
        print(f"run {name} ...", flush=True)
        rc, dt, ev = cloze(arm, cond) if kind == "cloze" else ar(arm, kind)
        if rc != 0 or ev is None:
            print(f"  失敗 rc {rc}", flush=True)
            res["runs"][name] = dict(failed=True, rc=rc, sec=round(dt, 1))
        else:
            res["runs"][name] = {k: ev[k] for k in ev if k in
                                 ("pos_bits", "word_bits", "best_epoch", "scale_bits",
                                  "unigram_bits", "pos_marginal_bits")}
            res["runs"][name]["sec"] = round(dt, 1)
            print(f"  pos {ev.get('pos_bits')}  ({dt:.0f}s)", flush=True)
        json.dump(res, open(OUT / "runs.json", "w"), ensure_ascii=False, indent=1)
    print("すべて完了", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""掃き出しの再計算 R1–R3(2026-10-18 事前登録)。

Track F のファイルは読むだけ。出力は results/sweep_posbins_20261018/ にのみ書く。
R1: train_grammar.py(変更なしのコピー)を原/補正の対で
R2: 10/09 の保存済みモデルで diagnose.py と fixplan_check.py を対で
R3: panel_composition.py を shim で原/補正の対で走らせ、型ラベルの ARI を測る
"""
import json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TRACKF = Path("/home/sh1/deepl/lineart-stroke-grammar")
POSB = ROOT / "results/posbins_20261009"
OUT = ROOT / "results/sweep_posbins_20261018"
PY = sys.executable
CORP = {"orig": str(TRACKF / "results/grammar_corpus_20260920/corpus.npz"),
        "fixed": str(POSB / "corpus_fixed.npz")}


def run(cmd, log, cwd=ROOT):
    t0 = time.time()
    with open(log, "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(cwd))
    return r.returncode, time.time() - t0


R3_SHIM = """
import sys
from pathlib import Path
sys.path.insert(0, {stroke!r})
import panel_composition as M
M.CORPUS = {corpus!r}
M.OUT = Path({out!r})
M.KS = (12,)
M.main()
"""


def r3():
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in ("orig", "fixed"):
        o = OUT / f"r3_composition_{arm}"
        code = R3_SHIM.format(stroke=str(TRACKF / "tools/stroke"), corpus=CORP[arm], out=str(o))
        rc, dt = run([PY, "-c", code], OUT / f"r3_composition_{arm}.log", cwd=TRACKF)
        print(f"r3 {arm}: rc={rc} {dt:.0f}s")
        if rc:
            print((OUT / f"r3_composition_{arm}.log").read_text()[-2000:]); sys.exit(1)


LAB_OLD = str(TRACKF / "results/panel_composition_20260921/labels_k12.npy")
LAB_NEW = str(OUT / "r3_composition_fixed/labels_k12.npy")


def r4():
    """型条件つき cloze を、補正corpus の上で旧ラベル/新ラベルの対で(2026-10-18 事前登録)"""
    import numpy as np
    lab_new = np.load(LAB_NEW)
    assert lab_new.shape == (5345,) and lab_new.max() + 1 == 12, lab_new.shape   # IC2
    for arm, lab in (("oldlab", LAB_OLD), ("newlab", LAB_NEW)):
        o = OUT / f"r4_cloze_fixed_{arm}"
        cmd = [PY, str(HERE.parent / "trackf/train_cloze.py"),
               "--data", CORP["fixed"], "--labels", lab, "--out", str(o)]
        rc, dt = run(cmd, OUT / f"r4_cloze_fixed_{arm}.log")
        ev = json.load(open(o / "eval.json")) if (o / "eval.json").exists() else None
        print(f"r4 {arm}: rc={rc} {dt:.0f}s "
              f"pos {ev['pos_bits']:.6f} word {ev['word_bits']:.6f}" if ev else f"r4 {arm}: rc={rc}")
        if rc:
            print((OUT / f"r4_cloze_fixed_{arm}.log").read_text()[-2000:]); sys.exit(1)


FIXPLAN_SHIM = """
import sys
sys.path.insert(0, {gen!r})
import common, fixplan_check as M
common.CORPUS = M.CORPUS = {corpus!r}
sys.argv = ['x', '--s2_ckpt', {ckpt!r}, '--out', {out!r}]
M.main()
"""


def r2():
    """診断値(G5)と sample_pos の差(G6)を、原/補正の対で(2026-10-18 事前登録)"""
    OUT.mkdir(parents=True, exist_ok=True)
    # IC1: 09-21 の ckpt + 原corpus で記録値を再現する
    jobs = [("record", "results/gen_smoke_20260921/set/gen_best.pt",
             "results/gen_smoke_20260921/twostage/stage2/gen_best.pt", CORP["orig"])]
    for arm in ("orig", "fixed"):
        jobs.append((arm, str(POSB / f"train_set_{arm}/gen_best.pt"),
                     str(POSB / f"train_2stage_{arm}/stage2/gen_best.pt"), CORP[arm]))
    for name, sck, s2ck, corpus in jobs:
        o = OUT / f"r2_diagnose_{name}.json"
        cmd = [PY, str(HERE / "diagnose.py"), "--set_ckpt", sck, "--s2_ckpt", s2ck,
               "--corpus", corpus, "--out", str(o)]
        rc, dt = run(cmd, OUT / f"r2_diagnose_{name}.log")
        print(f"r2 diagnose {name}: rc={rc} {dt:.0f}s")
        if rc:
            print((OUT / f"r2_diagnose_{name}.log").read_text()[-2000:]); sys.exit(1)
    for arm in ("orig", "fixed"):
        o = OUT / f"r2_fixplan_{arm}"
        code = FIXPLAN_SHIM.format(gen=str(HERE), corpus=CORP[arm],
                                   ckpt=str(POSB / f"train_2stage_{arm}/stage2/gen_best.pt"),
                                   out=str(o))
        rc, dt = run([PY, "-c", code], OUT / f"r2_fixplan_{arm}.log")
        print(f"r2 fixplan {arm}: rc={rc} {dt:.0f}s")
        if rc:
            print((OUT / f"r2_fixplan_{arm}.log").read_text()[-2000:]); sys.exit(1)


if __name__ == "__main__":
    {"r2": r2, "r3": r3, "r4": r4}[sys.argv[1]]()

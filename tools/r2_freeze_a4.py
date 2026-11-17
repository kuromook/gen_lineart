# -*- coding: utf-8 -*-
"""登録 R2(config/prereg_r2_name_ngram_3class_20261117.md)のモデルを凍らせる。

学習 = 便A の主81行 + R1b の学習244行(E の a v7 の門・空・重なりを外した後)。3つの段。
IC1(R1b の 81行で A4 が 75/81)を確かめてから、A4・A2 の埋め・A5・NC1・NC2 を凍らせて
results/r2/a4_frozen_20261117.pkl に保存する(追跡しない)。層名は印字しない。
使い方: R1_MERGE=1 ./venv/bin/python tools/r2_freeze_a4.py
"""
import contextlib, collections, csv, hashlib, io, os, pickle, random, sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, os.path.dirname(__file__))
import r1_stage_predictor as R  # noqa: E402

AGATE = "../lineart-census/data/census_full/a_verdict_v7_20261116.csv"
OUT = "results/r2/a4_frozen_20261117.pkl"


def main():
    if not R.MERGE:
        print("R1_MERGE=1 で走らせる(3つの段)")
        return 1
    gate = {(r["page_path"], r["layer_id"]) for r in csv.DictReader(io.open(AGATE, encoding="utf-8"))
            if r["a_verdict_v7"] == "template_excluded"}
    rt, excl = R.load()
    allA = R.build_eval(rt, excl)
    units_a = {r["unit"] for r in allA}
    ev = [r for r in allA if r["key"] not in gate and not r["empty"] and not r["mayer"] and r["stage"] in R.ORDER]
    with contextlib.redirect_stdout(io.StringIO()):
        tr, drop, *_ = R.build_train(rt, excl, units_a)
    tr = [r for r in tr if r["key"] not in gate]
    print("a の門 %s sha256 %s / 便A 主 %d 行 / R1b 学習 %d 行" % (AGATE, R.sha(AGATE), len(ev), len(tr)))

    # IC1
    with contextlib.redirect_stdout(io.StringIO()):
        y = np.array([r["stage"] for r in tr]); g = np.array([r["unit"] for r in tr])
        f = R.Feats(tr, "A4"); m = R.fit_gb(f.X(tr), y, g, "A4")
    hit = sum(p == r["stage"] for p, r in zip(m.predict(f.X(ev)), ev))
    print("IC1: R1b の 81行で A4 = %d/%d(登録の値 75/81)→ %s" % (hit, len(ev), "一致" if (hit, len(ev)) == (75, 81) else "**不一致。凍らせない**"))
    if (hit, len(ev)) != (75, 81):
        return 1

    full = ev + tr
    y = np.array([r["stage"] for r in full]); g = np.array([r["unit"] for r in full])
    print("\n凍らせる学習: %d 行 / 作品フォルダー %d / 段 %s" % (len(full), len(set(g)), dict(collections.Counter(y))))
    f4, f5 = R.Feats(full, "A4"), R.Feats(full, "A5")
    m4 = R.fit_gb(f4.X(full), y, g, "A4")
    m5 = R.fit_gb(f5.X(full), y, g, "A5(記録だけ)")
    maj = collections.Counter(y).most_common(1)[0][0]
    fill = collections.Counter(r["stage"] for r in full if r["t"]["name_rule"] == "なし").most_common(1)[0][0]
    rng = random.Random(0); ysh = y.copy()
    for u in sorted(set(g)):
        idx = [i for i, x in enumerate(g) if x == u]
        vals = [ysh[i] for i in idx]; rng.shuffle(vals)
        for i, v in zip(idx, vals):
            ysh[i] = v
    nc1 = GradientBoostingClassifier(n_estimators=m4.n_estimators, max_depth=m4.max_depth, learning_rate=0.1,
                                     random_state=0).fit(f4.X(full), ysh)
    years = sorted({r["t"]["year_label"] for r in full})

    def fp(rows):
        return np.nan_to_num(np.array([[R.fnum(r["t"]["canvas_px_w"]), R.fnum(r["t"]["canvas_px_h"]),
                                        R.fnum(r["t"]["n_leaf_layers"])] + [1.0 if r["t"]["year_label"] == yy else 0.0
                                                                            for yy in years] for r in rows]), nan=-1.0)
    nc2 = R.fit_gb(fp(full), y, g, "NC2")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    blob = {"a4": m4, "a4_feats": f4, "a5": m5, "a5_feats": f5, "nc1": nc1, "nc2": nc2, "nc2_years": years,
            "a0_majority": maj, "a2_fill": fill, "train_units": sorted(set(g)), "train_keys": sorted(r["key"] for r in full),
            "order": R.ORDER}
    with open(OUT, "wb") as fh:
        pickle.dump(blob, fh)
    print("\nA0 の多数派 = %s / A2 の埋め = %s" % (maj, fill))
    print("学習の作品フォルダー: %s" % ", ".join(sorted(set(g))))
    print("凍らせた: %s sha256 %s" % (OUT, R.sha(OUT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""登録 R3(config/prereg_r3_pixel_line_features_20261119.md、追補1・2)の走行器具。

行 = 段のラベル付き 325行(R2 の学習と同じ。E の a v7 の門・空・mayer を外した後)。主は作者のコメント「資料」の行を外す。
列 = census 経由の線の質・線の数の表 v2(α ≥ 8 の B〜D が主)+ 全層の画素の表の数の列(R1 の NUM)。
分け方 = 作品フォルダーを1つずつ抜く(主)/ 年代札を1つずつ抜く(副)。層名は印字しない。
使い方: ./venv/bin/python tools/r3_pixel_features.py [--alpha0] [--with-shiryou] [--split year]
"""
import argparse, collections, contextlib, csv, io, math, os, random, sys

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, os.path.dirname(__file__))
os.environ.pop("R1_MERGE", None)
import r1_stage_predictor as R  # noqa: E402

AGATE = "../lineart-census/data/census_full/a_verdict_v7_20261116.csv"
LINE = "data/census/r_line/r_line_columns_v2_20261119.csv"
IM85 = "data/census/shiryou_folder_sketch_im85_two_axis_answers_20261118.csv"
ORDER3 = ("あたり+下絵", "清書", "段ではない")
LINECOUNT = "n_endpoint_raw_a8"  # 追補1・2 の IC-L で選んだ列(α ≥ 8 の IC でも同じ)

A_COLS = ["n_drawn", "n_drawn_a8", "a_1_7", "a_8_25", "a_26_63", "a_64_127", "a_128_199", "a_200_254", "a_255",
          "gray_mean_a0", "gray_mean_a8", "pure_black_frac_a0", "pure_black_frac_a8", "sat_mean_a0", "sat_mean_a8"]
BD = ["w_med", "w_p10", "w_p90", "skel_len_full", "n_seg", "len_p50", "len_p90", "ori_entropy", "par_frac",
      "n_endpoint", "n_junction", "n_branch", "n_cc", "n_endpoint_raw", "n_junction_raw", "n_branch_raw", "n_spur_cut"]
COUNTS = ["skel_len_full", "n_seg", "n_endpoint", "n_junction", "n_branch", "n_cc", "n_endpoint_raw",
          "n_junction_raw", "n_branch_raw", "n_spur_cut"]
BANNED = {"name", "ancestors", "name_rule", "v_layer_kind", "layer_name"}  # 名前の文字の列(完全一致)。祖先の表示・マスクの数は構造の列なので可


def m3(s):
    return "あたり+下絵" if s in ("あたり", "下絵") else s


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_rows():
    gate = {(r["page_path"], r["layer_id"]) for r in csv.DictReader(io.open(AGATE, encoding="utf-8"))
            if r["a_verdict_v7"] == "template_excluded"}
    with contextlib.redirect_stdout(io.StringIO()):
        rt, excl = R.load()
        allA = R.build_eval(rt, excl)
        units = {r["unit"] for r in allA}
        tr, *_ = R.build_train(rt, excl, units)
    ev = [r for r in allA if r["key"] not in gate and not r["empty"] and not r["mayer"] and r["stage"] in R.ORDER4]
    tr = [r for r in tr if r["key"] not in gate]
    return ev + tr


def pixel_names(suffix):
    n = list(A_COLS) + [c + suffix for c in BD]
    n += [c + suffix + "/n_drawn_a8" for c in COUNTS] + [c + suffix + "/bbox_area" for c in COUNTS]
    return n + ["census:" + c for c in R.NUM_NAMES]


def pixel_vec(r, line, suffix):
    L = line[r["key"]]
    t = r["t"]
    bb = (f(t["bbox_x1"]) - f(t["bbox_x0"])) * (f(t["bbox_y1"]) - f(t["bbox_y0"]))
    nd = f(L["n_drawn_a8"])
    v = [f(L[c]) for c in A_COLS] + [f(L[c + suffix]) for c in BD]
    v += [f(L[c + suffix]) / nd if nd else float("nan") for c in COUNTS]
    v += [f(L[c + suffix]) / bb if bb and bb > 0 else float("nan") for c in COUNTS]
    return v + R.numeric(t)


def fp_vec(r, years):
    t = r["t"]
    return [f(t["canvas_px_w"]), f(t["canvas_px_h"]), f(t["n_leaf_layers"])] + [1.0 if t["year_label"] == y else 0.0
                                                                                 for y in years]


def size_vec(r, line):
    t = r["t"]
    bb = (f(t["bbox_x1"]) - f(t["bbox_x0"])) * (f(t["bbox_y1"]) - f(t["bbox_y0"]))
    return [f(line[r["key"]]["n_drawn_a8"]), bb]


def arr(v):
    return np.nan_to_num(np.array(v, dtype=float), nan=-1.0, posinf=1e12, neginf=-1e12)


def quiet_fit(X, y, g):
    with contextlib.redirect_stdout(io.StringIO()):
        return R.fit_gb(X, y, g, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha0", action="store_true", help="B〜D を α > 0 の列で(副)")
    ap.add_argument("--with-shiryou", action="store_true", help="資料の13行を外さない(副)")
    ap.add_argument("--split", default="unit", choices=("unit", "year"))
    a = ap.parse_args()
    suffix = "" if a.alpha0 else "_a8"
    print("登録 R3 / B〜D の列: %s / 資料の行: %s / 分け方: %s" % (
        "α > 0(副)" if a.alpha0 else "α ≥ 8(主)", "入れる(副)" if a.with_shiryou else "外す(主)",
        "作品フォルダーを1つずつ抜く(主)" if a.split == "unit" else "年代札を1つずつ抜く(副)"))
    print("a の門 %s sha256 %s / 列の表 %s sha256 %s" % (AGATE, R.sha(AGATE), LINE, R.sha(LINE)))

    rows = load_rows()
    line = {}
    sets = collections.Counter()
    for L in csv.DictReader(io.open(LINE, encoding="utf-8")):
        line[(L["page_path"], L["layer_id"])] = L
        sets[L["set"]] += 1
    print("\nIC2: 列の表 %d 行(組 %s)" % (len(line), dict(sets)))
    miss = [r for r in rows if r["key"] not in line]
    print("   ラベルの %d 行のうち列の表に無い行: %d" % (len(rows), len(miss)))
    print("   4段の内訳: %s" % dict(collections.Counter(r["stage"] for r in rows)))
    shir = [r for r in rows if "資料" in (r["t"].get("author_comment") or "")]
    print("   作者のコメント「資料」の行: %d(段 %s)→ %s" % (len(shir), dict(collections.Counter(r["stage"] for r in shir)),
                                                      "入れる" if a.with_shiryou else "外す"))
    if not a.with_shiryou:
        rows = [r for r in rows if r not in shir]
    rows = [r for r in rows if r["key"] in line]
    print("   読む行: %d" % len(rows))

    names = pixel_names(suffix)
    bad = [n for n in names if n.split(":")[-1] in BANNED or n.startswith("ngram:")]
    print("\nNC4(禁止の列): P2 の列 %d 本のうち、名前・祖先・name_rule・v_layer_kind から来た列 %d 本 %s"
          % (len(names), len(bad), "→ 合格" if not bad else "→ **不合格** %s" % bad))
    if bad:
        return 1

    y3 = np.array([m3(r["stage"]) for r in rows])
    groups = np.array([r["unit"] if a.split == "unit" else r["t"]["year_label"] for r in rows])
    Xp = arr([pixel_vec(r, line, suffix) for r in rows])
    years = sorted({r["t"]["year_label"] for r in rows})
    Xfp = arr([fp_vec(r, years) for r in rows])
    Xsz = arr([size_vec(r, line) for r in rows])
    nr = [r["t"]["name_rule"] for r in rows]

    arms = ("P0 多数派", "P1 名前+埋め", "P2 画素だけ", "P3 名前の文字+画素", "A4 名前の文字だけ",
            "NC1 ラベル並べ替え", "NC2 作品の指紋", "NC3 大きさだけ")
    pred = {k: [None] * len(rows) for k in arms}
    folds = sorted(set(groups))
    print("\n折り: %d(%s)" % (len(folds), "作品フォルダー" if a.split == "unit" else "年代札"))
    for gi, gname in enumerate(folds):
        te = [i for i in range(len(rows)) if groups[i] == gname]
        tr = [i for i in range(len(rows)) if groups[i] != gname]
        trr = [rows[i] for i in tr]
        ytr = y3[tr]
        gtr = np.array([rows[i]["unit"] for i in tr])  # 内側の選びはいつも作品フォルダー別
        maj = collections.Counter(ytr).most_common(1)[0][0]
        fill = collections.Counter(y3[i] for i in tr if nr[i] == "なし").most_common(1)
        fill = fill[0][0] if fill else maj
        for i in te:
            pred["P0 多数派"][i] = maj
            pred["P1 名前+埋め"][i] = m3(nr[i]) if nr[i] != "なし" else fill
        m = quiet_fit(Xp[tr], ytr, gtr)
        for i, p in zip(te, m.predict(Xp[te])):
            pred["P2 画素だけ"][i] = p
        fa = R.Feats(trr, "A4")
        Xa_tr, Xa_te = fa.X(trr), fa.X([rows[i] for i in te])
        m = quiet_fit(Xa_tr, ytr, gtr)
        for i, p in zip(te, m.predict(Xa_te)):
            pred["A4 名前の文字だけ"][i] = p
        m = quiet_fit(np.hstack([Xa_tr, Xp[tr]]), ytr, gtr)
        for i, p in zip(te, m.predict(np.hstack([Xa_te, Xp[te]]))):
            pred["P3 名前の文字+画素"][i] = p
        rng = random.Random(gi)
        ysh = ytr.copy()
        for u in sorted(set(gtr)):
            idx = [j for j, x in enumerate(gtr) if x == u]
            vals = [ysh[j] for j in idx]
            rng.shuffle(vals)
            for j, v in zip(idx, vals):
                ysh[j] = v
        m = quiet_fit(Xp[tr], ysh, gtr)
        for i, p in zip(te, m.predict(Xp[te])):
            pred["NC1 ラベル並べ替え"][i] = p
        m = quiet_fit(Xfp[tr], ytr, gtr)
        for i, p in zip(te, m.predict(Xfp[te])):
            pred["NC2 作品の指紋"][i] = p
        m = quiet_fit(Xsz[tr], ytr, gtr)
        for i, p in zip(te, m.predict(Xsz[te])):
            pred["NC3 大きさだけ"][i] = p

    def hits(k, idx):
        return [pred[k][i] == y3[i] for i in idx]

    all_i = list(range(len(rows)))
    two = [i for i in all_i if y3[i] in ("あたり+下絵", "清書")]
    nashi2 = [i for i in two if nr[i] == "なし"]
    print("\n== 3段の全 %d 行(記述)" % len(all_i))
    for k in arms:
        print("   %-14s %.4f" % (k, np.mean(hits(k, all_i))))
    print("IC1(器具の印): A4 名前の文字だけ、この折りで %d/%d" % (sum(hits("A4 名前の文字だけ", all_i)), len(all_i)))

    print("\n== C1(主): 名前に手がかりの無い2クラスの行 %d(%s)" % (
        len(nashi2), dict(collections.Counter(y3[i] for i in nashi2))))
    const = [y3[i] == "あたり+下絵" for i in nashi2]
    print("   定数「あたり+下絵」 %d/%d = %.4f" % (sum(const), len(nashi2), np.mean(const)))
    for k in arms:
        h = hits(k, nashi2)
        print("   %-14s %d/%d = %.4f  予測 %s" % (k, sum(h), len(h), np.mean(h),
                                                dict(collections.Counter(pred[k][i] for i in nashi2))))
    w, l, p = R.sign_p(hits("P2 画素だけ", nashi2), const)
    c1 = sum(hits("P2 画素だけ", nashi2)) > sum(const) and p < 0.05
    print("   P2 対 定数: P2 だけ当たり %d / 定数だけ %d / 片側 p = %.4f → C1 %s" % (w, l, p, "通る" if c1 else "届かない"))
    for k in ("NC1 ラベル並べ替え", "NC2 作品の指紋", "NC3 大きさだけ"):
        print("   %s は定数を%s(%d 対 %d)" % (k, "**超えた**" if sum(hits(k, nashi2)) > sum(const) else "超えない",
                                        sum(hits(k, nashi2)), sum(const)))

    print("\n== C2: 2クラスの行 %d(%s)" % (len(two), dict(collections.Counter(y3[i] for i in two))))
    for k in arms:
        print("   %-14s %.4f" % (k, np.mean(hits(k, two))))
    w, l, p = R.sign_p(hits("P3 名前の文字+画素", two), hits("A4 名前の文字だけ", two))
    c2 = sum(hits("P3 名前の文字+画素", two)) > sum(hits("A4 名前の文字だけ", two)) and p < 0.05
    print("   P3 対 A4: P3 だけ %d / A4 だけ %d / 片側 p = %.4f → C2 %s" % (w, l, p, "通る" if c2 else "届かない"))

    print("\n== 段ごと(2クラスの行と全行、P2・P3・A4)")
    for k in ("P2 画素だけ", "P3 名前の文字+画素", "A4 名前の文字だけ"):
        for s in ORDER3:
            na = sum(1 for i in all_i if y3[i] == s)
            npd = sum(1 for i in all_i if pred[k][i] == s)
            tp = sum(1 for i in all_i if pred[k][i] == s and y3[i] == s)
            print("   %s %s: 的中 %s / 取りこぼし %s(作者 %d・予測 %d)" % (
                k, s, "%.4f" % (tp / npd) if npd else "判定保留(母数0)", "%.4f" % ((na - tp) / na) if na else "-",
                na, npd))

    print("\n== 作品フォルダーごと(2クラスの行、P2 / A4 / 定数あたり+下絵)")
    for u in sorted({rows[i]["unit"] for i in two}):
        ii = [i for i in two if rows[i]["unit"] == u]
        print("   %-24s %2d 行  P2 %d  A4 %d  定数 %d  なし %d" % (u, len(ii), sum(hits("P2 画素だけ", ii)),
                                                           sum(hits("A4 名前の文字だけ", ii)),
                                                           sum(y3[i] == "あたり+下絵" for i in ii),
                                                           sum(nr[i] == "なし" for i in ii)))
    print("\n== 年代札ごと(2クラスの行、P2 / A4、プールと並べる。平均しない)")
    for yy in years:
        ii = [i for i in two if rows[i]["t"]["year_label"] == yy]
        if ii:
            print("   %-6s %2d 行  P2 %.4f  A4 %.4f" % (yy, len(ii), np.mean(hits("P2 画素だけ", ii)),
                                                    np.mean(hits("A4 名前の文字だけ", ii))))

    print("\n== C3(副): あたり 対 下絵、線の数 = %s" % (LINECOUNT if not a.alpha0 else "n_endpoint_raw(α > 0)"))
    lc = LINECOUNT if not a.alpha0 else "n_endpoint_raw"
    ata = [r for r in rows if r["stage"] == "あたり"]
    sht = [r for r in rows if r["stage"] == "下絵"]
    print("   あたり %d 行 / 下絵 %d 行 / 作品フォルダー %d" % (len(ata), len(sht), len({r["unit"] for r in ata + sht})))

    def bb(r):
        t = r["t"]
        return (f(t["bbox_x1"]) - f(t["bbox_x0"])) * (f(t["bbox_y1"]) - f(t["bbox_y0"]))
    for lab, fn in (("生の値(棒)", lambda r: f(line[r["key"]][lc])),
                    ("÷外接矩形の面積", lambda r: f(line[r["key"]][lc]) / bb(r) if bb(r) > 0 else float("nan")),
                    ("÷n_drawn_a8", lambda r: f(line[r["key"]][lc]) / f(line[r["key"]]["n_drawn_a8"])),
                    ("線幅 w_med%s(描き重ねの印)" % suffix, lambda r: f(line[r["key"]]["w_med" + suffix]))):
        xs = [fn(r) for r in sht]
        xa = [fn(r) for r in ata]
        res = mannwhitneyu(xs, xa, alternative="greater")
        auc = res.statistic / (len(xs) * len(xa))
        extra = ""
        if lab.startswith("生の値"):
            c3 = res.pvalue < 0.05 and auc >= 0.70
            extra = " → C3 %s" % ("通る" if c3 else "届かない")
        print("   %-24s 下絵の中央 %.4g / あたりの中央 %.4g / 片側 MWU p = %.4f / AUC %.4f%s" % (
            lab, np.median(xs), np.median(xa), res.pvalue, auc, extra))
    up = dn = 0
    for u in sorted({r["unit"] for r in ata} & {r["unit"] for r in sht}):
        ms = np.median([f(line[r["key"]][lc]) for r in sht if r["unit"] == u])
        ma = np.median([f(line[r["key"]]["" + lc]) for r in ata if r["unit"] == u])
        up += ms > ma
        dn += ms < ma
    print("   同じ作品に両方が在る作品: 下絵の方が多い %d / あたりの方が多い %d" % (up, dn))

    if not a.alpha0 and not a.with_shiryou and a.split == "unit":
        print("\n== 記録(棒ではない): im85 の68層(見た目 = ラフ・下絵)に、読む行の全部で学んだ P2 を当てる")
        im = [r for r in csv.DictReader(io.open(IM85, encoding="utf-8"))]
        L68 = {(r["page_path"], r["layer_id"]) for r in im}
        print("   列の表に在る: %d / %d" % (sum(k in line for k in L68), len(L68)))
        m = quiet_fit(Xp, y3, np.array([r["unit"] for r in rows]))
        # im85 の層は層の役割の表の行を持たないことがあるので、census の数の列は -1 で埋める
        Xi = []
        for k in sorted(L68):
            Lr = line[k]
            nd = f(Lr["n_drawn_a8"])
            v = [f(Lr[c]) for c in A_COLS] + [f(Lr[c + suffix]) for c in BD]
            v += [f(Lr[c + suffix]) / nd if nd else float("nan") for c in COUNTS]
            v += [float("nan")] * len(COUNTS) + [float("nan")] * len(R.NUM_NAMES)
            Xi.append(v)
        pi = m.predict(arr(Xi))
        print("   予測: %s(外接矩形と census の数の列は無い扱い。代用の値なので参考)" % dict(collections.Counter(pi)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

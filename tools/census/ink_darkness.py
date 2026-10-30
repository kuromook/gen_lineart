#!/usr/bin/env python
"""墨の濃さ(1段目・2段目)を**同じ述語で畳む。**

**事前登録は `doc/prereg_ink_darkness_20261030.md`**(2026-10-30 17:55、結果の到着前)。
**主値は3つだけ**: 鏡像のずれ / 墨のうち純黒 / `alpha=255` の割合。
**墨の輝度の「中央値」は主値にしない** —— 材料は16刻みのヒストグラムで、
**中央値そのものは入っていない**(ビンの名前までしか書けない)。

**1段目の表も、この器具で作り直す。** 2026-10-30 には手で書いた ——
**公開中の値をこの器具が1セルも違わず再現するかを、走行のたびに確かめる**
(再現しなければ止まる)。**畳み方が変わったことに後から気づけないのが怖いので。**

**work_id は匿名の番号**(条例10)。**1段目の W1〜W6 は動かさない** ——
**公開済みの行の名札を動かすと、過去の報告と食い違う。** 2段目は W7 から振る。
対応表は `data/census_full/ink_darkness_work_ids.csv`(`.gitignore` 配下)。

    ./venv/bin/python tools/census/ink_darkness.py
"""
import collections
import csv
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGE1 = "inbox/ink_darkness_stage1_20261030.csv"
STAGE2_GLOB = "inbox/ink_darkness_stage2_*_20261031.csv"
NBIN = 16

# **1段目で出た帯**(事前登録 §3。**形式2 の行**)。**結果を見てから動かさない**
BAND_PURE_BLACK = (0.5262, 0.9712)
MIRROR_BAR = 0.02   # **事前登録で決めた境**(1段目の最大 1.75% の上の切りのよい値)


def load(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def hist(s):
    if not s:
        return [0] * NBIN
    out = [int(x) for x in s.split(";")]
    assert len(out) == NBIN, s[:40]
    return out


def mirror_parts(gray, alpha, n):
    """**ずれを3つに分ける。** 同じ「ずれ」に少なくとも2つの機序が在るため:

    - **白のビン**(`i = 15`): **alpha 255 で白く描いた画素**は、輝度が 240〜255 に入るが
      alpha は 240〜255 のままなので、鏡像(alpha 0〜15)と合わない。
      **抽出側の言明: ずれた 282層のうち 227層は `alpha=255` の輝度の中央が 255(白)。**
      **つまり「絵の具が灰色」ではなく「白い絵の具」でもこの量は増える**
    - **黒のビン**(`i = 0`): 純黒の画素と alpha 240〜255 の数の差
    - **中のビン**(`i = 1〜14`): **ここが大きいときだけ「絵の具自体が灰色」に近づく**

    **どれも「絵の具の色」を直接測っていない**(合成後の輝度しか無い)。
    """
    if not n:
        return (None, None, None)
    w = abs(gray[15] - alpha[0]) / n
    b = abs(gray[0] - alpha[15]) / n
    mid = sum(abs(gray[i] - alpha[NBIN - 1 - i]) for i in range(1, 15)) / n
    return (w, b, mid)


def mirror_deviation(gray, alpha, n_alpha_gt0):
    """**絵の具が純黒なら、白の上では `gray = 255 - alpha`** ——
    輝度ヒストと alpha ヒストは**鏡像**になる。ずれは割合で返す。

    **届かない所**: 鏡像であることは**絵の具が純黒であることと同じではない** ——
    **純黒の絵の具でアルファを振ったもの**と、**灰色の絵の具を不透明度で薄めたもの**は、
    ヒストグラムの形では分けられない場合がある(層の不透明度の列を隣に置く)。
    """
    if not n_alpha_gt0:
        return None
    d = sum(abs(gray[i] - alpha[NBIN - 1 - i]) for i in range(NBIN))
    return d / n_alpha_gt0


def median_bin(gray):
    """**墨画素(gray < 128 = ビン 0〜7)の中央が入るビン。** 中央値そのものではない。"""
    ink = gray[:8]
    tot = sum(ink)
    if not tot:
        return ""
    half, run = tot / 2.0, 0
    for i, c in enumerate(ink):
        run += c
        if run >= half:
            return f"{i * NBIN}-{i * NBIN + NBIN - 1}"
    return ""


def fmt_label(formats):
    """**画素の形式の名札。** 畳んだ行は `+` で並べる(`2+5`)。"""
    fs = sorted(formats)
    return "+".join(fs) if fs else "なし"


def fold(rows, work_id, year_label):
    """1つの群(作品 × 層の種類 × 画素形式)を1行にする。**プールしない単位で呼ぶ。**"""
    g = dict(n_layers=len(rows),
             n_layers_with_pixels=sum(1 for r in rows if int(r["n_px_alpha_gt0"] or 0)))
    for col in ("n_px_alpha_gt0", "n_px_alpha_255", "n_px_ink", "n_px_pure_black"):
        g[col] = sum(int(r[col] or 0) for r in rows)
    gray = [0] * NBIN
    alpha = [0] * NBIN
    for r in rows:
        for i, v in enumerate(hist(r["hist_gray_alpha_gt0"])):
            gray[i] += v
        for i, v in enumerate(hist(r["hist_alpha"])):
            alpha[i] += v
    dev = mirror_deviation(gray, alpha, g["n_px_alpha_gt0"])
    w, b, mid = mirror_parts(gray, alpha, g["n_px_alpha_gt0"])
    return [work_id, year_label, rows[0]["stratum"],
            fmt_label({r["pixel_formats"] for r in rows}),
            g["n_layers"], g["n_layers_with_pixels"],
            g["n_px_alpha_gt0"], g["n_px_alpha_255"], g["n_px_ink"],
            g["n_px_pure_black"],
            f"{g['n_px_alpha_255'] / g['n_px_alpha_gt0']:.4f}"
            if g["n_px_alpha_gt0"] else "",
            f"{g['n_px_pure_black'] / g['n_px_ink']:.4f}" if g["n_px_ink"] else "",
            median_bin(gray),
            f"{dev:.4f}" if dev is not None else "",
            f"{w:.4f}" if w is not None else "",
            f"{b:.4f}" if b is not None else "",
            f"{mid:.4f}" if mid is not None else "",
            ";".join(str(x) for x in gray), ";".join(str(x) for x in alpha)]


HEADER = ["work_id", "year_label", "stratum", "pixel_format",
          "n_layers", "n_layers_with_pixels", "n_px_alpha_gt0", "n_px_alpha_255",
          "n_px_ink", "n_px_pure_black", "frac_alpha255",
          "frac_pure_black_among_ink", "median_gray_ink_bin", "mirror_deviation",
          "mirror_dev_white_bin", "mirror_dev_black_bin", "mirror_dev_mid_bins",
          "hist_gray_alpha_gt0", "hist_alpha"]


def build(rows, ids, pooled_year_rows):
    """**作品ごとの行**と、**畳んだ行**を作る。畳んだ行は `work_id = ALL`。"""
    out = []
    by_work = collections.defaultdict(list)
    for r in rows:
        by_work[r["work_dir"]].append(r)
    for wd in sorted(by_work, key=lambda w: (by_work[w][0]["year_label"], w)):
        sel = by_work[wd]
        yl = sel[0]["year_label"]
        for stratum in ("L1", "C1", "C2"):
            s_rows = [r for r in sel if r["stratum"] == stratum]
            if not s_rows:
                continue
            for f in sorted({r["pixel_formats"] for r in s_rows}):
                out.append(fold([r for r in s_rows if r["pixel_formats"] == f],
                                ids[wd], yl))
    # **札ごと**(世代もプールだが、**札は世代の単位なので畳んでよい**。
    # **札をまたいで足した行だけが「プール」である** —— 名札で見分けられるようにする)
    if pooled_year_rows:
        for yl in sorted({r["year_label"] for r in rows}):
            sel = [r for r in rows if r["year_label"] == yl]
            for stratum in ("L1", "C1", "C2"):
                s_rows = [r for r in sel if r["stratum"] == stratum]
                if not s_rows:
                    continue
                # **形式ごとにも出す** —— **形式5 は読みに疑いが在る**ので、
                # **外した数と入れた数の両方が読めるようにする**(条例4)
                formats = {r["pixel_formats"] for r in s_rows}
                if len(formats) > 1:
                    for f in sorted(formats):
                        out.append(fold([r for r in s_rows
                                         if r["pixel_formats"] == f], "ALL", yl))
                out.append(fold(s_rows, "ALL", yl))
    # **全部を足した行。** `work_id = ALL` の行は**プールである** ——
    # **形式ごと / 数字の形式だけ / 全部** の3段を出すので、
    # **形式5の読みの疑いを外した数と、入れた数の両方が読める**(条例4)
    for stratum in ("L1", "C1", "C2"):
        s_rows = [r for r in rows if r["stratum"] == stratum]
        if not s_rows:
            continue
        formats = {r["pixel_formats"] for r in s_rows}
        for f in sorted(formats):
            out.append(fold([r for r in s_rows if r["pixel_formats"] == f],
                            "ALL", ""))
        num = [r for r in s_rows if r["pixel_formats"].isdigit()]
        if num and len({r["pixel_formats"] for r in num}) > 1:
            out.append(fold(num, "ALL", ""))
        if len(formats) > 1:
            out.append(fold(s_rows, "ALL", ""))
    return out


# **公開中の1段目の表と、この器具の出力が違ってよい所。** これ以外が違えば止める。
# **2026-10-31 に見つけた1点**: 手で書いた表は、**用紙の層(墨画素が1つも無い)にも
# `median_gray_ink_bin = 0-15` を印字していた** —— **墨画素が 0 の群で「墨の輝度の中央」は
# 定義できない。** 空欄が正しい(**0 は「数えて 0」ではなく「材料が無い」**。T126 の家族)。
# **数の列は1つも動いていない** —— 動いたのはこの1列だけである
EXPECTED_FIX = {("C1", "用紙の色", "median_gray_ink_bin"): ("0-15", "")}


def compare_stage1(rows1) -> int:
    """**手で書いた表を、この器具が再現するか。** 宣言した1点以外が違えば止める。"""
    # **commit 済みの版と比べる。** 作業木のファイルと比べると、
    # **この器具が直前の走行で上書きしたものと比べることになり、点検が空になる**
    # (2026-10-31 に1度そうなった。**自分の出力と比べる点検は、何も見ていない**)
    out = subprocess.run(["git", "show", "HEAD:census/ink_darkness_stage1.csv"],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        print("  **公開済みの1段目の表が HEAD に無い。** 突き合わせを飛ばした")
        return 0
    raw = list(csv.reader(out.stdout.splitlines()))
    # **公開中のファイルの見出しで読む** —— 列を足したときに、
    # **並びで比べると全部の列がずれて「全部違う」と鳴る**(それでは何も分からない)
    cols = raw[0]
    prev = {(r[0], r[1], r[2], r[3]): dict(zip(cols, r)) for r in raw[1:]}
    now = {(str(r[0]), str(r[1]), str(r[2]), str(r[3])):
           {k: str(v) for k, v in zip(HEADER, r)} for r in rows1}
    bad, added, fixed = [], [], 0
    for k in sorted(set(prev) | set(now)):
        a, b = prev.get(k), now.get(k)
        if a is None:
            # **足した行は通す。** 守りたいのは**公開済みの数が消えたり動いたりしないこと**
            added.append(k)
            continue
        if b is None:
            bad.append((k, "**この器具が作らない行**(公開済みの行が消える)", ""))
            continue
        for col in cols:
            x, y = a[col], b.get(col, "(この器具に無い列)")
            if x == y:
                continue
            exp = EXPECTED_FIX.get((k[2], k[3], col))
            if exp == (x, y):
                fixed += 1
            else:
                bad.append((k, col, f"公開中 {x!r} / いま {y!r}"))
    if bad:
        print("  **1段目の表が再現しない(宣言していない違い)**:")
        for k, col, detail in bad[:20]:
            print(f"    {k} {col} {detail}")
        return 1
    print(f"  1段目: **公開中の表を再現した**"
          f"(**宣言した直し {fixed} セル**: 墨画素 0 の群の `median_gray_ink_bin` /"
          f" **足した行 {len(added)}**。消えた行・動いた数は 0)")
    return 0


def main() -> int:
    s1 = load(os.path.join(ROOT, STAGE1))
    s2 = []
    for p in sorted(glob.glob(os.path.join(ROOT, STAGE2_GLOB))):
        s2 += load(p)
    print(f"  1段目 {len(s1)}行 / 2段目 {len(s2)}行")
    # **落としたものを理由別に数える**(規律9)
    for name, rows in (("1段目", s1), ("2段目", s2)):
        st = collections.Counter(r["status"] for r in rows)
        empty = sum(1 for r in rows if not int(r["n_px_alpha_gt0"] or 0))
        print(f"    {name}: status {dict(st)} / `alpha>0` が 0 の層 {empty}")

    ids = {}
    for i, wd in enumerate(sorted({r["work_dir"] for r in s1},
                                  key=lambda w: (w.split("\\")[0], w)), start=1):
        ids[wd] = f"W{i}"
    n1 = len(ids)
    for i, wd in enumerate(sorted({r["work_dir"] for r in s2},
                                  key=lambda w: (w.split("\\")[0], w)), start=n1 + 1):
        ids[wd] = f"W{i}"
    print(f"  作品フォルダー: 1段目 {n1} / 2段目 {len(ids) - n1}"
          f"(**1段目の名札は動かさない**)")

    rows1 = build(s1, ids, pooled_year_rows=False)
    # **公開中の1段目の表を1セルずつ再現するかを確かめる**(手で書いた表との突き合わせ)
    if compare_stage1(rows1):
        return 1

    rows2 = build(s2, ids, pooled_year_rows=True)
    n = bc.write_csv(os.path.join(ROOT, "census", "ink_darkness_stage2.csv"),
                     HEADER, rows2)
    bc.write_csv(os.path.join(ROOT, "census", "ink_darkness_stage1.csv"),
                 HEADER, rows1)
    with open(os.path.join(ROOT, "data", "census_full",
                           "ink_darkness_work_ids.csv"), "w",
              encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["work_id", "work_dir", "stage"])
        for wd, wid in sorted(ids.items(), key=lambda kv: int(kv[1][1:])):
            w.writerow([wid, wd, "1" if int(wid[1:]) <= n1 else "2"])
    nw = write_white_layers(s2)
    print(f"  ink_darkness_stage2.csv {n}行 / "
          f"ink_darkness_white_layers.csv {nw}行 / 対応表は `data/census_full/` 側")
    read_prereg(rows1, rows2)
    return 0


WHITE_BAR = 240   # **白と呼ぶ輝度。** 器具の背景の閾(> 220)より厳しい側に置いた


def write_white_layers(rows):
    """**線画の述語が拾った「白い層」を数える。**

    **鏡像のずれの正体はこれである** —— **`alpha = 255` の画素の輝度の中央が白**の層。
    **絵の具が灰色なのではなく、白で描いた(または塗りつぶした)層が、
    名前の述語(`線画`/`主線`/`lineart`)に拾われている。**

    **これは述語についての測定であって、工程についての測定ではない**(条例6)——
    **「作者が白で修正した」とは書かない。** 読めるのは
    **「この名前の層の中に、全不透明で白い画素が多い層が在る」**までである。
    """
    out = []
    l1 = [r for r in rows
          if r["stratum"] == "L1" and int(r["n_px_alpha_255"] or 0) > 0]
    for yl in sorted({r["year_label"] for r in l1}) + ["ALL"]:
        sel = [r for r in l1 if yl == "ALL" or r["year_label"] == yl]
        for fmt in sorted({r["pixel_formats"] for r in sel}) + ["ALL"]:
            sub = [r for r in sel if fmt == "ALL" or r["pixel_formats"] == fmt]
            if not sub:
                continue
            mid = [r for r in sub if r["median_gray_at_alpha255"] not in ("", "不明")]
            white = [r for r in mid
                     if float(r["median_gray_at_alpha255"]) >= WHITE_BAR]
            black = [r for r in mid if float(r["median_gray_at_alpha255"]) == 0]
            tot = sum(int(r["n_px_alpha_255"]) for r in sub)
            px_w = sum(int(r["n_px_alpha_255"]) for r in white)
            out.append([yl if yl != "ALL" else "", fmt, len(sub), len(black),
                        len(white), px_w, tot,
                        f"{px_w / tot:.4f}" if tot else "", ""])
    out.append(["", "", "", "", "", "", "", "", NOTE_WHITE])
    return bc.write_csv(os.path.join(ROOT, "census",
                                     "ink_darkness_white_layers.csv"),
                        ["year_label", "pixel_format", "n_layers_with_alpha255",
                         "n_layers_black_at_alpha255", "n_layers_white_at_alpha255",
                         "n_px_alpha_255_white", "n_px_alpha_255",
                         "share_px_alpha255_white", "note"], out)


NOTE_WHITE = (
    "**`alpha = 255` の画素の輝度の中央**で層を分けた数。"
    "**白 = 240 以上**(器具の背景の閾 220 より厳しい側に置いた)。"
    "**層の数は少なく、画素の数は多い** —— **両方を並べて書く**(条例1)。"
    "**これは名前の述語についての測定である** —— "
    "**`線画`/`主線`/`lineart` という名前の層の中に、全不透明で白い画素が多い層が在る。** "
    "**誰が何のために白を置いたかは、この走行の出力ではない**(条例6)。"
    "**鏡像のずれの正体はこの層である** —— **白い画素は alpha も輝度も最大の側に入る**ので、"
    "**純黒の絵の具なら成り立つ鏡像(`gray = 255 - alpha`)を壊す。** "
    "**だから鏡像のずれは「絵の具が灰色」の検査として、この窓では使えない**"
)


def over_rows(rows):
    return [r for r in rows if r["mirror_deviation"]
            and float(r["mirror_deviation"]) >= MIRROR_BAR]


def read_prereg(rows1, rows2):
    """**事前登録 §4 の読み方で読む。** 左が出たら右の文、を決めてある。"""
    def sel(rows, stratum, pooled=False):
        return [dict(zip(HEADER, r)) for r in rows
                if r[2] == stratum and ((r[0] == "ALL") == pooled)]

    print("\n  [事前登録 §4 の読み]")
    for name, rows in (("1段目", rows1), ("2段目", rows2)):
        l1 = [r for r in sel(rows, "L1") if r["mirror_deviation"]]
        for fmt in ("2", "5"):
            f = [r for r in l1 if r["pixel_format"] == fmt]
            if not f:
                continue
            dev = [(float(r["mirror_deviation"]), r["work_id"], r["year_label"])
                   for r in f]
            over = [d for d in dev if d[0] >= MIRROR_BAR]
            print(f"    {name} L1 形式{fmt}: 行 {len(f)} / "
                  f"鏡像のずれ {min(d[0] for d in dev):.4f}〜{max(d[0] for d in dev):.4f}"
                  f" / **{MIRROR_BAR:.0%} 以上の行 {len(over)}**")
            for r in sorted(f, key=lambda r: -float(r["mirror_deviation"])):
                if float(r["mirror_deviation"]) < MIRROR_BAR:
                    continue
                print(f"      ずれ {float(r['mirror_deviation']):.4f}"
                      f"  {r['work_id']:4s} 札 {r['year_label']:5s}"
                      f"  白 {float(r['mirror_dev_white_bin']):.4f}"
                      f" / 黒 {float(r['mirror_dev_black_bin']):.4f}"
                      f" / 中 {float(r['mirror_dev_mid_bins']):.4f}"
                      f"  画素 {int(r['n_px_alpha_gt0']):>12,d}"
                      f"  層 {r['n_layers']}")
            mid_led = [r for r in over_rows(f)
                       if float(r["mirror_dev_mid_bins"])
                       > float(r["mirror_dev_white_bin"])]
            print(f"      **中のビンが白のビンより大きい行 {len(mid_led)}**"
                  f"(「絵の具自体が灰色」に近づくのはこの形だけ)")
        pb = [(float(r["frac_pure_black_among_ink"]), r["work_id"], r["year_label"],
               r["pixel_format"]) for r in l1 if r["frac_pure_black_among_ink"]]
        lo = [p for p in pb if p[0] < BAND_PURE_BLACK[0]]
        hi = [p for p in pb if p[0] > BAND_PURE_BLACK[1]]
        print(f"    {name} L1: 墨のうち純黒 行 {len(pb)} / "
              f"1段目の帯({BAND_PURE_BLACK[0]}〜{BAND_PURE_BLACK[1]})の"
              f"**下に出た行 {len(lo)} / 上に出た行 {len(hi)}**")


if __name__ == "__main__":
    raise SystemExit(main())

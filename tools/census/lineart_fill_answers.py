#!/usr/bin/env python
"""**名前が線画なのに塗りつぶしかもしれない層 68行の答え**を取り込む(2026-11-07)。

**聞き取りの表**: `lineart_fill_candidates.py` が 11-03 23:12 に作った表に作者が記入して返した
`data/census_full/hearing_lineart_fill_candidates_answered_20261107.xlsx`(sha256 `85ca7590…`)。
**白紙の表は上書きしていない。**

**fitness の空欄は census が埋めた** —— **ユーザーの指示(11-07)「fitness は回答はそちらでうめて」。**
作者の言葉(11-07):「fitness だけよくわからなかった。線画レイヤの範囲が画面ほぼ全体となっているが
見た目にはそのように見えなかった」。**census の読み**: 篩は透明度 1 以上を「描かれた」と数えたが、
fitness の候補では**描かれた画素の大半が透明度 1〜15(ヒストの最下段)**で、**16 以上だけならキャンバスの
1割に届かない**。**埋める前に層ごとにこれを確かめ、満たさない層は埋めない**(器具が数を印字する)。
**埋めた層は `answered_by` で作者の答えと分ける。作者が「分からない」と書いた2層は元の答えも残す。**

**答えの使い方は表に書いた規則のまま**(結果を見る前に決めた。規律10):
**「塗りつぶし」だけを墨の濃さの「作者の答えで外した値」から外す。** 「線画とベタ」「線画」は数に入れたまま。
**「その他」は外さない**(規則に無い。種類ごとに墨の割合を印字して、外すかはユーザーが決める)。
**主値 0.9177 は動かさない。**

    ./venv/bin/python tools/census/lineart_fill_answers.py
"""
import collections
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402
import ink_darkness as ik  # noqa: E402
import lineart_fill_candidates as lf  # noqa: E402
import worksheet_folder_reach as wf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANSWERED = "data/census_full/hearing_lineart_fill_candidates_answered_20261107.xlsx"
SHA_PREFIX = "85ca75906dea8a90"
OUT_PUB = "census/lineart_fill_answers.csv"
OUT_FULL = "data/census_full/lineart_fill_answers_20261107.csv"
EXCLUDE = {"塗りつぶし"}
# **【11-07 ユーザーの決定「外す」】作者が「その他」と答えた層も外す。** census が埋めた「その他」(作品1つ、
# 見た目は線だけ)は外さない —— ユーザーに見せた「その他」の内訳は作者の答えだけだった。
# **決定は種類ごとの墨の割合を見た後で、外した後の値を見る前**(母集団を動かしたので前後を両方印字する)
EXCLUDE_OTHER = "その他"
LABEL_OTHER = "さらに、作者が「その他」と答えた層も外した値(11-07 ユーザーの決定)"
BY_AUTHOR = "作者"
BY_CENSUS = "census(ユーザーの指示)"
FILL_WORK = "\\2024b\\fitness"
FILL_ANSWER = "その他"
FILL_COMMENT = "ほぼ透明な画素(透明度1〜15)が全面に在る。透明度16以上はキャンバスの1割未満で、見た目は線だけ"
FAINT_MIN = 0.85       # 描かれた画素のうち透明度 1〜15 の割合の下限(fitness の35層で 0.89〜0.96 だった)


def faint(r):
    """(描かれた画素のうち透明度 1〜15 の割合, 透明度 16 以上の被覆)。ヒストは16段で、最下段が 1〜15"""
    h = [int(x) for x in r["hist_alpha"].split(";")]
    g = int(r["n_px_alpha_gt0"])
    assert sum(h) == g, (r["page_path"], r["layer_id"])
    return h[0] / g, (g - h[0]) / int(r["n_px_total"])


def main() -> int:
    print("[線画の名前の塗りつぶし候補 68行の答え]")
    from openpyxl import load_workbook
    path = os.path.join(ROOT, ANSWERED)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if not sha.startswith(SHA_PREFIX):
        print(f"    **聞き取りの表の sha256 が違う**({sha[:16]})")
        return 1
    ws = load_workbook(path)["層"]
    sheet = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[1]]
    rows = {(r["page_path"], int(r["layer_id"])): r for r in lf.load_rows()
            if r["stratum"] == "L1" and r["pixel_formats"] == "2"}
    recs, bad = [], 0
    n_fill, n_fill_skip = 0, 0
    for s in sheet:
        work_dir, page, name, lid, _, ans, com, _, _, _, share, year, wid = s[:13]
        m = rows.get((page, int(lid)))
        if m is None:
            print(f"    **墨の濃さの表に無い行** {wid} 札 {year}")
            bad += 1
            continue
        raw = ans
        by = BY_AUTHOR if ans else ""
        if work_dir.endswith(FILL_WORK) and ans in (None, "分からない"):
            fs, c16 = faint(m)
            if fs >= FAINT_MIN and c16 < lf.cover(m) and c16 < 0.165:
                ans, com, by = FILL_ANSWER, FILL_COMMENT, BY_CENSUS
                n_fill += 1
            else:
                n_fill_skip += 1
        recs.append(dict(page=page, lid=int(lid), wid=wid, year=year, ans=ans or "", raw=raw or "",
                         com=com or "", by=by, ink=int(m["n_px_ink"] or 0), stage=m["ink_stage"], share=share))
    print(f"    行 {len(sheet)} / 取り込んだ {len(recs)}")
    print(f"    **fitness を census が埋めた {n_fill}層 / 確かめに通らず埋めなかった {n_fill_skip}層**"
          f"(透明度 1〜15 が描かれた画素の {FAINT_MIN} 以上、かつ 16 以上の被覆が 0.165 未満)")
    for (a, b), n in sorted(collections.Counter((r["ans"] or "空", r["by"] or "—") for r in recs).items()):
        print(f"      {a} / {b}: {n}")

    # **墨の濃さの「作者の答えで外した値」**(2段目・層別 L1 形式2 が主値の母集団)
    con = __import__("duckdb").connect(wf.DB, read_only=True)
    keys = {(p, int(i)) for p, i, _ in wf.inside_keys(con)}
    hearing = wf.load_hearing()
    st2 = []
    for p in sorted(glob.glob(os.path.join(ROOT, ik.STAGE2_GLOB))):
        rep = ik.STAGE2_REPLACED.get(os.path.basename(p))
        if rep:
            p = os.path.join(os.path.dirname(p), rep)
        st2 += list(__import__("csv").DictReader(open(p, encoding="utf-8")))
    main_pop = [r for r in st2 if r["stratum"] == "L1" and r["pixel_formats"] == "2"]
    k = lambda r: (r["page_path"], int(r["layer_id"]))  # noqa: E731
    prev_x = {k(r) for r in main_pop if k(r) in keys and hearing.get(r["page_path"]) in wf.HEARING_EXCLUDE}
    new_x = {(r["page"], r["lid"]) for r in recs if r["ans"] in EXCLUDE}
    other_x = {(r["page"], r["lid"]) for r in recs if r["ans"] == EXCLUDE_OTHER and r["by"] == BY_AUTHOR}
    new_x_in_pop = new_x & {k(r) for r in main_pop}
    print(f"    **「塗りつぶし」の層 {len(new_x)}**(うち主値の母集団=2段目に在る {len(new_x_in_pop)})")
    steps = [("墨の濃さの主値(登録した母集団。動かさない)", set()),
             (wf.LABEL_AUTH, prev_x),
             ("それに加えて、線画の名前の候補で作者が「塗りつぶし」と答えた層を外した値(11-07)", prev_x | new_x),
             (LABEL_OTHER, prev_x | new_x | other_x)]
    vals, out_vals = [], []
    for label, x in steps:
        keep = [r for r in main_pop if k(r) not in x]
        v, ink = wf.frac(keep)
        vals.append(v)
        print(f"    {label}: 層 {len(keep):,} / 外した層 {len(main_pop) - len(keep)} / 墨画素 {ink:,} / 純黒 {v}")
        out_vals.append([label, "", "", len(keep), len({r['page_path'] for r in keep}), "", v, ""])
    if vals[0] != wf.MAIN_VALUE:
        print(f"    **主値が {wf.MAIN_VALUE} を再現しない**({vals[0]})")
        bad += 1
    if vals[1] != "0.9156":
        print(f"    **11-03 の外した値 0.9156 を再現しない**({vals[1]})")
        bad += 1
    print(f"    **名前付きの前後**: {wf.LABEL_AUTH} {vals[1]} → 塗りつぶしも外した値 {vals[2]}"
          f" → その他も外した値 {vals[3]}(主値 {vals[0]} は動かさない)")

    # **「その他」は外さない。種類ごとの墨の割合を印字する**(外すかはユーザーが決める)
    pop_ink = sum(int(r["n_px_ink"] or 0) for r in main_pop)
    groups = collections.defaultdict(list)
    for r in recs:
        groups[(r["ans"] or "空", r["by"], r["com"])].append(r)
    out = []
    for (a, b, c), g in sorted(groups.items(), key=lambda x: (x[0][0], -len(x[1]))):
        ink2 = sum(r["ink"] for r in g if r["stage"] == "2")
        out.append(["", a, b, len(g), len({r["page"] for r in g}), f"{ink2 / pop_ink:.5f}", "", c])
        print(f"      {a} / {b or '—'} / {c or '(コメントなし)'}: 層 {len(g)} / 主値の母集団の墨の {ink2 / pop_ink:.3%}")
    rows_pub = out_vals + out + [[""] * 7 + [n] for n in NOTE]
    n_pub = bc.write_csv(os.path.join(ROOT, OUT_PUB),
                         ["subset_label", "answer_kind", "answered_by", "n_layers", "n_pages",
                          "share_ink_of_population", "frac_pure_black_among_ink", "author_comment"], rows_pub)
    with open(os.path.join(ROOT, OUT_FULL), "w", encoding="utf-8", newline="") as fh:
        w = __import__("csv").writer(fh)
        w.writerow(["page_path", "layer_id", "work_id", "year_label", "answer", "answer_raw_author",
                    "answered_by", "comment", "ink_stage", "n_px_ink"])
        for r in recs:
            w.writerow([r["page"], r["lid"], r["wid"], r["year"], r["ans"], r["raw"], r["by"], r["com"],
                        r["stage"], r["ink"]])
    print(f"    {OUT_PUB} {n_pub}行 / 層ごとの全量 `{OUT_FULL}`(非公開)")
    return 1 if bad else 0


NOTE = [
    "**名前が線画なのに塗りつぶしかもしれない層の聞き取り(2026-11-03 作成、11-07 回収)。** "
    "**篩**: 描かれた画素(透明度 1 以上)÷ キャンバスが 0.165 を超える層。0.165 は作者が仕上げの線と答えた層の最大値",
    "**答えの使い方は表を作ったときに書いた規則のまま**: **「塗りつぶし」だけを外す。** 「線画とベタ」「線画」は数に入れたまま。"
    "**「その他」は規則に無かったが、【11-07 ユーザーの決定「外す」】で作者が「その他」と答えた層も外した**(最後の行)。"
    "**決定は種類ごとの墨の割合を見た後** —— **母集団を動かしたので、外す前の値(塗りつぶしだけを外した値)も名前を付けて残す。** "
    "**census が埋めた「その他」(作品 W5、見た目は線だけ)は外していない。** **主値 0.9177 は動かさない**(規律10)",
    "**作品 W5(札 2024b)の層は census が埋めた**(`answered_by`)—— **ユーザーの指示(11-07)**。"
    "作者「範囲が画面ほぼ全体となっているが見た目にはそのように見えなかった」。"
    "**描かれた画素の大半が透明度 1〜15 で、16 以上だけならキャンバスの1割未満** —— **篩が透明度 1 以上を数えたので候補に入った**。"
    "**埋める前に層ごとに確かめた**(満たさない層は埋めない)。**作者が「分からない」と書いた2層は元の答えを非公開の全量に残した**",
    "**この薄い画素が何の工程で残ったかは分からない**(作者もすぐには分からなかった)。**census は推測で決めない**",
    "**`share_ink_of_population` は主値の母集団(2段目・層別 L1 形式2)の墨画素に占める割合。** 1段目の層は 0 に数える(主値の外)",
    "**層の名前・頁・作品の名前は書かない**(条例10)。**全量は `data/census_full/` 側**",
]

if __name__ == "__main__":
    raise SystemExit(main())

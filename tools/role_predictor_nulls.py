#!/usr/bin/env python3
"""層の役割の予測器 —— **帰無値を予測器の前に印字する**(規律9 / 登録簿 `role_predictor`)。

**この器具は予測しない。** 数えるだけである。出すのは3つ:
  1. 母集団と、落とした行を**理由別に**数えた数
  2. 帰無値1: **多数派の段**(登録簿では「走行前に計算する」)
  3. 帰無値2: **名前の当て方**(census の規則。下限の棒)—— **census の数を写さず、表から引き直す**

**何に反応しているかを名乗る**(条例9): 読むのは census の表 1本だけで、画素も画も見ない。
`author_stage` が空でない行だけが帰無値の母数で、それは **12頁・265層**しかない。
"""
import argparse, collections, csv, hashlib, pathlib, sys

STAGES = ("あたり", "下絵", "清書", "段ではない")          # 聞き取りで並べた語彙
OUT_OF_VOCAB_IS_DROPPED = True                             # 語彙の外の答えは**畳まず落として数える**


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def pct(k, n):
    return f"{k}/{n} = {k / n:.4f}" if n else f"{k}/{n} = —(母数0)"


def name_rule_bars(rows, label_col):
    """名前の当て方の段ごとの的中と取りこぼし。**表から引き直す**(写さない)。"""
    out = {}
    for s in STAGES[:3]:                                   # 段ではない は当て方が名乗らない
        said = [r for r in rows if r["name_rule"] == s]
        truth = [r for r in rows if r[label_col] == s]
        hit = sum(1 for r in said if r[label_col] == s)
        miss = sum(1 for r in truth if r["name_rule"] != s)
        out[s] = (hit, len(said), miss, len(truth))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/role_predictor/layer_role_table_20261110.csv")
    ap.add_argument("--label-col", default="author_stage",
                    help="author_stage(全層) / author_stage_excl_textish(描き文字・フキダシを空にした版)")
    a = ap.parse_args()
    p = pathlib.Path(a.table)
    rows = list(csv.DictReader(open(p, encoding="utf-8")))

    print("══ 0. 表(出所)")
    print(f"   {p}  行 {len(rows)}  sha256 先頭 {sha16(p)}")
    print("   出所: census `reply_e_census_layer_role_table_and_round2_20261110.md`(2026-11-10)")
    print(f"   ラベルの列: {a.label_col}")
    print("   **この器具は予測しない。数えるだけである。**")

    # ── 1. 母集団と落とした行(理由別)
    print()
    print("══ 1. 母集団と落とした行(**理由別**。規律9: 選別は測定である)")
    lab = [r for r in rows if r[a.label_col].strip()]
    print(f"   表の全行                          : {len(rows)}")
    print(f"   うち **ラベルの在る行**            : {len(lab)}"
          f"   頁 {len(set(r['page_path'] for r in lab))}"
          f" / 作品フォルダー {len(set(r['work_dir'] for r in lab))}"
          f" / 分け単位 {len(set(r['split_unit'] for r in lab))}")
    print(f"   うち ラベルの無い行(当てる相手)    : {len(rows) - len(lab)}")
    print("   —— **帰無値の母数はラベルの在る行だけ。** 当てる相手の側は帰無値に入らない")

    print()
    print("   ラベルの在る行の答えの内訳(**作者の答えをそのまま数える。畳まない**):")
    cnt = collections.Counter(r[a.label_col] for r in lab)
    for k, v in cnt.most_common():
        mark = "" if k in STAGES else "   ←—— **聞き取りで並べた語彙の外**"
        print(f"     {k:12s} {v:5d}{mark}")

    drops, kept = [], list(lab)
    oov = [r for r in kept if r[a.label_col] not in STAGES]
    if oov and OUT_OF_VOCAB_IS_DROPPED:
        kept = [r for r in kept if r[a.label_col] in STAGES]
        byv = collections.Counter(r[a.label_col] for r in oov)
        drops.append(("語彙の外の答え(**畳まずに落とした**)", len(oov), dict(byv)))
    unknown = cnt.get("分からない", 0)
    print()
    print("   落とした行(理由別):")
    print(f"     「分からない」                   : {unknown}"
          + ("   ←—— **0 である。登録簿の u = 0**" if unknown == 0 else ""))
    for why, n, detail in drops:
        print(f"     {why:30s} : {n}   {detail}")
    print(f"     `.lip` の頁                      : "
          f"{sum(1 for r in lab if r['page_path'].lower().endswith('.lip'))}")
    print(f"     描画できなかった層               : "
          f"{sum(1 for r in lab if r['render_status'] == '描画できなかった')}")
    print(f"     フォルダーでない通常・ベクター以外: "
          f"{sum(1 for r in lab if r['v_layer_kind'] not in ('raster', 'vector'))}"
          f"   (在る種別 {dict(collections.Counter(r['v_layer_kind'] for r in lab))})")
    print(f"   **落とさなかった**: 空の層(描画画素0) {sum(1 for r in kept if r['empty'] == '1')}"
          f" —— 下の §4。**作者の段が付いているので落とせない**")
    print(f"   帰無値を読む母数                  : {len(kept)}")

    # ── 2. 帰無値1: 多数派
    print()
    print("══ 2. 帰無値1: **多数派の段**(登録簿 `null_values`。走行前に出す数)")
    c = collections.Counter(r[a.label_col] for r in kept)
    top, n_top = c.most_common(1)[0]
    print(f"   多数派は **{top}** —— 的中 {pct(n_top, len(kept))}")
    for k in STAGES:
        print(f"     {k:12s} {c.get(k, 0):5d}")
    print("   **これは『名前の当て方で引いた12頁』の数で、原稿全体の割合ではない**"
          "(便A が来たら引き直す。登録簿 `next_hearing_design`)")

    # ── 3. 帰無値2: 名前の当て方
    print()
    print("══ 3. 帰無値2: **名前の当て方**(下限の棒。**census の数を写さず表から引き直した**)")
    print(f"   {'段':8s} {'的中(当てた層のうち作者も同じ)':34s} {'取りこぼし(作者の段のうち名前が拾わない)'}")
    for s, (hit, said, miss, truth) in name_rule_bars(kept, a.label_col).items():
        print(f"   {s:8s} {pct(hit, said):34s} {pct(miss, truth)}")
    nr = collections.Counter(r["name_rule"] for r in kept)
    print(f"   当て方が名乗った段の数: {dict(nr)}")
    print("   **あたりは当て方が1層も名乗らない** —— 名前では引けない(規律13: 代用しない)")
    print("   **片方だけでは勝ちにならない。§2 と §3 を並べて読む**(登録簿 `null_values`)")

    # ── 4. 空の層(前後を両方)
    print()
    print("══ 4. 空の層(描画画素0)—— **外す前と外した後を両方、名前を付けて印字**(規律9-5)")
    e = [r for r in kept if r["empty"] == "1"]
    print(f"   ラベルの在る空の層 {len(e)}: {dict(collections.Counter(r[a.label_col] for r in e))}")
    print("   **空の層は『役割が無い層』ではない** —— 作者は空の層にも段を付けている"
          "(`base`(ベースマスク)や髪ベタは本体が空でマスクが形を持つ。census の読みと作者のコメント)")
    for tag, sub in (("空を含む(**こちらを使う**)", kept), ("空を外した", [r for r in kept if r["empty"] != "1"])):
        c2 = collections.Counter(r[a.label_col] for r in sub)
        t2, n2 = c2.most_common(1)[0]
        print(f"   {tag}  母数 {len(sub):4d}  多数派 {t2} {pct(n2, len(sub))}")
        for s, (hit, said, miss, truth) in name_rule_bars(sub, a.label_col).items():
            print(f"       {s:8s} 的中 {pct(hit, said):26s} 取りこぼし {pct(miss, truth)}")
    print("   **外すと床が動く**ので、外すなら前後を両方印字する義務が残る"
          "(「床を厳しくした」は独立ではない)。**この器具の既定は外さない**")

    # ── 5. 名乗り
    print()
    print("══ 5. この器具が**見ていない**もの(条例9: 何に反応しているかを名乗る)")
    print("   画素の値・画・マスクの中身 —— 見ていない(表の数だけ)")
    print("   当てる相手の側(ラベルの無い 52,978層)—— 帰無値には1行も入れていない")
    print("   作品フォルダーで分けた hold-out —— **まだ作っていない。** 勝ち負けはそこで読む")
    print("   予測 —— **していない**")
    return 0


if __name__ == "__main__":
    sys.exit(main())

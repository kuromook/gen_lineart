#!/usr/bin/env python3
"""便B(薄い段を埋める便)の**選別の仮説を測る** —— 推測で返さない(条例1 の形)。

**何に反応しているかを名乗る**(条例9):
  読むのは census の表 1本。**画も画素の値も見ていない**(表の数だけ)。
  的中を測る母数は **ラベルの在る 263層(12頁)だけ** —— その12頁は
  **census の名前の当て方で引いた頁**なので、**ここで測った的中はその母集団の数**である。
  **これは棒ではない。** 便B は**学習専用で棒を読まない**(登録簿 `next_hearing_design`)。
  ここで選ぶのは「どの頁を作者に聞くか」だけで、**予測の主張は1つもしない**。

出すもの: 帰無値(段の薄い側の割合)/ 規則ごとの **的中・拾えた数・取りこぼし**(段別)/
  表の全体でその規則が選ぶ **層・頁・分け単位の数**(便B の実務の歩留まり)。
"""
import argparse, collections, csv, pathlib, sys

RARE = ("あたり", "下絵")
STAGES = ("あたり", "下絵", "清書", "段ではない")


def f(r, k, d=0.0):
    v = r.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def faint(r):
    """薄さ: アルファ 26 以上の画素の割合。**小さいほど薄い。** 画素が無い層は None"""
    a, b = f(r, "n_px_a_gt0"), f(r, "n_px_a_ge26")
    return (b / a) if a > 0 else None


def rules():
    """規則は**名前を付けて**並べる。名前に『名前』と入っているものは名前に触っている規則"""
    return [
      ("名前: 当て方が下絵と言う(既存の規則)", lambda r: r["name_rule"] == "下絵"),
      ("名前: 祖先に `sketch set`", lambda r: "sketch set" in r["ancestors"]),
      ("名前: 祖先に 下絵/sketch/rough/ラフ/あたり",
       lambda r: any(w in r["ancestors"].lower() for w in ("下絵", "sketch", "rough", "ラフ", "あたり", "ネーム"))),
      ("構造: 祖先フォルダーが非表示", lambda r: r["ancestors_shown_db"] == "0"),
      ("構造: 自身が非表示", lambda r: r["self_shown_db"] == "0"),
      ("構造: どちらかが非表示", lambda r: r["ancestors_shown_db"] == "0" or r["self_shown_db"] == "0"),
      ("構造: 不透明度 < 256", lambda r: 0 < f(r, "opacity") < 256),
      ("画素: 薄い(ge26/gt0 < 0.25)", lambda r: (faint(r) is not None and faint(r) < 0.25)),
      ("画素: 大面積(gt0 > 8,000,000)", lambda r: f(r, "n_px_a_gt0") > 8_000_000),
      ("画素: **大面積で薄い**", lambda r: f(r, "n_px_a_gt0") > 8_000_000
                                   and faint(r) is not None and faint(r) < 0.25),
      ("合わせ: 大面積で薄い **か** 祖先が非表示",
       lambda r: (f(r, "n_px_a_gt0") > 8_000_000 and faint(r) is not None and faint(r) < 0.25)
                 or r["ancestors_shown_db"] == "0"),
      ("合わせ: 祖先に `sketch set` **かつ** 当て方が清書でない",
       lambda r: "sketch set" in r["ancestors"] and r["name_rule"] != "清書"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/role_predictor/layer_role_table_20261110.csv")
    ap.add_argument("--holdout", default="data/role_predictor/holdout_units_round2_a_20261110.csv")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.table, encoding="utf-8")))
    hold = {r["split_unit"] for r in csv.DictReader(open(a.holdout, encoding="utf-8"))}
    lab = [r for r in rows if r["author_stage"] in STAGES]
    lab_units = {r["split_unit"] for r in lab}
    # **便B が引いてよい層**: 便A の評価単位と、既にラベルの在る単位を外した残り
    pool = [r for r in rows if r["split_unit"] not in hold and r["split_unit"] not in lab_units]

    print("══ 0. 何を測るか(**便B の選別の仮説。棒ではない**)")
    print(f"   的中の母数      : ラベルの在る {len(lab)}層 / 12頁 / 分け単位 {len(lab_units)}")
    print(f"   歩留まりの母数  : 表の {len(rows)}層 から 便A の評価単位 {len(hold)} と"
          f" ラベルの在る単位 {len(lab_units)} を外した **{len(pool)}層**"
          f" / 頁 {len(set(r['page_path'] for r in pool))} / 単位 {len(set(r['split_unit'] for r in pool))}")
    print("   **この12頁は名前の当て方で引いた頁** —— ここの的中はその母集団の数である")

    rare = [r for r in lab if r["author_stage"] in RARE]
    print()
    print("══ 1. 帰無値(**規則を1つも使わないとき**)")
    print(f"   薄い段(あたり+下絵)の割合: {len(rare)}/{len(lab)} = {len(rare)/len(lab):.4f}"
          f"   (あたり {sum(1 for r in rare if r['author_stage']=='あたり')} /"
          f" 下絵 {sum(1 for r in rare if r['author_stage']=='下絵')})")
    print("   **あたりは4層しかない。** 1層 = 25% —— **『0/4』は厳密なゼロではない**(条例5)")

    print()
    print("══ 2. 規則ごと(的中 = 選んだ層のうち あたり+下絵 / 拾えた数 = 19層のうち)")
    hdr = f"   {'規則':44s} {'的中':18s} {'拾えた':12s} {'あたり':7s} {'下絵':7s} {'倍率':6s} {'歩留まり(層/頁/単位)'}"
    print(hdr)
    base = len(rare) / len(lab)
    for name, fn in rules():
        said = [r for r in lab if fn(r)]
        hit = [r for r in said if r["author_stage"] in RARE]
        prec = (len(hit) / len(said)) if said else None
        got_a = sum(1 for r in hit if r["author_stage"] == "あたり")
        got_s = sum(1 for r in hit if r["author_stage"] == "下絵")
        psel = [r for r in pool if fn(r)]
        pages = len({r["page_path"] for r in psel})
        units = len({r["split_unit"] for r in psel})
        ps = f"{len(hit)}/{len(said)} = {prec:.3f}" if said else "母数0"
        lift = f"{prec/base:5.1f}x" if prec else "  —  "
        print(f"   {name:44s} {ps:18s} {len(hit):2d}/{len(rare):2d} = {len(hit)/len(rare):.2f}"
              f"  {got_a}/4    {got_s}/15    {lift} {len(psel):6d} / {pages:5d} / {units:4d}")

    print()
    print("══ 3. 段ごとに、規則が**選んでしまう**数(清書・段ではないを混ぜる量)")
    for name, fn in rules():
        said = [r for r in lab if fn(r)]
        c = collections.Counter(r["author_stage"] for r in said)
        print(f"   {name:44s} " + "  ".join(f"{s}={c.get(s,0)}" for s in STAGES))

    print()
    print("══ 4. この器具が**見ていない**もの / **確かめていない**もの")
    print("   画・画素の値 —— 見ていない(表の数だけ)")
    print("   **`pos_from_top` の向き** —— 『手前から数えた位置』の手前がどちらかを確かめていないので使わなかった"
          "(census に問う。推測で素性にしない。規律13)")
    print("   **描線の色(青・赤)** —— ラベルの在る層に値が0件なので**測れない**(抽出側に発注中)")
    print("   **`colour_type_index`(表現色)** —— あたり4層は全部 1 だが 清書126層も 1。分けない")
    print("   **当てる相手の側のラベル** —— 存在しない。歩留まりは層の数で、段の数ではない")
    print("   **この器具は予測しない。便B の頁を選ぶための測りである**")
    return 0


if __name__ == "__main__":
    sys.exit(main())

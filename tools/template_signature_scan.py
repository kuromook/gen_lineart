#!/usr/bin/env python3
"""**決定論的に落とせる層(テンプレート)を、名前を使わずに見つける。**

作者の言明(2026-11-10、逐語)「PICOトンボや枠などのテンプレートは全く同じレイヤでしかない　
レイヤ名もぶれていない　これらは決定論的に除外できるはずだ」を、**署名の一致**で確かめる器具。

**何に反応しているかを名乗る**(条例9):
  署名 = (層名, アルファ>0 の画素数, 外接矩形, キャンバス寸法)。**画素の値も画も見ていない。**
  **同じ物が別の作品に現れること**を見ているだけで、「テンプレートらしさ」は見ていない。
  **閾値(いくつの作品単位にまたがればテンプレートとするか)は、候補が2組しかない表で選んだ** ——
  **当て込みである**。対照(2単位の組)を必ず並べて印字する。

**この器具は予測しない。** 落とす名前の表は人が固定する(表に無い名前は落とさない。許可表と同じ考え方)。
"""
import argparse, collections, csv, pathlib, sys

SIG = ("name", "n_px_a_gt0", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "canvas_px_w", "canvas_px_h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/role_predictor/layer_role_table_20261110.csv")
    ap.add_argument("--min-units", type=int, default=3, help="テンプレートとみなす作品単位の数(既定3。当て込み)")
    ap.add_argument("--unit-col", default="split_unit")
    a = ap.parse_args()
    p = pathlib.Path(a.table)
    rows = list(csv.DictReader(open(p, encoding="utf-8")))

    print("══ 0. 何を見ているか")
    print(f"   表 {p}  行 {len(rows)}")
    print(f"   署名 = {' + '.join(SIG)}")
    print("   **画素の値も画も見ていない。同じ物が別の作品に現れることだけを見ている**")

    # ── 落とした行を理由別に(規律9)
    drop_empty = [r for r in rows if r["n_px_a_gt0"] in ("", "0")]
    kept = [r for r in rows if r["n_px_a_gt0"] not in ("", "0")]
    print()
    print("══ 1. 母集団と落とした行(理由別。規律9)")
    print(f"   表の全行                    : {len(rows)}")
    print(f"   画素が0 または 空の行(除く) : {len(drop_empty)}"
          "   —— 署名が『0画素』で全部一致してしまうので除く")
    print(f"   署名を作った行              : {len(kept)}")
    print(f"   **この表の母集団は『名前の当て方で引いた段の候補』** —— "
          f"テンプレート層は候補の外なので、ここに在るのは聞き取りの頁ぶんだけ")

    g_pages, g_units, g_rows = {}, {}, {}
    for r in kept:
        k = tuple(r[c] for c in SIG)
        g_pages.setdefault(k, set()).add(r["page_path"])
        g_units.setdefault(k, set()).add(r[a.unit_col])
        g_rows.setdefault(k, []).append(r)
    print(f"   署名の組                    : {len(g_pages)}")

    # ── 単位の数の分布(**閾値を見えるようにする**)
    print()
    print("══ 2. 署名が何作品にまたがるか(**閾値をここで選ぶ。分布を隠さない**)")
    dist = collections.Counter(len(v) for v in g_units.values())
    for n in sorted(dist):
        mark = "  ←—— **既定の閾値以上**" if n >= a.min_units else ""
        print(f"   {n:2d}作品にまたがる署名: {dist[n]:6d} 組{mark}")

    # ── 閾値以上(= テンプレートの候補)
    print()
    print(f"══ 3. {a.min_units}作品以上にまたがる署名(**テンプレートの候補**)")
    cand = sorted(((k, g_units[k], g_pages[k]) for k in g_units if len(g_units[k]) >= a.min_units),
                  key=lambda x: -len(x[1]))
    if not cand:
        print("   なし")
    for k, u, pg in cand:
        lab = collections.Counter(r["author_stage"] for r in g_rows[k] if r["author_stage"].strip())
        labs = ("  作者の答え " + str(dict(lab))) if lab else "  作者の答え なし"
        print(f"   作品{len(u):2d} 頁{len(pg):3d}  name={k[0]!r} 画素={k[1]} キャンバス={k[6]}x{k[7]}{labs}")
    n_layers = sum(len(g_rows[k]) for k, _, _ in cand)
    print(f"   候補の層 {n_layers} / 名前 {len({k[0] for k, _, _ in cand})} 種")

    # ── 対照: 閾値の1つ下(**これを拾ったら誤り**)
    print()
    print(f"══ 4. 対照 —— {a.min_units - 1}作品にまたがる署名(**ここを拾ったら誤りである**)")
    ctrl = sorted(((k, g_units[k], g_pages[k]) for k in g_units if len(g_units[k]) == a.min_units - 1),
                  key=lambda x: -len(x[2]))
    print(f"   組 {len(ctrl)}  層 {sum(len(g_rows[k]) for k, _, _ in ctrl)}")
    names = collections.Counter(k[0] for k, _, _ in ctrl)
    for nm, c in names.most_common(8):
        print(f"     {c:4d} 組  name={nm!r}")
    print("   **多くは作画である**(同じ作品の版が2単位に分かれている可能性。**仮説。確かめていない**)")
    print(f"   —— だから閾値を {a.min_units - 1} に下げると作画を落とす")

    # **閾値が刃の上かどうかを印字する**(当て込みを隠さないための印字。条例9)
    below = [n for n in dist if n < a.min_units]
    above = [n for n in dist if n >= a.min_units]
    if below and above:
        lo, hi = max(below), min(above)
        print(f"   **閾値の余裕**: 候補でない側の最大は {lo} 作品、候補の側の最小は {hi} 作品 —— "
              f"**その間({lo + 1}〜{hi})に署名の組は1つも無い**。"
              f"だから **{lo + 1} から {hi} までどこに引いても同じ結果**になる"
              if hi - lo > 1 else
              f"   **閾値は刃の上である**: {lo} 作品と {hi} 作品が隣り合っている —— 1つずらすと結果が動く")
    print(f"   ただし **候補が {len(cand)} 組しかない** —— 余裕が在るのは『いまの表では』である")

    # ── 1作品の中だけの繰り返し(= 使い回し。テンプレートではない)
    inner = [(k, g_pages[k]) for k in g_units if len(g_units[k]) == 1 and len(g_pages[k]) >= 3]
    print()
    print("══ 5. 1作品の中だけで繰り返す署名(**使い回し。テンプレートではない**)")
    print(f"   組 {len(inner)}  例: "
          + ", ".join(f"{k[0]!r}×{len(pg)}頁" for k, pg in sorted(inner, key=lambda x: -len(x[1]))[:5]))

    print()
    print("══ 6. この器具が**していない**こと")
    print("   落とす名前の表を作ること —— **人が固定する。表に無い名前は落とさない**(許可表と同じ)")
    print("   テンプレートかどうかの判定 —— 見ているのは『同じ物が別の作品に在る』だけ")
    print("   予測 —— **していない**")
    return 0


if __name__ == "__main__":
    sys.exit(main())

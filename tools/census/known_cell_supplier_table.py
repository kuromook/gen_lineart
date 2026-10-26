#!/usr/bin/env python
"""既知セル: 供給元が**納品書に手で書いた表**と、こちらの集計を突き合わせる。

**既知セルには、試験対象の器具が作っていない入力を少なくとも1つ入れる**(T132)。
ここで使うのは `inbox/deliver_f_request3_layer_kind_20261026.md` の種別ごとの表 ——
**この器具が作っていない**(供給元が DB から別に集計して書いた)。

**この点検が届く範囲を、名前より狭く書く**(T132 の規則):

- **届く**: 受領 CSV の行が途中で切れている / 読み落としている /
  種別の綴りを取り違えている / `rendered` と `drawn_ratio` の桶を間違えている
- **届かない**: **DB そのものの誤り。** 供給元の表もこちらの集計も
  同じ DB から出ているので、**壊れた DB でも一致する**。
  一致は「転記が合っている」であって「材料が正しい」ではない

    ./venv/bin/python tools/census/known_cell_supplier_table.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402

NOTICE = "inbox/deliver_f_request3_layer_kind_20261026.md"
ROW = re.compile(r"^\|\s*([A-Za-z_0-9]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|")


def parse_notice(path):
    """納品書の markdown 表を読む。**この器具が作っていない入力。**"""
    out = {}
    for line in open(path, encoding="utf-8"):
        m = ROW.match(line.strip())
        if m:
            kind, n, rendered, gt0 = m.groups()
            out[kind] = (int(n.replace(",", "")),
                         int(rendered.replace(",", "")),
                         int(gt0.replace(",", "")))
    return out


def main() -> int:
    if not os.path.exists(NOTICE):
        print(f"納品書が無い: {NOTICE} —— **この既知セルは回せない**")
        return 2
    supplier = parse_notice(NOTICE)
    rows, pages, _ = bc.load_req3(bc.sources.by_key("req3_layer_kind")["path"])

    mine = {}
    for r in rows:
        k = r["v_layer_kind"]
        n, rend, gt0 = mine.get(k, (0, 0, 0))
        mine[k] = (n + 1,
                   rend + (1 if r["rendered"] == "true" else 0),
                   gt0 + (1 if r["drawn_bucket"] == "gt0" else 0))

    print("既知セル: 供給元が納品書に手で書いた表 対 こちらの集計")
    print(f"  供給元の表: {NOTICE}(**この器具が作っていない入力**)")
    print(f"  こちらの集計: {len(rows)} 層 / {len(pages)} 頁\n")
    print(f"  {'種別':16s} {'層(供/当)':>16s} {'描けた(供/当)':>18s} {'dr>0(供/当)':>18s}  判定")
    bad = 0
    only_supplier = sorted(set(supplier) - set(mine))
    only_mine = sorted(set(mine) - set(supplier))
    for k in sorted(set(supplier) & set(mine)):
        s, m = supplier[k], mine[k]
        ok = s == m
        bad += 0 if ok else 1
        print(f"  {k:16s} {s[0]:>7d}/{m[0]:<7d} {s[1]:>8d}/{m[1]:<8d} "
              f"{s[2]:>8d}/{m[2]:<8d}  {'一致' if ok else '**食い違い**'}")
    if only_supplier:
        print(f"  **供給元の表にしか無い種別: {only_supplier}**")
        bad += len(only_supplier)
    if only_mine:
        print(f"  **こちらにしか無い種別: {only_mine}**")
        bad += len(only_mine)

    print()
    if bad:
        print(f"**{bad} 件の食い違い。受領 CSV か読み手のどちらかが違う**")
        return 1
    print("全種別で一致。**ただしこれは転記の一致である** —— "
          "供給元の表もこちらの集計も同じ DB から出ているので、"
          "**壊れた DB でも一致する。** 材料が正しいことは、この点検では分からない")
    return 0


if __name__ == "__main__":
    sys.exit(main())

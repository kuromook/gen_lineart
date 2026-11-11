# -*- coding: utf-8 -*-
"""第3段の規則が**届く範囲の上限**を、キャンバスの集計から出す(走る前に)。

census の答え(2026-11-12): テンプレートの名前の層は **.clip で16種のキャンバス**に分かれる。
署名はキャンバスを含むので、**「同じ署名が3作品単位以上」はキャンバスの中で数えることになる**。
→ **キャンバスの単位数が3未満のキャンバスでは、規則は構造上1単位も確かめられない。**
母数を先に印字する(規律9: 選別は測定である)。
"""
import csv, collections, hashlib, io, sys

SRC = "inbox/template_names_canvas_20261112.csv"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    print("材料: %s sha256 %s" % (SRC, sha(SRC)))
    rows = list(csv.DictReader(io.open(SRC, encoding="utf-8")))
    print("行: %d(名前 × 種別 × 形式 × キャンバス)\n" % len(rows))

    for fmt in ("clip", "lip"):
        sub = [r for r in rows if r["format"] == fmt]
        print("== 形式 %s ==" % fmt)
        per_canvas = collections.Counter()
        for r in sub:
            per_canvas[r["canvas"]] = max(per_canvas[r["canvas"]], int(r["n_split_units"]))
        tot = sum(per_canvas.values())
        ok = {k: v for k, v in per_canvas.items() if v >= 3}
        ng = {k: v for k, v in per_canvas.items() if v < 3}
        print("   キャンバスの種類: %d / 単位の和(重複あり): %d" % (len(per_canvas), tot))
        print("   **3単位以上のキャンバス: %d 種(単位の和 %d)**" % (len(ok), sum(ok.values())))
        print("   **3単位未満のキャンバス: %d 種(単位の和 %d)—— 規則は構造上届かない**"
              % (len(ng), sum(ng.values())))
        if tot:
            print("   → **規則が届き得る単位の割合の上限: %d / %d = %.4f**"
                  % (sum(ok.values()), tot, sum(ok.values()) / tot))
        for k, v in sorted(per_canvas.items(), key=lambda x: -x[1]):
            print("     %-14s %2d単位 %s" % (k, v, "" if v >= 3 else "**届かない**"))
        print()

    print("== 名前ごと(.clip) ==")
    for nm in sorted({r["layer_name"] for r in rows}):
        sub = [r for r in rows if r["layer_name"] == nm and r["format"] == "clip"]
        if not sub:
            print("   %-16s .clip に無い" % nm)
            continue
        cv = {r["canvas"]: int(r["n_split_units"]) for r in sub}
        reach = sum(v for v in cv.values() if v >= 3)
        tot = sum(cv.values())
        print("   %-16s キャンバス %2d種 / 単位 %2d / **届き得る %2d(%.3f)**"
              % (nm, len(cv), tot, reach, reach / tot if tot else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""**聞き取りの層の表で「3D背景」を名前と祖先で先に埋める手順**(Track Q の頼み 2026-11-17、Q の登録 `q8_3d_background_prefill_in_other_hearings_registration_20261117.md`)。

**規則は Q のもの。census は当てるだけ**(数を見る前にこのファイルに書いた。形は `kuro_keshi_prefill.py` と同じ):
- 当たる層: 種別(`v_layer_kind`)が `image_material`、層の名前が **ASCII だけ**(空でない)、かつ祖先(`page_order` の `/` で繋いだ名前)に
  `BG` を含む(大文字小文字を区別しない)。**ラスタの層には当てない**(モノクロの3D は Track R の印の側)
- 初期値「3D背景」、**欄は閉じない**、札 `機械の初期値`、表の頭に「3D背景の初期値は層の名前とフォルダー名で入れました。違えば書き直してください」
- **黒消しの規則にも当たる層は、どちらも入れない**(両方の記録の列に「重なり」と書き、数を印字する)。重なりの層は、どちらの見張りの区切りにも入れない
- 見張り: 黒消しと同じ(行 id の順に5行ずつ、`random.Random(20261117)` で1行、最後の区切りは 行数/5 の確率)。**乱数は規則ごと・表ごとに新しく作る**
- 記録の列: `d3_rule_hit` / `d3_prefilled` / `d3_sentinel` /(答えの後)`d3_final_answer` / `d3_rewritten`
- 「3D背景」の選択肢が無い聞き取りには入れない
- 単独で走らせると、全層で当たる層を形式・時期ごとに数え、黒消しとの重なりを印字する(点数ではない)

    ./venv/bin/python tools/census/d3_background_prefill.py
"""
import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kuro_keshi_prefill as kk  # noqa: E402

SEED = 20261117
LABEL = "機械の初期値"
HEAD = "3D背景の初期値は層の名前とフォルダー名で入れました。違えば書き直してください"


def hit(kind, name, ancestors):
    name = name or ""
    return kind == "image_material" and name != "" and name.isascii() and "bg" in (ancestors or "").lower()


def _chunks(ids, rng):
    out = {}
    for i in range(0, len(ids), 5):
        chunk = ids[i:i + 5]
        pick = rng.randrange(len(chunk)) if len(chunk) == 5 or rng.random() < len(chunk) / 5 else None
        for j, rid in enumerate(chunk):
            out[rid] = j == pick
    return out


def plan_both(rows):
    """rows: [(row_id, kind, name, ancestors)] → {row_id: dict(kk_*, d3_*)}。重なりはどちらも入れない"""
    out = {}
    hk = {r[0] for r in rows if kk.hit(r[2], r[3])}
    hd = {r[0] for r in rows if hit(r[1], r[2], r[3])}
    both = hk & hd
    sk = _chunks(sorted(hk - both), random.Random(SEED))
    sd = _chunks(sorted(hd - both), random.Random(SEED))
    for r in rows:
        i = r[0]
        x = {}
        for pre, h, s in (("kk", hk, sk), ("d3", hd, sd)):
            if i in both:
                x.update({f"{pre}_rule_hit": "重なり", f"{pre}_prefilled": 0, f"{pre}_sentinel": 0})
            else:
                x.update({f"{pre}_rule_hit": int(i in h), f"{pre}_prefilled": int(i in h and not s[i]),
                          f"{pre}_sentinel": int(i in h and s[i])})
        out[i] = x
    return out


def main() -> int:
    import db
    import exclusions
    from layer_role_table import page_order
    print("[3D背景の先埋めの規則に当たる層(全層。点数ではない)]")
    con = db.connect()
    ex = exclusions.load()
    pages = [p for (p,) in con.execute("select page_path from pages").fetchall() if p not in ex]
    fmt = dict(con.execute("select page_path, format from pages").fetchall())
    yr = dict(con.execute("select page_path, year from pages").fetchall())
    info = {(r[0], r[1]): (r[2], r[3]) for r in con.execute(
        """select l.page_path, l.layer_id, coalesce(l.name, ''), k.v_layer_kind
             from layers l join v_layer_kind k using (page_path, layer_id) where not l.folder""").fetchall()}
    po = page_order(con, pages)
    c, both = collections.Counter(), collections.Counter()
    for k, o in po.items():
        name, kind = info.get(k, ("", ""))
        if hit(kind, name, o[2]):
            per = "新" if int(str(yr[k[0]])[:4]) >= 2020 else "旧"
            c[(fmt[k[0]], per)] += 1
            if kk.hit(name, o[2]):
                both[(fmt[k[0]], per)] += 1
    print(f"    母集団: 恒久除外の頁を外した {len(pages):,}頁のフォルダーでない層 {len(po):,}")
    for k, n in sorted(c.items()):
        print(f"    .{k[0]} {k[1]}: 当たる {n:,}(うち黒消しの規則にも当たる {both[k]})")
    print(f"    計 {sum(c.values()):,} / 重なり {sum(both.values())}")
    demo = plan_both([(1, "image_material", "bg01", "BG"), (2, "image_material", "成人修正", "BG"),
                      (3, "image_material", "x", "BG/モザイク"), (4, "raster", "bg", "BG")])
    print("    点検(4行: 当たる / 名前が ASCII でない / 両方に当たる / ラスタ): "
          + " / ".join(f"{k}:{v['d3_rule_hit']}" for k, v in demo.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

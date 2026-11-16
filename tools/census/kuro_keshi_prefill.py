#!/usr/bin/env python
"""**聞き取りの層の表で「黒消し」を名前で先に埋める手順**(Track Q の頼み 2026-11-17、Q の登録 `q6_kuro_keshi_prefill_in_other_hearings_registration_20261117.md`)。

**規則は Q のもの。census は当てるだけ**(数を見る前にこのファイルに書いた):
- 当たる層: 層の名前がちょうど `成人修正`、または祖先(フォルダーの名前を `/` で繋いだ `page_order` の祖先)に `モザイク` を含む層
- 当たった層の答えの欄に初期値「黒消し」を入れる。**欄は閉じない**(空・テンプレートと違う)。札は `機械の初期値`
  (灰色の字)。表の頭に「黒消しの初期値は名前で入れました。違えば書き直してください」
- **見張りの行**: 当たった層を行 id の順に5行ずつ区切り、各区切りから1行を `random.Random(20261117)` で選んで
  初期値を入れない(札も付けない)。**5行に満たない最後の区切り**は、乱数 `rng.random() < 区切りの行数 / 5` のときだけ1行選ぶ
  (5行に1行の割合を保つため。census の解釈で、Q に返事で確かめる)。乱数は表ごとに新しく作る
- 記録の列: row_id / kk_rule_hit / kk_prefilled / kk_sentinel /(答えが返った後)kk_final_answer / kk_rewritten
- **問いに「黒消し」の選択肢が無い聞き取りには入れない**。作者に渡す前の承認はそれぞれの聞き取りの側で
- 単独で走らせると、全層で規則に当たる層を形式・時期(新 2020〜 / 旧 〜2019b、頁の札)ごとに数える(点数ではない)

    ./venv/bin/python tools/census/kuro_keshi_prefill.py
"""
import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEED = 20261117
NAME, ANC = "成人修正", "モザイク"
LABEL = "機械の初期値"
HEAD = "黒消しの初期値は名前で入れました。違えば書き直してください"


def hit(name, ancestors):
    return (name or "") == NAME or ANC in (ancestors or "")


def plan(rows):
    """rows: [(row_id, name, ancestors)] → {row_id: dict(kk_rule_hit, kk_prefilled, kk_sentinel)}"""
    rng = random.Random(SEED)
    out = {r[0]: dict(kk_rule_hit=0, kk_prefilled=0, kk_sentinel=0) for r in rows}
    hits = sorted(r[0] for r in rows if hit(r[1], r[2]))
    for i in range(0, len(hits), 5):
        chunk = hits[i:i + 5]
        pick = rng.randrange(len(chunk)) if len(chunk) == 5 or rng.random() < len(chunk) / 5 else None
        for j, rid in enumerate(chunk):
            out[rid].update(kk_rule_hit=1, kk_sentinel=int(j == pick), kk_prefilled=int(j != pick))
    return out


def main() -> int:
    import db
    import exclusions
    from layer_role_table import page_order
    print("[黒消しの先埋めの規則に当たる層(全層。点数ではない)]")
    con = db.connect()
    ex = exclusions.load()
    pages = [p for (p,) in con.execute("select page_path from pages").fetchall() if p not in ex]
    fmt = dict(con.execute("select page_path, format from pages").fetchall())
    yr = dict(con.execute("select page_path, year from pages").fetchall())
    names = {(r[0], r[1]): r[2] for r in con.execute("select page_path, layer_id, coalesce(name, '') from layers where not folder").fetchall()}
    po = page_order(con, pages)
    c = collections.Counter()
    for k, o in po.items():
        if hit(names.get(k, ""), o[2]):
            per = "新" if int(str(yr[k[0]])[:4]) >= 2020 else "旧"
            c[(fmt[k[0]], per, "名前" if names.get(k, "") == NAME else "祖先")] += 1
    print(f"    母集団: 恒久除外の頁を外した {len(pages):,}頁のフォルダーでない層 {len(po):,}")
    for k, n in sorted(c.items()):
        print(f"    .{k[0]} {k[1]} {k[2]}で当たる: {n:,}")
    print(f"    計 {sum(c.values()):,}(名前と祖先の両方に当たる層は「名前」で数えた)")
    demo = plan([(i, NAME, "") for i in range(1, 13)])
    print(f"    見張りの選び方の点検(当たる12行): 見張り {sum(v['kk_sentinel'] for v in demo.values())} 行 "
          f"= 行 id {[k for k, v in demo.items() if v['kk_sentinel']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

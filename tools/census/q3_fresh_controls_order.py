#!/usr/bin/env python
"""**黒消しの規則 v3 の新しい対照の一覧**(Track Q の頼み `note_q_census_q3_fresh_controls_20261115.md`、ユーザーの承認「y」2026-11-15)。

**引く前に決めたこと**(数を見る前にこのファイルに書いた。census は一覧を引くだけで、点数は無い):
- 全系統で `.clip` だけ。恒久除外の頁は入れない(`exclusions`)
- **写しを避ける**:
  - 便1 の一覧(`grey_stats_bin1_layers_20261111.csv`、825層)の層は入れない
  - 825層のどれかと **(頁のファイル名, 層の番号)** が同じ層も入れない(Q の頼みどおり)
  - **ファイルの sha256 の束で代表でない頁の層は入れない**(`page_representative.csv`。代表は最も古い札 = 作者の裁定)。表に無い頁は落として数える
  - 新しい一覧の中でも (頁のファイル名, 層の番号) が重なれば、引いた順の先だけ残す
- **時期**: 年代札の先頭4桁が 2020 以上を「新」、それより前(〜2019b)を「旧」。**各系統、新から 25・旧から 25**。足りなければそのまま(他方で埋めない)
- **K′ コマ枠**: 種別 frame_folder で、名前に `モザイク` を含まないもの(便1 と同じ)
- **P′ 用紙**: 種別 paper(便1 と同じ)
- **H′ 髪や服のベタ**(便1 と同じ出所 + 作者の答え):
  - H′a = 作者が答えた層: 便B の段の答えが `ベタ`(作者が答えた行)、または便A・便B・役割の表 v3 の作者のコメントに `ベタ` を含む層。種別 raster
  - H′b = 名前がちょうど `ベタ` の raster の層
  - **取る順**(時期ごと): H′a を全部 → H′b のうち **描かれた画素が頁の 2% を超える層**を作品単位で均等に →
    残りの H′b を作品単位で均等に。25 まで
  - **「描かれた画素が頁の 2% を超える」は、Q の追補1 の条件(1つの繋がった面が頁の 2% を超える)の必要条件にすぎない**
    (面の画素 ≤ 層の描かれた画素)。**全面ベタだと言っているのではない。** 1つの面かどうかは抽出側の成分の列で分かる
  - 描かれた画素 = 全頁の画素の表の `n_px_a_gt0`、頁 = `v_canvas_px_w × v_canvas_px_h`。画素の表に無い層・描画できなかった層は、この層の印を「測れない」として残りの H′b に入れる
- 均等の引き方: 便1 と同じ(作品単位を乱数で並べ、単位を順に1層ずつ)。乱数 `random.Random(20261115)`、系統と時期の順に同じ乱数を使い続ける
- 系統が重なる層は先の系統(K′ → H′ → P′)に置く
- 落としたものを理由別に印字する(規律9)

**v2(同じ日、引いた後に直した器具の誤り。規則は変えていない)**: v1 は均等の引き方で 25 を引いてから、一覧の中の重なりで飛ばしていたので、
**飛ばした分が埋まらなかった**(K′ 47 / P′ 33、計 87層。`q3_fresh_controls_20261115_v1_wasted_draws.csv` に残す)。
v2 は、均等の順に候補を全部並べ、飛ばしながら 25 になるまで取る。**乱数の消費の順が変わるので、v1 と v2 は同じ層を引いていない**。前後の数を両方返す

    ./venv/bin/python tools/census/q3_fresh_controls_order.py
"""
import collections
import csv
import ntpath
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
import split_units  # noqa: E402
from grey_stats_bin1_order import even  # noqa: E402
from p_split_columns import wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 20261115
N_HALF = 25
BIN1 = "outbox/grey_stats_bin1_layers_20261111.csv"
REP = "data/census_full/page_representative.csv"
PIX = "data/census_full/full_clip_pixels_layers_merged.csv"
A = "data/census_full/hearing_round2_a_answers.csv"
B = "data/census_full/hearing_round2_b_answers.csv"
V3 = "data/census_full/layer_role_table_20261112_v3.csv"
OUT = "data/census_full/q3_fresh_controls_20261115.csv"
MOZA, BETA = "モザイク", "ベタ"


def rd(p):
    return list(csv.DictReader(open(os.path.join(ROOT, p), encoding="utf-8")))


def main() -> int:
    print(f"[黒消しの規則 v3 の新しい対照の一覧] seed {SEED}")
    con = db.connect()
    ex = exclusions.load()
    un = split_units.units(con)
    rows = con.execute("""select l.page_path, l.layer_id, coalesce(l.name, ''), k.v_layer_kind, p.format, p.year,
                                 p.v_canvas_px_w * p.v_canvas_px_h
                            from layers l join v_layer_kind k using (page_path, layer_id) join pages p using (page_path)""").fetchall()
    info = {(r[0], r[1]): r for r in rows}
    bin1 = {(r["page_path"], int(r["layer_id"])) for r in rd(BIN1)}
    bin1_fn = {(ntpath.basename(p), i) for p, i in bin1}
    rep = {r["page_path"]: r["is_representative"] == "yes" for r in rd(REP)}
    pix = {(r["page_path"], int(r["layer_id"])): r for r in rd(PIX)}
    drop = collections.Counter()

    def ok(r):
        k = (r[0], r[1])
        for why, bad in ((".lip", r[4] != "clip"), ("恒久除外の頁", r[0] in ex), ("便1 の825層", k in bin1),
                         ("便1 と (ファイル名, 層の番号) が同じ", (ntpath.basename(r[0]), r[1]) in bin1_fn),
                         ("sha256 の束の表に無い頁", r[0] not in rep), ("sha256 の束で代表でない頁", not rep.get(r[0], True))):
            if bad:
                drop[why] += 1
                return False
        return True

    def period(k):
        return "新" if int(str(info[k][5])[:4]) >= 2020 else "旧"

    def big(k):
        x = pix.get(k)
        if not x or x["render_status"] != "描いた" or not x["n_px_a_gt0"] or not info[k][6]:
            return None
        return int(x["n_px_a_gt0"]) / info[k][6] > 0.02

    koma = [(r[0], r[1]) for r in rows if r[3] == "frame_folder" and MOZA not in r[2] and ok(r)]
    paper = [(r[0], r[1]) for r in rows if r[3] == "paper" and ok(r)]
    ha = {(r["page_path"], int(r["layer_id"])) for r in rd(B) if r["answer_source"] == "author" and r["stage"] == BETA}
    for p in (A, B, V3):
        ha |= {(r["page_path"], int(r["layer_id"])) for r in rd(p) if BETA in (r.get("author_comment") or "")}
    n_ha_raw = len(ha)
    ha = sorted(k for k in ha if k in info and info[k][3] == "raster" and ok(info[k]))
    hb = sorted({(r[0], r[1]) for r in rows if r[2] == BETA and r[3] == "raster" and ok(r)} - set(ha))
    print(f"    候補(落とした後): K′ {len(koma):,} / P′ {len(paper):,} / H′a 作者の答え {len(ha)}(落とす前 {n_ha_raw})/ H′b 名前 {len(hb):,}")
    print("    落とした(候補を作るときに走査した延べ。理由は先に当たったもの1つ): "
          + " / ".join(f"{a} {b:,}" for a, b in drop.most_common()))
    bb = collections.Counter(big(k) for k in hb)
    print(f"    H′b の『描かれた画素が頁の 2% を超える』: 超える {bb[True]} / 超えない {bb[False]} / 測れない {bb[None]}")

    rng = random.Random(SEED)
    out, seen, seen_fn = [], set(), set()

    def take(group, sub, ks, cap):
        n = 0
        for k in ks:
            if n >= cap:
                break
            fn = (ntpath.basename(k[0]), k[1])
            if k in seen:
                drop[f"{group}: 先の系統と重なる"] += 1
                continue
            if fn in seen_fn:
                drop[f"{group}: 一覧の中で (ファイル名, 層の番号) が重なる"] += 1
                continue
            seen.add(k)
            seen_fn.add(fn)
            out.append(dict(group=group, sub=sub, period=period(k), page_path=k[0], layer_id=k[1],
                            v_layer_kind=info[k][3], split_unit=un[wd(k[0])],
                            drawn_gt_2pct={True: 1, False: 0, None: ""}[big(k)] if group.startswith("H") else ""))
            n += 1
        return n

    for group, cands in (("K′ コマ枠(枠そのもの)", koma), ("H′ 髪や服のベタ", None), ("P′ 用紙", paper)):
        for per in ("新", "旧"):
            if cands is not None:
                c = [k for k in cands if k not in seen and period(k) == per]
                take(group, "", even(c, un, rng, len(c)), N_HALF)
                continue
            got = take(group, "H′a 作者の答え", [k for k in ha if period(k) == per], N_HALF)
            b = [k for k in hb if period(k) == per and k not in seen]
            c = [k for k in b if big(k)]
            got += take(group, "H′b 名前・2% を超える", even(c, un, rng, len(c)), N_HALF - got)
            c = [k for k in b if not big(k) and k not in seen]
            take(group, "H′b 名前", even(c, un, rng, len(c)), N_HALF - got)

    for group in ("K′ コマ枠(枠そのもの)", "H′ 髪や服のベタ", "P′ 用紙"):
        s = [x for x in out if x["group"] == group]
        line = " / ".join(f"{per} {sum(x['period'] == per for x in s)}" for per in ("新", "旧"))
        print(f"    {group}: {len(s)}層({line})/ 頁 {len({x['page_path'] for x in s})} / 作品単位 {len({x['split_unit'] for x in s})}")
        if group.startswith("H"):
            print("        内訳: " + " / ".join(f"{k} {v}" for k, v in collections.Counter(x["sub"] for x in s).items())
                  + f" / 描かれた画素が頁の 2% を超える {sum(x['drawn_gt_2pct'] == 1 for x in s)}"
                  f"(超えない {sum(x['drawn_gt_2pct'] == 0 for x in s)} / 測れない {sum(x['drawn_gt_2pct'] == '' for x in s)})")
    late = {k: v for k, v in drop.items() if ":" in k}
    if late:
        print("    引いた後に飛ばした: " + " / ".join(f"{a} {b}" for a, b in late.items()))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["no"] + list(out[0]))
        w.writeheader()
        for i, x in enumerate(out, 1):
            w.writerow(dict(no=i, **x))
    print(f"    計 {len(out)}層 → `{OUT}`(非公開。発注は Q の追補で固定した後)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

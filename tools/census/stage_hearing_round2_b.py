#!/usr/bin/env python
"""**便B(学習専用の便。棒を読まない)の頁を引き、20行ずつの束の聞き取りの表を作る** —— Track E の決定
(`note_census_b_hypotheses_measured_20261110.md` §2、`note_census_exclude_axis_and_20row_split_20261110.md` §2・§3)。

**引く前に決めたこと(札に書く)**:
- 母集団: 層の役割の表 v2(`layer_role_table_20261110_v2.csv`、53,289層)のうち、
  **便A の評価12単位を外し、ラベルの在る12頁も外した残り**(E の歩留まりと同じ切り方)
- 層別は3つ。**名前を使わない**(E の決定):
  - **芯**: 覆い率 `n_px_a_gt0 / (canvas_px_w × canvas_px_h)` > 0.5 **かつ** 薄さ `n_px_a_ge26 / n_px_a_gt0` < 0.25 の層を持つ頁 → 4頁
    (**閾値は E がラベル19層を見て決めた —— 当て込み**。便A では使わない)
  - **広**: 祖先フォルダーが非表示(`ancestors_shown_db = 0`)の層を持つ頁 → 4頁
  - **描画色**: `draw_color_enable = 1` の層を持つ頁 → 2頁(**この軸を測れるようにするため**。E)
- **1単位1頁**(`split_units` の単位。10頁で単位が重ならない)。層別の順(芯 → 広 → 描画色)に、
  頁を `random.Random(20261110)` で並べて先頭から取る。前の層別で使った単位は飛ばす(飛ばした数を印字)
- 頁が複数の層別に当たっても、引いた層別の札だけを書き、他に当たる層別も別の列に印字する
- 層: その頁の、フォルダーでない `raster` と `vector` の全部(便A と同じ。名前・層別に関わらず)
- **束**: 頁の順のまま、20行を超えない所で切る。**頁の途中では切らない**。1頁で20行を超える頁は単独の束(例外、印字)
- 問いは軸1(工程の段)だけ。最初の束は試し
- **表の形(E 11-11 `note_census_a1_keep_choices_20261111.md`)**: 便B は選択肢に「空(1画素も描かれていない)」を足し、
  「空」の印を画素の列(層の役割の表 v2 の `empty`)から機械が付ける。頁と行は変えない(引き直さない)
- **「空」の定義を変える(E 11-11 `note_census_empty_definition_20261111.md`、作者の裁定「画素無くてもマスクがあればその層は空ではない
  完全に空であるという定義が必要」)**: 一度は E の提案の定義D(祖先のマスクまで見る)にした
- **作者が C を選んだ(E `note_census_empty_definition_fixed_20261111.md`。定義D は採られなかった)**: 機械の印は `empty_c()`
  (自分のマスクだけ見る + 画素は在るが自分のマスクで全部消えている層も空)。**E の頼みで「空」を段の選択肢から外す**
  (作者は空の層に段を付けている = 排他ではない)。「空」は機械の印の列だけに残す。**表は作者に出す前なので、行も頁も変わらない**
- **v2(2026-11-10、ユーザーの決定「20以下」)**: v1 の10頁は 550行で、8頁が1頁で20行を超えた(作者の数「Excel は 20」)。
  **候補を「並べる層(raster と vector)が 20 以下の頁」に絞って同じ seed・同じ手順で引き直す。** 作者が答える前に決めた。
  v1 の出力は消さずに残す(`*_20261110.csv` / `*_bundleB-*`)。v2 は `_le20` を付ける。絞る前後の候補の数を両方印字する
- **v3 の表(2026-11-12、`ROUND2B_PREFILL=1`)**: 頁と行は v2 のまま(引き直さない)。2つを変える
  - **機械の答えを先に入れて閉じる**(ユーザー 11-12「OKだ」、E 同意): `empty_c` = 空 の行 → `機械: 空`、
    E の除外の表の 状態=確定 の名前のうちテンプレートの名前(`PICOトンボ` / `断ち落とし枠（白）` / `外枠`)の行 → `機械: テンプレート`。
    第2回の層名の聞き取りで「よい」の名前は閉じない(問いが軸2 で、段を含意しない。E)
  - **段の選択肢に「テンプレート(トンボ・裁ち落とし枠など、印刷のための層)」を足す**(ユーザーの指示「a 選択肢にテンプレートが必要」、E 11-12)
  - 出力は `_le20_prefill`。v2 の表は消さない。閉じた行の数と、作者が答える行の数を印字する

    ./venv/bin/python tools/census/stage_hearing_round2_b.py
"""
import collections
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402
from stage_hearing import OUT_KEY, STAGES, name_rule  # noqa: E402
from stage_hearing_round2_a_sheets import LIMIT, layers_top_down, write_xlsx  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLE = "data/census_full/layer_role_table_20261110_v2.csv"   # 頁を引いた母集団(v2 のまま。引き直さない)
TABLE_V3 = "data/census_full/layer_role_table_20261112_v3.csv"   # 空の印だけ v3 から(便B の残り 65層の画素が入った)
HOLDOUT = "data/census_full/holdout_units_round2_a_20261110.csv"
SEED = 20261110
ORDER_OUT = "outbox/stage_layers_order_round2b_20261111.csv"
MAX_ROWS = int(os.environ.get("ROUND2B_MAX_ROWS", "20"))   # 0 = 絞らない(v1)
STRATA = [("芯", 4), ("広", 4), ("描画色", 2)]
OUT = "data/census_full/hearing_round2_b_pages_20261110" + ("_le20" if MAX_ROWS else "") + ".csv"
KEY_OUT = "data/census_full/hearing_round2_b_key_20261110" + ("_le20" if MAX_ROWS else "") + ".csv"
PREFILL = os.environ.get("ROUND2B_PREFILL", "") == "1"
XLSX = "data/census_full/hearing_round2_b_bundle{b}_20261110" + ("_le20" if MAX_ROWS else "") + ("_prefill" if PREFILL else "") + ".xlsx"
MACHINE_OUT = "data/census_full/hearing_round2_b_machine_answers_20261112.csv"
TEMPLATE_NAMES = {"PICOトンボ", "断ち落とし枠（白）", "外枠"}   # E の表の 状態=確定 のテンプレートの名前(11-12)
TEMPLATE_CHOICE = "テンプレート(トンボ・裁ち落とし枠など、印刷のための層)"


def f(x):
    return float(x) if x not in ("", None) else None


def empty_c(r) -> str:
    """「完全に空」の正本(作者の裁定 2026-11-11 = C + トーンの形)。E `tools/empty_tone_masked_out.py` の `is_empty()` と同じ規則。
    祖先のマスクは見ない。読めない層は "0"(空ではない = 測れない)"""
    px = int(r["n_px_a_gt0"]) if r["n_px_a_gt0"] not in ("", None) else None
    if px is None or r["mask_status"] not in ("ok", ""):
        return "0"
    own = int(r["n_px_a_gt0_own_mask"]) if r["n_px_a_gt0_own_mask"] not in ("", None) else None
    if px == 0:
        return "0" if (r["has_own_mask"] == "1" and (own or 0) > 0) else "1"
    return "1" if own == 0 else "0"


def main() -> int:
    print(f"[便B の頁(学習専用の便)] seed {SEED}")
    con = db.connect()
    un = split_units.units(con)
    ev = {r["split_unit"] for r in csv.DictReader(open(os.path.join(ROOT, HOLDOUT), encoding="utf-8"))
          if r["side"].startswith("評価")}
    labelled = {r["page_path"] for r in csv.DictReader(open(os.path.join(ROOT, OUT_KEY), encoding="utf-8"))}
    hits = collections.defaultdict(set)
    empty = {}
    n_all, drop = 0, collections.Counter()
    pages_all = set()
    for r in csv.DictReader(open(os.path.join(ROOT, TABLE), encoding="utf-8")):
        n_all += 1
        empty[(r["page_path"], r["layer_id"])] = empty_c(r)
        if r["split_unit"] in ev:
            drop["便A の評価単位"] += 1
            continue
        if r["page_path"] in labelled:
            drop["ラベルの在る12頁"] += 1
            continue
        p = r["page_path"]
        pages_all.add(p)
        gt0, ge26, w, h = f(r["n_px_a_gt0"]), f(r["n_px_a_ge26"]), f(r["canvas_px_w"]), f(r["canvas_px_h"])
        if gt0 and w and h and gt0 / (w * h) > 0.5 and ge26 / gt0 < 0.25:
            hits[p].add("芯")
        if r["ancestors_shown_db"] == "0":
            hits[p].add("広")
        if r["draw_color_enable"] == "1":
            hits[p].add("描画色")
    # 空の印だけ v3 から(便B の残り 65層の画素が入った。頁を引く母集団は v2 のまま)
    empty = {(r["page_path"], r["layer_id"]): empty_c(r)
             for r in csv.DictReader(open(os.path.join(ROOT, TABLE_V3), encoding="utf-8"))}
    print(f"    表 {n_all:,}層 → 外した " + " / ".join(f"{a} {b:,}" for a, b in drop.items())
          + f" → 残り {n_all - sum(drop.values()):,}層 / {len(pages_all):,}頁")
    for s, _ in STRATA:
        ps = [p for p in hits if s in hits[p]]
        print(f"      {s}: 当たる頁 {len(ps):,} / 単位 {len({un[wd(p)] for p in ps})}")
    q0 = ",".join("?" * len(hits))
    allp = con.execute(f"""select l.page_path, l.layer_id, l.parent_id, l.order_in_parent, coalesce(l.name, ''),
                                  l.folder, l.visibility, k.v_layer_kind
                             from layers l join v_layer_kind k using (page_path, layer_id)
                            where l.page_path in ({q0})""", sorted(hits)).fetchall()
    bp = collections.defaultdict(list)
    for r in allp:
        bp[r[0]].append(r)
    nrow = {p: len(layers_top_down(bp[p])[0]) for p in hits}
    if MAX_ROWS:
        print(f"    **v2: 並べる層が {MAX_ROWS} 以下の頁に絞る**(ユーザーの決定)")
        for s, _ in STRATA:
            ps = [p for p in hits if s in hits[p]]
            ok = [p for p in ps if nrow[p] <= MAX_ROWS]
            print(f"      {s}: 絞る前 {len(ps):,}頁 / {len({un[wd(p)] for p in ps})}単位 → 絞った後 {len(ok):,}頁 / "
                  f"{len({un[wd(p)] for p in ok})}単位(落とした 20行超 {len(ps) - len(ok):,}頁)")
        hits = {p: v for p, v in hits.items() if nrow[p] <= MAX_ROWS}
    rng = random.Random(SEED)
    used, picks = set(), []
    for s, k in STRATA:
        cand = sorted(p for p in hits if s in hits[p])
        rng.shuffle(cand)
        got = skipped = 0
        for p in cand:
            u = un[wd(p)]
            if u in used:
                skipped += 1
                continue
            used.add(u)
            picks.append((s, p, u))
            got += 1
            if got == k:
                break
        print(f"    {s}: 引いた {got}(使用済みの単位で飛ばした頁 {skipped})")
    q = ",".join("?" * len(picks))
    ls = con.execute(f"""select l.page_path, l.layer_id, l.parent_id, l.order_in_parent, coalesce(l.name, ''),
                                l.folder, l.visibility, k.v_layer_kind
                           from layers l join v_layer_kind k using (page_path, layer_id)
                          where l.page_path in ({q})""", [p for _, p, _ in picks]).fetchall()
    by_page = collections.defaultdict(list)
    for r in ls:
        by_page[r[0]].append(r)
    rows, items = [], []
    for i, (s, p, u) in enumerate(picks, 1):
        lay, _ = layers_top_down(by_page[p])
        items.append(dict(item=i, stratum=s, other_strata="/".join(sorted(hits[p] - {s})), page_path=p, split_unit=u,
                          n_layers_raster_vector=len(lay), wip_status=""))
        for x in lay:
            rows.append(dict(item=i, kind_item=s, page_path=p, wip="", rule=name_rule(x["name"]), **x))
    # 束: 頁の順のまま、20行を超えない所で切る
    bundles, cur, n = [], [], 0
    for it in items:
        m = it["n_layers_raster_vector"]
        if cur and n + m > LIMIT:
            bundles.append(cur)
            cur, n = [], 0
        cur.append(it["item"])
        n += m
    bundles.append(cur)
    names = [(f"B-{j}", tuple(b)) for j, b in enumerate(bundles, 1)]
    for it in items:
        print(f"      項目 {it['item']}: {it['stratum']}" + (f"(他に {it['other_strata']})" if it["other_strata"] else "")
              + f" / {it['n_layers_raster_vector']}行")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(items[0]))
        w.writeheader()
        w.writerows(items)
    bmap = {i: b for b, its in names for i in its}
    with open(os.path.join(ROOT, KEY_OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["bundle"] + list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(dict(bundle=bmap[r["item"]], **r))
    miss = [r for r in rows if (r["page_path"], str(r["layer_id"])) not in empty]
    if miss:
        # 「空」の印は画素の列から付ける(E 11-11)。画素の行の無い層があれば、表は作らず抽出側への一覧だけ作る
        print(f"    **画素の行の無い層 {len(miss)} / {len(rows)}(頁 {len({r['page_path'] for r in miss})})** —— "
              f"「空」の印が付けられないので表は作らない。抽出側への一覧 `{ORDER_OUT}`(発注はユーザーの承認の後)")
        with open(os.path.join(ROOT, ORDER_OUT), "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["no", "page_path", "layer_id", "name_family", "why_listed", "v_layer_kind"])
            for i, r in enumerate(miss, 1):
                w.writerow([i, r["page_path"], r["layer_id"], "", "便B", r["kind"]])
        print(f"    計 {len(rows)}行 / {len(names)}束 → `{OUT}` / 鍵 `{KEY_OUT}`")
        return 0
    machine, stages = None, STAGES
    if PREFILL:
        machine = {}
        for r in rows:
            k = (r["page_path"], str(r["layer_id"]))
            if empty[k] == "1":
                machine[k] = "空"
            elif r["name"] in TEMPLATE_NAMES:
                machine[k] = "テンプレート"
        stages = STAGES[:-1] + [TEMPLATE_CHOICE, STAGES[-1]]
        c = collections.Counter(machine.values())
        print(f"    **機械の答えで閉じた行 {len(machine)}**(空 {c['空']} / テンプレート {c['テンプレート']})/ "
              f"**作者が答える行 {len(rows) - len(machine)}**(全 {len(rows)} 行)")
        with open(os.path.join(ROOT, MACHINE_OUT), "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["bundle", "page_path", "layer_id", "name", "machine_answer", "answer_tag"])
            for r in rows:
                k = (r["page_path"], str(r["layer_id"]))
                if k in machine:
                    w.writerow([bmap[r["item"]], r["page_path"], r["layer_id"], r["name"], machine[k],
                                "machine_empty_c" if machine[k] == "空" else "machine_exclusion_confirmed"])
    for b, its in names:
        sub = [r for r in rows if r["item"] in its]
        over = "  **20行を超える(1頁で切れない)**" if len(sub) > LIMIT else ""
        print(f"    {b}: 項目 {', '.join(map(str, its))} → {len(sub)}行{over}")
        write_xlsx(b, its, items, sub, None, label="便B", n_bundles=len(names), path=XLSX, trial=(b == "B-1"),
                   stages=stages, empty=None if PREFILL else empty, machine=machine)
    print(f"    計 {len(rows)}行 / {len(names)}束 → `{OUT}` / 鍵 `{KEY_OUT}` / 表 `{XLSX.format(b='B-1')}` ほか(非公開)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

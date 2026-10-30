#!/usr/bin/env python3
"""m92: 作者に見せる**窓を選ぶ**(追補142)。器具①。

**登録の正本は `doc/prereg_author_division_20261030.md`(修正1 を含む)。**
ここは登録を実装するだけで、**規則を作らない**:

- **各頁 4 窓・頁ごとに同数**(面の数に比例させない)/ **1024x1024 原寸** /
  **一様無作為・種 20261030** / **引き直す条件は1つだけ**(窓の中に標準の面が1つも無い)
- **引き直した回数を理由別に印字する**(条例 項7 / 規律9)
- **窓が重ならないという規則は登録に無いので足さない。** 重なった対の数は**観察として印字する**
- **IC2** の退化した窓(標準も A0 も1面)は、**32 窓を全部引いた後に**同じ乱数列の続きで
  探し、**探すのに要した引きの数を印字する**

標準と A0 の作り方は `m89` から**借りずに import する**(写すと分かれる)。
**判断を2か所に置かない。**

**修正2 を実装している**(2026-10-30): `P0437` は標準が空なので**窓を 0 個出す**。
**28 窓 / 31 問**。登録本体の 32 窓・35 問は**撤回しないので両方印字する**。

**登録に書いていないので器具が決めたこと**(印字もする):
- IC2 の引きでは**頁も一様に引く**。登録は「同じ乱数列の続き」しか言っていないので、
  32 窓と同じ分布の素直な延長を採った。上限 `IC2_TRIES` 引きで見つからなければ出さない
- ラベル地図は**1頁1回だけ作って控えに置く**(2回作ると判断が分かれる危険より、
  単に遅い。`--fresh` で控えを無視して作り直す)
"""
import collections, csv, json, os, pathlib, sys, time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                     # noqa: E402
import m16_score_corrected_standard as m16           # noqa: E402
import m77_standard_visible as m77                   # noqa: E402
import m89_rescore_island_unit as m89                # noqa: E402

OUT = pathlib.Path("results/m92_windows")
CACHE = pathlib.Path(os.environ.get(
    "M92_CACHE", "/tmp/claude-1000/-home-sh1-deepl-lineart-colour-manuscripts/"
                 "56c20ca8-6335-4886-8b3b-961887500cde/scratchpad/m92_labels"))
SEED = 20261030          # **登録に書いた種。変えない**
WIN = 1024               # **原寸 1024x1024。縮小しない(NC3)**
PER_PAGE = 4
IC2_TRIES = 20000
# **器具が止まらないための上限であって棒ではない**(修正2-D)。当たった頁は窓を出さない
MAX_REDRAW = 200000
ALIAS = pathlib.Path("results/alias/page_alias.csv")
# **作者の裁定(追補128)。別名で書く** —— 原稿の固有名は書かない
DIFF = {"P0166", "P0432", "P0437", "P0444", "P0446"}


def aliases():
    """実路 → 別名。`manifest` の `source` は `page_alias.csv` の `page_path` と 1:1。"""
    with open(ALIAS, encoding="utf-8") as fh:
        return {r["page_path"]: r["alias"] for r in csv.DictReader(fh)}


def n_areas(a):
    """a の中の面の数(0 = 面でない画素)。"""
    return int(np.count_nonzero(np.bincount(a.ravel())[1:]))


def labels_for(p, byp, drops, fresh):
    """1頁の **島の単位の標準**(原寸)と **A0**(preview の側)。控えが在れば読む。"""
    std_f = CACHE / f"{p['page']}.std.npy"
    a0_f = CACHE / f"{p['page']}.a0.npy"
    rows_all = byp[p["source"]]
    rows_vis, d = m77.visible_rows(rows_all)
    drops.update(d)
    s = int(round(p["preview"]["縮小率"]))
    pw, ph = p["preview"]["size"]
    grid = (ph * s, pw * s)                          # **作業画布 = preview × s**
    floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))

    if not fresh and std_f.exists() and a0_f.exists():
        lab_i = np.load(std_f)
        a0_small = np.load(a0_f)
        return s, pw, ph, grid, lab_i, a0_small, "控え"

    comps = []
    (_, _), (lab_a, garea_l), st = m16.measure_standard(
        p, rows_vis, grid, floor_px, out_comps=comps)
    comps = [c for c in comps if c["area"] >= floor_px]
    lab_i, garea_i = m89.island_labels(lab_a, comps, floor_px, grid)
    del lab_a, comps
    if lab_i.max() > 32000:
        raise SystemExit(f"面が多すぎて int16 に入らない: {p['page']} {lab_i.max()}")
    lab_i = lab_i.astype(np.int16)

    im = cv2.imread(str(m12.D / p["preview"]["file"]), cv2.IMREAD_COLOR)
    a0_small, _ = m12.filtered_labels(
        m12.equal_value_components(m12.quantise(im, m12.LEVELS)),
        max(1, round(m12.FLOOR * pw * ph)))
    del im
    a0_small = a0_small.astype(np.int16)

    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(std_f, lab_i)
    np.save(a0_f, a0_small)
    return s, pw, ph, grid, lab_i, a0_small, "作った"


def main():
    fresh = "--fresh" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding="utf-8"))["pages"]
    rng = np.random.default_rng(SEED)
    drops = collections.Counter()
    redraw = collections.Counter()
    rows, geo = [], {}
    alias = aliases()
    skipped, skip_note = collections.Counter(), []
    t0 = time.time()

    print(f"種 {SEED} / 窓 {WIN}x{WIN} 原寸 / 頁ごと {PER_PAGE} 窓 / 頁 {len(pages)}")
    print("**窓の中に標準の面が1つも無いときだけ引き直す。それ以外の引き直しはしない**")
    print(f"控え: {CACHE}  (--fresh で作り直す)\n")

    # ---- 第1巡: 頁ごとに標準と A0 を作り、4 窓を引く ----
    for p in pages:
        s, pw, ph, grid, lab_i, a0_small, how = labels_for(p, byp, drops, fresh)
        H, W = grid
        cv_ = p.get("canvas") or []
        rem = f"{cv_[0] - W}x{cv_[1] - H}" if len(cv_) == 2 else ""
        canvas = f"{cv_[0]}x{cv_[1]}" if len(cv_) == 2 else ""
        n_std_page = n_areas(lab_i)
        al = alias.get(p["source"], "")
        st_ = "差分" if al in DIFF else ("非差分" if al else "**別名が解けない**")
        head = f"  {al or '????'} {st_:4s} s={s} 作業画布 {W}x{H} 余り {rem or '0x0'} 標準の面 {n_std_page:4d}"

        if int(np.count_nonzero(lab_i)) == 0:
            skipped["標準が空(standard_visible)なので窓を取れない"] += 1
            skip_note.append(f"{al} ({st_})")
            print(f"{head}  **窓 0 個 —— 修正2-A の裁定**", flush=True)
            del lab_i, a0_small
            continue
        if H < WIN or W < WIN:
            skipped["作業画布が窓より小さい"] += 1
            skip_note.append(f"{al} ({st_})")
            print(f"{head}  **窓 0 個 —— 画布が窓より小さい**", flush=True)
            del lab_i, a0_small
            continue
        geo[p["page"]] = {"H": H, "W": W, "s": s, "canvas": canvas, "rem": rem,
                          "alias": al, "stratum": st_}

        def draw(H=H, W=W):
            return (int(rng.integers(0, H - WIN + 1)),
                    int(rng.integers(0, W - WIN + 1)))

        def counts(y, x):
            ys, xs = y // s, x // s
            return (n_areas(lab_i[y:y + WIN, x:x + WIN]),
                    n_areas(a0_small[ys:ys + WIN // s, xs:xs + WIN // s]))

        picked, tries = [], 0
        while len(picked) < PER_PAGE and tries < MAX_REDRAW:
            tries += 1
            y, x = draw()
            n_std, n_a0 = counts(y, x)
            if n_std == 0:
                redraw["窓の中に標準の面が1つも無い"] += 1
                continue
            picked.append({"page": p["page"], "alias": al, "stratum": st_,
                           "y": y, "x": x, "w": WIN, "h": WIN,
                           "s": s, "grid": f"{W}x{H}", "canvas": canvas,
                           "remainder": rem, "n_std_areas": n_std,
                           "n_a0_areas": n_a0})
        if len(picked) < PER_PAGE:
            # **頁ごとに同数**(登録)なので、埋まらない頁は 1 つも出さない
            skipped[f"引き直しの上限 {MAX_REDRAW} に当たった"] += 1
            skip_note.append(f"{al} ({st_}) 引き {tries} で {len(picked)}/{PER_PAGE}")
            del geo[p["page"]], lab_i, a0_small
            print(f"{head}  **窓 0 個 —— 上限に当たった**", flush=True)
            continue
        rows += picked
        print(f"{head}  {how}  引き {tries}  窓 "
              + ", ".join(f"({r['x']},{r['y']})std{r['n_std_areas']}/a0{r['n_a0_areas']}"
                          for r in picked), flush=True)
        del lab_i, a0_small

    # ---- 第2巡: **32 窓の後**、同じ乱数列の続きで IC2 の退化窓を探す ----
    pl = sorted(geo)
    plan = []
    for k in range(1, IC2_TRIES + 1):
        pg = pl[int(rng.integers(0, len(pl)))]
        g = geo[pg]
        y = int(rng.integers(0, g["H"] - WIN + 1))
        x = int(rng.integers(0, g["W"] - WIN + 1))
        plan.append((k, pg, y, x))
    by_pg = collections.defaultdict(list)
    for k, pg, y, x in plan:
        by_pg[pg].append((k, y, x))

    ic2, looked = None, 0
    for pg in pl:
        lab_i = np.load(CACHE / f"{pg}.std.npy")
        a0_small = np.load(CACHE / f"{pg}.a0.npy")
        s = geo[pg]["s"]
        for k, y, x in by_pg[pg]:
            if ic2 and k > ic2["draws"]:
                break                      # これ以降の引きは既に当たった k より後
            looked += 1
            ys, xs = y // s, x // s
            if n_areas(a0_small[ys:ys + WIN // s, xs:xs + WIN // s]) != 1:
                continue                   # A0 の側が安いので先に見る(結果は順序に依らない)
            if n_areas(lab_i[y:y + WIN, x:x + WIN]) != 1:
                continue
            if ic2 is None or k < ic2["draws"]:
                ic2 = {"page": pg, "y": y, "x": x, "w": WIN, "h": WIN, "s": s,
                       "grid": f"{geo[pg]['W']}x{geo[pg]['H']}",
                       "canvas": geo[pg]["canvas"], "remainder": geo[pg]["rem"],
                       "alias": geo[pg]["alias"], "stratum": geo[pg]["stratum"],
                       "n_std_areas": 1, "n_a0_areas": 1, "role": "IC2", "draws": k}
            break
        del lab_i, a0_small
        print(f"  IC2 探索 {geo[pg]['alias']:6s} 済み"
              + (f" / いまの最良 引き{ic2['draws']}" if ic2 else ""), flush=True)

    # 重なり(**規則ではなく観察**)
    ov = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a["page"] == b["page"] and abs(a["x"] - b["x"]) < WIN \
                    and abs(a["y"] - b["y"]) < WIN:
                ov += 1

    print("\n== 母集団(**両方を名前を付けて書く**。規律9 / 修正2-B)==")
    print(f"  登録集合8頁 → 窓 32 / 問 35   (**登録本体。撤回しない**)")
    print(f"  窓を出せた頁 {len(geo)} → 窓 {len(rows)} / 問 {len(rows) + 3}  "
          f"(差分 {sum(1 for r in rows if r['stratum']=='差分')} / "
          f"非差分 {sum(1 for r in rows if r['stratum']=='非差分')})")
    print("\n== 窓を出せなかった頁(理由別。規律9)==")
    for k, v in skipped.most_common():
        print(f"  {v:6d}  {k}")
    print(f"  {sum(skipped.values()):6d}  計  —— {', '.join(skip_note) or 'なし'}")

    print("\n== 引き直し(理由別。規律9)==")
    for k, v in redraw.most_common():
        print(f"  {v:6d}  {k}")
    print(f"  {sum(redraw.values()):6d}  計")
    print("\n== 層の選別で落ちた数(m77。理由別)==")
    for k, v in collections.Counter(drops).most_common():
        print(f"  {v:6d}  {k}")

    print("\n== 観察(規則ではない)==")
    print(f"  同じ頁で重なる窓の対: {ov}  "
          f"(**重ならないという規則は登録に無いので足していない**)")
    deg = [r for r in rows if r["n_std_areas"] == 1 and r["n_a0_areas"] == 1]
    print(f"  **引き当ててしまった退化窓(標準も A0 も1面)の数: {len(deg)}**  "
          + (", ".join(f"{r['alias']}({r['x']},{r['y']})" for r in deg) or "なし"))
    print(f"    → **取り除かない。** 内容を見て窓を外すのは NC1 の誤り。"
          f"**既知セルが {1 + len(deg)} 個になることを IC2 の読みに書く**(条例 項7)")
    print(f"  標準 1 面だけの窓: "
          + (", ".join(f"{r['alias']}({r['x']},{r['y']})a0={r['n_a0_areas']}"
                       for r in rows if r["n_std_areas"] == 1) or "なし"))
    print(f"  窓の外に落ちる帯(余り): "
          + ", ".join(f"{geo[p]['alias']}={geo[p]['rem'] or '0x0'}" for p in pl))
    if ic2:
        print(f"  IC2 の退化窓: {ic2['page']} ({ic2['x']},{ic2['y']}) "
              f"—— **32 窓の後、引き {ic2['draws']} 回目**で当たった"
              f"(窓を {looked} 個見た)")
    else:
        print(f"  **IC2 の退化窓が見つからない。** {IC2_TRIES} 引きで標準も A0 も1面の窓が "
              f"無い → **IC2 は出せない。登録に戻って書き直す**")

    # IC3: **原寸 1:1 の点検**(大きさと、作業画布の内側に在ること)
    bad = [r for r in rows if r["w"] != WIN or r["h"] != WIN
           or r["x"] < 0 or r["y"] < 0
           or r["x"] + WIN > int(r["grid"].split("x")[0])
           or r["y"] + WIN > int(r["grid"].split("x")[1])]
    print("\n== IC3(原寸 1:1 と範囲の点検)==")
    print(f"  窓 {len(rows)} / 1024x1024 でないか範囲外: {len(bad)}  "
          f"→ {'**PASS**' if not bad else '**FAIL。出さない**'}")

    allr = rows + [ic2] if ic2 else list(rows)
    for i, r in enumerate(allr):
        r.setdefault("role", "ask")
        r["item_id"] = f"W{i:02d}"
    with open(OUT / "windows.csv", "w", encoding="utf-8") as fh:
        cols = ["item_id", "role", "alias", "stratum", "page", "x", "y", "w", "h",
                "s", "grid", "canvas", "remainder", "n_std_areas", "n_a0_areas"]
        fh.write(",".join(cols) + "\n")
        for r in allr:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\n書いた: {OUT/'windows.csv'}({len(allr)} 行 = "
          f"{len(rows)} 窓 + IC2 {1 if ic2 else 0})")
    print(f"**IC1 の複製2つと並べ替えは器具②で行う**({len(rows) + 3} 問。"
          f"登録本体の 35 問は修正2-B で 31 になった)")
    print(f"{time.time()-t0:.0f} 秒")
    return 0 if not bad and ic2 else 1


if __name__ == "__main__":
    sys.exit(main())

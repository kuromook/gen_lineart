#!/usr/bin/env python3
"""遮蔽の深さの採点(登録 `doc/prereg_occlusion_edge_20261107.md`、改訂1・改訂2 込み)。

    ./venv/bin/python tools/occlusion/score_occlusion_edge.py > results/occlusion_edge_scoring_20261109/edge_score.txt

材料は inbox/ の納品物(.gitignore 配下)と census の作者の答え(非公開)。印字するのは数と年代の札だけ。
"""
import collections
import csv
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score_occlusion import EXCLUDED, FILES, INBOX, ROOT, load_side, load_structure  # noqa: E402

EDGE = INBOX / "occ_edge_layers_20261109.csv"
FILL = ROOT.parent / "lineart-census" / "data" / "census_full" / "lineart_fill_answers_20261107.csv"
BINS = ["le1", "le2", "le3", "le4", "le6", "le8", "le12", "le16", "le24", "le32", "le48", "le64",
        "le128", "le256", "gt256", "nocontour"]
UPPER = {"le1": 1, "le2": 2, "le3": 3, "le4": 4, "le6": 6, "le8": 8, "le12": 12, "le16": 16, "le24": 24,
         "le32": 32, "le48": 48, "le64": 64, "le128": 128, "le256": 256}
FLOOR, BAR = 200, 0.10
P = print


def med(xs):
    return st.median(xs) if xs else float("nan")


def near(r, pre, T, denom):
    d = int(r[denom] or 0)
    if d == 0:
        return None
    return sum(int(r[f"{pre}{b}"]) for b in BINS if b in UPPER and UPPER[b] <= T) / d


def load_population():
    """登録の母集団と、改訂1 で外す層の鍵を返す(印字はしない)。"""
    info = load_structure()
    v4 = {}
    for f in FILES["v4"]:
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                v4[(r["page_path"], r["layer_id"])] = (r["value_status"], r["cov26_wide"])
    excl = set()
    with open(FILL, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["answered_by"] == "作者" and r["answer_raw_author"] in ("塗りつぶし", "その他"):
                excl.add((r["page_path"], r["layer_id"]))

    drop = collections.Counter()
    vis_mismatch = 0
    rows = []
    with open(EDGE, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            drop["行(納品)"] += 1
            k = (r["page_path"], r["layer_id"])
            if r["page_path"].endswith(EXCLUDED):
                drop["作者が恒久に外した頁"] += 1
                continue
            s = info.get(k)
            if s is None:
                drop["構造の表に無い"] += 1
                continue
            vis_mismatch += s["visible"] != (r["self_shown"] == "1" and r["ancestors_shown"] == "1")
            if not s["visible"]:
                drop["表示されていない(measured_visible でない)"] += 1
                continue
            if s["name"] != "線画":
                drop["名前が厳密に 線画 でない"] += 1
                continue
            vs = v4.get(k)
            if vs is None or vs[0] != "ok":
                drop["_v4 で ok でない・無い"] += 1
                continue
            if int(vs[1] or 0) <= 0:
                drop["cov26_wide = 0"] += 1
                continue
            r["_ic1"] = int(r["n_cov"]) == int(vs[1])
            rows.append(r)
    return rows, excl, drop, vis_mismatch


def main():
    rep, year, colour = load_side()
    rows, excl, drop, vis_mismatch = load_population()

    P("# 遮蔽の深さの採点(登録 `prereg_occlusion_edge_20261107.md` 改訂1・2)\n")
    P("## 母集団(落とした数を理由別に。規律9)")
    for k, v in drop.items():
        P(f"- {k}: {v:,}")
    P(f"- 母集団(線画 ∧ measured_visible ∧ ok ∧ cov26_wide > 0): {len(rows):,}")

    P("\n## 器具の点検")
    P(f"- IC1 `n_cov` ≠ `_v4` `cov26_wide`: {sum(not r['_ic1'] for r in rows)}(0 でなければ読まない)")
    P(f"- 見える の食い違い(registered_cells 対 納品の self_shown∧ancestors_shown、全行): {vis_mismatch}")
    bad = sum(1 for r in rows if sum(int(r[f"r_{b}"]) for b in BINS) != int(r["n_cov"]))
    P(f"- R の度数の和 ≠ n_cov: {bad}")
    in_excl = [r for r in rows if (r["page_path"], r["layer_id"]) in excl]
    P(f"- 改訂1 の外す層(作者が 塗りつぶし・その他): census の表で {len(excl)} 層、うち母集団に在る {len(in_excl)}")

    def score(pop, label, floor_s=FLOOR, show=True):
        d = collections.Counter()
        res = []
        for r in pop:
            if int(r["n_cov"]) < FLOOR:
                d["R の覆われた画素 < 200"] += 1
                continue
            nr = {T: near(r, "r_", T, "n_cov") for T in (8, 16, 32)}
            sk = {}
            for k in range(1, 6):
                if int(r[f"s{k}_n_cov"]) >= floor_s:
                    sk[k] = {T: near(r, f"s{k}_", T, f"s{k}_n_cov") for T in (8, 16, 32)}
            if not sk:
                d["S の使える回が 0"] += 1
                continue
            u = near(r, "u_", 16, "occ_area")
            res.append((r, nr, sk, u))
        out = {"n": len(res), "drop": d}
        diff = {T: [nr[T] - med([s[T] for s in sk.values()]) for _, nr, sk, _ in res] for T in (8, 16, 32)}
        out["E1"] = {T: med(diff[T]) for T in diff}
        perk = {}
        smed = {}
        for k in range(1, 6):
            xs = [(nr[16], sk[k][16]) for _, nr, sk, _ in res if k in sk]
            perk[k] = med([a - b for a, b in xs])
            smed[k] = med([b for _, b in xs])
        out["perk"], out["smed"] = perk, smed
        out["scatter"] = max(smed.values()) - min(smed.values())
        out["R16"] = med([nr[16] for _, nr, _, _ in res])
        out["S16"] = med([med([s[16] for s in sk.values()]) for _, _, sk, _ in res])
        out["U16"] = med([u for *_, u in res if u is not None])
        bands = collections.Counter("≥0.8" if nr[16] >= 0.8 else ("≤0.2" if nr[16] <= 0.2 else "0.2〜0.8")
                                    for _, nr, _, _ in res)
        out["bands"] = bands
        out["res"] = res
        if show:
            P(f"\n### {label}")
            P(f"- 読んだ層: {len(res):,}" + "".join(f" / 落とした({k}) {v:,}" for k, v in d.items()))
            P(f"- near_16 の層の中央値: R {out['R16']:.4f} / S(5回の中央値) {out['S16']:.4f} / U(併記、帰無値ではない) {out['U16']:.4f}")
            P(f"- **E1 主値 = median[near_16(R) − near_16(S 中央値)] = {out['E1'][16]:+.4f}**(棒 +{BAR:.2f})"
              f" / T=8 {out['E1'][8]:+.4f} / T=32 {out['E1'][32]:+.4f}")
            P("- 回ごと(near_16): " + " / ".join(f"S{k} {perk[k]:+.4f}" for k in perk))
            P(f"- S の散らばり(回ごとの near_16(S) 中央値の 最大−最小): {out['scatter']:.4f}")
            P("- E2 帯(near_16(R)、仮の名前): " + " / ".join(f"{b} {bands.get(b, 0):,}" for b in ("≥0.8", "0.2〜0.8", "≤0.2")))
        return out

    def verdict(o):
        e = o["E1"][16]
        if e < 0:
            return "向きが逆 —— 読まない(器具を見直す)"
        if e <= o["scatter"]:
            return "判定保留(主値が S の散らばり以下)"
        if any(v < BAR for v in o["perk"].values()):
            return "判定保留(S の回のどれかで届かない)" if e >= BAR else "届かない"
        return "超えた" if e >= BAR else "届かない"

    main_pop = [r for r in rows if (r["page_path"], r["layer_id"]) not in excl]
    P("\n## E1(主: 改訂1 で外した後)")
    o = score(main_pop, "主(作者が線でないと答えた層を外した後)")
    P(f"\n**判定: {verdict(o)}**")
    P("\n## 隣に印字(棒ではない)")
    o_all = score(rows, "改訂1 で外す前")
    P(f"  判定の形: {verdict(o_all)}")
    o_f1 = score(main_pop, "S の床を 1 にした版(改訂2 §2)", floor_s=1)
    P(f"  判定の形: {verdict(o_f1)}")

    P("\n## 層(畳まない。判定は上の全体で)")
    def strat(name, key):
        g = collections.defaultdict(list)
        for r in main_pop:
            g[key(r)].append(r)
        P(f"\n#### {name}")
        P("| 層 | 読んだ層 | R near_16 | S near_16 | E1 | 回ごと最小 | 判定の形 |")
        P("|---|---:|---:|---:|---:|---:|---|")
        for k in sorted(g, key=str):
            oo = score(g[k], "", show=False)
            if oo["n"] == 0:
                P(f"| {k} | 0 | | | | | |")
                continue
            P(f"| {k} | {oo['n']:,} | {oo['R16']:.4f} | {oo['S16']:.4f} | {oo['E1'][16]:+.4f} | {min(oo['perk'].values()):+.4f} | {verdict(oo)} |")
    strat("year_label", lambda r: year.get(r["page_path"], "?"))
    strat("色原稿か", lambda r: "色" if r["page_path"] in colour else "色でない")
    strat("上の遮蔽物にクリップ層を含むか", lambda r: "含む" if r["has_clip_occluder"] == "1" else "含まない")
    strat("代表のみ", lambda r: "代表" if rep.get(r["page_path"]) else "代表でない")
    P("\n## NC2(全面の遮蔽物の下)")
    strat("full_canvas_occluder", lambda r: "全面あり" if r["full_canvas_occluder"] == "1" else "全面なし")


if __name__ == "__main__":
    main()

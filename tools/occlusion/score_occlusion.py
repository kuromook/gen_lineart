#!/usr/bin/env python3
"""遮蔽の採点(事前登録 `doc/prereg_occlusion_20261027.md`、読み方は記録12・20・21)。

  ./venv/bin/python tools/occlusion/score_occlusion.py --version v4 > <out.txt>
  ./venv/bin/python tools/occlusion/score_occlusion.py --version v3   # 「作業用の層を含む」

材料はどれも inbox/ の納品物(.gitignore 配下)。印字するのは数と年代の札だけで、頁の名前は出さない。
母集団の述語は registered_cells.py のものを import する(述語を2か所に持たない)。
"""
import argparse
import collections
import csv
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from registered_cells import ancestor_hidden, is_folder, self_on  # noqa: E402

INBOX = ROOT / "inbox"
FILES = {
    "v4": [INBOX / f"occ_full_p{k}_layers_v4_20261103.csv" for k in (1, 2, 3)],
    "v3": [INBOX / f"occ_full_p{k}_layers_v3_20261101.csv" for k in (1, 2, 3)],
}
EXCLUDED = "2018\\illust20172\\page0031.clip"   # 作者が恒久に外した頁(名前で落とす。中身は出さない)
C1_BAR, C3_BAR, C2_BAR, C2_TOL, C2_MIN = 0.05, 0.05, 0.80, 0.1, 0.1


def med(xs):
    return st.median(xs) if xs else float("nan")


def load_structure():
    pages = collections.defaultdict(list)
    with open(INBOX / "census_r8_layer_structure_20261027.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pages[r["page_path"]].append(r)
    info = {}
    for pp, rows in pages.items():
        by_id = {r["layer_id"]: r for r in rows}
        for r in rows:
            if is_folder(r) or r["z_global"] == "":
                continue
            info[(pp, r["layer_id"])] = {
                "name": r["layer_name"], "kind": r["v_layer_kind"],
                "visible": self_on(r) and not ancestor_hidden(r, by_id),
            }
    return info


def load_side():
    rep, year = {}, {}
    with open(INBOX / "page_representative.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rep[r["page_path"]] = r["is_representative"] == "yes"
            year[r["page_path"]] = r["year_label"]
    colour = set()
    with open(INBOX / "m_colour_page_layers_872.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            colour.add(r["page_path"])
    return rep, year, colour


def five_class(r):
    """五分法(T60 の分け方を列から組み直した形): 初期値 × 保存の量。"""
    init, blocks, stored = r["body_init"], int(r["body_blocks"] or 0), int(r["body_stored"] or 0)
    if init == "255":
        return 1 if stored == 0 else (4 if stored >= blocks else 5)
    return 2 if stored == 0 else 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=FILES, default="v4")
    a = ap.parse_args()
    P = print
    info = load_structure()
    rep, year, colour = load_side()

    drop = collections.Counter()
    rows = []
    for f in FILES[a.version]:
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                drop["行(材料)"] += 1
                if r["page_path"].endswith(EXCLUDED):
                    drop["作者が恒久に外した頁"] += 1
                    continue
                s = info.get((r["page_path"], r["layer_id"]))
                if s is None:
                    drop["構造の表に無い"] += 1
                    continue
                if not s["visible"]:
                    drop["表示されていない(自身か祖先)"] += 1
                    continue
                r.update(s)
                rows.append(r)

    P(f"# 遮蔽の採点 —— 材料 `_{a.version}`" + ("(作業用の層を含む)" if a.version == "v3" else "(主。作者の 11-03 の裁定入り)"))
    P("\n## 母集団(落とした数を理由別に。規律9)")
    for k, v in drop.items():
        P(f"- {k}: {v:,}")
    P(f"- measured_visible: {len(rows):,}")
    vs = collections.Counter(r["value_status"] for r in rows)
    P("- value_status: " + " / ".join(f"{k} {v:,}" for k, v in vs.most_common()))

    # ---- 器具の点検(数を読む前に) ----
    P("\n## 器具の点検")
    bad_cap = sum(1 for r in rows if r["value_status"] == "ok" and any(
        int(r[c] or 0) > int(r["n_drawn26"]) for c in ("cov26_strict", "cov26_wide", "cov26_all",
                                                       "p1_cov26", "p2_cov26", "p3_cov26", "p4_cov26", "p5_cov26")))
    P(f"- 覆われた画素 > 描画画素 の行: {bad_cap}(0 でなければ読まない)")
    for o in ("strict", "wide", "all"):
        kc = [r for r in rows if r["value_status"] == "ok" and r[f"n_occ_above_{o}"] == "0"]
        nz = sum(1 for r in kc if int(r[f"cov26_{o}"] or 0) != 0)
        P(f"- IC1 `{o}`: 上に遮蔽物が無い ok の層 {len(kc):,} のうち被覆が 0 でない {nz}(0 でなければ読まない)")
    mono = sum(1 for r in rows if r["value_status"] == "ok" and not (
        int(r["cov26_strict"] or 0) <= int(r["cov26_wide"] or 0) <= int(r["cov26_all"] or 0)))
    P(f"- 包含 strict ≤ wide ≤ all が破れる ok の層: {mono}(集合が包含なので 0 のはず)")
    occ = {o: sum(1 for r in rows if r[f"is_occ_{o}"] == "1") for o in ("strict", "wide", "all")}
    P(f"- 遮蔽物の数(IC4 の種類の量): strict {occ['strict']:,} / wide {occ['wide']:,} / all {occ['all']:,}")
    if bad_cap or mono:
        P("**器具の点検に落ちた。採点しない。**")
        return

    ok = [r for r in rows if r["value_status"] == "ok"]
    for r in ok:
        d = int(r["n_drawn26"])
        r["c"] = {o: int(r[f"cov26_{o}"]) / d for o in ("strict", "wide", "all")}
        r["c1"] = int(r["cov1_wide"]) / int(r["n_alpha1"]) if int(r["n_alpha1"] or 0) else None
        r["p"] = [int(r[f"p{k}_cov26"]) / d for k in range(1, 6)]

    # ---- A0p の用紙層 ----
    by_page = collections.defaultdict(list)
    for r in rows:
        by_page[r["page_path"]].append(r)
    paper = {pp: [r for r in rs if r["kind"] == "paper" and r["is_occ_wide"] == "1"] for pp, rs in by_page.items()}
    npap = collections.Counter(min(len(v), 2) for v in paper.values())
    P(f"- 用紙の `wide` 遮蔽物が 0 / 1 / 2枚以上 の頁: {npap[0]:,} / {npap[1]:,} / {npap[2]:,}")

    def a0p_values(rs):
        vals, cnt = [], collections.Counter()
        for r in rs:
            ps = paper[r["page_path"]]
            if len(ps) >= 2:
                cnt["用紙層が2枚以上の頁"] += 5
                continue
            for k in range(5):
                if not ps:
                    vals.append(r["p"][k]); cnt["用紙層が無い頁(全部入れる)"] += 1
                    continue
                pr = ps[0]
                if pr is r:
                    cnt["用紙層そのもの"] += 1
                    continue
                np_, nl = int(pr[f"p{k+1}_n_occ_above"]), int(r[f"p{k+1}_n_occ_above"])
                if np_ > nl:
                    vals.append(r["p"][k]); cnt["用紙層が奥(入れる)"] += 1
                elif np_ < nl:
                    cnt["用紙層が手前(外す)"] += 1
                else:
                    cnt["同値(決められない。外す)"] += 1
        return vals, cnt

    # ---- C1 ----
    def c1(rs, label, detail=True):
        m = med([r["c"]["wide"] for r in rs])
        pooled = med([v for r in rs for v in r["p"]])
        per = [med([r["p"][k] for r in rs]) for k in range(5)]
        vp, cnt = a0p_values(rs)
        mp = med(vp)
        diff, diffp = m - pooled, m - mp
        if diff >= C1_BAR and all(m - x >= C1_BAR for x in per):
            v = "超えた"
        elif diff >= C1_BAR:
            v = "判定保留(置換のどれかで届かない)"
        else:
            v = "届かない"
        if detail:
            P(f"\n## C1 重なり順は情報を持つか —— {label}(線画 {len(rs):,} 層)")
            P(f"- 実測 `wide` の中央値: **{m:.4f}**")
            P(f"- A0(主、置換。層×5回を畳む): **{pooled:.4f}** → 差 **{diff:+.4f}**(棒 +{C1_BAR})")
            P(f"- A0 の置換ごとの中央値: " + " / ".join(f"{x:.4f}" for x in per) +
              f" → 差の最小 {min(m - x for x in per):+.4f}")
            P(f"- A0p(併記。用紙層が奥の置換だけ): **{mp:.4f}** → 差 **{diffp:+.4f}**({len(vp):,} 対)")
            P("  - A0p の内訳: " + " / ".join(f"{k} {n:,}" for k, n in cnt.most_common()))
            P(f"- **判定(A0、主): {v}**")
            vp_ = "越えた" if diffp >= C1_BAR else "届かない"
            P(f"- A0p では: {vp_}" + ("(A0 で届かず A0p で越えた → 判定保留。記録12 §2)" if v == "届かない" and diffp >= C1_BAR else ""))
            c1s = [r["c1"] for r in rs if r["c1"] is not None]
            P(f"- 診断: アルファ ≥ 1 の分母で読んだ実測の中央値 {med(c1s):.4f}({len(c1s):,} 層)/ "
              f"`strict` {med([r['c']['strict'] for r in rs]):.4f} / `all` {med([r['c']['all'] for r in rs]):.4f}(帯の端。帰無値と比べない)")
        return m, pooled, diff, v

    line = [r for r in ok if r["name"] == "線画"]
    lvs = collections.Counter(r["value_status"] for r in rows if r["name"] == "線画")
    P(f"\n- `線画` 層の measured_visible: {sum(lvs.values()):,}(" + " / ".join(f"{k} {v:,}" for k, v in lvs.most_common()) + ")")
    c1(line, "全体(判定はこれで下す)")

    # ---- C3 ----
    P(f"\n## C3 透過するものの調査は先か後か —— 線画 {len(line):,} 層")
    a_ = med([r["c"]["wide"] for r in line]) - med([r["c"]["strict"] for r in line])
    b_ = med([r["c"]["wide"] - r["c"]["strict"] for r in line])
    P(f"- (a) 中央値の差 wide − strict: **{a_:+.4f}** / (b) 層ごとの差の中央値: **{b_:+.4f}**(棒 {C3_BAR})")
    P(f"- 層ごとの差が 0 でない層: {sum(1 for r in line if r['c']['wide'] != r['c']['strict']):,}")
    v3 = "超えた" if a_ >= C3_BAR and b_ >= C3_BAR else ("届かない" if a_ < C3_BAR and b_ < C3_BAR else "判定保留(2つの形で割れた)")
    P(f"- **判定: {v3}**")

    # ---- C2 ----
    P(f"\n## C2 画素走行は構造の表に何を足すか —— ok の全層 {len(ok):,}")
    nodec = 0
    for pp, rs in by_page.items():
        n = len(rs)
        zs = sorted(int(r["z_global"]) for r in rs)
        for r in rs:
            if n < 2:
                r["front"] = None
                continue
            z = int(r["z_global"])
            ahead = n - 1 - _rank_le(zs, z)
            r["front"] = ahead / (n - 1)
    for r in ok:
        if r.get("front") is None:
            nodec += 1
            r["dec"] = None
        else:
            r["dec"] = min(int(r["front"] * 10), 9)
    P(f"- 前方位置: 0 = 最も手前、1 = 最も奥。等幅10区間 [0,0.1) … [0.9,1.0]。測る層が1つで位置の無い ok の層: {nodec:,}(A3 の升を持たないので A3 では外す)")
    cells = collections.defaultdict(list)
    for r in ok:
        if r["dec"] is not None:
            cells[(r["dec"], r["kind"])].append(r["c"]["wide"])
    a3 = {k: med(v) for k, v in cells.items()}
    n0 = med([r["c"]["wide"] for r in ok])
    pmed = {pp: med([r["c"]["wide"] for r in rs if r["value_status"] == "ok"] or [float("nan")]) for pp, rs in by_page.items()}
    tgt = [r for r in ok if r["c"]["wide"] >= C2_MIN]
    tgt_a3 = [r for r in tgt if r["dec"] is not None]
    hit = lambda pred, r: abs(pred - r["c"]["wide"]) <= C2_TOL  # noqa: E731
    h_n0 = sum(hit(n0, r) for r in tgt) / len(tgt)
    h_n0p = sum(hit(pmed[r["page_path"]], r) for r in tgt) / len(tgt)
    h_a3 = sum(hit(a3[(r["dec"], r["kind"])], r) for r in tgt_a3) / len(tgt_a3)
    P(f"- 採点の層(被覆率 ≥ {C2_MIN}): {len(tgt):,}(うち A3 の升を持つ {len(tgt_a3):,})。A3 の升 {len(a3)}")
    P(f"- N0(主の帰無値、定数 {n0:.4f}): **{h_n0:.1%}**")
    P(f"- N0'(併記、頁ごとの中央値): {h_n0p:.1%}")
    P(f"- A3(十分位 × kind の頻度表、同じ層で作って当てる): **{h_a3:.1%}**(棒 {C2_BAR:.0%})")
    if h_n0 >= C2_BAR:
        P("- **判定: 読まない**(N0 が棒に届いた → C1 の答に合流する。改訂1)")
    else:
        P(f"- **判定: {'超えた(構造の表で足りる)' if h_a3 >= C2_BAR else '届かない(画素の実測にしか無い情報が在る)'}**")

    # ---- 内訳(判定に使わない) ----
    P("\n## 内訳(判定に使わない。改訂2)")
    rep_line = [r for r in line if rep.get(r["page_path"], False)]
    m, pooled, diff, v = c1(rep_line, "代表のみ", detail=False)
    P(f"- **代表のみ**(作者の裁定「古い方」、`is_representative = yes`): 線画 {len(rep_line):,} 層、実測 {m:.4f} / A0 {pooled:.4f} / 差 {diff:+.4f} → {v}")
    P("\n| 群 | 札 | 線画の層 | 実測 | A0 | 差 |\n|---|---|---:|---:|---:|---:|")
    grp = collections.defaultdict(list)
    for r in line:
        grp[("色原稿" if r["page_path"] in colour else "それ以外", year.get(r["page_path"], "?"))].append(r)
    for (g, y), rs in sorted(grp.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        m, pooled, diff, _ = c1(rs, "", detail=False)
        P(f"| {g} | `{y}` | {len(rs):,} | {m:.4f} | {pooled:.4f} | {diff:+.4f} |")
    P("\n- 五分法のクラス(線画の ok の層。class 1・5 は分母が帯なので値は片端):")
    cl = collections.defaultdict(list)
    for r in line:
        cl[five_class(r)].append(r)
    for k in sorted(cl):
        rs = cl[k]
        P(f"  - class {k}: {len(rs):,} 層、実測 {med([r['c']['wide'] for r in rs]):.4f} / A0 {med([v for r in rs for v in r['p']]):.4f}"
          + ("(片端)" if k in (1, 5) else ""))


def _rank_le(zs, z):
    """z 以下の値の数 − 1 = 自分より奥の層の数(z は頁内で重ならない)。"""
    import bisect
    return bisect.bisect_right(zs, z) - 1


if __name__ == "__main__":
    main()

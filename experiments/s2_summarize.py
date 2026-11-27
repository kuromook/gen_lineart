"""登録 S2 の本番のまとめ: 作品ごとに1行(プールしない)。コマの区分の数と、下絵の在るコマ(保留を除く)の 残/動/足/捨 の中央値。
S1 の頁の値(保留を除く、同じ頁)を並べる。H1/H1b で作者が一言を書いた頁は、器具の区分と並べて印字する。
"""
import csv
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from s2_panel_delta import s1_rows  # noqa: E402

FULL = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results/s2_full")
OUT = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "results/s2_full_by_work.csv")
H1_NOTES = ["h02", "h03", "h04b", "h05", "h08", "h11", "h16", "h20"]


def med(xs):
    xs = [float(x) for x in xs if x not in ("", "nan") and float(x) == float(x)]
    return round(st.median(xs), 3) if xs else float("nan")


def main():
    S1 = s1_rows()
    s1_by_line = {q["line"]: s for q, s in S1.values()}
    out, allk, allp = [], [], []
    for d in sorted(p for p in FULL.iterdir() if p.is_dir()):
        K = list(csv.DictReader(open(d / "s2_panels.csv", encoding="utf-8"))) if (d / "s2_panels.csv").exists() else []
        Pg = list(csv.DictReader(open(d / "s2_pages.csv", encoding="utf-8"))) if (d / "s2_pages.csv").exists() else []
        log = (d / "log.txt").read_text(encoding="utf-8").strip().splitlines()
        drops = log[-1] if log else ""
        allk += K; allp += Pg
        present = [k for k in K if k["kind"] == "下絵の在るコマ"]
        absent = [k for k in K if k["kind"] == "下絵の無いコマ"]
        few = [k for k in K if k["kind"] == "線の少ないコマ"]
        use = [k for k in present if k["hold"] != "True"]
        Lsum = sum(int(k["L"]) for k in present + absent) or 1
        s1use = [s1_by_line[p["line"]] for p in Pg if s1_by_line[p["line"]]["hold"] != "True"]
        out.append(dict(work_id=d.name, pages=len(Pg), panels=len(K), present=len(present), absent=len(absent), few_line=len(few),
                        absent_share_n=round(len(absent) / max(len(present) + len(absent), 1), 3),
                        absent_share_L=round(sum(int(k["L"]) for k in absent) / Lsum, 3),
                        held=len(present) - len(use), used=len(use),
                        keep=med(k["keep"] for k in use), moved=med(k["moved"] for k in use),
                        added=med(k["added"] for k in use), dropped=med(k["dropped"] for k in use),
                        N1_delta=med(k["N1_delta_keep"] for k in use), near_edge=med(k["near_edge_share"] for k in use),
                        s1_pages_used=len(s1use), s1_keep=med(s["B_keep"] for s in s1use), s1_added=med(s["B_added"] for s in s1use),
                        ground=sum(int(p["ground"]) for p in Pg), small=sum(int(p["small_dropped"]) for p in Pg),
                        C1_fail=sum(float(k["C1_keep"]) < 0.999 for k in K),
                        faint=med(p.get("faint_share", "nan") for p in Pg), drops=drops))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    kinds = {}
    for k in allk:
        kinds[k["kind"]] = kinds.get(k["kind"], 0) + 1
    print(f"頁 {len(allp)} / コマ {len(allk)} {kinds} / 地 {sum(o['ground'] for o in out)} / 小さいコマ落とし {sum(o['small'] for o in out)}"
          f" / C1 失敗 {sum(o['C1_fail'] for o in out)}")
    print(f"{'work_id':26s} 頁 コマ 在 無 少 | 無の割(数/線) | 保留 使 | 残 動 足 捨 | S1頁 残 足 | N1Δ 境")
    for o in out:
        print(f"{o['work_id']:26s} {o['pages']:3d} {o['panels']:3d} {o['present']:3d} {o['absent']:3d} {o['few_line']:2d} | "
              f"{o['absent_share_n']:.2f}/{o['absent_share_L']:.2f} | {o['held']:3d} {o['used']:3d} | {o['keep']:.2f} {o['moved']:.2f} "
              f"{o['added']:.2f} {o['dropped']:.2f} | {o['s1_pages_used']:3d} {o['s1_keep']:.2f} {o['s1_added']:.2f} | "
              f"{o['N1_delta']:+.2f} {o['near_edge']:.2f} 薄{o['faint']:.2f}  {o['drops']}")
    # H1/H1b の一言の頁
    sel = {r["item_id"]: r for r in csv.DictReader(open(ROOT / "data/h1_sketch_stage_20261120/selection.csv", encoding="utf-8"))}
    sel.update({r["item_id"]: r for r in csv.DictReader(open(ROOT / "data/h1b_reask_20261126/selection.csv", encoding="utf-8"))})
    ans = {}
    import json
    for f in ("data/h1_sketch_stage_20261120/s_h1_sketch_stage_20261120_answers.jsonl", "data/h1b_reask_20261126/s_h1b_reask_20261126_answers.jsonl"):
        for l in open(ROOT / f, encoding="utf-8"):
            x = json.loads(l); ans[x["item_id"]] = x
    print("\nH1/H1b で作者が「一部のコマに下絵が無い」と書いた頁と、器具の区分")
    for i in H1_NOTES:
        s = sel[i]
        line = s.get("line") or next(q["line"] for q, r in S1.values()
                                     if q["work_id"] == s["work_id"] and q["page"] == s["page"] and r["B_keep"] == s["B_keep"])
        ks = [k for k in allk if k["line"] == line]
        desc = " ".join(f"k{k['koma_id']}:{ {'下絵の在るコマ': '在', '下絵の無いコマ': '無', '線の少ないコマ': '少'}[k['kind']]}" for k in ks)
        print(f"  {i} {s['work_id']} {s['page']} コマ {len(ks)} 無 {sum(k['kind'] == '下絵の無いコマ' for k in ks)} [{desc}] | 作者: {ans[i]['comment']}")
    print(f"表: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

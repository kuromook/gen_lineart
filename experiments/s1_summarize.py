"""登録 S1 の本番のまとめ: 作品ごとに1行(プールしない)。頁ごとの値の中央値と、判定保留・ずれ優勢の頁の数。

主値 = 腕B の 残/動/足(線画の骨格、塗りを外す)と 捨(下絵の骨格)。判定保留の頁は主値の中央値から外し、数を印字する。
"""
import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results/s1_full"
OUT = ROOT / "results/s1_full_by_work.csv"


def med(rows, k):
    v = [float(r[k]) for r in rows if r[k] not in ("nan", "")]
    v = [x for x in v if x == x]
    return round(st.median(v), 3) if v else float("nan")


def main():
    allrows, out = [], []
    for f in sorted(FULL.glob("*/s1_pages.csv")):
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        allrows += rows
        hold = [r for r in rows if r["hold"] == "True"]
        hold_noN2 = sum(r["N2_keep"] in ("nan", "") for r in hold)  # 床が測れない(同じ大きさの別頁が無い)
        use = [r for r in rows if r["hold"] != "True"]
        mis = sum(r["moved_mostly_misalignment"] == "True" for r in use)
        clip = sum(r["B_clipped"] == "True" for r in rows)
        c1bad = sum(float(r["C1_keep"]) < 0.99 for r in rows)
        out.append(dict(work_id=f.parent.name, pages=len(rows), hold=len(hold), hold_noN2=hold_noN2, hold_below_floor=len(hold) - hold_noN2, used=len(use),
                        B_keep=med(use, "B_keep"), B_moved=med(use, "B_moved"), B_added=med(use, "B_added"),
                        B_dropped=med(use, "B_dropped"), A_keep=med(use, "A_keep"),
                        N1_delta_keep=med(use, "N1_delta_keep"), N2_keep=med(rows, "N2_keep"),
                        moved_mostly_misalignment=mis, fill_skel_excluded=med(rows, "fill_skel_excluded"),
                        shift_clipped=clip, C1_fail=c1bad))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"頁 {len(allrows)} / 作品 {len(out)} / 判定保留 {sum(o['hold'] for o in out)} 頁"
          f"(床が測れない {sum(o['hold_noN2'] for o in out)} / 床と分けられない {sum(o['hold_below_floor'] for o in out)}) / "
          f"C1 失敗 {sum(o['C1_fail'] for o in out)} 頁 / 移動が端 {sum(o['shift_clipped'] for o in out)} 頁")
    hdr = f"{'work_id':26s} {'頁':>3s} {'保留':>3s} {'無床':>3s} | {'残':>5s} {'動':>5s} {'足':>5s} {'捨':>5s} | {'A残':>5s} {'N1Δ':>6s} {'N2残':>5s} {'ずれ優勢':>4s} {'塗り外':>5s}"
    print(hdr)
    for o in out:
        print(f"{o['work_id']:26s} {o['pages']:3d} {o['hold']:3d} {o['hold_noN2']:3d} | {o['B_keep']:5.3f} {o['B_moved']:5.3f} {o['B_added']:5.3f} "
              f"{o['B_dropped']:5.3f} | {o['A_keep']:5.3f} {o['N1_delta_keep']:+6.3f} {o['N2_keep']:5.3f} {o['moved_mostly_misalignment']:4d} "
              f"{o['fill_skel_excluded']:5.3f}")
    print(f"表: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""登録 S5: コマの中の 64px 格子の濃さの相関 G(抽出側 pair_qc.grid_correlation のコマ版)。S4 の「下絵の在るコマ」に付ける。"""
import csv, glob, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from s1_cleanup_delta import find  # noqa: E402
from s2_panel_delta import PAGES, SHAPES, s1_rows  # noqa: E402
from h2_build_hearing import SKIP_WORKS, _shift_gray  # noqa: E402

CELL, MIN_CELLS = 64, 30


def main():
    S1 = s1_rows()
    q_by_line = {q["line"]: q for q, _ in S1.values()}
    s1_by_line = {q["line"]: s for q, s in S1.values()}
    pages = {r["page_path"]: r for r in csv.DictReader(open(PAGES, encoding="utf-8"))}
    K = []
    for f in sorted(glob.glob(str(ROOT / "results/s4_full/*/s2_panels.csv"))):
        K += [k for k in csv.DictReader(open(f, encoding="utf-8")) if k["kind"] == "下絵の在るコマ"]
    K = [k for k in K if k["work_id"] not in SKIP_WORKS and q_by_line[k["line"]]["is_primary"] == "True"]
    by_line = {}
    for k in K:
        by_line.setdefault(k["line"], []).append(k)
    out, dropped = [], 0
    for n, (line, ks) in enumerate(sorted(by_line.items())):
        q, s = q_by_line[line], s1_by_line[line]
        lab = cv2.imread(str(SHAPES / Path(pages[q["source_path"]]["out_file"]).name), cv2.IMREAD_UNCHANGED)
        ln = 255 - cv2.imread(str(find(q, "line")), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        sk = 255 - _shift_gray(cv2.imread(str(find(q, "sketch")), cv2.IMREAD_GRAYSCALE), int(s["B_dx"]), int(s["B_dy"])).astype(np.float32)
        for k in ks:
            m = (lab == int(k["koma_id"]))
            ys, xs = np.nonzero(m)
            y0, x0 = ys.min(), xs.min()
            h, w = (ys.max() + 1 - y0) // CELL, (xs.max() + 1 - x0) // CELL
            if h * w == 0:
                dropped += 1; continue
            sl = (slice(y0, y0 + h * CELL), slice(x0, x0 + w * CELL))
            cell = lambda a: a[sl].reshape(h, CELL, w, CELL).mean(axis=(1, 3))
            inside = cell(m.astype(np.float32)) >= 0.5
            a, b = cell(ln)[inside], cell(sk)[inside]
            if inside.sum() < MIN_CELLS or a.std() < 1e-6 or b.std() < 1e-6:
                dropped += 1; continue
            out.append(dict(work_id=k["work_id"], line=line, koma_id=k["koma_id"], cells=int(inside.sum()),
                            G=round(float(np.corrcoef(a, b)[0, 1]), 4), C=k["C"], keep=k["keep"]))
        if n % 50 == 0:
            print(f"  {n}/{len(by_line)} 頁", flush=True)
    out.sort(key=lambda r: r["G"])
    for i, r in enumerate(out):
        r["rank"] = i + 1
    with open(ROOT / "results/s5_grid_corr.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    n5 = max(1, round(len(out) * 0.05))
    print(f"コマ {len(K)} → G あり {len(out)}(落とした: セル {MIN_CELLS} 未満・濃さが一様 {dropped})/ 下位5% = {n5}")
    from collections import Counter
    print("下位5% の作品:", dict(Counter(r["work_id"] for r in out[:n5]).most_common()))
    p04 = [r for r in out if r["line"] == "015_2017_pink_pink_page0041_line.jpg" and r["koma_id"] == "3"]
    print("既知セル H2 p04:", p04[0] if p04 else "無し")
    qs = np.quantile([r["G"] for r in out], [0.05, 0.25, 0.5, 0.75])
    print("G の 5/25/50/75%:", np.round(qs, 3))


if __name__ == "__main__":
    main()

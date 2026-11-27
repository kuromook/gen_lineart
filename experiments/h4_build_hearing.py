"""聞き取り H4 の問いを作る(登録: doc/work_log.md 2026-11-27「登録 S5」)。G の下位5% から14・上位半分から対照6、作品を巡回、種 20261128。
H1〜H3 に出た頁・コマは除く。出力 data/h4_composition_20261128/(追跡しない)。"""
import csv, json, random, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from s1_cleanup_delta import binarize, find, shift  # noqa: E402
from s2_panel_delta import PAGES, SHAPES, s1_rows  # noqa: E402
from s3_binarize import binarize_sketch  # noqa: E402
from h2_build_hearing import _shift_gray, pick  # noqa: E402

OUT = ROOT / "data/h4_composition_20261128"
SEED = 20261128


def main():
    S1 = s1_rows()
    q_by_line = {q["line"]: q for q, _ in S1.values()}
    s1_by_line = {q["line"]: s for q, s in S1.values()}
    h1 = list(csv.DictReader(open(ROOT / "data/h1_sketch_stage_20261120/selection.csv", encoding="utf-8")))
    skip_lines = {next(q["line"] for q, s in S1.values() if q["work_id"] == r["work_id"] and q["page"] == r["page"] and s["B_keep"] == r["B_keep"]) for r in h1}
    skip_lines |= {r["line"] for r in csv.DictReader(open(ROOT / "data/h1b_reask_20261126/selection.csv", encoding="utf-8"))}
    asked = set()
    for d in ("h2_panel_sketch_20261127", "h3_panel_sketch_20261127"):
        asked |= {(r["line"], r["koma_id"]) for r in csv.DictReader(open(ROOT / "data" / d / "selection.csv", encoding="utf-8"))}
    G = list(csv.DictReader(open(ROOT / "results/s5_grid_corr.csv", encoding="utf-8")))
    n5 = max(1, round(len(G) * 0.05))
    ok = lambda r: r["line"] not in skip_lines and (r["line"], r["koma_id"]) not in asked
    low = [r for r in G[:n5] if ok(r)]
    high = [r for r in G[len(G) // 2:] if ok(r)]
    rng = random.Random(SEED)
    sel = [("low", r) for r in pick(low, 14, rng)] + [("control", r) for r in pick(high, 6, rng)]
    print(f"下位5% {n5}(除いて {len(low)})/ 上位半分 {len(G) - len(G) // 2}(除いて {len(high)})→ {len(sel)}")
    rng.shuffle(sel)
    pages = {r["page_path"]: r for r in csv.DictReader(open(PAGES, encoding="utf-8"))}
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    items = []
    with open(OUT / "selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "kind", "work_id", "line", "koma_id", "G", "rank", "C"])
        for i, (kind, k) in enumerate(sel, 1):
            iid = f"c{i:02d}"
            q = q_by_line[k["line"]]; s = s1_by_line[k["line"]]
            dx, dy = int(s["B_dx"]), int(s["B_dy"])
            lab = cv2.imread(str(SHAPES / Path(pages[q["source_path"]]["out_file"]).name), cv2.IMREAD_UNCHANGED)
            inside = lab == int(k["koma_id"])
            ys, xs = np.nonzero(inside)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            m = inside[y0:y1, x0:x1]
            ln = cv2.imread(str(find(q, "line")), cv2.IMREAD_GRAYSCALE)
            sk = _shift_gray(cv2.imread(str(find(q, "sketch")), cv2.IMREAD_GRAYSCALE), dx, dy)
            Lb = binarize(find(q, "line"))[0]
            Sb = shift(binarize_sketch(find(q, "sketch"), "T1"), dx, dy)
            ov = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
            ov[Sb[y0:y1, x0:x1]] = (170, 170, 170)
            ov[Lb[y0:y1, x0:x1]] = (0, 0, 0)
            for name, im in (("sketch", cv2.cvtColor(sk[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)),
                             ("line", cv2.cvtColor(ln[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)), ("overlay", ov)):
                im = im.copy()
                im[~m] = (im[~m] * 0.25 + 255 * 0.75).astype(np.uint8)
                cv2.imwrite(str(OUT / "pages" / f"{iid}_{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            w.writerow([iid, kind, k["work_id"], k["line"], k["koma_id"], k["G"], k["rank"], k["C"]])
            items.append({"id": iid, "version": "s2_panel_shapes_20261126", "images": [
                {"src": f"pages/{iid}_sketch.jpg", "caption": "下絵(このコマの所)"},
                {"src": f"pages/{iid}_line.jpg", "caption": "線画(このコマの所)", "target": True},
                {"src": f"pages/{iid}_overlay.jpg", "caption": "重ね(下絵=灰・線画=黒)"}]})
            print(f"  {iid} {kind:7s} {k['work_id']} {Path(k['line']).stem} koma{k['koma_id']} G {k['G']} 順位 {k['rank']}")
    spec = {"interview_id": "s_h4_composition_20261128", "page_title": "コマの構図 H4",
            "question": "この下絵は、このコマ(薄くしていない所)の線画と同じ絵の下絵ですか?", "unit": "コマ",
            "decides": "下絵と線画の構図が違うコマを、機械で集められるかを数えるのに使います",
            "choices": [{"value": "same", "label": "同じ絵・おおむねそのまま", "help": "下絵の構図のまま線画にしている"},
                        {"value": "same_changed", "label": "同じ絵・線画で構図や形を変えた", "help": "同じ場面だが、線画で描き直している"},
                        {"value": "other", "label": "別の絵の下絵", "help": "このコマの線画とは別の絵・別の場面の下絵"},
                        {"value": "atari_only", "label": "あたりだけで判断できない", "help": "下絵が大まかすぎて同じ絵か決められない"}],
            "items": items}
    (OUT / "questions.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"問い {len(items)} 件")


if __name__ == "__main__":
    main()

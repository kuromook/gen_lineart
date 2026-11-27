"""聞き取り H2 の問いを作る(登録: doc/work_log.md 2026-11-26「登録 S2」・2026-11-27「H2 の作り」)。
出力 data/h2_panel_sketch_20261127/(追跡しない): questions.json / pages/*.jpg / selection.csv(器具の区分。画面には出さない)
"""
import csv, glob, json, random, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from s1_cleanup_delta import binarize, find, shift  # noqa: E402
from s2_panel_delta import PAGES, SHAPES, s1_rows  # noqa: E402

OUT = ROOT / "data/h2_panel_sketch_20261127"
SEED = 20261126
SKIP_WORKS = {"031_2019b_ako7", "053_2022b_akocult", "070_2024b_hamlabi"}


def pick(rows, n, rng):
    by = {}
    for r in rows:
        by.setdefault(r["work_id"], []).append(r)
    works = sorted(by); rng.shuffle(works)
    for w in works:
        rng.shuffle(by[w])
    out = []
    while len(out) < n and any(by.values()):
        for w in works:
            if by[w] and len(out) < n:
                out.append(by[w].pop())
    return out


def main():
    S1 = s1_rows()
    q_by_line = {q["line"]: q for q, _ in S1.values()}
    s1_by_line = {q["line"]: s for q, s in S1.values()}
    # H1/H1b の一言の頁(線画の名で)
    notes = {"h02", "h03", "h05", "h08", "h11", "h16", "h20"}
    sel1 = [r for r in csv.DictReader(open(ROOT / "data/h1_sketch_stage_20261120/selection.csv", encoding="utf-8")) if r["item_id"] in notes]
    skip_lines = {next(q["line"] for q, s in S1.values() if q["work_id"] == r["work_id"] and q["page"] == r["page"] and s["B_keep"] == r["B_keep"]) for r in sel1}
    skip_lines |= {r["line"] for r in csv.DictReader(open(ROOT / "data/h1b_reask_20261126/selection.csv", encoding="utf-8")) if r["item_id"] == "h04b"}
    K = []
    for f in sorted(glob.glob(str(ROOT / "results/s2_full/*/s2_panels.csv"))):
        K += list(csv.DictReader(open(f, encoding="utf-8")))
    K = [k for k in K if k["work_id"] not in SKIP_WORKS and k["line"] not in skip_lines and q_by_line[k["line"]]["is_primary"] == "True"]
    print(f"候補: 在る {sum(k['kind'] == '下絵の在るコマ' for k in K)} / 無い {sum(k['kind'] == '下絵の無いコマ' for k in K)}(一言の頁 {len(skip_lines)} を除いた)")
    rng = random.Random(SEED)
    sel = [("在る", k) for k in pick([k for k in K if k["kind"] == "下絵の在るコマ"], 10, rng)]
    sel += [("無い", k) for k in pick([k for k in K if k["kind"] == "下絵の無いコマ"], 10, rng)]
    rng.shuffle(sel)
    pages = {r["page_path"]: r for r in csv.DictReader(open(PAGES, encoding="utf-8"))}
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    items = []
    with open(OUT / "selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "instrument", "work_id", "line", "koma_id", "K_over_L", "keep", "added", "bbox"])
        for i, (cls, k) in enumerate(sel, 1):
            iid = f"p{i:02d}"
            q = q_by_line[k["line"]]; s = s1_by_line[k["line"]]
            lab = cv2.imread(str(SHAPES / Path(pages[q["source_path"]]["out_file"]).name), cv2.IMREAD_UNCHANGED)
            inside = lab == int(k["koma_id"])
            ys, xs = np.nonzero(inside)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            m = inside[y0:y1, x0:x1]
            ln = cv2.imread(str(find(q, "line")), cv2.IMREAD_GRAYSCALE)
            sk = cv2.imread(str(find(q, "sketch")), cv2.IMREAD_GRAYSCALE)
            sk = _shift_gray(sk, int(s["B_dx"]), int(s["B_dy"]))
            Lb, _, _ = binarize(find(q, "line")); Sb, _, _ = binarize(find(q, "sketch"))
            Sb = shift(Sb, int(s["B_dx"]), int(s["B_dy"]))
            ov = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
            ov[Sb[y0:y1, x0:x1]] = (170, 170, 170)
            ov[Lb[y0:y1, x0:x1]] = (0, 0, 0)
            for name, im in (("sketch", cv2.cvtColor(sk[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)),
                             ("line", cv2.cvtColor(ln[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)), ("overlay", ov)):
                im = im.copy()
                im[~m] = (im[~m] * 0.25 + 255 * 0.75).astype(np.uint8)  # コマの外は薄く
                cv2.imwrite(str(OUT / "pages" / f"{iid}_{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            w.writerow([iid, cls, k["work_id"], k["line"], k["koma_id"], k["K_over_L"], k["keep"], k["added"], f"{x0},{y0},{x1},{y1}"])
            items.append({"id": iid, "version": "s2_panel_shapes_20261126", "images": [
                {"src": f"pages/{iid}_sketch.jpg", "caption": "下絵(このコマの所)"},
                {"src": f"pages/{iid}_line.jpg", "caption": "線画(このコマの所)", "target": True},
                {"src": f"pages/{iid}_overlay.jpg", "caption": "重ね(下絵=灰・線画=黒)"}]})
            print(f"  {iid} 器具={cls} {k['work_id']} {Path(k['line']).stem} koma{k['koma_id']} K/L {k['K_over_L']}")
    spec = {"interview_id": "s_h2_panel_sketch_20261127", "page_title": "コマの下絵 H2",
            "question": "このコマ(薄くしていない所)に、この線画の下絵はありますか?", "unit": "コマ",
            "decides": "コマごとに下絵の在る・無いを分ける器具が、作者の目に合うかを数えるのに使います",
            "choices": [{"value": "present", "label": "ある", "help": "このコマの絵の下絵が下絵の層にある"},
                        {"value": "partial", "label": "一部だけ", "help": "コマの一部(人物の一部など)にだけある"},
                        {"value": "absent", "label": "無い", "help": "このコマの下絵は下絵の層に無い"}],
            "items": items}
    (OUT / "questions.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"問い {len(items)} 件")


def _shift_gray(g, dx, dy):
    out = np.full_like(g, 255)
    h, w = g.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = g[ys, xs]
    return out


if __name__ == "__main__":
    main()

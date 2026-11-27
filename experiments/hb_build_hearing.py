"""聞き取り B(HB)の問いを作る(登録: doc/work_log.md 2026-11-28「登録 HB」)。出力 data/hb_why_changed_20261128/(追跡しない)。"""
import csv, glob, json, random, sys
from pathlib import Path
import cv2, numpy as np
from skimage.morphology import skeletonize
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from s1_cleanup_delta import FILL_PAD, R2, binarize, classify, dist_to_ink, fill_mask, find, shift  # noqa: E402
from s2_panel_delta import PAGES, SHAPES, s1_rows  # noqa: E402
from s3_binarize import binarize_sketch  # noqa: E402
from h2_build_hearing import SKIP_WORKS, _shift_gray, pick  # noqa: E402

OUT = ROOT / "data/hb_why_changed_20261128"
SEED = 20261128
STRATA = [("2015〜2016", ("2015", "2016")), ("2017", ("2017",)), ("2018", ("2018",)), ("2019〜2019b", ("2019", "2019b")),
          ("2021〜2024", ("2021", "2022", "2022b", "2023", "2023b", "2024", "2024b", "2024c"))]


def main():
    S1 = s1_rows()
    q_by_line = {q["line"]: q for q, _ in S1.values()}
    s1_by_line = {q["line"]: s for q, s in S1.values()}
    h1 = list(csv.DictReader(open(ROOT / "data/h1_sketch_stage_20261120/selection.csv", encoding="utf-8")))
    skip_lines = {next(q["line"] for q, s in S1.values() if q["work_id"] == r["work_id"] and q["page"] == r["page"] and s["B_keep"] == r["B_keep"]) for r in h1}
    skip_lines |= {r["line"] for r in csv.DictReader(open(ROOT / "data/h1b_reask_20261126/selection.csv", encoding="utf-8"))}
    asked = set()
    for d in ("h2_panel_sketch_20261127", "h3_panel_sketch_20261127", "h4_composition_20261128"):
        asked |= {(r["line"], r["koma_id"]) for r in csv.DictReader(open(ROOT / "data" / d / "selection.csv", encoding="utf-8"))}
    K = []
    for f in sorted(glob.glob(str(ROOT / "results/s4_full/*/s2_panels.csv"))):
        K += list(csv.DictReader(open(f, encoding="utf-8")))
    K = [k for k in K if k["kind"] == "下絵の在るコマ" and k["hold"] != "True" and k["work_id"] not in SKIP_WORKS
         and q_by_line[k["line"]]["is_primary"] == "True" and k["line"] not in skip_lines and (k["line"], k["koma_id"]) not in asked]
    rng = random.Random(SEED)
    sel = []
    for name, ys in STRATA:
        pool = [k for k in K if k["work_id"].split("_")[1] in ys]
        ch = sorted(1 - float(k["keep"]) for k in pool)
        cut = ch[int(len(ch) * 0.75)] if ch else 1
        top = [k for k in pool if 1 - float(k["keep"]) >= cut]
        got = pick(top, 4, rng)
        print(f"{name}: コマ {len(pool)} → 上位25% {len(top)}(動+足 ≥ {cut:.2f})→ {len(got)}")
        sel += [(name, k) for k in got]
    rng.shuffle(sel)
    pages = {r["page_path"]: r for r in csv.DictReader(open(PAGES, encoding="utf-8"))}
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    items = []
    with open(OUT / "selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "stratum", "work_id", "line", "koma_id", "keep", "moved", "added", "dropped", "C"])
        for i, (name, k) in enumerate(sel, 1):
            iid = f"b{i:02d}"
            q = q_by_line[k["line"]]; s = s1_by_line[k["line"]]
            dx, dy = int(s["B_dx"]), int(s["B_dy"])
            lab = cv2.imread(str(SHAPES / Path(pages[q["source_path"]]["out_file"]).name), cv2.IMREAD_UNCHANGED)
            inside = lab == int(k["koma_id"])
            ys_, xs_ = np.nonzero(inside)
            y0, y1, x0, x1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
            m = inside[y0:y1, x0:x1]
            ln = cv2.imread(str(find(q, "line")), cv2.IMREAD_GRAYSCALE)
            sk = _shift_gray(cv2.imread(str(find(q, "sketch")), cv2.IMREAD_GRAYSCALE), dx, dy)
            Lb = binarize(find(q, "line"))[0]
            Sb = shift(binarize_sketch(find(q, "sketch"), "T1"), dx, dy)
            F = fill_mask(Lb)
            Fpad = cv2.dilate(F.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FILL_PAD + 1,) * 2)) > 0
            win = (slice(y0, y1), slice(x0, x1))
            Lsk = (skeletonize(Lb[win]) & ~Fpad[win])
            Ssk = skeletonize(Sb[win])
            dS = dist_to_ink(Sb[win]); dL = dist_to_ink(Lb[win])
            ov = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
            ov[Sb[win]] = (225, 225, 225)
            ly, lx = np.nonzero(Lsk)
            kk, mm, aa = classify(dS[ly, lx])
            sy, sx = np.nonzero(Ssk)
            drop = dL[sy, sx] > R2
            ov[sy[drop], sx[drop]] = (0, 140, 255)  # 捨 = 橙
            for mask, col in ((kk, (0, 0, 0)), (mm, (255, 80, 0)), (aa, (0, 0, 230))):
                ov[ly[mask], lx[mask]] = col
            ov = cv2.erode(ov, np.ones((3, 3), np.uint8))
            for nm, im in (("sketch", cv2.cvtColor(sk[win], cv2.COLOR_GRAY2BGR)), ("line", cv2.cvtColor(ln[win], cv2.COLOR_GRAY2BGR)), ("colored", ov)):
                im = im.copy()
                im[~m] = (im[~m] * 0.25 + 255 * 0.75).astype(np.uint8)
                cv2.imwrite(str(OUT / "pages" / f"{iid}_{nm}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            w.writerow([iid, name, k["work_id"], k["line"], k["koma_id"], k["keep"], k["moved"], k["added"], k["dropped"], k["C"]])
            items.append({"id": iid, "version": "s4_t1_c050", "images": [
                {"src": f"pages/{iid}_sketch.jpg", "caption": "下絵(このコマの所)"},
                {"src": f"pages/{iid}_line.jpg", "caption": "線画(このコマの所)"},
                {"src": f"pages/{iid}_colored.jpg", "caption": "色分け: 黒=残した 青=動かした 赤=足した 橙=捨てた(下絵の線)", "target": True}]})
            print(f"  {iid} {name:10s} {k['work_id']} {Path(k['line']).stem} koma{k['koma_id']} 残{k['keep']} 動{k['moved']} 足{k['added']}")
    spec = {"interview_id": "s_hb_why_changed_20261128", "page_title": "線を変えた理由 HB",
            "question": "このコマで、線画が下絵から変わった所(青=動かした・赤=足した)の主な理由はどれですか?", "unit": "コマ",
            "decides": "下絵から線画で線を動かした・足した理由を、作者の言葉で種類に分けるのに使います",
            "choices": [{"value": "refine", "label": "清書で形を整えた", "help": "下絵の線を、線画で正確に確定した"},
                        {"value": "detail", "label": "下絵に無い細部を描き足した", "help": "髪・服のしわ・影の線・背景の細部など"},
                        {"value": "change", "label": "構図・ポーズ・形を変えた", "help": "下絵と違う形に描き直した"},
                        {"value": "atari", "label": "下絵があたり程度で、形は線画で決めた", "help": "下絵は大まかな位置だけ"},
                        {"value": "instrument", "label": "色分けの誤り", "help": "ずれ・塗り・抽出の失敗などで色が正しくない(一言に書いてください)"}],
            "items": items}
    (OUT / "questions.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"問い {len(items)} 件")


if __name__ == "__main__":
    main()

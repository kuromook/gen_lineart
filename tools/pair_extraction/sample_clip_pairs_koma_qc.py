"""Ad-hoc QC sampler: regenerate aligned-crop thumbnails for an evenly-spread
sample of rows from results/clip_pairs_koma_panels_20260821.csv, since the
chunked driver overwrote --qc-out each chunk (only the last chunk's first 60
panels survived on disk). Re-crops directly from the zip using each row's
recorded scale/dx/dy, reusing the same extract_scaled_roi/crop_page/
load_gray/overlay_edges/make_qc helpers as the real run.
"""
import csv
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from match_koma_panels import crop_page, extract_scaled_roi, load_gray, make_qc, overlay_edges
from tile_region_manifest_480 import edge_map

ZIP_PATH = "dataset/raw_zips/dataset_clip_pairs_v2.zip"
CSV_IN = "results/clip_pairs_koma_panels_20260821.csv"
QC_CSV_MEMBER = "clip_pairs/clip_pairs_qc.csv"


def main():
    rows = list(csv.DictReader(open(CSV_IN)))
    n = len(rows)
    sample_n = 24
    idx = sorted(set(int(i * (n - 1) / (sample_n - 1)) for i in range(sample_n)))
    sample = [rows[i] for i in idx]

    with zipfile.ZipFile(ZIP_PATH) as zf:
        qc_text = zf.read(QC_CSV_MEMBER).decode("utf-8")
        qc_by_key = {
            (r["work_id"], r["slug"], r["page"]): r
            for r in csv.DictReader(qc_text.splitlines())
        }

        qc_rows = []
        for r in sample:
            work_id, slug, page = r["work_id"], r["slug"], r["page"]
            zip_root = f"clip_pairs/{work_id}/{slug}"
            qc_entry = qc_by_key.get((work_id, slug, page))
            if qc_entry is None:
                print(f"skip {r['pid']}: no matching row in clip_pairs_qc.csv")
                continue
            line_name, sketch_name = qc_entry["line"], qc_entry["sketch"]
            try:
                line_gray = load_gray(zf, zip_root, line_name)
                rough_gray = load_gray(zf, zip_root, sketch_name, autocontrast=True)
            except KeyError:
                print(f"skip {r['pid']}: could not resolve filenames ({line_name} / {sketch_name})")
                continue

            x0, y0, x1, y1 = int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])
            out_w, out_h = x1 - x0, y1 - y0
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            line_crop = crop_page(line_gray, x0, y0, x1, y1)
            line_edge = edge_map(line_crop)

            scale, dx, dy = float(r["scale"]), int(r["dx"]), int(r["dy"])
            rough_base = extract_scaled_roi(rough_gray, cx, cy, out_w, out_h, 0, 1.0)
            pad = max(64, abs(dx) + 8, abs(dy) + 8)
            roi_best = extract_scaled_roi(rough_gray, cx, cy, out_w, out_h, pad, scale)
            if roi_best is not None:
                y0w, x0w = pad + dy, pad + dx
                rough_best_aligned = roi_best[y0w:y0w + out_h, x0w:x0w + out_w]
                roi_edge = edge_map(roi_best)
                win_edge = roi_edge[y0w:y0w + out_h, x0w:x0w + out_w]
            else:
                rough_best_aligned = rough_base if rough_base is not None else line_crop
                win_edge = line_edge

            qc_rows.append({
                "housei": r["pid"], "panel_index": r["panel_index"],
                "line_img": line_crop,
                "rough_base_img": rough_base if rough_base is not None else line_crop,
                "rough_best_img": rough_best_aligned,
                "overlay_img": overlay_edges(win_edge, line_edge),
                "base_edge_f1": float(r["base_edge_f1"]), "base_chamfer": float(r["base_chamfer"]),
                "edge_f1": float(r["edge_f1"]), "chamfer": float(r["chamfer"]),
                "scale": scale, "dx": dx, "dy": dy,
            })
            print(f"sampled {r['pid']} panel{r['panel_index']}")

    make_qc(qc_rows, 220, Path("results/clip_pairs_koma_panels_20260821_qc_spread.png"))
    print(f"wrote results/clip_pairs_koma_panels_20260821_qc_spread.png ({len(qc_rows)} rows)")


if __name__ == "__main__":
    main()

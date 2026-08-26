"""Panel-boundary-first alignment search for `clip_pairs` (see
`doc/preprocess/raw_dataset_storage_policy.md`'s `dataset_clip_pairs_v2.zip`
entry and `doc/work_log.md`'s 2026-08-21 "Extraction Tool Reply" entries),
reusing the panel-detection/alignment-search machinery already proven on
ako5ver2/hamlabi/fitness/housei/gakuen in `match_koma_panels.py`.

`clip_pairs` differs structurally from those 5 sources in a way
`match_koma_panels.py`'s single-`--zip-root` design doesn't fit directly:
it packs ~69 works (each with its own subproject folder, own
`manifest.json`/`koma_manifest.json`) into one zip, rather than one root
per archive. Sidesteps that by driving entirely off the extraction tool's
own cross-work `clip_pairs/clip_pairs_qc.csv` (work_id/slug/page/line/
sketch/koma filenames, pair_quality, is_primary, review_rank already
resolved there) instead of re-deriving zip roots or joining
manifest.json/koma_manifest.json per folder -- the per-page pixel logic
(`detect_panels`, `search_panel_alignment`, etc.) is imported unchanged.

Filter recommended by the extraction tool's own reply (not `aligned=true`,
confirmed too strict -- 275 of 341 `aligned=false` pairs are QC-clean):
`pair_quality=ok` + `is_primary=True` + `koma` present -> 1272 starting
pairs (verified against their own count).
"""

import argparse
import csv
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from match_koma_panels import (
    crop_page,
    detect_panels,
    extract_scaled_roi,
    load_gray,
    make_page_overlay,
    make_qc,
    member_exists,
    overlay_edges,
    read_zip_member,
    search_panel_alignment,
)
from tile_region_manifest_480 import ALIGNMENT_CLOSE_PX, edge_map

ZIP_PATH = "dataset/raw_zips/dataset_clip_pairs_v2.zip"
QC_CSV_MEMBER = "clip_pairs/clip_pairs_qc.csv"
CSV_OUT = "results/clip_pairs_koma_panels_20260821.csv"
JSON_OUT = "results/clip_pairs_koma_panels_20260821.json"
QC_OUT = "results/clip_pairs_koma_panels_20260821_qc.png"


def load_qc_rows(zf):
    text = read_zip_member(zf, "", QC_CSV_MEMBER).decode("utf-8")
    return list(csv.DictReader(text.splitlines()))


def filtered_pairs(qc_rows):
    return [
        r for r in qc_rows
        if r["pair_quality"] == "ok" and r["is_primary"] == "True" and r["koma"].strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=ZIP_PATH, dest="zip_path")
    parser.add_argument("--min-area-ratio", type=float, default=0.01)
    parser.add_argument("--close-px", type=int, default=7)
    parser.add_argument("--max-shift", type=int, default=64)
    parser.add_argument("--shift-step", type=int, default=16)
    parser.add_argument("--coarse-max-shift", type=int, default=480)
    parser.add_argument("--coarse-downscale", type=int, default=4)
    parser.add_argument("--coarse-shift-step", type=int, default=32)
    parser.add_argument("--start", type=int, default=0, help="start index into the filtered pair list")
    parser.add_argument("--end", type=int, default=0, help="0 = all remaining")
    parser.add_argument("--append", action="store_true", help="append to existing csv-out/json-out, for chunked runs")
    parser.add_argument("--csv-out", default=CSV_OUT)
    parser.add_argument("--json-out", default=JSON_OUT)
    parser.add_argument("--qc-out", default=QC_OUT)
    parser.add_argument("--overlay-dir", default="results/clip_pairs_koma_panels_20260821_overlays")
    parser.add_argument("--qc-thumb", type=int, default=220)
    parser.add_argument("--qc-max-rows", type=int, default=60)
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip_path) as zf:
        qc_rows = load_qc_rows(zf)
        pairs = filtered_pairs(qc_rows)
        print(f"[match_clip_pairs_koma_panels] {len(qc_rows)} total qc rows, {len(pairs)} pass pair_quality=ok+is_primary+koma")
        pairs = pairs[args.start:args.end] if args.end else pairs[args.start:]

        rows = []
        qc_thumb_rows = []
        for index, pair in enumerate(pairs, 1):
            work_id, slug, page = pair["work_id"], pair["slug"], pair["page"]
            zip_root = f"clip_pairs/{work_id}/{slug}"
            pid = f"{work_id}__{slug}__{Path(page).stem}"
            koma_name, line_name, sketch_name = pair["koma"], pair["line"], pair["sketch"]

            if not member_exists(zf, zip_root, line_name) or not member_exists(zf, zip_root, sketch_name) or not member_exists(zf, zip_root, koma_name):
                print(f"skip {pid}: asset missing from zip (data gap, not a bug)")
                continue

            koma_gray = load_gray(zf, zip_root, koma_name)
            line_gray = load_gray(zf, zip_root, line_name)
            rough_gray = load_gray(zf, zip_root, sketch_name, autocontrast=True)

            panels = detect_panels(koma_gray, min_area_ratio=args.min_area_ratio, close_px=args.close_px)
            make_page_overlay(line_gray, panels, Path(args.overlay_dir) / f"{pid}_panels.png")

            for panel_index, panel in enumerate(panels, 1):
                x0, y0, x1, y1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
                out_w, out_h = x1 - x0, y1 - y0
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                line_crop = crop_page(line_gray, x0, y0, x1, y1)
                line_edge = edge_map(line_crop)
                if line_edge.sum() < 20:
                    continue
                close_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (int(ALIGNMENT_CLOSE_PX) * 2 + 1, int(ALIGNMENT_CLOSE_PX) * 2 + 1)
                )
                line_support = cv2.dilate(line_edge.astype(np.uint8), close_kernel) > 0
                line_dist = cv2.distanceTransform((~line_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

                best, base = search_panel_alignment(rough_gray, line_edge, line_support, line_dist, cx, cy, out_w, out_h, args)

                row = {
                    "work_id": work_id, "slug": slug, "page": page, "pid": pid,
                    "content_fingerprint": pair["content_fingerprint"],
                    "review_rank": pair["review_rank"],
                    "panel_index": panel_index,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "area_ratio": round(panel["area_ratio"], 4), "fill_ratio": round(panel["fill_ratio"], 4),
                    "base_edge_f1": base["edge_f1"], "base_chamfer": base["chamfer"],
                    "edge_f1": best["edge_f1"], "chamfer": best["chamfer"],
                    "strict_edge_f1": best["strict_edge_f1"], "strict_edge_recall": best["strict_edge_recall"],
                    "strict_edge_precision": best["strict_edge_precision"],
                    "scale": best["scale"], "dx": best["dx"], "dy": best["dy"],
                }
                rows.append(row)

                if len(qc_thumb_rows) < args.qc_max_rows:
                    rough_base = extract_scaled_roi(rough_gray, cx, cy, out_w, out_h, 0, 1.0)
                    rough_best = extract_scaled_roi(rough_gray, cx, cy, out_w, out_h, 0, best["scale"])
                    if rough_best is not None:
                        pad = max(args.max_shift, abs(best["dx"]) + 8, abs(best["dy"]) + 8)
                        roi_best = extract_scaled_roi(rough_gray, cx, cy, out_w, out_h, pad, best["scale"])
                        y0w, x0w = pad + best["dy"], pad + best["dx"]
                        rough_best_aligned = roi_best[y0w:y0w + out_h, x0w:x0w + out_w]
                    else:
                        rough_best_aligned = rough_base
                    qc_thumb_rows.append({
                        "housei": pid, "panel_index": panel_index,
                        "line_img": line_crop, "rough_base_img": rough_base if rough_base is not None else line_crop,
                        "rough_best_img": rough_best_aligned,
                        "overlay_img": overlay_edges(best["win_edge"], line_edge),
                        "base_edge_f1": base["edge_f1"], "base_chamfer": base["chamfer"],
                        "edge_f1": best["edge_f1"], "chamfer": best["chamfer"],
                        "scale": best["scale"], "dx": best["dx"], "dy": best["dy"],
                    })
                print(
                    f'{pid} panel{panel_index} chamfer {base["chamfer"]:.1f}->{best["chamfer"]:.1f} '
                    f'F1 {base["edge_f1"]:.2f}->{best["edge_f1"]:.2f} scale={best["scale"]:.2f} '
                    f'd=({best["dx"]},{best["dy"]})',
                    flush=True,
                )
            print(f"[{index}/{len(pairs)}] {pid}: {len(panels)} panels", flush=True)

    csv_path, json_path = Path(args.csv_out), Path(args.json_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = rows
    if args.append and json_path.exists():
        import json
        all_rows = json.loads(json_path.read_text()) + rows
    fields = list(all_rows[0].keys()) if all_rows else []
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    import json
    with open(json_path, "w") as file:
        json.dump(all_rows, file, indent=2)
        file.write("\n")
    make_qc(qc_thumb_rows, args.qc_thumb, Path(args.qc_out))

    if rows:
        base_med = float(np.median([r["base_chamfer"] for r in rows]))
        best_med = float(np.median([r["chamfer"] for r in rows]))
        print(f"this run: panels={len(rows)} chamfer_median base={base_med:.2f} best={best_med:.2f}")
    print(f"total accumulated panels={len(all_rows)}")
    print(f"wrote: {args.csv_out}, {args.json_out}, {args.qc_out}, overlays in {args.overlay_dir}/")


if __name__ == "__main__":
    main()

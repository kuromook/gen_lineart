"""Segment housei-style pages into panel regions using a koma (panel-border) layer.

This is the panel-boundary-first segmentation step planned in
`doc/raw_dataset_extraction_knowledge.md` ("Planned Fix: Panel-Boundary-First
Region Segmentation"), now unblocked for `housei` by a newly delivered koma
layer (`housei_NNN_koma.jpg`, one per page, in `dataset_housei_v2.zip`).

Two stages, both dry-run/candidate-generation only:

1. `detect_panels`: the koma layer contains only panel-border ink on an
   otherwise blank page. Connected components of the *non-ink* area, after
   closing small border gaps and dropping the component(s) touching the page
   edge (outer margin/background), are the panel interiors.
2. Per-panel alignment search: reuses the fixed alignment-gate primitives
   (`edge_map` / `support_f1` / `chamfer`, `ALIGNMENT_*` constants) from
   `tile_region_manifest_480.py`, extended with a uniform-scale search
   (0.85-1.15), since the finishing pass can rescale content per panel (see
   "Residual Misalignment" in `doc/raw_dataset_extraction_knowledge.md`). For
   each candidate scale, edge/support/distance maps are computed once for a
   padded ROI and then translation candidates are scored by cheap array
   slicing rather than recomputing Canny per offset.

Output: a candidate CSV/JSON manifest and a QC montage (line panel / rough
original crop / rough aligned+scaled crop / edge overlay) for review. Does not
write a training manifest or materialize tiles.
"""

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tile_region_manifest_480 import (
    ALIGNMENT_CLOSE_PX,
    ALIGNMENT_STRICT_CLOSE_PX,
    ALIGNMENT_TRUNCATE_PX,
    chamfer,
    edge_map,
    support_f1,
)


ZIP_PATH = "dataset/raw_zips/dataset_housei_v2.zip"
ZIP_ROOT = "dataset_housei"
CSV_OUT = "results/housei_koma_panels_20260726.csv"
JSON_OUT = "results/housei_koma_panels_20260726.json"
QC_OUT = "results/housei_koma_panels_20260726_qc.png"
OVERLAY_OUT = "results/housei_koma_panels_20260726_overlay.png"

SCALES = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]


def read_zip_member(zf, zip_root, name):
    for candidate in (f"{zip_root}/{name}", f"{zip_root}\\{name}", name):
        try:
            return zf.read(candidate)
        except KeyError:
            pass
    raise KeyError(f"missing zip member for {name!r}")


def load_json_member(zf, zip_root, name):
    return json.loads(read_zip_member(zf, zip_root, name))


def load_gray(zf, zip_root, name, autocontrast=False):
    image = Image.open(io.BytesIO(read_zip_member(zf, zip_root, name))).convert("L")
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=0)
    return np.asarray(image)


def page_id(entry):
    """Generic per-page identifier across manifest schemas: housei's manifest
    has a dedicated 'housei' field (e.g. 'housei_004'); ako5/hamlabi/kurip's
    koma-layer manifests only have 'file' (e.g. 'page0001.clip'), so fall back
    to that stem.
    """
    if entry.get("housei"):
        return entry["housei"]
    source = entry.get("file") or entry.get("page") or "unknown"
    return Path(source).stem


def koma_lookup(koma_manifest):
    """Join key: koma_manifest's 'page' field matches the main manifest's
    'page' (housei's schema) or 'file' (ako5/hamlabi/kurip's schema) field.
    """
    return {km["page"]: km for km in koma_manifest}


def join_key(entry):
    return entry.get("page") or entry.get("file")


def detect_panels(koma_gray, ink_thresh=200, min_area_ratio=0.01, close_px=7):
    """Panel interiors = connected non-ink components, excluding the margin."""
    height, width = koma_gray.shape
    ink = koma_gray < ink_thresh
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))
    ink_closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    free = ~ink_closed
    labels, num = ndimage.label(free, structure=np.ones((3, 3), dtype=np.uint8))
    page_area = height * width
    panels = []
    for index, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        ys, xs = sl
        touches_border = ys.start == 0 or xs.start == 0 or ys.stop == height or xs.stop == width
        if touches_border:
            continue
        area = int((labels[sl] == index).sum())
        ratio = area / page_area
        if ratio < min_area_ratio:
            continue
        bbox_area = (ys.stop - ys.start) * (xs.stop - xs.start)
        panels.append({
            "x0": int(xs.start), "y0": int(ys.start), "x1": int(xs.stop), "y1": int(ys.stop),
            "area_ratio": ratio, "fill_ratio": area / bbox_area,
        })
    panels.sort(key=lambda p: (p["y0"], p["x0"]))
    return panels


def extract_scaled_roi(page_gray, cx, cy, out_w, out_h, pad, scale):
    """Crop a scale*content-sized source region and resize to (out_w+2pad, out_h+2pad)."""
    src_w = (out_w + 2 * pad) * scale
    src_h = (out_h + 2 * pad) * scale
    x0 = int(round(cx - src_w / 2))
    y0 = int(round(cy - src_h / 2))
    x1 = int(round(x0 + src_w))
    y1 = int(round(y0 + src_h))
    height, width = page_gray.shape
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(width, x1), min(height, y1)
    if x1c <= x0c or y1c <= y0c:
        return None
    crop = page_gray[y0c:y1c, x0c:x1c]
    # pad back out to the requested source size so the resize factor stays exact
    pad_top, pad_left = y0c - y0, x0c - x0
    pad_bottom, pad_right = y1 - y1c, x1 - x1c
    if pad_top or pad_left or pad_bottom or pad_right:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)
    out_size = (out_w + 2 * pad, out_h + 2 * pad)
    interp = cv2.INTER_AREA if scale > 1 else cv2.INTER_LINEAR
    return cv2.resize(crop, out_size, interpolation=interp)


def search_panel_alignment(rough_page, line_edge, line_support, line_dist, cx, cy, out_w, out_h, args):
    """Joint translation+scale search; returns metrics for the best candidate and scale=1/dx=0/dy=0 baseline.

    Per-candidate scoring uses sparse edge-pixel coordinates rather than
    full-array boolean indexing, so cost scales with ink density, not panel
    pixel area; a panel-sized ROI is still large (thousands of px), but the
    number of edge pixels drawn on it is a small fraction of that.
    """
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(ALIGNMENT_CLOSE_PX) * 2 + 1, int(ALIGNMENT_CLOSE_PX) * 2 + 1)
    )
    offsets = []
    for dy in range(-args.max_shift, args.max_shift + 1, args.shift_step):
        for dx in range(-args.max_shift, args.max_shift + 1, args.shift_step):
            offsets.append((dx, dy))
    offsets.sort(key=lambda item: item[0] ** 2 + item[1] ** 2)

    line_ys, line_xs = np.nonzero(line_edge)
    n_line = len(line_ys)

    best = None
    baseline = None
    pad = args.max_shift
    for scale in SCALES:
        roi = extract_scaled_roi(rough_page, cx, cy, out_w, out_h, pad, scale)
        if roi is None:
            continue
        roi_edge = edge_map(roi)
        roi_support = cv2.dilate(roi_edge.astype(np.uint8), close_kernel) > 0
        rough_ys, rough_xs = np.nonzero(roi_edge)

        for dx, dy in offsets:
            y0, x0 = pad + dy, pad + dx
            if n_line == 0:
                recall = 0.0
            else:
                recall = float(roi_support[line_ys + y0, line_xs + x0].mean())
            wy = rough_ys - y0
            wx = rough_xs - x0
            inside = (wy >= 0) & (wy < out_h) & (wx >= 0) & (wx < out_w)
            n_inside = int(inside.sum())
            if n_inside < 10 or n_line < 10:
                f1 = 0.0
            else:
                precision = float(line_support[wy[inside], wx[inside]].mean())
                f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
            candidate = {"scale": scale, "dx": dx, "dy": dy, "edge_f1": f1, "roi_edge": roi_edge, "y0": y0, "x0": x0}
            if scale == 1.0 and dx == 0 and dy == 0:
                baseline = candidate
            if best is None or f1 > best["edge_f1"]:
                best = candidate

    def finalize(candidate):
        win_edge = candidate["roi_edge"][candidate["y0"]:candidate["y0"] + out_h, candidate["x0"]:candidate["x0"] + out_w]
        ch = chamfer(win_edge, line_edge, ALIGNMENT_TRUNCATE_PX) if win_edge.sum() >= 10 else float(ALIGNMENT_TRUNCATE_PX)
        strict_f1, strict_precision, strict_recall = support_f1(win_edge, line_edge, ALIGNMENT_STRICT_CLOSE_PX)
        return {
            "scale": candidate["scale"], "dx": candidate["dx"], "dy": candidate["dy"],
            "edge_f1": candidate["edge_f1"], "chamfer": ch,
            "strict_edge_f1": strict_f1, "strict_edge_precision": strict_precision, "strict_edge_recall": strict_recall,
            "win_edge": win_edge,
        }

    return finalize(best), finalize(baseline) if baseline is not None else finalize(best)


def crop_page(page_gray, x0, y0, x1, y1):
    return page_gray[y0:y1, x0:x1]


def overlay_edges(rough_edge, line_edge):
    canvas = np.full((*line_edge.shape, 3), 255, dtype=np.uint8)
    canvas[rough_edge] = (255, 60, 60)
    canvas[line_edge] = (40, 80, 255)
    canvas[rough_edge & line_edge] = (40, 160, 40)
    return canvas


def make_qc(rows, thumb, output_path):
    if not rows:
        return
    columns = ["line panel", "rough (dx=0 scale=1)", "rough aligned+scaled", "edge overlay"]
    header_h, label_h = 28, 34
    canvas = Image.new("RGB", (thumb * len(columns), header_h + (thumb + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except OSError:
        font = small_font = ImageFont.load_default()
    for col, title in enumerate(columns):
        width = draw.textlength(title, font=font)
        draw.text((col * thumb + (thumb - width) / 2, 6), title, fill=0, font=font)
    for row_idx, row in enumerate(rows):
        y = header_h + row_idx * (thumb + label_h)
        for col, image in enumerate((row["line_img"], row["rough_base_img"], row["rough_best_img"], row["overlay_img"])):
            canvas.paste(Image.fromarray(image).convert("RGB").resize((thumb, thumb)), (col * thumb, y))
        text = (
            f'{row["housei"]} panel{row["panel_index"]} '
            f'chamfer {row["base_chamfer"]:.1f}->{row["chamfer"]:.1f} '
            f'F1 {row["base_edge_f1"]:.2f}->{row["edge_f1"]:.2f} '
            f'scale={row["scale"]:.2f} d=({row["dx"]},{row["dy"]})'
        )
        draw.text((4, y + thumb + 2), text, fill=0, font=small_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def make_page_overlay(line_full, panels, output_path, scale_down=2200):
    height, width = line_full.shape
    factor = min(1.0, scale_down / max(height, width))
    small = cv2.resize(line_full, (int(width * factor), int(height * factor)))
    canvas = Image.fromarray(small).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for i, panel in enumerate(panels, 1):
        box = [panel["x0"] * factor, panel["y0"] * factor, panel["x1"] * factor, panel["y1"] * factor]
        draw.rectangle(box, outline=(255, 0, 0), width=2)
        draw.text((box[0] + 3, box[1] + 3), str(i), fill=(255, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=ZIP_PATH, dest="zip_path")
    parser.add_argument("--zip-root", default=ZIP_ROOT)
    parser.add_argument("--min-area-ratio", type=float, default=0.01)
    parser.add_argument("--close-px", type=int, default=7)
    parser.add_argument("--max-shift", type=int, default=64)
    parser.add_argument("--shift-step", type=int, default=16)
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--end-page", type=int, default=0, help="0 = all pages")
    parser.add_argument("--append", action="store_true", help="append rows to existing csv-out/json-out instead of overwriting; use for chunked runs")
    parser.add_argument("--csv-out", default=CSV_OUT)
    parser.add_argument("--json-out", default=JSON_OUT)
    parser.add_argument("--qc-out", default=QC_OUT)
    parser.add_argument("--overlay-dir", default="results/housei_koma_panels_20260726_overlays")
    parser.add_argument("--qc-thumb", type=int, default=220)
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip_path) as zf:
        manifest = load_json_member(zf, args.zip_root, "manifest.json")
        koma_manifest = load_json_member(zf, args.zip_root, "koma_manifest.json")
        koma_by_page = koma_lookup(koma_manifest)
        entries = manifest[args.start_page:args.end_page] if args.end_page else manifest[args.start_page:]

        rows = []
        qc_rows = []
        for index, entry in enumerate(entries, 1):
            pid = page_id(entry)
            koma_entry = koma_by_page.get(join_key(entry))
            if koma_entry is None:
                print(f"skip {pid}: no koma layer")
                continue
            koma_gray = load_gray(zf, args.zip_root, koma_entry["koma"])
            line_gray = load_gray(zf, args.zip_root, entry["line"])
            rough_gray = load_gray(zf, args.zip_root, entry["sketch"], autocontrast=True)

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
                    "housei": pid, "page": koma_entry.get("page", ""), "panel_index": panel_index,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "area_ratio": round(panel["area_ratio"], 4), "fill_ratio": round(panel["fill_ratio"], 4),
                    "base_edge_f1": base["edge_f1"], "base_chamfer": base["chamfer"],
                    "edge_f1": best["edge_f1"], "chamfer": best["chamfer"],
                    "strict_edge_f1": best["strict_edge_f1"], "strict_edge_recall": best["strict_edge_recall"],
                    "strict_edge_precision": best["strict_edge_precision"],
                    "scale": best["scale"], "dx": best["dx"], "dy": best["dy"],
                }
                rows.append(row)

                if len(qc_rows) < 60:
                    rough_base = extract_scaled_roi(rough_gray, cx, cy, out_w, out_h, 0, 1.0)
                    rough_best = extract_scaled_roi(rough_gray, cx, cy, out_w, out_h, 0, best["scale"])
                    if rough_best is not None:
                        pad = args.max_shift
                        roi_best = extract_scaled_roi(rough_gray, cx, cy, out_w, out_h, pad, best["scale"])
                        y0w, x0w = pad + best["dy"], pad + best["dx"]
                        rough_best_aligned = roi_best[y0w:y0w + out_h, x0w:x0w + out_w]
                    else:
                        rough_best_aligned = rough_base
                    qc_rows.append({
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
            print(f"[{index}/{len(entries)}] {pid}: {len(panels)} panels", flush=True)

    csv_path, json_path = Path(args.csv_out), Path(args.json_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = rows
    if args.append and json_path.exists():
        all_rows = json.loads(json_path.read_text()) + rows
    fields = list(all_rows[0].keys()) if all_rows else []
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    with open(json_path, "w") as file:
        json.dump(all_rows, file, indent=2)
        file.write("\n")
    make_qc(qc_rows, args.qc_thumb, Path(args.qc_out))

    if rows:
        base_med = float(np.median([r["base_chamfer"] for r in rows]))
        best_med = float(np.median([r["chamfer"] for r in rows]))
        print(f"this run: panels={len(rows)} chamfer_median base={base_med:.2f} best={best_med:.2f}")
    print(f"total accumulated panels={len(all_rows)}")
    print(f"wrote: {args.csv_out}, {args.json_out}, {args.qc_out}, overlays in {args.overlay_dir}/")


if __name__ == "__main__":
    main()

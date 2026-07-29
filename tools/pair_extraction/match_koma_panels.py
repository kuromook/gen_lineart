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


def member_exists(zf, zip_root, name):
    if not name:
        return False
    names = set(zf.namelist())
    return any(candidate in names for candidate in (f"{zip_root}/{name}", f"{zip_root}\\{name}", name))


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


def resolve_files(entry):
    """Return a dict with 'sketch'/'line'/'koma' filenames regardless of
    manifest schema: housei/ako5/hamlabi/fitness keep 'sketch'/'line' as
    top-level entry keys, with 'koma' on a separate koma_manifest.json entry
    (looked up via join_key()/koma_lookup()); gakuen nests all three under a
    single entry's 'files' dict instead, with no koma_manifest.json at all.
    """
    return entry.get("files", entry)


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


def _score_offset(dx, dy, pad, out_w, out_h, rough_support, rough_ys, rough_xs, line_ys, line_xs, line_support, n_line):
    y0, x0 = pad + dy, pad + dx
    recall = float(rough_support[line_ys + y0, line_xs + x0].mean()) if n_line else 0.0
    wy = rough_ys - y0
    wx = rough_xs - x0
    inside = (wy >= 0) & (wy < out_h) & (wx >= 0) & (wx < out_w)
    n_inside = int(inside.sum())
    if n_inside < 10 or n_line < 10:
        f1 = 0.0
    else:
        precision = float(line_support[wy[inside], wx[inside]].mean())
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return f1, y0, x0


MIN_COARSE_EDGE_PIXELS = 40
MIN_ALIGNED_ROUGH_STD = 8.0


def coarse_offset_estimate(rough_page, line_edge, cx, cy, out_w, out_h, downscale, max_shift, shift_step):
    """Wide, cheap translation-only pre-search on downscaled images.

    Per-panel/per-character finishing shifts can exceed what's affordable to
    search exhaustively at native resolution and step size (see
    doc/raw_dataset_extraction_knowledge.md, "Residual Misalignment"): an
    exhaustive native search is roughly O(max_shift^2 * edge_count), so
    widening it to chase a larger true offset gets expensive fast. Searching
    the same *effective* native range at 1/downscale resolution instead costs
    roughly O((max_shift/downscale)^2) with edge counts also shrinking by
    downscale^2, so a much wider net is affordable here; the fine, scale-aware
    search in `search_panel_alignment` then only needs to refine a small
    residual around this coarse estimate rather than hunt for it from scratch.

    Scores every candidate offset with the same sparse-edge-coordinate F1
    metric used by the fine search below (`_score_offset`), not a generic
    image-similarity metric: a `cv2.matchTemplate` (normalized cross-
    correlation) version was tried and measured faster, but a real test run
    showed it silently picks *worse* neighborhoods on real panels (this
    dataset's median chamfer got worse after "search", base=40.21 ->
    best=43.83, vs. base=40.21 -> best=34.92 with the F1 loop below on the
    same panels) — cross-correlation between binary edge maps can be
    maximized by incidental dense/repeated structure that has nothing to do
    with genuine rough/line correspondence, so it is not a safe substitute
    for the F1 metric the rest of this pipeline was validated against.
    `shift_step` therefore keeps its real effect here (grid spacing at
    downscaled resolution); increase it (or `downscale`) to trade search
    density for speed rather than switching metrics.

    Guards against noise-driven false offsets on near-empty panels: a panel
    with too little inked content (weak baseline correspondence, e.g. a
    mostly-blank panel) has no real signal for translation search to lock
    onto, and an earlier test run showed exactly this on the F1 loop too — a
    spurious large coarse offset made one panel's chamfer slightly *worse*
    than doing no coarse correction at all. Below `MIN_COARSE_EDGE_PIXELS`
    line edge pixels, skip the coarse stage and let the fine search run at
    the panel's own center, same as if `--coarse-max-shift 0` had disabled it
    entirely.
    """
    if int(line_edge.sum()) < MIN_COARSE_EDGE_PIXELS:
        return 0, 0

    d = max(1, int(downscale))
    pad = max(1, int(max_shift))
    roi = extract_scaled_roi(rough_page, cx, cy, out_w, out_h, pad, 1.0)
    if roi is None:
        return 0, 0
    small_roi = cv2.resize(roi, (max(1, roi.shape[1] // d), max(1, roi.shape[0] // d)), interpolation=cv2.INTER_AREA)
    small_edge = edge_map(small_roi)
    small_ys, small_xs = np.nonzero(small_edge)

    small_out_w, small_out_h = max(1, out_w // d), max(1, out_h // d)
    small_pad = max(1, pad // d)
    small_line_edge = cv2.resize(
        line_edge.astype(np.uint8), (small_out_w, small_out_h), interpolation=cv2.INTER_AREA
    ) > 0
    small_line_support = cv2.resize(
        (cv2.dilate(line_edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0).astype(np.uint8),
        (small_out_w, small_out_h), interpolation=cv2.INTER_AREA,
    ) > 0
    line_ys, line_xs = np.nonzero(small_line_edge)
    n_line = len(line_ys)
    if n_line == 0 or len(small_ys) == 0:
        return 0, 0

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    small_rough_support = cv2.dilate(small_edge.astype(np.uint8), close_kernel) > 0

    step = max(1, shift_step // d)
    best_dx, best_dy, best_f1 = 0, 0, -1.0
    for dy in range(-small_pad, small_pad + 1, step):
        for dx in range(-small_pad, small_pad + 1, step):
            f1, _, _ = _score_offset(
                dx, dy, small_pad, small_out_w, small_out_h,
                small_rough_support, small_ys, small_xs, line_ys, line_xs, small_line_support, n_line,
            )
            if f1 > best_f1:
                best_f1, best_dx, best_dy = f1, dx, dy
    return best_dx * d, best_dy * d


def search_panel_alignment(rough_page, line_edge, line_support, line_dist, cx, cy, out_w, out_h, args):
    """Joint translation+scale search; returns metrics for the best candidate and scale=1/dx=0/dy=0 baseline.

    Per-candidate scoring uses sparse edge-pixel coordinates rather than
    full-array boolean indexing, so cost scales with ink density, not panel
    pixel area; a panel-sized ROI is still large (thousands of px), but the
    number of edge pixels drawn on it is a small fraction of that.

    If `args.coarse_max_shift` is set, a cheap downscaled pre-search
    (`coarse_offset_estimate`) finds a coarse neighborhood first and the fine
    search below re-centers on it, so `args.max_shift` only needs to cover the
    residual rather than the true (possibly much larger) offset. Returned
    `dx`/`dy` are always the total native-pixel offset from the panel's own
    `(cx, cy)`, coarse contribution included, so downstream consumers
    (`materialize_koma_panels.py` etc.) don't need to change.
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

    coarse_dx, coarse_dy = 0, 0
    if getattr(args, "coarse_max_shift", 0):
        coarse_dx, coarse_dy = coarse_offset_estimate(
            rough_page, line_edge, cx, cy, out_w, out_h,
            args.coarse_downscale, args.coarse_max_shift, args.coarse_shift_step,
        )
    cx_search, cy_search = cx + coarse_dx, cy + coarse_dy

    best = None
    pad = args.max_shift
    for scale in SCALES:
        roi = extract_scaled_roi(rough_page, cx_search, cy_search, out_w, out_h, pad, scale)
        if roi is None:
            continue
        roi_edge = edge_map(roi)
        roi_support = cv2.dilate(roi_edge.astype(np.uint8), close_kernel) > 0
        rough_ys, rough_xs = np.nonzero(roi_edge)

        for dx, dy in offsets:
            f1, y0, x0 = _score_offset(
                dx, dy, pad, out_w, out_h, roi_support, rough_ys, rough_xs, line_ys, line_xs, line_support, n_line
            )
            candidate = {
                "scale": scale, "dx": coarse_dx + dx, "dy": coarse_dy + dy,
                "edge_f1": f1, "roi": roi, "roi_edge": roi_edge, "y0": y0, "x0": x0,
            }
            if best is None or f1 > best["edge_f1"]:
                best = candidate

    # True zero-shift baseline (scale=1, dx=0, dy=0 at the panel's own original
    # center), computed independently of the coarse pre-search so base_chamfer/
    # base_edge_f1 stay a stable "no correction at all" reference regardless of
    # --coarse-* settings.
    baseline = None
    baseline_roi = extract_scaled_roi(rough_page, cx, cy, out_w, out_h, pad, 1.0)
    if baseline_roi is not None:
        baseline_edge = edge_map(baseline_roi)
        baseline_support = cv2.dilate(baseline_edge.astype(np.uint8), close_kernel) > 0
        baseline_ys, baseline_xs = np.nonzero(baseline_edge)
        f1, y0, x0 = _score_offset(
            0, 0, pad, out_w, out_h, baseline_support, baseline_ys, baseline_xs, line_ys, line_xs, line_support, n_line
        )
        baseline = {"scale": 1.0, "dx": 0, "dy": 0, "edge_f1": f1, "roi_edge": baseline_edge, "y0": y0, "x0": x0}

    # Do-no-harm guarantee: the coarse pre-search can occasionally lock onto a
    # spurious distant local optimum (seen on real data — repetitive/ambiguous
    # panel content scoring a similar or higher F1 far from the true offset,
    # making the reported "best" chamfer meaningfully *worse* than doing no
    # correction at all). The baseline is always a valid, already-scored
    # candidate, so if the search's own best is no better than it, just report
    # the baseline instead of a confidently-wrong distant offset.
    if baseline is not None and (best is None or baseline["edge_f1"] >= best["edge_f1"]):
        best = baseline

    # Second do-no-harm check, orthogonal to the F1 one above: F1 alone can
    # look "improved" while the found offset has actually walked the crop
    # into a degenerate region — off the real page content into a black
    # scanner-bed margin or similar blank void — because a near-empty target
    # can score a higher (but still low) F1 than an equally-near-empty
    # baseline by chance. Observed on real data (gakuen): two panels' "best"
    # showed a solid near-black aligned crop despite edge_f1 nominally
    # improving. Guard with the same rough_std convention already used
    # elsewhere in this pipeline's style gates (e.g. match_kurip_regions.py's
    # --min-rough-std) — real manuscript content has texture; a blank/void
    # crop does not.
    if best is not baseline and baseline is not None and "roi" in best:
        best_window = best["roi"][best["y0"]:best["y0"] + out_h, best["x0"]:best["x0"] + out_w]
        if float(best_window.std()) < MIN_ALIGNED_ROUGH_STD:
            best = baseline

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
    # np.array() forces an independent copy: a plain slice is a view that
    # keeps the entire source page buffer (tens of MB) alive for as long as
    # the crop is referenced, e.g. when stored in qc_rows across a whole run.
    return np.array(page_gray[y0:y1, x0:x1])


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
    parser.add_argument("--max-shift", type=int, default=64,
                         help="fine-search radius in native px, around the coarse estimate when --coarse-max-shift is set (otherwise around the panel's own center)")
    parser.add_argument("--shift-step", type=int, default=16)
    parser.add_argument("--coarse-max-shift", type=int, default=480,
                         help="wide pre-search radius in native px on a downscaled image, to find the neighborhood before the expensive fine search; 0 disables coarse pre-search (old behavior, single native-resolution search only)")
    parser.add_argument("--coarse-downscale", type=int, default=4)
    parser.add_argument("--coarse-shift-step", type=int, default=32,
                         help="native-px step for the coarse pre-search grid (internally divided by --coarse-downscale); "
                              "coarser than --shift-step by default since the coarse stage only needs a neighborhood, "
                              "not a precise offset, and grid size (hence Python-loop cost) scales with the square of "
                              "max_shift/step")
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
        has_koma_manifest = member_exists(zf, args.zip_root, "koma_manifest.json")
        koma_by_page = koma_lookup(load_json_member(zf, args.zip_root, "koma_manifest.json")) if has_koma_manifest else None
        entries = manifest[args.start_page:args.end_page] if args.end_page else manifest[args.start_page:]

        rows = []
        qc_rows = []
        for index, entry in enumerate(entries, 1):
            pid = page_id(entry)
            files = resolve_files(entry)

            if koma_by_page is not None:
                koma_entry = koma_by_page.get(join_key(entry))
                if koma_entry is None:
                    print(f"skip {pid}: no koma layer")
                    continue
                if not koma_entry.get("koma"):
                    print(f"skip {pid}: koma manifest entry present but koma field is null/empty (ink={koma_entry.get('ink')}, sources={koma_entry.get('sources')}) (data gap, not a bug)")
                    continue
                koma_name, page_field = koma_entry["koma"], koma_entry.get("page", "")
            else:
                # gakuen-style: koma is embedded in the main manifest entry
                # itself (files['koma']), no separate koma_manifest.json.
                koma_name = files.get("koma")
                if not koma_name:
                    print(f"skip {pid}: no koma field in manifest entry (data gap, not a bug)")
                    continue
                page_field = entry.get("page", "")

            sketch_name, line_name = files.get("sketch"), files.get("line")
            if not member_exists(zf, args.zip_root, line_name) or not member_exists(zf, args.zip_root, sketch_name):
                print(f"skip {pid}: koma layer present but line/sketch asset missing from zip (data gap, not a bug)")
                continue
            koma_gray = load_gray(zf, args.zip_root, koma_name)
            line_gray = load_gray(zf, args.zip_root, line_name)
            rough_gray = load_gray(zf, args.zip_root, sketch_name, autocontrast=True)

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
                    "housei": pid, "page": page_field, "panel_index": panel_index,
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
                        # pad must exceed |dx|/|dy| or the y0w/x0w crop below
                        # goes negative or past the ROI edge (silent numpy
                        # wraparound / a bogus black-looking crop, not a clean
                        # error) — best["dx"]/["dy"] are the *combined*
                        # coarse+fine offset relative to this panel's own
                        # (cx, cy), which the coarse pre-search can push well
                        # past args.max_shift alone.
                        pad = max(args.max_shift, abs(best["dx"]) + 8, abs(best["dy"]) + 8)
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

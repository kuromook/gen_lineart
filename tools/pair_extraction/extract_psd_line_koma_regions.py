"""Prototype panel-region detector for `dataset_psd_line_v2.zip`.

Unlike this project's existing 6 koma-pipeline sources (ako5ver2, hamlabi,
fitness, gakuen, housei, "4th"), this source has no separate panel-border
layer -- it is 275 flattened line-art page PNGs, already extracted from PSDs
by the user's own tool (see `manifest.json`/`index.tsv` inside the zip).
Panel border lines, where a page has them, are baked directly into the
flattened line raster. It is also a mix of real multi-panel manga pages and
single-panel/no-border standalone illustration pages -- both must be handled.

A prior attempt at this exact problem (recorded in
`doc/preprocess/raw_dataset_extraction_knowledge.md`, "Planned Fix:
Panel-Boundary-First Region Segmentation") tried plain long-line morphology
(open with a long horizontal/vertical structuring element) directly as the
border mask and failed on a real ako5ver2 page: it flagged character hair
strokes as false panel borders. That failure is specifically why the other 6
sources route through a separately-extracted clean koma layer instead of
running any detector on their flattened line art at all -- for those sources
panel borders are composited from a layer that the finished line art does not
contain, so there was nothing to reliably detect from the raster alone.

This source is different: the user visually confirmed panel borders here are
actually drawn into the finished line layer. So a raster detector is not
hopeless here, but the earlier failure mode (thick, gently-curved hair
mistaken for a straight ruled border) is still a real risk and this detector
is designed defensively around it, in two independent ways:

1. **Long minimum line length, in page-relative terms.** The directional
   opening kernel length is a fraction of page width/height (default 6%),
   which for real page sizes (3600-7000 px) is several hundred pixels --
   far longer than any single hair strand, even a long straight one.
2. **Enclosure, not just line presence.** A page is scanned for *free-space
   connected components* bounded by the border-candidate mask (same
   recipe as `match_koma_panels.detect_panels()`, applied here to a detected
   mask instead of a clean pre-extracted layer). A component only becomes a
   panel candidate if it is fully enclosed (does not touch the page edge)
   and reasonably rectangular (`fill_ratio` close to 1). Long straight hair
   strands essentially never *close off* an axis-aligned rectangular region
   on all four sides; even if a strand passes the line-length gate, it will
   not by itself create a bounded interior. Round/cloud-shaped speech
   balloons are excluded for a related reason: their outlines are curved, not
   long straight horizontal/vertical runs, so they mostly do not survive the
   directional opening at all.

Full-bleed ("buchinuki") panels that cross off the page edge on some sides
are a known, explicitly out-of-scope limitation, consistent with the same
carve-out already recorded for the koma-layer pipeline: such a panel's
free-space component touches the page edge and gets folded into the
"background/margin" component, so it is silently dropped rather than
detected. Left for a later pass if this source is productionized.

Two page shapes handled:

- multi-panel manga pages: `detect_panels()` finds >=1 real bordered
  interior; each becomes a panel region.
- single-panel / no-border standalone illustrations (e.g. character sheets):
  `detect_panels()` finds nothing (or one large low-value candidate) and the
  whole page is kept as one fallback region, consistent with how
  `region_dataset_extraction_policy.md` already treats large/whole-panel
  regions as valid parent candidates rather than something to reject.

Within each panel/region, ink-dense sub-regions are proposed with the same
line-ink connected-component logic already used for housei
(`split_koma_panel_subregions.subregion_proposals`, imported directly rather
than re-implemented) before native 480x480 windows are cut out -- koma
panels/pages are mostly blank background, so tiling the whole thing wastes
almost all candidates on near-empty windows (documented yield problem from
the housei work).

This source currently has **no rough counterpart** (line-only batch, "0
sketch/pairs -- line-only by design", per `doc/preprocess/raw_dataset_
storage_policy.md`), so there is no rough/line alignment step here and no
`tile_region_manifest_480.py` gates can run (those all compare rough vs.
line edges). Output tiles are line-only, but geometrically follow the same
convention as the rest of the pipeline: exactly 480x480, native resolution
(no resizing), cut by sliding window with stride, from a masked/cropped
region -- so they slot into that pipeline's tile format once a paired rough
extraction exists for this source.

Dry-run by default (QC montages only); pass --save to write region/tile PNGs
and manifests.
"""

import argparse
import csv
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from split_koma_panel_subregions import subregion_proposals  # noqa: E402


ZIP_PATH = "dataset/raw_zips/dataset_psd_line_v2.zip"
ZIP_ROOT = "dataset_psd_line"

TILE = 480


# ---------------------------------------------------------------------------
# Zip / manifest access
# ---------------------------------------------------------------------------

def read_zip_member(zf, zip_root, name):
    for candidate in (f"{zip_root}/{name}", f"{zip_root}\\{name}", name):
        try:
            return zf.read(candidate)
        except KeyError:
            pass
    raise KeyError(f"missing zip member for {name!r}")


def load_manifest(zf, zip_root):
    return json.loads(read_zip_member(zf, zip_root, "manifest.json"))


def load_gray(zf, zip_root, name):
    image = Image.open(io.BytesIO(read_zip_member(zf, zip_root, name))).convert("L")
    return np.asarray(image)


def list_line_pages(manifest):
    """One entry per page that has a resolved line-layer output PNG.

    `outputs.line` in the manifest is a Windows path from the user's
    extraction-tool machine (e.g. `C:\\...\\dataset_psd_line\\0001_..._line.png`);
    only the basename matters for locating the file inside this zip.
    """
    pages = []
    for doc in manifest.get("documents", []):
        line_out = (doc.get("outputs") or {}).get("line")
        if not line_out:
            continue
        name = line_out.replace("\\", "/").rsplit("/", 1)[-1]
        page_id = Path(name).stem  # e.g. "0001_名称未設定 1_line"
        pages.append({"page_id": page_id, "file": name, "width": doc.get("width"), "height": doc.get("height")})
    pages.sort(key=lambda p: p["page_id"])
    return pages


# ---------------------------------------------------------------------------
# Panel-border detection (see module docstring for the two-guard rationale)
# ---------------------------------------------------------------------------

def border_candidate_mask(gray, ink_thresh, min_h_len, min_v_len, close_px, max_thickness_radius):
    """Directional-opening border mask, with a stroke-thinness pre-filter.

    The directional open alone (keep only pixels lying on a straight run of
    at least `min_h_len`/`min_v_len` ink pixels) is not sufficient on its
    own: `cv2.MORPH_OPEN` with a 1-px-tall/1-px-wide kernel only checks run
    *length*, not stroke *thickness*, so any large solid black fill (hair
    masses, screentone blocks, effect flashes -- all common in this source)
    trivially survives it too, since a filled blob's rows/columns are each
    longer than the kernel. Measured on a real page (`0032_05`): without this
    filter the border mask picks up large ragged fill-blob fragments that
    fracture the free-space enclosure test. The thinness pre-filter (ink
    pixels within `max_thickness_radius` of the ink/background boundary, via
    a distance transform -- i.e. local stroke radius, not run length) removes
    solid fills while leaving thin ruled border strokes untouched (measured
    ~7px thick at native resolution on the same page). It does not, on its
    own, distinguish a thin ruled border from thin character linework --
    that discrimination still relies on the run-length requirement below.
    """
    ink = (gray < ink_thresh).astype(np.uint8)
    if max_thickness_radius > 0:
        dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
        ink = ((dist <= max_thickness_radius) & (ink > 0)).astype(np.uint8)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, min_h_len), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, min_v_len)))
    h_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, v_kernel)
    border = ((h_lines > 0) | (v_lines > 0)).astype(np.uint8)
    if close_px > 0:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))
        border = cv2.morphologyEx(border, cv2.MORPH_CLOSE, close_kernel)
    return border > 0


def _best_span_run(coverage, min_span_frac, edge_margin):
    """Find the strongest divider position in a 1-D coverage profile.

    `coverage[i]` is the fraction of border-candidate pixels along the line
    perpendicular to the axis being tested (a row, for a horizontal divider;
    a column, for a vertical one). A real full-span divider produces a short
    run of indices near 1.0 at the line's thickness. Indices within
    `edge_margin` of either end are zeroed first so the region's own
    already-established boundary (from the parent split, or the page edge)
    is never re-detected as a fresh internal divider. Returns the (start,
    end) index range of the strongest qualifying run, or None.
    """
    coverage = coverage.copy()
    if edge_margin > 0:
        coverage[:edge_margin] = 0
        coverage[-edge_margin:] = 0
    above = coverage >= min_span_frac
    if not above.any():
        return None
    # Group consecutive above-threshold indices into runs; keep the run with
    # the highest mean coverage (thickest/cleanest single divider line).
    runs = []
    start = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(above)))
    best = max(runs, key=lambda r: coverage[r[0]:r[1]].mean())
    return best


def recursive_xy_split(border, min_span_frac, min_region_frac, min_region_px, max_depth, _depth=0):
    """Recursively partition `border` (a detect-scale bool mask, True where a
    border-candidate line pixel was detected) into leaf panel cells.

    Unlike enclosed-free-space-island detection (which requires a divider to
    form a fully closed loop on all four sides), this only requires a single
    divider line that spans most of the *current region's* width or height --
    so it correctly splits grids whose outer frame is undrawn (content
    bleeds to the page trim, or the artist never ruled an enclosing
    rectangle around an internal cross-divider) as long as the internal
    dividers themselves are long enough. This mirrors the recursive X-Y cut
    / binary space partition used for general document layout segmentation,
    and also naturally handles staircase (non-uniform-grid) layouts because
    each half is split independently after a cut, rather than assuming one
    global grid across the whole page.

    Returns a list of (y0, y1, x0, x1) leaf boxes in the input array's local
    coordinate system.
    """
    h, w = border.shape
    if _depth >= max_depth or h < min_region_px or w < min_region_px:
        return [(0, h, 0, w)]

    row_margin = max(1, int(round(h * 0.015)))
    col_margin = max(1, int(round(w * 0.015)))
    row_cov = border.mean(axis=1)
    col_cov = border.mean(axis=0)
    best_row = _best_span_run(row_cov, min_span_frac, row_margin) if h >= 2 * row_margin + 1 else None
    best_col = _best_span_run(col_cov, min_span_frac, col_margin) if w >= 2 * col_margin + 1 else None

    candidates = []
    if best_row is not None:
        candidates.append(("h", best_row, row_cov[best_row[0]:best_row[1]].mean()))
    if best_col is not None:
        candidates.append(("v", best_col, col_cov[best_col[0]:best_col[1]].mean()))
    if not candidates:
        return [(0, h, 0, w)]

    axis, (s, e), _score = max(candidates, key=lambda c: c[2])
    mid = (s + e) // 2

    if axis == "h":
        if mid < h * min_region_frac or (h - mid) < h * min_region_frac:
            return [(0, h, 0, w)]
        top = recursive_xy_split(border[:mid], min_span_frac, min_region_frac, min_region_px, max_depth, _depth + 1)
        bot = recursive_xy_split(border[mid:], min_span_frac, min_region_frac, min_region_px, max_depth, _depth + 1)
        bot = [(y0 + mid, y1 + mid, x0, x1) for (y0, y1, x0, x1) in bot]
        return top + bot
    else:
        if mid < w * min_region_frac or (w - mid) < w * min_region_frac:
            return [(0, h, 0, w)]
        left = recursive_xy_split(border[:, :mid], min_span_frac, min_region_frac, min_region_px, max_depth, _depth + 1)
        right = recursive_xy_split(border[:, mid:], min_span_frac, min_region_frac, min_region_px, max_depth, _depth + 1)
        right = [(y0, y1, x0 + mid, x1 + mid) for (y0, y1, x0, x1) in right]
        return left + right


def detect_panels(gray, args):
    """Panel interiors via recursive X-Y cut on detected divider lines (see
    `recursive_xy_split` docstring for why this replaced the earlier
    enclosed-free-space-island approach). Returns (panels, border_mask,
    detect_scale); panel bboxes are in native page pixel coordinates.
    """
    height, width = gray.shape
    long_side = max(height, width)
    detect_scale = min(1.0, args.detect_long_side / float(long_side))
    if detect_scale < 1.0:
        small = cv2.resize(
            gray, (max(1, int(round(width * detect_scale))), max(1, int(round(height * detect_scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = gray
    sh, sw = small.shape

    min_h_len = max(9, int(round(sw * args.min_line_length_frac)))
    min_v_len = max(9, int(round(sh * args.min_line_length_frac)))
    # Radius is specified directly in detect-resolution pixels, not native
    # pixels: every page is downscaled to the same `--detect-long-side`
    # target first, so stroke thickness in that common frame is already
    # roughly comparable across pages of very different native size --
    # converting back through each page's own native/detect scale factor
    # (tried first) over- or under-shrinks the radius inconsistently
    # depending on native page resolution and was measurably worse.
    border = border_candidate_mask(small, args.ink_thresh, min_h_len, min_v_len, args.border_close_px, args.max_border_thickness_radius)

    min_region_px = max(9, int(round(min(sh, sw) * args.min_panel_dim_frac)))
    leaves = recursive_xy_split(border, args.min_divider_span_frac, args.min_panel_dim_frac, min_region_px, args.max_split_depth)

    page_area = sh * sw
    inv = 1.0 / detect_scale
    panels = []
    for y0, y1, x0, x1 in leaves:
        bbox_h, bbox_w = y1 - y0, x1 - x0
        ratio = (bbox_h * bbox_w) / page_area
        if ratio < args.min_panel_area_ratio:
            continue
        if bbox_w < sw * args.min_panel_dim_frac or bbox_h < sh * args.min_panel_dim_frac:
            continue
        nx0, ny0 = int(round(x0 * inv)), int(round(y0 * inv))
        nx1, ny1 = int(round(x1 * inv)), int(round(y1 * inv))
        nx1, ny1 = min(nx1, width), min(ny1, height)
        panels.append({
            "x0": nx0, "y0": ny0, "x1": nx1, "y1": ny1,
            "area_ratio": round(ratio, 5), "fill_ratio": 1.0,
        })
    panels.sort(key=lambda p: (p["y0"], p["x0"]))
    return panels, border, detect_scale


# ---------------------------------------------------------------------------
# Sub-region ink-island proposal (reused) + native 480 tiling (no rough gates)
# ---------------------------------------------------------------------------

def make_subregion_args(args):
    return argparse.Namespace(
        line_threshold=args.line_threshold,
        min_line_component_area=args.min_line_component_area,
        region_close=args.region_close,
        line_margin=args.line_margin,
        min_region_area=args.min_region_area,
        min_region_size=args.min_region_size,
        min_line_ink=args.min_line_ink,
        max_subregions_per_panel=args.max_subregions_per_panel,
    )


def sliding_tiles(region_gray, tile, stride, ink_min, ink_max):
    h, w = region_gray.shape
    if h < tile or w < tile:
        return []
    ys = list(range(0, h - tile + 1, stride)) or [0]
    if ys[-1] != h - tile:
        ys.append(h - tile)
    xs = list(range(0, w - tile + 1, stride)) or [0]
    if xs[-1] != w - tile:
        xs.append(w - tile)
    out = []
    for y in ys:
        for x in xs:
            crop = region_gray[y:y + tile, x:x + tile]
            ink = float((crop < 128).mean())
            if ink_min <= ink <= ink_max:
                out.append((x, y, round(ink, 4)))
    return out


# ---------------------------------------------------------------------------
# QC rendering
# ---------------------------------------------------------------------------

def draw_panel_overlay(gray, panels, is_multi, max_side=900):
    height, width = gray.shape
    scale = min(1.0, max_side / float(max(height, width)))
    small = cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    canvas = Image.fromarray(small).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for i, p in enumerate(panels, 1):
        box = [p["x0"] * scale, p["y0"] * scale, p["x1"] * scale, p["y1"] * scale]
        draw.rectangle(box, outline=(220, 30, 30), width=3)
        draw.text((box[0] + 4, box[1] + 4), str(i), fill=(220, 30, 30))
    label = "multi-panel" if is_multi else "single/no-panel (fallback whole page)"
    draw.rectangle([0, 0, canvas.width, 22], fill=(255, 255, 255))
    draw.text((4, 4), label, fill=(0, 0, 0))
    return canvas


def make_contact_sheet(items, output_path, thumb=260, cols=5):
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    label_h = 18
    canvas = Image.new("RGB", (thumb * cols, (thumb + label_h) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    for i, (image, label) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = c * thumb, r * (thumb + label_h)
        thumbnail = image.copy()
        thumbnail.thumbnail((thumb, thumb))
        canvas.paste(thumbnail, (x, y))
        draw.text((x + 2, y + thumb + 1), label[:36], fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def select_pages(all_pages, args):
    if args.pages:
        wanted = {p.strip() for p in args.pages.split(",") if p.strip()}
        selected = [p for p in all_pages if p["page_id"] in wanted or p["page_id"].split("_")[0] in wanted]
        missing = wanted - {p["page_id"] for p in selected} - {p["page_id"].split("_")[0] for p in selected}
        if missing:
            print(f"warning: requested pages not found: {sorted(missing)}")
        return selected
    if args.sample_n:
        rng = random.Random(args.sample_seed)
        return rng.sample(all_pages, min(args.sample_n, len(all_pages)))
    return all_pages[: args.limit] if args.limit else all_pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default=ZIP_PATH, dest="zip_path")
    parser.add_argument("--zip-root", default=ZIP_ROOT)
    parser.add_argument("--pages", default="", help="comma-separated page_id or numeric-prefix list; default = all/--sample-n/--limit")
    parser.add_argument("--sample-n", type=int, default=0, help="randomly sample N pages instead of an explicit --pages list")
    parser.add_argument("--sample-seed", type=int, default=20260808)
    parser.add_argument("--limit", type=int, default=0, help="first N pages (manifest order); 0 = all")

    # panel-border detection
    parser.add_argument("--detect-long-side", type=int, default=1600, help="downscale long side for border detection")
    parser.add_argument("--ink-thresh", type=int, default=190)
    parser.add_argument("--min-line-length-frac", type=float, default=0.06, help="min straight-run length as a fraction of (downscaled) page width/height")
    parser.add_argument("--border-close-px", type=int, default=8, help="gap-bridging closure, at detect resolution")
    parser.add_argument("--max-border-thickness-radius", type=float, default=4.0, help="detect-resolution stroke-thickness-radius cap for the border-candidate pre-filter (excludes solid fills/hair masses, see border_candidate_mask docstring); 0 disables the filter")
    parser.add_argument("--min-divider-span-frac", type=float, default=0.7, help="recursive X-Y cut: a row/column is treated as a divider only if this fraction of it is border-candidate pixels (see recursive_xy_split)")
    parser.add_argument("--max-split-depth", type=int, default=6, help="recursive X-Y cut: max recursion depth (caps panel count at 2**depth)")
    parser.add_argument("--min-panel-area-ratio", type=float, default=0.01)
    parser.add_argument("--min-panel-dim-frac", type=float, default=0.06)

    # sub-region ink-island proposal (reused from split_koma_panel_subregions.py)
    parser.add_argument("--line-threshold", type=int, default=192)
    parser.add_argument("--min-line-component-area", type=int, default=12)
    parser.add_argument("--region-close", type=int, default=121)
    parser.add_argument("--line-margin", type=float, default=0.12)
    parser.add_argument("--min-region-area", type=int, default=3500)
    parser.add_argument("--min-region-size", type=int, default=180)
    parser.add_argument("--min-line-ink", type=float, default=0.004)
    parser.add_argument("--max-subregions-per-panel", type=int, default=6)

    # native 480 tiling (line-only; no rough alignment gates exist for this source yet)
    parser.add_argument("--tile", type=int, default=TILE)
    parser.add_argument("--tile-stride", type=int, default=240)
    parser.add_argument("--tile-ink-min", type=float, default=0.010)
    parser.add_argument("--tile-ink-max", type=float, default=0.12)
    parser.add_argument("--max-tiles-per-subregion", type=int, default=3)

    parser.add_argument("--out-dir", default="dataset/psd_line_koma_extraction_20260808")
    parser.add_argument("--results-dir", default="results/psd_line_koma_extraction_20260808")
    parser.add_argument("--overlay-qc-count", type=int, default=40)
    parser.add_argument("--tile-qc-count", type=int, default=80)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    line_dir = out_dir / "line"
    if args.save:
        line_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip_path) as zf:
        manifest = load_manifest(zf, args.zip_root)
        all_pages = list_line_pages(manifest)
        pages = select_pages(all_pages, args)
        print(f"pages available (line layer resolved)={len(all_pages)}; selected for this run={len(pages)}")

        page_rows = []
        panel_rows = []
        tile_rows = []
        overlay_items = []
        tile_qc_items = []
        n_multi, n_single = 0, 0
        missing_pages = []

        for page in pages:
            try:
                gray = load_gray(zf, args.zip_root, page["file"])
            except KeyError:
                missing_pages.append(page["page_id"])
                continue
            height, width = gray.shape
            panels, _border, _scale = detect_panels(gray, args)
            is_multi = len(panels) >= 2
            if is_multi:
                n_multi += 1
            else:
                n_single += 1

            if not panels:
                panels = [{"x0": 0, "y0": 0, "x1": width, "y1": height, "area_ratio": 1.0, "fill_ratio": 1.0, "fallback_whole_page": True}]

            page_rows.append({
                "page_id": page["page_id"], "file": page["file"], "width": width, "height": height,
                "panel_count": len(panels), "is_multi_panel": is_multi,
            })

            if len(overlay_items) < args.overlay_qc_count:
                overlay = draw_panel_overlay(gray, panels, is_multi)
                overlay_items.append((overlay, f"{page['page_id'][:20]} n={len(panels)}"))

            sub_args = make_subregion_args(args)
            for panel_index, panel in enumerate(panels, 1):
                x0, y0, x1, y1 = panel["x0"], panel["y0"], panel["x1"], panel["y1"]
                panel_crop = np.array(gray[y0:y1, x0:x1])
                panel_row = {
                    "page_id": page["page_id"], "panel_index": panel_index,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "panel_width": x1 - x0, "panel_height": y1 - y0,
                    "area_ratio": panel.get("area_ratio"), "fill_ratio": panel.get("fill_ratio"),
                    "fallback_whole_page": bool(panel.get("fallback_whole_page", False)),
                    "is_multi_panel_page": is_multi,
                }
                panel_rows.append(panel_row)

                boxes = subregion_proposals(panel_crop, sub_args)
                if not boxes:
                    ph, pw = panel_crop.shape
                    boxes = [{"box": (0, 0, pw, ph), "region_area": pw * ph, "line_ink": float((panel_crop < args.line_threshold).mean())}]

                for sub_index, sub in enumerate(boxes, 1):
                    sx0, sy0, sx1, sy1 = sub["box"]
                    sub_crop = np.array(panel_crop[sy0:sy1, sx0:sx1])
                    tiles = sliding_tiles(sub_crop, args.tile, args.tile_stride, args.tile_ink_min, args.tile_ink_max)
                    tiles = tiles[: args.max_tiles_per_subregion]
                    for tile_index, (tx, ty, ink) in enumerate(tiles, 1):
                        tile_crop = sub_crop[ty:ty + args.tile, tx:tx + args.tile]
                        name = f"psdline_{page['page_id']}_p{panel_index}_s{sub_index}_t{tile_index}.png"
                        tile_row = {
                            "page_id": page["page_id"], "panel_index": panel_index, "sub_index": sub_index,
                            "tile_index": tile_index, "name": name,
                            "page_x0": x0 + sx0 + tx, "page_y0": y0 + sy0 + ty,
                            "line_ink": ink, "is_multi_panel_page": is_multi,
                        }
                        if args.save:
                            path = line_dir / name
                            Image.fromarray(tile_crop).save(path)
                            tile_row["line_path"] = str(path)
                        tile_rows.append(tile_row)
                        if len(tile_qc_items) < args.tile_qc_count:
                            img = Image.fromarray(tile_crop).convert("RGB")
                            tile_qc_items.append((img, f"{page['page_id'][:14]} p{panel_index}s{sub_index}t{tile_index} ink={ink:.3f}"))

    if missing_pages:
        print(f"warning: {len(missing_pages)} page(s) in manifest have no resolvable zip member, skipped: {missing_pages}")
    print(f"page classification: multi-panel={n_multi} single/no-panel={n_single} (of {len(pages) - len(missing_pages)} pages processed)")
    print(f"panels detected (incl. whole-page fallbacks): {len(panel_rows)}")
    print(f"tiles: {len(tile_rows)}")

    make_contact_sheet(overlay_items, results_dir / "panel_detection_overlay_qc.png", thumb=320, cols=4)
    make_contact_sheet(tile_qc_items, results_dir / "tile_qc.png", thumb=200, cols=6)
    print(f"QC: {results_dir / 'panel_detection_overlay_qc.png'}")
    print(f"QC: {results_dir / 'tile_qc.png'}")

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "pages.csv", "w", newline="") as f:
        if page_rows:
            writer = csv.DictWriter(f, fieldnames=list(page_rows[0].keys()))
            writer.writeheader()
            writer.writerows(page_rows)
    with open(results_dir / "panels.csv", "w", newline="") as f:
        if panel_rows:
            writer = csv.DictWriter(f, fieldnames=list(panel_rows[0].keys()))
            writer.writeheader()
            writer.writerows(panel_rows)
    with open(results_dir / "tiles.csv", "w", newline="") as f:
        if tile_rows:
            writer = csv.DictWriter(f, fieldnames=list(tile_rows[0].keys()))
            writer.writeheader()
            writer.writerows(tile_rows)
    (results_dir / "pages.json").write_text(json.dumps(page_rows, indent=2, ensure_ascii=False) + "\n")

    if args.save:
        print(f"saved {len(tile_rows)} line tiles to {line_dir}")
    else:
        print("dry-run: pass --save to write region/tile PNGs to disk (manifests/CSVs above are always written)")


if __name__ == "__main__":
    main()

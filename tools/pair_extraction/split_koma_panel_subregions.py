"""Split accepted koma panels into ink-dense sub-regions before 480px tiling.

Diagnosed via a per-gate funnel measurement (stride=480 sample over all 59
originally-accepted housei koma panels): `ink_range` alone rejected 80.9% of
candidate tiles (median line ink density 0.0018, far below the 0.012 floor),
while the alignment gate rejected only 0.7%. Koma panels are defined by
panel-border geometry, not content density, so a large fraction of a panel's
area can be blank background/margin; a naive 480px sliding window over the
whole panel wastes most candidates on sparse windows.

This reuses the ink-connected-component region-proposal logic already
validated for hamlabi's page-level region finding
(`match_hamlabi_regions.py`'s `region_proposals()` / `line_ink_mask()`),
scoped to one already-aligned koma panel instead of a whole page: find dense
line-ink islands within the panel and crop just those as smaller sub-regions.

Sub-regions start from the panel's already-verified alignment, but a single
panel-level `(dx, dy, scale)` is only the best *average* transform for the
whole panel; individual content islands within a busy panel (e.g. two
characters at different depths) can have their own small residual offset
(the same non-uniform-deformation finding documented for whole panels, just
recurring one level down). `--refine-alignment` (on by default) re-runs a
small local translation+scale search per sub-region, using the panel-level
alignment as the starting point rather than the wide +-160px search used at
the panel level (the panel is already roughly right, so this only needs to
find a small residual, not search from scratch).

If no sub-region is found in a panel (e.g. content too sparse everywhere),
the whole panel is kept as a single fallback region so no panel is silently
dropped.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from match_koma_panels import SCALES as PANEL_SCALES, extract_scaled_roi
from tile_region_manifest_480 import ALIGNMENT_CLOSE_PX, ALIGNMENT_TRUNCATE_PX, chamfer, edge_map


def line_ink_mask(line, threshold, min_component_area):
    mask = (line < threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            cleaned[labels == label] = 1
    return cleaned


def clamp_box(box, width, height):
    x0, y0, x1, y1 = box
    x0 = max(0, min(width - 1, int(round(x0))))
    y0 = max(0, min(height - 1, int(round(y0))))
    x1 = max(x0 + 1, min(width, int(round(x1))))
    y1 = max(y0 + 1, min(height, int(round(y1))))
    return x0, y0, x1, y1


def expand_box(box, margin, width, height):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    pad = max(bw, bh) * margin
    return clamp_box((x0 - pad, y0 - pad, x1 + pad, y1 + pad), width, height)


def subregion_proposals(line, args):
    h, w = line.shape
    mask = line_ink_mask(line, args.line_threshold, args.min_line_component_area)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.region_close, args.region_close))
    grouped = cv2.dilate(mask, close_kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    boxes = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < args.min_region_area or bw < args.min_region_size or bh < args.min_region_size:
            continue
        box = expand_box((x, y, x + bw, y + bh), args.line_margin, w, h)
        bx0, by0, bx1, by1 = box
        line_crop = line[by0:by1, bx0:bx1]
        line_ink = float((line_crop < args.line_threshold).mean())
        if line_ink < args.min_line_ink:
            continue
        boxes.append({"box": box, "region_area": area, "line_ink": line_ink})
    boxes.sort(key=lambda item: (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]), reverse=True)
    return boxes[: args.max_subregions_per_panel] if args.max_subregions_per_panel else boxes


def refine_subregion_alignment(rough_panel, line_sub, cx, cy, out_w, out_h, args):
    """Small local translation+scale search, starting from the panel's own
    alignment. `rough_panel` is the whole already-panel-aligned rough image
    (dx=dy=0, scale=1.0 relative to it *is* the panel-level alignment), so
    this only needs a small range to find a sub-region's residual offset, not
    a wide from-scratch search.
    """
    line_edge = edge_map(line_sub)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(ALIGNMENT_CLOSE_PX) * 2 + 1, int(ALIGNMENT_CLOSE_PX) * 2 + 1)
    )
    line_support = cv2.dilate(line_edge.astype(np.uint8), close_kernel) > 0
    line_ys, line_xs = np.nonzero(line_edge)
    n_line = len(line_ys)

    pad = args.refine_max_shift
    offsets = []
    for dy in range(-pad, pad + 1, args.refine_shift_step):
        for dx in range(-pad, pad + 1, args.refine_shift_step):
            offsets.append((dx, dy))
    offsets.sort(key=lambda item: item[0] ** 2 + item[1] ** 2)

    best = None
    baseline = None
    for scale in args.refine_scales:
        roi = extract_scaled_roi(rough_panel, cx, cy, out_w, out_h, pad, scale)
        if roi is None:
            continue
        roi_edge = edge_map(roi)
        roi_support = cv2.dilate(roi_edge.astype(np.uint8), close_kernel) > 0
        rough_ys, rough_xs = np.nonzero(roi_edge)
        for dx, dy in offsets:
            y0, x0 = pad + dy, pad + dx
            recall = float(roi_support[line_ys + y0, line_xs + x0].mean()) if n_line else 0.0
            wy, wx = rough_ys - y0, rough_xs - x0
            inside = (wy >= 0) & (wy < out_h) & (wx >= 0) & (wx < out_w)
            n_inside = int(inside.sum())
            if n_inside < 10 or n_line < 10:
                f1 = 0.0
            else:
                precision = float(line_support[wy[inside], wx[inside]].mean())
                f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
            candidate = {"scale": scale, "dx": dx, "dy": dy, "f1": f1, "roi_edge": roi_edge, "y0": y0, "x0": x0}
            if scale == 1.0 and dx == 0 and dy == 0:
                baseline = candidate
            if best is None or f1 > best["f1"]:
                best = candidate

    def finalize(candidate):
        win_edge = candidate["roi_edge"][candidate["y0"]:candidate["y0"] + out_h, candidate["x0"]:candidate["x0"] + out_w]
        ch = chamfer(win_edge, line_edge, ALIGNMENT_TRUNCATE_PX) if win_edge.sum() >= 10 else float(ALIGNMENT_TRUNCATE_PX)
        return ch

    best_chamfer = finalize(best)
    base_chamfer = finalize(baseline) if baseline is not None else best_chamfer
    roi_final = extract_scaled_roi(rough_panel, cx, cy, out_w, out_h, pad, best["scale"])
    y0f, x0f = pad + best["dy"], pad + best["dx"]
    rough_sub = roi_final[y0f:y0f + out_h, x0f:x0f + out_w]
    return rough_sub, best["scale"], best["dx"], best["dy"], base_chamfer, best_chamfer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-manifest", default="dataset/regions_housei_koma_panels_20260726/manifest.csv")
    parser.add_argument("--out-dir", default="dataset/regions_housei_koma_subregions_20260726")
    parser.add_argument("--max-chamfer", type=float, default=20.0)
    parser.add_argument("--exclude-housei", default="")
    # same defaults as match_hamlabi_regions.py's page-level region_proposals(),
    # since koma panels are native-resolution crops from the same source pages.
    parser.add_argument("--line-threshold", type=int, default=192)
    parser.add_argument("--min-line-component-area", type=int, default=12)
    parser.add_argument("--region-close", type=int, default=121)
    parser.add_argument("--line-margin", type=float, default=0.12)
    parser.add_argument("--min-region-area", type=int, default=3500)
    parser.add_argument("--min-region-size", type=int, default=180)
    parser.add_argument("--min-line-ink", type=float, default=0.004)
    parser.add_argument("--max-subregions-per-panel", type=int, default=8)
    parser.add_argument("--refine-alignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-max-shift", type=int, default=48)
    parser.add_argument("--refine-shift-step", type=int, default=8)
    parser.add_argument("--refine-scales", type=float, nargs="+", default=PANEL_SCALES)
    parser.add_argument("--qc-out", default="results/housei_koma_subregions_20260726_qc.png")
    parser.add_argument("--offset-panels", type=int, default=0, help="for chunking")
    parser.add_argument("--limit-panels", type=int, default=0, help="0 = all; for quick testing/chunking")
    parser.add_argument("--append", action="store_true", help="append to out-dir manifest instead of overwriting")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    excluded = {h.strip() for h in args.exclude_housei.split(",") if h.strip()}

    with open(args.panel_manifest, newline="") as file:
        panel_rows = list(csv.DictReader(file))
    accepted_panels = [
        r for r in panel_rows
        if r["housei"] not in excluded and float(r["chamfer"]) <= args.max_chamfer
    ]
    if args.offset_panels:
        accepted_panels = accepted_panels[args.offset_panels :]
    if args.limit_panels:
        accepted_panels = accepted_panels[: args.limit_panels]
    print(f"panels: total={len(panel_rows)} accepted(chamfer<={args.max_chamfer})={len(accepted_panels)}")

    out_dir = Path(args.out_dir)
    rough_dir, line_dir = out_dir / "rough", out_dir / "line"
    if args.save:
        rough_dir.mkdir(parents=True, exist_ok=True)
        line_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    fallback_count = 0
    qc_items = []
    for panel_row in accepted_panels:
        rough = np.asarray(Image.open(panel_row["native_rough_path"]).convert("L"))
        line = np.asarray(Image.open(panel_row["native_line_path"]).convert("L"))
        h, w = line.shape

        boxes = subregion_proposals(line, args)
        if not boxes:
            boxes = [{"box": (0, 0, w, h), "region_area": w * h, "line_ink": float((line < args.line_threshold).mean())}]
            fallback_count += 1

        for sub_index, sub in enumerate(boxes, 1):
            x0, y0, x1, y1 = sub["box"]
            out_w, out_h = x1 - x0, y1 - y0
            line_sub = line[y0:y1, x0:x1]

            if args.refine_alignment:
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                rough_sub, r_scale, r_dx, r_dy, r_base_chamfer, r_chamfer = refine_subregion_alignment(
                    rough, line_sub, cx, cy, out_w, out_h, args
                )
            else:
                rough_sub = rough[y0:y1, x0:x1]
                r_scale, r_dx, r_dy, r_base_chamfer, r_chamfer = 1.0, 0, 0, None, None

            name = f"sub_{panel_row['name'].rsplit('.', 1)[0]}_s{sub_index}.png"
            out_row = {
                **panel_row,
                "sub_index": sub_index,
                "sub_x0": x0, "sub_y0": y0, "sub_x1": x1, "sub_y1": y1,
                "sub_line_ink": round(sub["line_ink"], 4),
                "sub_refine_scale": r_scale, "sub_refine_dx": r_dx, "sub_refine_dy": r_dy,
                "sub_base_chamfer": r_base_chamfer, "sub_chamfer": r_chamfer,
                "native_long_side": max(out_w, out_h),
                "native_width": out_w,
                "native_height": out_h,
                "name": name,
            }
            if args.save:
                rough_path, line_path = rough_dir / name, line_dir / name
                Image.fromarray(rough_sub).save(rough_path)
                Image.fromarray(line_sub).save(line_path)
                out_row["native_rough_path"] = str(rough_path)
                out_row["native_line_path"] = str(line_path)
            out_rows.append(out_row)
            if len(qc_items) < 80:
                label = f'{panel_row["housei"]}p{panel_row["panel_index"]}s{sub_index} ink={sub["line_ink"]:.3f} {out_w}x{out_h}'
                if r_chamfer is not None:
                    label += f' cham={r_base_chamfer:.1f}->{r_chamfer:.1f} scale={r_scale:.2f} d=({r_dx},{r_dy})'
                qc_items.append((rough_sub, line_sub, label))

    print(f"sub-regions: {len(out_rows)} from {len(accepted_panels)} panels "
          f"({fallback_count} panels had no dense sub-region, kept whole panel as fallback)")
    if args.refine_alignment:
        chamfers_base = [r["sub_base_chamfer"] for r in out_rows if r["sub_base_chamfer"] is not None]
        chamfers_best = [r["sub_chamfer"] for r in out_rows if r["sub_chamfer"] is not None]
        if chamfers_base:
            print(f"sub-region chamfer median: base={np.median(chamfers_base):.2f} refined={np.median(chamfers_best):.2f}")

    if qc_items:
        from PIL import ImageDraw, ImageFont
        thumb = 220
        canvas = Image.new("RGB", (thumb * 2, (thumb + 30) * len(qc_items)), "white")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
        except OSError:
            font = ImageFont.load_default()
        for i, (r, l, label) in enumerate(qc_items):
            y = i * (thumb + 30)
            canvas.paste(Image.fromarray(r).convert("RGB").resize((thumb, thumb)), (0, y))
            canvas.paste(Image.fromarray(l).convert("RGB").resize((thumb, thumb)), (thumb, y))
            draw.text((4, y + thumb + 2), label, fill=0, font=font)
        Path(args.qc_out).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(args.qc_out)
        print(f"QC: {args.qc_out}")

    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "manifest.json"
        all_rows = out_rows
        if args.append and json_path.exists():
            all_rows = json.loads(json_path.read_text()) + out_rows
        json_path.write_text(json.dumps(all_rows, indent=2) + "\n")
        fields = list(all_rows[0].keys()) if all_rows else []
        with open(out_dir / "manifest.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"saved {len(out_rows)} rows this run, total accumulated={len(all_rows)}, to {out_dir}/manifest.csv")
    else:
        print(f"dry-run: would save {len(out_rows)} rows to {out_dir}/manifest.csv; pass --save to write")


if __name__ == "__main__":
    main()

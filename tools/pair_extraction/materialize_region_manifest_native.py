"""Re-materialize a reviewed region manifest near native source resolution.

The 768 px long-side normalization used by
`tools/pair_extraction/materialize_region_candidates.py` is not scale-neutral:
a 121 px region and a 6,947 px region both become 768 px, so downstream 480 px
tiles mix 6x upscaled fragments with 9x downscaled page composition. See
`doc/region_dataset_extraction_policy.md`.

This tool re-crops the same reviewed regions from the source pages at
`source_size / --source-scale-divisor`, so every output pixel corresponds to a
fixed number of manuscript pixels across the whole dataset.

Alignment is redone here. The reviewed `align_dx` / `align_dy` were found in the
768 px space, so their residual grows with the region's downscale factor; they
are used only as a starting offset and refined at the output scale.

Dry-run is the default; use --save to write images and the manifest.
"""

import argparse
import ast
import csv
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def read_zip_member(archive, root, name):
    for candidate in (f"{root}/{name}", f"{root}\\{name}", name):
        try:
            return archive.read(candidate)
        except KeyError:
            pass
    raise KeyError(f"missing zip member for {name!r}")


def load_page_index(archive, root):
    manifest = json.loads(read_zip_member(archive, root, "manifest.json"))
    index = {}
    for entry in manifest:
        page = Path(entry.get("file", entry["line"])).stem.replace("page", "")
        index[page] = entry
    return index


def load_page_pair(archive, root, entry):
    rough = Image.open(io.BytesIO(read_zip_member(archive, root, entry["sketch"]))).convert("L")
    line = Image.open(io.BytesIO(read_zip_member(archive, root, entry["line"]))).convert("L")
    return np.asarray(ImageOps.autocontrast(rough, cutoff=0)), np.asarray(line)


def parse_box(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(round(float(v))) for v in value)
    return tuple(int(round(float(v))) for v in ast.literal_eval(value))


def crop_box(page, box):
    x0, y0, x1, y1 = box
    height, width = page.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return page[y0:y1, x0:x1]


def resize_to(image, width, height):
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS))


def edge_map(gray, sigma):
    return cv2.Canny(cv2.GaussianBlur(gray, (0, 0), sigma), 45, 135) > 0


def content_windows(line, size, count):
    """Pick the most ink-dense non-overlapping windows to score alignment on."""
    height, width = line.shape
    if height <= size or width <= size:
        return [(0, 0, min(width, size), min(height, size))]
    integral = cv2.integral((line < 128).astype(np.uint8))
    stride = max(size // 2, 1)
    scored = []
    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            total = (
                integral[y + size, x + size]
                - integral[y, x + size]
                - integral[y + size, x]
                + integral[y, x]
            )
            scored.append((int(total), x, y))
    scored.sort(reverse=True)
    picked = []
    for _, x, y in scored:
        if any(abs(x - px) < size and abs(y - py) < size for px, py, _, _ in picked):
            continue
        picked.append((x, y, size, size))
        if len(picked) >= count:
            break
    return picked or [(0, 0, size, size)]


def prepare_windows(rough, line, windows, blur_sigma):
    """Precompute edge coordinates and distance fields once per window.

    Scoring a shift then costs two coordinate lookups instead of a dilation, so a
    wide search range stays affordable at native resolution.
    """
    prepared = []
    for x, y, w, h in windows:
        rough_edge = edge_map(rough[y : y + h, x : x + w], blur_sigma)
        line_edge = edge_map(line[y : y + h, x : x + w], blur_sigma)
        if rough_edge.sum() < 20 or line_edge.sum() < 20:
            continue
        rough_y, rough_x = np.nonzero(rough_edge)
        line_y, line_x = np.nonzero(line_edge)
        prepared.append({
            "line_dist": cv2.distanceTransform((~line_edge).astype(np.uint8), cv2.DIST_L2, 3),
            "rough_dist": cv2.distanceTransform((~rough_edge).astype(np.uint8), cv2.DIST_L2, 3),
            "rough_y": rough_y,
            "rough_x": rough_x,
            "line_y": line_y,
            "line_x": line_x,
            "shape": rough_edge.shape,
        })
    return prepared


def matched_ratio(field, ys, xs, tolerance, shape):
    """Share of `ys`/`xs` landing within `tolerance` of the field's structure.

    Points shifted outside the window count as unmatched, so large shifts are
    penalized instead of rewarded by dropping pixels.
    """
    height, width = shape
    inside = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
    if not inside.any():
        return 0.0
    matched = field[ys[inside], xs[inside]] <= tolerance
    return float(matched.sum() / len(ys))


def shift_score(prepared, tolerance, dx, dy):
    total = 0.0
    weight = 0.0
    for window in prepared:
        precision = matched_ratio(
            window["line_dist"], window["rough_y"] + dy, window["rough_x"] + dx, tolerance, window["shape"]
        )
        recall = matched_ratio(
            window["rough_dist"], window["line_y"] - dy, window["line_x"] - dx, tolerance, window["shape"]
        )
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        count = len(window["line_y"])
        total += f1 * count
        weight += count
    return total / weight if weight else -1.0


def refine_shift(rough, line, args):
    """Two-stage translation search at output scale over ink-dense windows.

    The coarse stage uses a wide match tolerance so a large step cannot step over
    the score peak; the fine stage re-scores at the target tolerance.
    """
    windows = content_windows(line, args.align_window, args.align_samples)
    prepared = prepare_windows(rough, line, windows, args.blur_sigma)
    if not prepared:
        return {"dx": 0, "dy": 0, "score": 0.0, "base_score": 0.0, "score_gain": 0.0, "windows": 0, "boundary": 0}

    step = args.align_coarse_step
    coarse_best = (shift_score(prepared, args.align_coarse_tolerance, 0, 0), 0, 0)
    for dy in range(-args.align_range, args.align_range + 1, step):
        for dx in range(-args.align_range, args.align_range + 1, step):
            if dx == 0 and dy == 0:
                continue
            score = shift_score(prepared, args.align_coarse_tolerance, dx, dy)
            if score > coarse_best[0]:
                coarse_best = (score, dx, dy)

    base = shift_score(prepared, args.align_tolerance, 0, 0)
    best = {"dx": 0, "dy": 0, "score": base, "base_score": base, "windows": len(prepared)}
    _, cx, cy = coarse_best
    for dy in range(cy - step, cy + step + 1, max(args.align_refine_step, 1)):
        for dx in range(cx - step, cx + step + 1, max(args.align_refine_step, 1)):
            if abs(dx) > args.align_range or abs(dy) > args.align_range:
                continue
            score = shift_score(prepared, args.align_tolerance, dx, dy)
            if score > best["score"]:
                best.update(dx=dx, dy=dy, score=score)
    best["score_gain"] = best["score"] - base
    if best["score_gain"] < args.align_min_gain:
        best.update(dx=0, dy=0, score=base, score_gain=0.0)
    best["boundary"] = int(abs(best["dx"]) >= args.align_range or abs(best["dy"]) >= args.align_range)
    return best


def shift_image(image, dx, dy, fill=255):
    height, width = image.shape
    out = np.full_like(image, fill)
    sx0, sy0 = max(0, -dx), max(0, -dy)
    sx1, sy1 = min(width, width - dx), min(height, height - dy)
    dx0, dy0 = max(0, dx), max(0, dy)
    if sx1 > sx0 and sy1 > sy0:
        out[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = image[sy0:sy1, sx0:sx1]
    return out


def make_qc(items, path, thumb):
    if not items:
        return
    columns = ["rough", "line", "edge overlay"]
    header, label = 26, 32
    canvas = Image.new("RGB", (thumb * len(columns), header + (thumb + label) * len(items)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except OSError:
        font = small = ImageFont.load_default()
    for column, title in enumerate(columns):
        draw.text((column * thumb + 6, 5), title, fill=0, font=font)
    for index, item in enumerate(items):
        y = header + index * (thumb + label)
        rough, line = item["rough"], item["line"]
        overlay = np.full((*line.shape, 3), 255, np.uint8)
        overlay[edge_map(rough, 1.0)] = (255, 60, 60)
        overlay[edge_map(line, 1.0)] = (40, 80, 255)
        for column, image in enumerate((rough, line, overlay)):
            canvas.paste(Image.fromarray(image).convert("RGB").resize((thumb, thumb)), (column * thumb, y))
        draw.text((4, y + thumb + 2), item["label"], fill=0, font=small)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_csv(rows, path):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--zip-root", required=True)
    parser.add_argument("--out-base", required=True)
    parser.add_argument("--source-scale-divisor", type=float, default=1.0)
    parser.add_argument("--max-long-side", type=int, default=0)
    parser.add_argument("--min-output-size", type=int, default=480)
    parser.add_argument("--name-prefix", default="native")
    parser.add_argument("--prior-align-long-side", type=int, default=768)
    parser.add_argument("--no-prior-align", action="store_true")
    parser.add_argument("--align-range", type=int, default=96)
    parser.add_argument("--align-coarse-step", type=int, default=8)
    parser.add_argument("--align-coarse-tolerance", type=int, default=8)
    parser.add_argument("--align-refine-step", type=int, default=1)
    parser.add_argument("--align-tolerance", type=int, default=3)
    parser.add_argument("--align-window", type=int, default=480)
    parser.add_argument("--align-samples", type=int, default=6)
    parser.add_argument("--align-min-gain", type=float, default=0.005)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--qc-count", type=int, default=30)
    parser.add_argument("--qc-thumb", type=int, default=260)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with manifest_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if args.limit:
        rows = rows[: args.limit]

    out_base = Path(args.out_base)
    rough_dir, line_dir = out_base / "rough", out_base / "line"
    if args.save:
        rough_dir.mkdir(parents=True, exist_ok=True)
        line_dir.mkdir(parents=True, exist_ok=True)

    out_rows, qc_items = [], []
    skipped = {}
    # Source pages are large, so group work by page to decode each page once.
    ordered = sorted(enumerate(rows, 1), key=lambda item: (item[1].get("page") or "", item[0]))
    with zipfile.ZipFile(args.zip_path) as archive:
        pages = load_page_index(archive, args.zip_root)
        cache = {}
        done = 0
        for index, row in ordered:
            page = row.get("page")
            if page not in pages:
                skipped["missing_page"] = skipped.get("missing_page", 0) + 1
                continue
            if page not in cache:
                cache.clear()
                cache[page] = load_page_pair(archive, args.zip_root, pages[page])
            rough_page, line_page = cache[page]

            line_box = parse_box(row["line_box"])
            rough_box = parse_box(row["rough_box"])
            line_long = max(line_box[2] - line_box[0], line_box[3] - line_box[1])
            rough_long = max(rough_box[2] - rough_box[0], rough_box[3] - rough_box[1])

            scale = 1.0 / max(args.source_scale_divisor, 1e-6)
            if args.max_long_side:
                scale = min(scale, args.max_long_side / max(line_long, 1))
            out_w = max(1, int(round((line_box[2] - line_box[0]) * scale)))
            out_h = max(1, int(round((line_box[3] - line_box[1]) * scale)))
            if min(out_w, out_h) < args.min_output_size:
                skipped["below_min_output"] = skipped.get("below_min_output", 0) + 1
                continue

            prior_dx = prior_dy = 0
            if not args.no_prior_align:
                ratio = rough_long / max(args.prior_align_long_side, 1)
                prior_dx = int(round(float(row.get("align_dx") or 0.0) * ratio))
                prior_dy = int(round(float(row.get("align_dy") or 0.0) * ratio))
            shifted_rough_box = (
                rough_box[0] - prior_dx,
                rough_box[1] - prior_dy,
                rough_box[2] - prior_dx,
                rough_box[3] - prior_dy,
            )

            line_crop = crop_box(line_page, line_box)
            rough_crop = crop_box(rough_page, shifted_rough_box)
            if line_crop is None or rough_crop is None:
                skipped["empty_crop"] = skipped.get("empty_crop", 0) + 1
                continue
            line_out = resize_to(line_crop, out_w, out_h)
            rough_out = resize_to(rough_crop, out_w, out_h)

            shift = refine_shift(rough_out, line_out, args)
            if shift["dx"] or shift["dy"]:
                rough_out = shift_image(rough_out, shift["dx"], shift["dy"])

            name = f"{args.name_prefix}_{index:04d}_p{page}_r{row.get('rank') or 0}.png"
            if args.save:
                Image.fromarray(rough_out).save(rough_dir / name, compress_level=args.png_compress_level)
                Image.fromarray(line_out).save(line_dir / name, compress_level=args.png_compress_level)
            out_rows.append({
                **row,
                "_native_order": index,
                "native_name": name,
                "native_rough_path": str(rough_dir / name),
                "native_line_path": str(line_dir / name),
                "native_width": out_w,
                "native_height": out_h,
                "native_long_side": max(out_w, out_h),
                "source_line_long_side": line_long,
                "source_scale_divisor": args.source_scale_divisor,
                "native_src_per_out": line_long / max(max(out_w, out_h), 1),
                "prior_align_dx": prior_dx,
                "prior_align_dy": prior_dy,
                "native_align_dx": shift["dx"],
                "native_align_dy": shift["dy"],
                "native_align_score": shift["score"],
                "native_align_base_score": shift["base_score"],
                "native_align_score_gain": shift["score_gain"],
                "native_align_windows": shift["windows"],
                "native_align_boundary": shift["boundary"],
            })
            if len(qc_items) < args.qc_count:
                qc_items.append({
                    "rough": rough_out,
                    "line": line_out,
                    "label": (
                        f"{index:03d} p{page} {out_w}x{out_h} prior=({prior_dx},{prior_dy}) "
                        f"native=({shift['dx']},{shift['dy']}) "
                        f"f1={shift['base_score']:.3f}->{shift['score']:.3f} "
                        f"bnd={shift['boundary']}"
                    ),
                })
            done += 1
            if done % 20 == 0 or done == len(ordered):
                print(f"rows: {done}/{len(ordered)} kept={len(out_rows)}", flush=True)

    out_rows.sort(key=lambda row: row["_native_order"])
    qc_path = out_base / "native_materialize_qc.png"
    make_qc(qc_items, qc_path, args.qc_thumb)
    if args.save:
        write_csv(out_rows, out_base / "manifest.csv")
        (out_base / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (out_base / "manifest.json").write_text(json.dumps(out_rows, indent=2) + "\n")

    boundary = sum(row["native_align_boundary"] for row in out_rows)
    shifted = sum(1 for row in out_rows if row["native_align_dx"] or row["native_align_dy"])
    gain = sum(row["native_align_score_gain"] for row in out_rows) / max(len(out_rows), 1)
    pixels = sum(row["native_width"] * row["native_height"] for row in out_rows)
    action = "saved" if args.save else "dry-run"
    print(f"{action}: source_rows={len(rows)} kept={len(out_rows)} pixels={pixels / 1e6:.0f}Mpx")
    print(f"align: shifted={shifted} boundary_hits={boundary} mean_gain={gain:.4f}")
    if skipped:
        print("skipped: " + " ".join(f"{key}={value}" for key, value in sorted(skipped.items())))
    print(f"wrote: {qc_path}" + (f", {out_base}/manifest.csv" if args.save else ""))


if __name__ == "__main__":
    main()

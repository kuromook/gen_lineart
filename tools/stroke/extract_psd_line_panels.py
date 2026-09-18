#!/usr/bin/env python
"""Panel crops from the line-only psd_line source (no pairs, no rough).

Track F's corpus is panels; psd_line exists only as 480px tiles, which this
track has banned. The panel detector already written for this source
(`../lineart/tools/pair_extraction/extract_psd_line_koma_regions.py`,
recursive X-Y cut on border candidates) is reused as a library -- only the
output changes: native-resolution panel crops instead of tiles.

QC: an overlay contact sheet, because the detector's known failure mode is
mistaking thick hair for a ruled border. The pre-registered bar (plan
2026-09-18) is that 80% of sampled panels are cut sensibly.
"""
import argparse, csv, sys, zipfile
from pathlib import Path

import cv2
import numpy as np

LINEART = Path("/home/sh1/deepl/lineart")
sys.path.insert(0, str(LINEART / "tools/pair_extraction"))
import extract_psd_line_koma_regions as ex  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", dest="zip_path", default=str(LINEART / "dataset/raw_zips/dataset_psd_line_v3.zip"))
    p.add_argument("--zip-root", default="dataset_psd_line")
    p.add_argument("--out-dir", default=str(LINEART / "dataset/psd_line_koma_panels_20260918"))
    p.add_argument("--results-dir", default="results/psd_line_panels_20260918")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--min-panel-px", type=int, default=400, help="drop panels whose short side is below this")
    p.add_argument("--min-ink", type=float, default=0.005)
    p.add_argument("--qc", type=int, default=40)
    a = p.parse_args()
    # the detector's own knobs, at their defaults
    args = ex.argparse.Namespace(
        detect_long_side=1600, ink_thresh=190, min_line_length_frac=0.06, border_close_px=8,
        max_border_thickness_radius=4.0, min_divider_span_frac=0.7, max_split_depth=6,
        min_panel_area_ratio=0.01, min_panel_dim_frac=0.06)
    out_line = Path(a.out_dir) / "line"
    out_line.mkdir(parents=True, exist_ok=True)
    res = Path(a.results_dir); res.mkdir(parents=True, exist_ok=True)
    rows, overlays = [], []
    with zipfile.ZipFile(a.zip_path) as zf:
        manifest = ex.load_manifest(zf, a.zip_root)
        pages = ex.list_line_pages(manifest)
        if a.limit:
            pages = pages[: a.limit]
        print(f"pages {len(pages)}", flush=True)
        for i, page in enumerate(pages, 1):
            try:
                gray = ex.load_gray(zf, a.zip_root, page["file"])
            except KeyError:
                continue
            h, w = gray.shape
            panels, _b, _s = ex.detect_panels(gray, args)
            multi = len(panels) >= 2
            if not panels:
                panels = [{"x0": 0, "y0": 0, "x1": w, "y1": h, "fallback_whole_page": True}]
            if len(overlays) < a.qc:
                overlays.append((ex.draw_panel_overlay(gray, panels, multi),
                                 f"{page['page_id'][:20]} n={len(panels)}"))
            for k, pan in enumerate(panels, 1):
                x0, y0, x1, y1 = pan["x0"], pan["y0"], pan["x1"], pan["y1"]
                crop = np.array(gray[y0:y1, x0:x1])
                ph, pw = crop.shape
                ink = float((crop < 128).mean())
                if min(ph, pw) < a.min_panel_px or ink < a.min_ink:
                    continue
                name = f"psdline_{page['page_id'][:40].replace('/', '_')}_p{k}.png"
                cv2.imwrite(str(out_line / name), crop)
                rows.append({"name": name, "page_id": page["page_id"], "panel_index": k,
                             "x0": x0, "y0": y0, "x1": x1, "y1": y1, "w": pw, "h": ph,
                             "mp": round(pw * ph / 1e6, 2), "ink": round(ink, 4),
                             "panels_on_page": len(panels),
                             "whole_page": bool(pan.get("fallback_whole_page", False))})
            if i % 25 == 0:
                print(f"{i}/{len(pages)} panels {len(rows)}", flush=True)
    with open(res / "panels.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, list(rows[0])); w_.writeheader(); w_.writerows(rows)
    ex.make_contact_sheet(overlays, res / "overlay_qc.png")
    mp = np.array([r["mp"] for r in rows])
    print(f"\npanels {len(rows)} -> {out_line}")
    print(f"MP 中央値 {np.median(mp):.1f} / p90 {np.percentile(mp,90):.1f} / 最大 {mp.max():.1f}")
    print(f"1ページ複数コマ: {np.mean([r['panels_on_page']>1 for r in rows]):.1%}  "
          f"ページ全体を1コマ扱い: {np.mean([r['whole_page'] for r in rows]):.1%}")
    print(f"QC: {res/'overlay_qc.png'}")


if __name__ == "__main__":
    main()

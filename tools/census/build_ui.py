#!/usr/bin/env python
"""集計を1枚の HTML に差し込む。

**公開した CSV をそのまま読んで埋め込む** —— 画面が集計より古くなる経路を作らない。
`fetch` を使わないので **`clone` して `file://` で開けばどこでも動く**(サーバー不要)。

**段 A だけを埋め込む。** 段 C は画面のファイル選択で手元の
`data/census_full/` を読む(読んだものはそのページから出ない)。

    ./venv/bin/python tools/census/build_ui.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caveats import CAVEATS  # noqa: E402
import build_census as bc  # noqa: E402

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_template.html")


# **表の列 → その列の「答えない問い」**(`caveats.py` の `column` で引く)。
# 仕様 §5: **各数の隣に答えない問いを出す。** 空欄だけでなく**値のある列にも**出す。
# **対応の無い列はここに書かない** —— 書かないことで「但し書きが無い」と分かる
COLUMN_CAVEATS = {
    "year_label": ["年(`page_path` の第1区画)"],
    "n_works": ["作品(`page_path` の第2区画)"],
    "n_pages_without_work": ["作品(`page_path` の第2区画)"],
    "n_pages": ["ファイル数 / 異なり数"],
    "n_distinct_pages": ["ファイル数 / 異なり数", "ファイル 9,653 と 異なり 7,418 の差"],
    "n": ["ファイル 9,653 と 異なり 7,418 の差"],
    "share": ["重複の割合(双子を持つファイルの %)"],
    "format": ["重複の割合(双子を持つファイルの %)"],
    "category": ["納品をまたいだ同じ名前", "どちらの納品が現行か"],
    "cluster_size": ["ファイル 9,653 と 異なり 7,418 の差"],
    "n_clusters": ["ファイル 9,653 と 異なり 7,418 の差"],
    "ext_clip": ["ファイル数 / 異なり数"],
    "ext_lip": ["`ext_lip`(この集計に入っている `.lip` の頁数)",
                "ファイル数 / 異なり数"],
    "n_layers": ["層数(集計どうしの比較)"],
    "layer_type": ["layer_type", "layer_type の2の位(コマ枠フォルダー)"],
    "layer_type_and_2": ["layer_type の2の位(コマ枠フォルダー)",
                         "マスクの向き(`InitColor`)"],
    "n_drawn_blank": ["drawn_ratio(空欄)"],
    "drawn_ratio_usable": ["drawn_ratio"],
    "has_mask": ["has_mask"],
    "init": ["InitColor(本体のブロックの埋め値)", "マスクの向き(`InitColor`)"],
    "kind": ["InitColor(本体のブロックの埋め値)"],
    "rc_bucket": ["raster_cover"],
    "bound_class": ["被覆の値が、値か下限か"],
    "occluder_class": ["遮蔽物の層(`occluder_classes.csv`)"],
    "found_by": ["遮蔽物の層(`occluder_classes.csv`)"],
    "n_drawn_gt0": ["drawn_ratio", "被覆の値が、値か下限か"],
    "n_drawn_zero": ["drawn_ratio", "被覆の値が、値か下限か"],
    "n_png_files": ["納品の PNG の枚数", "納品の範囲"],
    "n_distinct_page_keys": ["納品の PNG の枚数"],
    "is_mask": ["納品の PNG の枚数", "層 PNG"],
    "n_png_names": ["納品をまたいだ同じ名前"],
    "content_match": ["納品をまたいだ同じ名前"],
    "n_common_names_bytes_differ": ["納品をまたいだ同じ名前"],
    "n_pixels_in_both_renamed": ["同じ画素が別の名前で在る数",
                                 "どちらの納品が現行か"],
    "n_pixels_in_both": ["同じ画素が別の名前で在る数"],
    "n_distinct_pixels_a": ["同じ画素が別の名前で在る数"],
    "n_distinct_pixels_b": ["同じ画素が別の名前で在る数"],
    "n_pixels_only_a": ["どちらの納品が現行か"],
    "n_pixels_only_b": ["どちらの納品が現行か"],
    "measured_by": ["この集計と供給元の集計が一致すること"],
    "canvas_w": ["canvas_w / canvas_h"],
    "canvas_h": ["canvas_w / canvas_h"],
    "dpi": ["canvas_w / canvas_h"],
    "canvas_unit_resolved": ["canvas_w / canvas_h"],
    "direction": ["visibility"],
    "sum_n_layers_incl_folders": ["層数(集計どうしの比較)"],
    "sum_n_folders": ["層数(集計どうしの比較)"],
}


def column_caveats():
    """対応表を、但し書きの本文を添えた形に解く。**引けない名前は落とさず叫ぶ。**"""
    by_col = {c["column"]: c for c in CAVEATS}
    out, missing = {}, []
    for col, names in COLUMN_CAVEATS.items():
        got = []
        for n in names:
            if n not in by_col:
                missing.append((col, n))
                continue
            c = by_col[n]
            got.append(dict(column=c["column"], not_answers=c["not_answers"],
                            trap=c["trap"]))
        if got:
            out[col] = got
    if missing:
        raise SystemExit(f"但し書きが引けない対応: {missing} —— "
                         "**対応表か `caveats.py` のどちらかが古い**")
    uncovered = sorted(set(by_col) - {n for ns in COLUMN_CAVEATS.values() for n in ns})
    return out, uncovered


def read_csv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="census")
    ap.add_argument("--out", default=os.path.join("census", "index.html"))
    args = ap.parse_args()
    as_of = bc.today()

    pages = read_csv(os.path.join(args.census, "pages_by_year.csv"))
    layers = read_csv(os.path.join(args.census, "layers_by_year_kind_type.csv"))
    pops = read_csv(os.path.join(args.census, "populations.csv"))
    prov = read_csv(os.path.join(args.census, "provenance.csv"))
    extra = {name: read_csv(os.path.join(args.census, name + ".csv"))
             for name in ("layer_type_values", "layer_type_bits", "mask_by_kind",
                          "visibility_exceptions", "body_initcolor",
                          "delivery_coverage", "delivery_duplicates",
                          "canvas_declared", "delivery_duplicates_by_content",
                          "duplicate_quantities", "cluster_sizes",
                          "grad_v1_v2_by_layer", "coverage_bound_classes",
                          "occluder_classes")}

    n_pages = sum(int(r["n_pages"]) for r in pages if r["n_pages"])
    n_layers = sum(int(r["n_layers"]) for r in layers if r["n_layers"])
    mask_fill = sum(int(r["n_layers"]) for r in layers
                    if r["v_layer_kind"] == "mask_fill" and r["n_layers"])

    col_cav, uncovered = column_caveats()
    # **空欄の理由は、器具の宣言から渡す。** 列名から推測させない ——
    # 推測した理由は、確かめた理由として読まれる(2026-10-26、T134 (h))
    empty_reasons = {
        "suppressed": {c: bc.SUPPRESSED[("drawn_ratio", "mask_fill")]
                       for c in ("n_drawn_gt0", "n_drawn_zero", "n_drawn_blank")},
        "by_population": {p: sorted(cols)
                          for p, cols in bc.POPULATION_METRICS.items()},
        # `ext_lip` は **0 が正しい**(その母集団に `.lip` の頁が無い)ので、
        # 空欄の札ではなく**列見出しの但し書き**で扱う。
        # 空欄でないものに空欄の理由を付けない
        "absent": {},
    }
    data = dict(
        column_caveats=col_cav,
        populations=pops, provenance=prov, pages=pages, layers=layers, **extra,
        caveats=[dict(column=c["column"], answers=c["answers"],
                      not_answers=c["not_answers"], trap=c["trap"],
                      by=c["by"], as_of=c["as_of"], detail=c["detail"])
                 for c in CAVEATS],
        empty_reasons=empty_reasons,
        reasons={
            "lip": "手元の材料に `.lip` が1頁も入っていない。"
                   "0 は「数えて 0 だった」ではない(ファイルは 1,986 在る)",
            "mask_fill": f"mask_fill {mask_fill:,}層({mask_fill:,} 層分)。"
                         "本体は InitColor 255 = 全面で、0 は空を意味しない。"
                         "抽出側が読み手を直すと値が動くので集計に入れない",
        },
        runline=[
            f"走行 {as_of}",
            f"段 A: {len(pages)} 行 × 頁 / {len(layers)} 行 × 層",
            f"集計した頁 {n_pages:,} / 層 {n_layers:,}",
            f"但し書き {len(CAVEATS)} 行",
            "固有名: 0",
        ],
        foot=f"走行 {as_of}。集計した頁 {n_pages:,}・層 {n_layers:,}(すべて .clip)。"
             f"但し書き {len(CAVEATS)} 行。段 A に固有名は 0。",
    )

    html = open(TEMPLATE, encoding="utf-8").read()
    assert "%%DATA%%" in html, "テンプレートに差し込み口が無い"
    payload = json.dumps(data, ensure_ascii=False)
    assert "</script" not in payload, "埋め込む JSON が script を閉じている"
    html = html.replace("%%DATA%%", payload)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)

    size = os.path.getsize(args.out)
    print(f"画面を書いた: {args.out}({size / 1024:.0f} KB)")
    print(f"  埋め込んだ: 母集団 {len(pops)} / 出所 {len(prov)} / "
          f"頁 {len(pages)} / 層 {len(layers)} / 但し書き {len(CAVEATS)}")
    for name, recs in extra.items():
        print(f"    {len(recs):4d} 行  {name}")
    print(f"  [列と但し書きの対応] {len(col_cav)} 列に対応あり")
    print(f"    **どの列にも結び付いていない但し書き {len(uncovered)} 行**"
          f"(表の列ではなく集計全体にかかるもの): " + " / ".join(uncovered))

    return audit_ui(args.out, payload)


def audit_ui(path: str, payload: str) -> int:
    """画面も公開物。**ただし走査先を2つに分ける。**

    **埋め込んだデータは語彙で見る** —— ここが固有名の漏れうる場所。
    **テンプレート(CSS と JS)は語彙で見ない** —— `color` と `cover` は
    実在のフォルダー名なので、CSS の語に当たって24行鳴る(2026-10-26 に実際そうなった)。
    代わりに**全文をパスの印で見る**。

    **届かないもの**: テンプレートの散文に手で書いた固有名(静的で、目で読んである)、
    2文字以下の名前、材料に出てこない名前。
    """
    _, p3, _ = bc.load_req3(bc.sources.by_key("req3_layer_kind")["path"])
    a33, _ = bc.load_added33(bc.sources.by_key("added33_structure")["path"])
    long_words, short_words = bc.private_vocabulary(p3, a33)
    matcher = bc.ProseMatcher(long_words)

    hits = []
    w = matcher.find(payload)
    if w is not None:
        i = payload.index(w)
        hits.append(("埋め込んだデータ", f"固有名「{len(w)}文字」",
                     payload[max(0, i - 40):i + 40]))
    whole = open(path, encoding="utf-8").read()
    for mark in ("F:\\", "\\manuscripts", "manuscripts\\"):
        if mark in whole:
            i = whole.index(mark)
            hits.append(("画面の全文", f"パスの印 {mark}", whole[max(0, i - 40):i + 40]))

    print(f"  [画面の点検] 埋め込んだデータを禁止語彙 {len(long_words)} 語で見る"
          f" / 全文はパスの印で見る")
    print(f"    **盲点: 2文字以下の名前 {len(short_words)} 語、"
          f"テンプレートの散文(静的・目で読んである)、CSS と JS の語**")
    print(f"  **指摘: {len(hits)}**")
    for h in hits:
        print(f"    {h[0]} —— {h[1]}: ...{h[2]}...")
    if hits:
        print("\n**指摘あり。この画面は commit しない**")
        return 1
    print("  指摘なし(埋め込んだデータに固有名は無い)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

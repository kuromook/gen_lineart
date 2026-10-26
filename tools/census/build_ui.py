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

    n_pages = sum(int(r["n_pages"]) for r in pages if r["n_pages"])
    n_layers = sum(int(r["n_layers"]) for r in layers if r["n_layers"])
    mask_fill = sum(int(r["n_layers"]) for r in layers
                    if r["v_layer_kind"] == "mask_fill" and r["n_layers"])

    data = dict(
        populations=pops, provenance=prov, pages=pages, layers=layers,
        caveats=[dict(column=c["column"], answers=c["answers"],
                      not_answers=c["not_answers"], trap=c["trap"],
                      by=c["by"], as_of=c["as_of"], detail=c["detail"])
                 for c in CAVEATS],
        reasons={
            "lip": "手元の材料に `.lip` が1頁も入っていない。"
                   "0 は「数えて 0 だった」ではない(ファイルは 1,986 在る)",
            "mask_fill": f"mask_fill {mask_fill:,}層。本体は InitColor 255 = 全面で、"
                         "0 は空を意味しない。抽出側が読み手を直すと値が動くので集計に入れない",
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
    _, p3, _ = bc.load_req3(bc.sources.LOCAL_CSV[0]["path"])
    a33, _ = bc.load_added33(bc.sources.LOCAL_CSV[1]["path"])
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

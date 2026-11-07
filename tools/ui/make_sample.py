#!/usr/bin/env python3
"""部品の煙試験用の見本を作る。**原稿は使わない**(合成した図形だけ)。出力は追跡しない場所へ。

    ./venv/bin/python tools/ui/make_sample.py <out_dir>
    ./venv/bin/python tools/ui/interview_server.py <out_dir>/questions.json
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for n in range(3):
        W, H = 2400, 1700  # 画面に収まらない大きさ(等倍の確認用)
        comp = Image.new("L", (W, H), 255)
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d, dl = ImageDraw.Draw(comp), ImageDraw.Draw(lay)
        for k in range(12):
            x = 150 + k * 180
            d.line([(x, 200), (x + 300, 1500)], fill=0, width=1 + (k % 3))
        dl.ellipse([700 + n * 200, 500, 1500 + n * 200, 1200], outline=(220, 30, 30, 255), width=3)
        d.ellipse([700 + n * 200, 500, 1500 + n * 200, 1200], outline=0, width=3)
        comp.save(out / f"s{n}_comp.png")
        lay.save(out / f"s{n}_layer.png")
        items.append({"id": f"s{n}", "version": "sample", "images": [
            {"src": f"s{n}_comp.png", "caption": "合成した絵"},
            {"src": f"s{n}_layer.png", "caption": "問う層だけ", "target": True}]})
    spec = {"interview_id": "ui_sample", "question": "印の付いた層の楕円は、線画ですか?", "unit": "層",
            "decides": "(見本)この層を線画として数えるかが決まる",
            "choices": [{"value": "line", "label": "線画", "help": "仕上げの線"},
                        {"value": "rough", "label": "下絵", "help": "下描き・あたり"}],
            "items": items}
    (out / "questions.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out / "questions.json")


if __name__ == "__main__":
    main()

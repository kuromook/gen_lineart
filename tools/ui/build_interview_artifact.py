#!/usr/bin/env python3
"""聞き取りの UI を claude.ai の artifact 用の1枚の HTML にする(絵は data URI で埋める)。

使い方:
    ./venv/bin/python tools/ui/build_interview_artifact.py <questions.json> -o <out.html> [--image-root <dir>]

- 出力は **追跡しない場所**に置く(原稿の画素を含む。Claude の scratchpad か track の `data/`)
- 公開は Artifact の publish に `capabilities: {"db": {}, "user": {}}` を付ける。答えは db の `answers` に1問1行で足される
- **artifact は使い捨て**(2026-11-07 ユーザー決定「聞き取り終わったらデータは残して削除」):
  1. ArtifactData の list で `answers` を全部読み、JSONL で track の `data/` に書く
  2. `tools/ui/verify_answers.py <questions.json> <answers.jsonl> --expect-rows <db の行数>` で一致を印字する
  3. ユーザーの確認を取ってから artifact を消す。URL・作った日・消した日を track の work_log に1行ずつ
"""
import argparse
import base64
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from interview_spec import load_questions, render_page  # noqa: E402

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
LIMIT = 15 * 1024 * 1024  # artifact の上限 16MB に余裕を残す


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--image-root")
    a = ap.parse_args()
    qpath = Path(a.questions).resolve()
    spec = load_questions(qpath)
    root = Path(a.image_root).resolve() if a.image_root else qpath.parent
    cache = {}
    for it in spec["items"]:
        for im in it["images"]:
            p = (root / im["src"]).resolve()
            if root not in p.parents or not p.is_file():
                sys.exit(f"絵が無いか、image-root の外: {im['src']}")
            if p.suffix.lower() not in MIME:
                sys.exit(f"埋められない形式: {p.suffix}")
            if p not in cache:
                cache[p] = f"data:{MIME[p.suffix.lower()]};base64," + base64.b64encode(p.read_bytes()).decode()
            im.setdefault("id", im["src"])
            im["src"] = cache[p]
    html = render_page(spec, mode="artifact")
    size = len(html.encode("utf-8"))
    if size > LIMIT:
        sys.exit(f"大きすぎる: {size/1e6:.1f}MB > {LIMIT/1e6:.0f}MB。問いを分けるか、絵を切り詰める(縮小はしない。規律6)")
    Path(a.out).write_text(html, encoding="utf-8")
    print(f"書いた: {a.out}({size/1e6:.2f}MB、問い {len(spec['items'])}、絵 {len(cache)})")
    print('公開: Artifact publish に capabilities {"db": {}, "user": {}} を付ける。答えは db の `answers` に入る')
    print("使い捨て: 聞き取りが終わったら、答えを移して verify_answers.py で突き合わせ、ユーザーの確認を取ってから消す")


if __name__ == "__main__":
    main()

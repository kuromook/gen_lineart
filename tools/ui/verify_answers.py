#!/usr/bin/env python3
"""答えの JSONL を問いと突き合わせて印字する。artifact を消す前の点検(行の数の一致)にも使う。

使い方:
    ./venv/bin/python tools/ui/verify_answers.py <questions.json> <answers.jsonl> [--expect-rows N]

印字: 行の数 / 答えのある問い・無い問い / 選び直した問い / 「どれでもない」「分からない」の数と一言の数 /
問いに無い item の行。`--expect-rows`(artifact の db で数えた行の数)と合わなければ終了コード1 —— **そのとき artifact を消さない**。
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from interview_spec import load_questions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("answers")
    ap.add_argument("--expect-rows", type=int)
    a = ap.parse_args()
    spec = load_questions(a.questions)
    rows = [json.loads(x) for x in Path(a.answers).read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r.get("interview_id") == spec["interview_id"]]
    by_item = defaultdict(list)
    for r in sorted(rows, key=lambda r: r.get("ts", "")):
        by_item[r.get("item_id")].append(r)
    stray = sorted(k for k in by_item if k not in spec["item_ids"])
    answered = [i for i in spec["item_ids"] if i in by_item]
    latest = {k: v[-1] for k, v in by_item.items() if k in spec["item_ids"]}
    vals = Counter(r["value"] for r in latest.values())
    ok = True
    print(f"聞き取り: {spec['interview_id']}")
    print(f"行: {len(rows)}(この interview_id の行だけ)")
    if a.expect_rows is not None:
        same = a.expect_rows == len(rows)
        ok &= same
        print(f"保存先で数えた行: {a.expect_rows} → **{'一致' if same else '不一致 —— 消さない'}**")
    print(f"答えのある問い: {len(answered)} / {len(spec['item_ids'])}(無い {len(spec['item_ids']) - len(answered)})")
    print(f"選び直した問い: {sum(1 for k, v in by_item.items() if len(v) > 1)}(前の行は残っている)")
    print(f"最新の答え: どれでもない {vals.get('__none__', 0)} / 分からない {vals.get('__unknown__', 0)} / "
          f"一言つき {sum(1 for r in latest.values() if r.get('comment'))}")
    if vals.get("__none__", 0):
        print("  → 「どれでもない」が来た問いは、問いの作りの欠陥として記録する(指針 §2-3)")
    if stray:
        ok = False
        print(f"**問いに無い item の行**: {len(stray)} 種 {stray[:5]}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

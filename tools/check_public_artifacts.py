#!/usr/bin/env python3
"""追跡されている画像と重みが許可表に在るかを見る。

CLAUDE.md §4 の器具。**木**を見る(差分ではない) —— 差分を見て「0件」と書くと、
読み手は木について書いたと読む(2026-10-26、基盤が自分で踏んだ)。

終了コード: 0 = 表と一致 / 1 = 表に無い追跡画像か追跡重みが在る / 2 = 器具自身が走れない
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = ROOT / "doc" / "public_artifacts_allowlist.txt"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".psd", ".clip"}
# 重みの公開はユーザーの承認が要る (2026-10-26)。表に無い追跡重みは終了コード1。
WEIGHT_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".bin", ".h5", ".pb"}
# **原稿の中身そのもの。** 追跡されていたら**無条件で失敗**させる —— 許可表は作らない
# (2026-10-28、構造 DB の控えを受け取ったときに足した。**作業木の外に置いてある**ので
#  ここに出ることは無いはずで、**出たら置き場を間違えている**)。
# 2026-10-23 に 99MB の DB を public repo に上げた事故がある。**force-push では消えない**
CORPUS_SUFFIXES = {".duckdb", ".db", ".sqlite", ".sqlite3", ".jsonl"}
BIG_FILE_MB = 5.0


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True, check=True).stdout


def main():
    if not ALLOWLIST.exists():
        print(f"許可表が無い: {ALLOWLIST}", file=sys.stderr)
        return 2

    allowed = set()
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            allowed.add(entry)

    tracked = [p for p in git("ls-files").splitlines() if p]
    images = sorted(p for p in tracked if Path(p).suffix.lower() in IMAGE_SUFFIXES)
    weights = sorted(p for p in tracked if Path(p).suffix.lower() in WEIGHT_SUFFIXES)
    watched = images + weights

    print(f"追跡ファイル {len(tracked)} / うち画像 {len(images)} / "
          f"重み {len(weights)} / 許可表 {len(allowed)} 行")

    unlisted = [p for p in watched if p not in allowed]
    stale = sorted(allowed - set(watched))

    # 落としたものを理由別に (規律9)。種類ごとに印字する —— 空でも 0 を出す (T134 l)
    for label, group in (("画像", images), ("重み", weights)):
        bad = [p for p in group if p not in allowed]
        print(f"  {label}: 許可表に在る {len(group) - len(bad)} / **無い {len(bad)}**")
    print(f"  表に在るが追跡されていない(消えた行): {len(stale)}")
    for p in stale:
        print(f"      {p}")

    # 木の大きいもの —— 画像でなくても公開されている。報告のみ、失敗させない
    print(f"\n木に在る {BIG_FILE_MB:g}MB 超の追跡ファイル(画像以外も。**差分では出ない**):")
    big = []
    for line in git("ls-tree", "-r", "-l", "HEAD").splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5 and parts[3].isdigit():
            mb = int(parts[3]) / 1048576
            if mb >= BIG_FILE_MB:
                big.append((mb, parts[4]))
    for mb, path in sorted(big, reverse=True):
        print(f"      {mb:8.1f} MB  {path}")
    print(f"  計 {len(big)} 件 / {sum(mb for mb, _ in big):.1f} MB")

    # **原稿の中身は許可表を持たない** —— 在ったら失敗。**承認では解除しない**
    corpus = sorted(p for p in tracked
                    if Path(p).suffix.lower() in CORPUS_SUFFIXES)
    print(f"  原稿の中身(DB・JSONL。**追跡されていてはいけない**): {len(corpus)}")
    for p in corpus:
        print(f"      {p}")

    if corpus:
        print("\n**原稿の中身が追跡されている。** 置き場を作業木の外に移すこと "
              "(`doc/db_snapshot.md`)—— **許可表では解除しない**", file=sys.stderr)
        return 1
    if unlisted:
        print("\n表に無い追跡画像・追跡重み(公開にはユーザーの承認が要る):", file=sys.stderr)
        for p in unlisted:
            print(f"      {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

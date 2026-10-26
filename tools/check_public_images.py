#!/usr/bin/env python3
"""追跡されている画像が許可表に在るかを見る。

CLAUDE.md §4 の器具。**木**を見る(差分ではない) —— 差分を見て「0件」と書くと、
読み手は木について書いたと読む(2026-10-26、基盤が自分で踏んだ)。

終了コード: 0 = 表と一致 / 1 = 表に無い追跡画像が在る / 2 = 器具自身が走れない
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = ROOT / "doc" / "public_images_allowlist.txt"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".psd", ".clip"}
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

    print(f"追跡ファイル {len(tracked)} / うち画像 {len(images)} / 許可表 {len(allowed)} 行")

    unlisted = [p for p in images if p not in allowed]
    stale = sorted(allowed - set(images))

    # 落としたものを理由別に (規律9)
    print(f"  許可表に在る: {len(images) - len(unlisted)}")
    print(f"  **許可表に無い: {len(unlisted)}**")
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

    if unlisted:
        print("\n表に無い追跡画像:", file=sys.stderr)
        for p in unlisted:
            print(f"      {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

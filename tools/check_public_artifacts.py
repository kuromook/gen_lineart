#!/usr/bin/env python3
"""追跡されている画像と重みが許可表に在るかを見る。

CLAUDE.md §4 の器具。**木**を見る(差分ではない) —— 差分を見て「0件」と書くと、
読み手は木について書いたと読む(2026-10-26、基盤が自分で踏んだ)。

そして **どの木か** を言う。`git ls-files` はいま座っている作業木の index だけで、
**公開されているのは origin の全 branch** である(2026-10-26、`foundation` の門が
通っているのに他の3 branch に表に無い画像15件が公開されていた)。既定で
**手元の全 branch と origin の全 branch** を見る。

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
BIG_FILE_MB = 5.0
# **原稿の中身そのもの。追跡されていたら無条件で失敗させる —— 許可表では解除しない**
# (表は「選んだものの表」で、原稿の中身は選ばない)。2026-10-23 に 99MB の DB を public repo
# に上げた事故の規則。`census` が 2026-10-28 に自分の木の器具へ足し、2026-10-30 に正本へ採った。
# **定数は役割ごとに引き直した(規律9)**: census の版は `.jsonl` も無条件で落とすが、
# **作者の判断を記録した `labels/*.jsonl` が3 branch で 13 件、意図して追跡されている**
# (最大 243KB、2026-10-30 に実測)。借りた定数をそのまま入れると材料を一種類まるごと黙って
# 消すので、**DB は無条件、`.jsonl` は 1MB 超だけ**を落とす。
CORPUS_SUFFIXES = {".duckdb", ".db", ".sqlite", ".sqlite3"}
CORPUS_BY_SIZE_SUFFIXES = {".jsonl"}
CORPUS_SIZE_MB = 1.0


def corpus_in(ref):
    """その木で追跡されている原稿の中身。`(道, MB, 理由)` を返す。"""
    hits = []
    if ref is None:  # index: 大きさは作業木の実体から
        for path in git("ls-files").splitlines():
            suffix = Path(path).suffix.lower()
            if suffix not in CORPUS_SUFFIXES | CORPUS_BY_SIZE_SUFFIXES:
                continue
            f = ROOT / path
            mb = (f.stat().st_size / 1048576) if f.exists() else 0.0
            if suffix in CORPUS_SUFFIXES:
                hits.append((path, mb, "DB"))
            elif mb >= CORPUS_SIZE_MB:
                hits.append((path, mb, f"{CORPUS_SIZE_MB:g}MB 超の jsonl"))
        return hits
    for line in git("ls-tree", "-r", "-l", ref, check=False).splitlines():
        parts = line.split(None, 4)
        if len(parts) != 5 or not parts[3].isdigit():
            continue
        path, mb = parts[4], int(parts[3]) / 1048576
        suffix = Path(path).suffix.lower()
        if suffix in CORPUS_SUFFIXES:
            hits.append((path, mb, "DB"))
        elif suffix in CORPUS_BY_SIZE_SUFFIXES and mb >= CORPUS_SIZE_MB:
            hits.append((path, mb, f"{CORPUS_SIZE_MB:g}MB 超の jsonl"))
    return hits


def git(*args, check=True):
    r = subprocess.run(["git", "-C", str(ROOT), *args],
                       capture_output=True, text=True, check=check)
    return r.stdout


def watched_in(listing):
    """監視対象の拡張子だけを返す。"""
    out = []
    for path in listing.splitlines():
        if path and Path(path).suffix.lower() in IMAGE_SUFFIXES | WEIGHT_SUFFIXES:
            out.append(path)
    return out


def refs_to_scan():
    """見る木の一覧。いまの index + 手元の全 branch + origin の全 branch。"""
    refs = [("index", None)]
    for line in git("branch", "--format=%(refname:short)").splitlines():
        if line.strip():
            refs.append((line.strip(), line.strip()))
    for line in git("branch", "-r", "--format=%(refname:short)").splitlines():
        name = line.strip()
        if name and "->" not in name:
            refs.append((name, name))
    return refs


def main():
    if not ALLOWLIST.exists():
        print(f"許可表が無い: {ALLOWLIST}", file=sys.stderr)
        return 2

    allowed = set()
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            allowed.add(entry)

    refs = refs_to_scan()
    print(f"許可表 {len(allowed)} 行 / 見る木 {len(refs)}"
          f"(index + 手元の branch + origin の branch)\n")

    per_ref = {}
    seen = set()
    for label, ref in refs:
        listing = git("ls-files") if ref is None else git("ls-tree", "-r", "--name-only", ref, check=False)
        found = sorted(watched_in(listing))
        seen.update(found)
        per_ref[label] = [p for p in found if p not in allowed]

    unlisted = sorted({p for bad in per_ref.values() for p in bad})
    stale = sorted(allowed - seen)

    # 落としたものを理由別に (規律9)。空でも 0 を印字する (T134 l)
    for label, _ in refs:
        bad = per_ref[label]
        mark = f"**表に無い {len(bad)}**" if bad else "表に無い 0"
        print(f"  {label:34s} {mark}")
        for p in bad:
            print(f"        {p}")
    print(f"\n  表に在るが、どの木にも無い(消えた行): {len(stale)}")
    for p in stale:
        print(f"      {p}")
    n_img = sum(1 for p in unlisted if Path(p).suffix.lower() in IMAGE_SUFFIXES)
    print(f"  **表に無い監視対象 計 {len(unlisted)} 件(画像 {n_img} / "
          f"重み {len(unlisted) - n_img})**")

    # 木の大きいもの —— 画像でなくても公開されている。報告のみ、失敗させない
    print(f"\nいまの HEAD に在る {BIG_FILE_MB:g}MB 超の追跡ファイル(画像以外も。**差分では出ない**):")
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

    # 原稿の中身。**どの木でも**見る —— 公開されているのは origin の全 branch である
    corpus = {}
    for label, ref in refs:
        for path, mb, why in corpus_in(ref):
            corpus.setdefault((path, why), []).append(label)
    print(f"\n原稿の中身(DB は無条件 / jsonl は {CORPUS_SIZE_MB:g}MB 超。"
          f"**許可表では解除しない**): {len(corpus)} 件")
    for (path, why), labels in sorted(corpus.items()):
        print(f"      {path}  [{why}]  {len(labels)} 木: {', '.join(labels[:4])}")

    if corpus:
        print("\n**原稿の中身が追跡されている。** 置き場を作業木の外に移すこと "
              "—— **許可表では解除しない**", file=sys.stderr)
        return 1
    if unlisted:
        print("\n表に無い追跡画像・追跡重み(公開にはユーザーの承認が要る):", file=sys.stderr)
        for p in unlisted:
            print(f"      {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""追跡されている画像と重みが許可表に在るかを見る。

CLAUDE.md §4 の器具。**木**を見る(差分ではない) —— 差分を見て「0件」と書くと、
読み手は木について書いたと読む(2026-10-26、基盤が自分で踏んだ)。

そして **どの木か** を言う。`git ls-files` はいま座っている作業木の index だけで、
**公開されているのは origin の全 branch** である(2026-10-26、`foundation` の門が
通っているのに他の3 branch に表に無い画像15件が公開されていた)。既定で
**手元の全 branch と origin の全 branch** を見る。

**表は正本と手元の和を取る**(2026-10-31、Track N の便とユーザーの指示)。
2026-10-31 14:15 に Track M が承認済みの画像5枚を追跡に入れ、基盤が **14:31 に正本の表へ
採った**。それでも **14:40 に Track N の無関係な commit が落ちた** —— 見る範囲は全54木なのに、
表は「いま座っている木の複製」1つしか読んでいなかったからである。**承認は正本に在ったのに、
9分後の別の木がそれを使えなかった。** 1件の承認を16回書かせるのは配り方の欠陥で、
規則の強さの話ではない(T161)。いまは **正本(基盤の木の `foundation`、無ければ
`origin/foundation`)+ この木の表**の和で判定し、**どちらから読んだかと件数を必ず印字する**。
**和を取っても規則は緩まない** —— 足せるのはユーザーの承認が記録された行だけで、
**この木にだけ在る行は「正本へ上げる」として数えて印字する**。

**落ちたら、誰の何かを言う**(同じ便の (イ)(ウ))。各ファイルについて **入れた commit /
在る木の数 / 既に `origin` に在るか**を出す。**`origin` に在るものは既に匿名で読める** ——
止めても公開は取り消されないので、文を変える。**止める効果は圧力であって保護ではない。**

終了コード: 0 = 表と一致 / 1 = 表に無い追跡画像か追跡重みが在る / 2 = 器具自身が走れない
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANON_PATH = "doc/public_artifacts_allowlist.txt"
ALLOWLIST = ROOT / CANON_PATH
# **正本の木。** 承認は1回で足り、配るのは機械である(T161)。読めなければ手元の表に落ちる
FOUNDATION_TREE = Path("/home/sh1/deepl/lineart")
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


def parse_table(text):
    """許可表の行。`#` 以降は注。"""
    out = set()
    for line in text.splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            out.add(entry)
    return out


def canonical_table():
    """正本の表と、どこから読んだか。読めなければ `(None, 理由)`。"""
    r = subprocess.run(["git", "-C", str(FOUNDATION_TREE), "show", f"foundation:{CANON_PATH}"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        return parse_table(r.stdout), f"基盤の木の `foundation`({FOUNDATION_TREE})"
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"origin/foundation:{CANON_PATH}"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        return parse_table(r.stdout), "この木の `origin/foundation`(基盤の木が読めなかった)"
    return None, "**正本が読めない。この木の表だけで判定した** —— 件数は信じる前に引き直す"


def provenance(path, labels):
    """`(既に公開されている木, 入れた commit の1行)`。**誰の何かを言う**(T161 の (イ))。"""
    public = [l for l in labels if l.startswith("origin/")]
    r = subprocess.run(["git", "-C", str(ROOT), "log", "--all", "--diff-filter=A", "-1",
                        "--format=%h %cd %s", "--date=format:%m-%d %H:%M", "--", path],
                       capture_output=True, text=True)
    line = r.stdout.strip().splitlines()
    return public, (line[0][:96] if line else "(入れた commit が辿れない)")


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
    # **この木に表が無くても、正本が読めるなら点検できる**(別クローン2本に表が無い。
    # 2026-10-31)。どちらも無いときだけ「点検していない」と言って 2 を返す
    local = parse_table(ALLOWLIST.read_text(encoding="utf-8")) if ALLOWLIST.exists() else set()
    canon, where = canonical_table()
    if not ALLOWLIST.exists():
        if canon is None:
            print(f"許可表がこの木に無く({ALLOWLIST})、正本も読めない。**点検していない**",
                  file=sys.stderr)
            return 2
        print(f"**この木に許可表が無い**({CANON_PATH})。正本だけで判定する —— {where}")
    only_local = []
    if canon is None:
        allowed = set(local)
        print(f"許可表: この木 {len(local)} 行だけ。{where}")
    else:
        allowed = local | canon
        only_local = sorted(local - canon)
        print(f"許可表: 和 {len(allowed)} 行 —— 正本 {len(canon)} 行({where})"
              f" + この木 {len(local)} 行 / 正本だけに在る {len(canon - local)} 件"
              f" / この木だけに在る {len(only_local)} 件")
        if only_local:
            print("  **この木だけに在る承認。正本の著者は基盤なので上げる**(T161):")
            for path in only_local:
                print(f"      {path}")

    refs = refs_to_scan()
    print(f"見る木 {len(refs)}(index + 手元の branch + origin の branch)\n")

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
              "(控えの書き方は `census` の `doc/db_snapshot.md` —— 数・日時・sha256 だけで、"
              "原稿の固有名を書かない)—— **許可表では解除しない**", file=sys.stderr)
        return 1
    if unlisted:
        print("\n表に無い追跡画像・追跡重み(公開にはユーザーの承認が要る):", file=sys.stderr)
        where_from = {}
        for label, bad in per_ref.items():
            for path in bad:
                where_from.setdefault(path, []).append(label)
        for p in unlisted:
            labels = where_from.get(p, [])
            public, first = provenance(p, labels)
            tail = " …" if len(labels) > 4 else ""
            print(f"      {p}", file=sys.stderr)
            print(f"        入れた commit: {first}", file=sys.stderr)
            print(f"        在る木 {len(labels)}: {', '.join(labels[:4])}{tail}", file=sys.stderr)
            if public:
                print("        **既に origin に在る = 匿名で読める。止めても公開は取り消されない**"
                      " —— 表に記録するか、取り下げるかを決める", file=sys.stderr)
            else:
                print("        まだ origin に無い。**ここで止めれば公開は防げる**", file=sys.stderr)
        print("\n  **承認はユーザーのもの。Claude は表に自分で足さない。**", file=sys.stderr)
        print(f"  承認が出たら、基盤の木の `{CANON_PATH}` に1行足す —— "
              "**和を取るので各木に写す必要は無い**(T161)", file=sys.stderr)
        print("  取り下げるなら、入れた木で `git rm --cached <道>` と ignore —— "
              "**origin に在る分はそれでは消えない**", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

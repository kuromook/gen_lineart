#!/usr/bin/env python3
"""**止まっていた木を追いつかせる。** 法に属するファイルだけを基盤の版と突き合わせる。

なぜ要るか: 法を `.claude/settings.json` の追跡で配ると(置き場の案 (a))、
**rebase も merge もしていない木には届かない**。2026-10-30 の実測で
`lineart-face-words` と `lineart-panel-generation` は `origin/foundation` から
**346 commit 遅れ**、各木の `CLAUDE.md` は 4,288〜6,649 字とばらばら
(**同じファイルの別版が 5 つ在る**。T92)。**休止した track が復活するときに
最初に回すのがこれである。**

**T84 の教訓を組み込んである: 直し方は文字列ではなく点検である。** 基盤の ref は
**木ごとに解決する** —— 手元の `foundation`、無ければ `origin/foundation`(何 commit
遅れかを印字)、どちらも無ければ「基盤の木から直接写せ」と言う。2026-10-25 に audit の
助言が `git checkout foundation -- .gitignore` と言い、別クローンの2本にその ref が
無くて、Track H が善意で古い ref に揃えた事故がこれである。

使い方:
  catch_up.py                 # いま居る木を見る
  catch_up.py <木への道> ...   # 他の木を見る(読み取りのみ。基盤からの一覧用)

**状態は3つある。2つで書くと事故になる。** 2026-10-30、merge のリハーサルで
この器具の2番目の版が **merge 済みの木を「遅れている」と言い、`.gitignore` を元に戻す
命令を印字した** —— その木は遅れておらず先にいた。回したら仕事が消える。
3番目の版はそれを直したが、**今度は分岐した木(自分の commit を持ち、かつ法の commit を
持っていない)を「先にいる」と誤判定した**。木の大半はこの状態になる。

  - **遅れている**(HEAD が法の commit に含まれる)→ `git checkout <sha> -- <道>` を出す
  - **先にいる**(HEAD が法の commit を含む)→ 違いはこの木の仕事。**命令は出さない**
  - **分岐している**(どちらでもない。**木の大半はこれ**)→ 違いがこの木の変更か古い写しか、
    この器具では区別できない。**法のファイルは基盤が著者なので、普通は写しが古い** ——
    だから per-file の命令を出す(基盤の audit もそう薦めている)。
    **例外は、その木が法を起草している場合**(`ops-review` がそれ)。両方印字し、
    **見分け方を書く** —— 選ぶのは座っている者

終了コード: 0 = 追いついている(先にいる場合も) / 1 = 遅れている物が在る / 2 = 器具が走れない
**何も書き換えない。** 印字した命令を回すのは、その木に座っている者。
"""
import re
import subprocess
import sys
from pathlib import Path

# 法に属するファイル。**正本は基盤で、各木はその写しを持つだけ**
LAW_FILES = [
    "CLAUDE.md",
    ".gitignore",
    "doc/ordinance_spec.md",
    "doc/public_artifacts_allowlist.txt",
    "tools/check_public_artifacts.py",
    "tools/audit_track_ledger.py",
    "tools/catch_up.py",
    "tools/hooks/guard_forced_add.py",
    "tools/hooks/guard_public_artifacts.py",
    "tools/hooks/session_brief.py",
    "tools/hooks/pre-commit",
    "tools/hooks/settings.hooks.json",
    ".claude/settings.json",   # 置き場の案 (a) を採った後に意味を持つ
]
FOUNDATION_TREE = Path("/home/sh1/deepl/lineart")


def git(tree, *args):
    r = subprocess.run(["git", "-C", str(tree), *args],
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def truth():
    """**基盤の木の `foundation` が指す commit。** 法の現在の版はこれ1つ。"""
    code, sha, _ = git(FOUNDATION_TREE, "rev-parse", "foundation^{commit}")
    return sha if code == 0 else None


def resolve_ref(tree, want):
    """この木で使える基盤の ref。**比べる相手は現在の commit そのもの**(T84)。

    `origin/foundation` のような**名前**で比べてはいけない —— 2026-10-30 にこの器具の
    最初の版がそれで `lineart-face-words` の `.gitignore` を「同じ」と報告した。
    その木の `origin/foundation` は **346 commit 遅れ**で、`data/` 規則が入っていない版と
    一致していただけである。**T84 をこの器具が再生産していた。**
    """
    if want is None:
        return None, "**基盤の木の `foundation` が読めない。** 比較できない"
    code, _, _ = git(tree, "cat-file", "-e", f"{want}^{{commit}}")
    if code == 0:
        return want, ""
    return None, ("**法の現在の commit がこの木に無い**(`%s`)。"
                  "**fetch するまで比較は無意味** —— 名前で比べると古い版と一致して"
                  "「同じ」と出る(T84)。先に `git fetch origin foundation`、"
                  "それでも入らなければ基盤の木から直接写す" % want[:10])


def standing(tree, want):
    """"behind" / "ahead" / "diverged" のどれか。**2値で書くと事故になる。**"""
    code, _, _ = git(tree, "merge-base", "--is-ancestor", want, "HEAD")
    if code == 0:
        return "ahead"
    code, _, _ = git(tree, "merge-base", "--is-ancestor", "HEAD", want)
    return "behind" if code == 0 else "diverged"


def blob(tree, ref, path):
    code, out, _ = git(tree, "rev-parse", f"{ref}:{path}")
    return out if code == 0 else None


def local_blob(tree, path):
    p = Path(tree) / path
    if not p.exists():
        return None
    code, out, _ = git(tree, "hash-object", "--", str(p))
    return out if code == 0 else None


def check(tree, want):
    tree = Path(tree).resolve()
    code, br, _ = git(tree, "branch", "--show-current")
    print(f"\n── {tree.name}  枝 `{br or '(detached)'}`")

    ref, note = resolve_ref(tree, want)
    if ref is None:
        print(f"  {note}")
        print(f"  **一件ずつの比較は出さない。** 出すと「同じ」が嘘になる")
        return 1
    how = standing(tree, ref)
    label = {"ahead": "**この木はそれを含んでいる(遅れていない)**",
             "behind": "**この木は遅れている**",
             "diverged": "**分岐している** —— 自分の commit を持ち、法の commit を持っていない"}[how]
    print(f"  法の現在の版: `{ref[:10]}`(基盤の木の `foundation`)—— {label}")

    findings = []
    for path in LAW_FILES:
        here = local_blob(tree, path)
        there = blob(tree, ref, path) if ref else None
        if ref and there is None:
            state = "基盤に無い" if here is None else "この木だけに在る"
            if here is None:
                continue          # どちらにも無い = まだ存在しない法。黙る
        elif here is None:
            state = "この木で消してある" if how == "ahead" else "**無い**"
            if how != "ahead":
                findings.append(path)
        elif there is None:
            state = "基盤の版が読めない"
        elif here == there:
            state = "同じ"
        elif how == "ahead":
            # 法を含んでいる木で中身が違う = **この木の仕事**。戻させてはいけない
            state = "この木で変えてある"
        elif how == "diverged":
            state = "**分岐**"
            findings.append(path)
        else:
            state = "**違う**"
            findings.append(path)
        print(f"    {state:<18} {path}")

    f = tree / "doc" / "initial_notice.md"
    if f.exists():
        # **行頭の見出しだけ。** 引き継ぎの中の「`## 条例` を読む」という案内に当たると、
        # 節が無い木を「在る」と報告する(T160。注入の器具が同じ形で壊れていた)
        has = re.search(r"^## 条例\s*$", f.read_text(encoding="utf-8"), re.M) is not None
        print(f"    {'同じ' if has else '**無い**':<18} doc/initial_notice.md § 条例"
              f"{'' if has else '  ← 型は doc/ordinance_spec.md'}")
        if not has:
            findings.append("doc/initial_notice.md § 条例")

    if not findings:
        print("  **追いついている。**"
              + ("**この木で変えた物は基盤へ上げる側**(`outbox` 経由)。"
                 if how == "ahead" else ""))
        return 0

    if how == "diverged":
        print(f"  **{len(findings)} 件が食い違っている。** この木の変更か古い写しか、"
              f"この器具では区別できない。**見分け方は1つ: この木は法を起草しているか。**")
        print(f"  **していない(track は普通こちら)** —— 法の著者は基盤なので写しが古い:")
        for path in findings:
            if path.endswith("§ 条例"):
                print(f"    # `doc/initial_notice.md` に `## 条例` 節を作る(型: doc/ordinance_spec.md)")
            else:
                print(f"    git checkout {ref} -- {path}")
        print(f"  **している(`ops-review` がそれ)** —— 上を回すと自分の仕事が消える。代わりに:")
        print(f"    git merge {ref}        # 法を取り込み、衝突だけ手で読む")
        return 1

    print(f"  **{len(findings)} 件遅れている。** 回す命令(この木で):")
    for path in findings:
        if path.endswith("§ 条例"):
            print(f"    # `doc/initial_notice.md` に `## 条例` 節を作る(型: doc/ordinance_spec.md)")
        elif ref:
            print(f"    git checkout {ref} -- {path}")
        else:
            print(f"    cp {FOUNDATION_TREE / path} {tree / path}")
    return 1


def default_tree():
    """引数が無いときに見る木。**いま居る木であって、この器具が在る木ではない。**

    2026-10-30 に基盤が踏んだ: track の木には `venv` も `tools/catch_up.py` も無いことが
    在るので、**基盤の器具を絶対パスで呼ぶ**のが普通の使い方になる。そのとき
    `__file__` の親を見ると**基盤自身を点検して「追いついている」と答える** ——
    呼んだ人は自分の木について聞いたと読む。器具の docstring は最初から
    「いま居る木を見る」と書いていたので、**言っていたことに合わせた。**
    (`names_are_not_checks` の形。点検が別の対象を見ていても、返事は信じられる形で返る。)
    """
    code, top, _ = git(Path.cwd(), "rev-parse", "--show-toplevel")
    return Path(top.strip()) if code == 0 and top.strip() else Path.cwd()


def main():
    trees = sys.argv[1:] or [default_tree()]
    want = truth()
    worst = 0
    for t in trees:
        if not (Path(t) / ".git").exists() and not Path(t).is_dir():
            print(f"{t}: 木が無い", file=sys.stderr)
            worst = max(worst, 2)
            continue
        worst = max(worst, check(t, want))
    return worst


if __name__ == "__main__":
    sys.exit(main())

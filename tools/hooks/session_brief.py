#!/usr/bin/env python3
"""SessionStart。**立場と点検の結果を、思い出す前に渡す。**

担保するもの:

  - **CLAUDE.md §0(立場の確定)** —— どの枝に座っているかと、どの節が自分の役割かを先に書く。
    いま `git branch --show-current` を Claude が自分で打つ形で、打たなければ何も起きない
  - **規律4(台帳を開いたら audit を回す)** —— `foundation` のときだけ回す。
    **track の木で回すとその track を基盤と見るので無意味**(器具は自分の親を基盤と見る)
  - **許可表の門** —— 件数だけ。詳細は `tools/check_public_artifacts.py` を自分で回す
  - **法の鮮度**(track のみ)—— `tools/catch_up.py` を読み取りで回し、**基盤の版と食い違う
    法のファイルの件数**を言う。これが「法整備をまわす」の実体である: 配るのは基盤だが、
    **取り込みはその木に座っている者しかできない**ので、**座った時に件数が目に入る**形にする。
    **命令は印字しない**(14 行の `git checkout` を毎回貼るのは注意力の無駄で、器具を回せば出る)
  - **条例**(track のみ)—— `doc/initial_notice.md` の `## 条例` 節**だけ**を注入する。
    これが唯一の格上げ経路である: 条例は「基本法 §0 が読めと命じている文書」の層に在り、
    skill にすると「モデルが関係ありそうだと判断したら読む」層に**下がる**。
    注入すれば基本法と同じ**常に在る**層に上がる。**節だけを採るので引き継ぎ全体は入らない**
    (実測 2026-10-30: 条例 231〜1,924 字 対 引き継ぎ全体 4,185〜15,420 字)

**止めない。** SessionStart は additionalContext を返すだけで、通さない道が無い。
数は**この起動のもの**で、引用する前に引き直すこと(`names_are_not_checks`)。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FOUNDATION = "foundation"


def run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, cwd=str(ROOT), **kw)


def branch():
    r = run(["git", "branch", "--show-current"])
    return r.stdout.strip() or "(detached)"


# 条例の予算。**超えたら切らずに警告する** —— 法を黙って落とすのは規律9 が禁じている形
# (選別は測定である)。収まらない条例は起草の欠陥で、実行時の条件ではない。
ORDINANCE_BUDGET = 2000
HEADING = "## 条例"


def ordinance():
    """`doc/initial_notice.md` の `## 条例` 節だけを返す。"""
    f = ROOT / "doc" / "initial_notice.md"
    if not f.exists():
        return ["**引き継ぎ書が無い**(`doc/initial_notice.md`)—— 基盤に問う。"]
    text = f.read_text(encoding="utf-8")
    if HEADING not in text:
        return [f"**この track に `{HEADING}` 節が無い。** 条例は引き継ぎの中に散っている可能性が"
                f"ある。型は基盤の `doc/ordinance_spec.md`。**無いことを無いと言う** —— "
                "節が無いのは「規則が無い」ではない。"]
    body = text.split(HEADING, 1)[1]
    body = body.split("\n## ", 1)[0].strip()
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S).strip()  # 編集者向けの注は入れない
    n = len(body)
    head = f"**条例**({n} 字"
    head += f" / 予算 {ORDINANCE_BUDGET} —— **超えている。短くせよ**)" if n > ORDINANCE_BUDGET \
        else ")"
    return [head, body]


def law_freshness(py):
    """基盤の版と食い違う法のファイルの件数。**器具が無ければ「点検していない」と言う。**

    `catch_up.py` は何も書き換えず、終了コードは 0(追いついている/先にいる)/
    1(遅れている)/ 2(走れない)。**「分岐」は 0 で返るが食い違いは在る**ので、
    件数は出力から読む —— 終了コードだけ見ると、木の大半が占める状態を見落とす。
    """
    tool = ROOT / "tools" / "catch_up.py"
    if not tool.exists():
        return ["**法の鮮度を点検していない**(この木に `tools/catch_up.py` が無い)。"
                "基盤の木から `../lineart/tools/catch_up.py` を読み取りで回せる。"]
    r = run(py + [str(tool)])
    if r.returncode == 2:
        return [f"**法の鮮度の点検が走れなかった(終了コード {r.returncode})** —— 一致は不明。"]
    line = next((x for x in r.stdout.splitlines() if "食い違っている" in x), "")
    n = re.search(r"(\d+) 件が食い違っている", line)
    if not n:
        return ["**法**: 基盤の版に追いついている。"]
    return [f"**法のファイルが基盤の版と {n.group(1)} 件食い違っている。** 取り込みはこの木に"
            "座っている者しかできない —— `./venv/bin/python ../lineart/tools/catch_up.py` が "
            "per-file の命令を印字する(読み取りのみ。**何も書き換えない**)。"]


def main():
    sys.stdin.read()
    br = branch()
    py = [sys.executable, "-I"]
    out = [f"**この木**: `{ROOT.name}` / **枝**: `{br}`"]

    if br == FOUNDATION:
        out.append("立場: **共通基盤**。記録と回送で、実験はしない(規律5)。")
        r = run(py + [str(ROOT / "tools" / "audit_track_ledger.py")])
        if r.returncode == 1:
            lines = [x for x in r.stdout.splitlines() if x.strip().startswith("- `")]
            n = next((x for x in r.stdout.splitlines() if "finding(s)" in x), "").strip()
            out.append(f"**規律4 の audit を回した: {n or '指摘あり'}**(ハーネスが回したので、"
                       "思い出す必要は無い)。先頭3件:")
            out += ["  " + x.strip() for x in lines[:3]]
            out.append("  全文は `tools/audit_track_ledger.py` を回す。")
        elif r.returncode == 0:
            out.append("**規律4 の audit: 指摘なし**(台帳と実体が一致)。")
        else:
            out.append(f"**規律4 の audit が走れなかった(終了コード {r.returncode})** —— "
                       "**点検していない**。一致は不明。")
    else:
        notice = ROOT / "doc" / "initial_notice.md"
        out.append(f"立場: **track**。"
                   f"{'`doc/initial_notice.md` と その `## 条例` から始める' if notice.exists() else '引き継ぎ書が無い —— 基盤に問う'}。"
                   "**`CLAUDE.md` §1 と §5 は基盤の役割。§2〜§4 と §6 は自分にも効く。**")
        out.append("手元の `CLAUDE.md` と `doc/CURRENT.md` は **branch 時点で凍った複製**。"
                   "通達と `../lineart/doc/CURRENT.md` が正本。")

        out += law_freshness(py)
        out += ordinance()

    g = run(py + [str(ROOT / "tools" / "check_public_artifacts.py")])
    if g.returncode == 1:
        n = next((x for x in g.stdout.splitlines() if "表に無い監視対象 計" in x), "").strip()
        out.append(f"**許可表の門: {n or '表に無いものが在る'}** —— 公開はユーザーの承認。"
                   "Claude は表に自分で足さない。")
    elif g.returncode != 0:
        out.append(f"**許可表の門が走れなかった(終了コード {g.returncode})** —— 点検していない。")

    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": "\n".join(out),
    }}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

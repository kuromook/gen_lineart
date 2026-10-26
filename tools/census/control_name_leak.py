#!/usr/bin/env python
"""負の対照: **`audit_public` が本当に鳴るか。**

点検の名前は存在を含意するが、含意は点検ではない —— **沈黙が正常動作の点検が最も危険**。
だからこの器具は、**既知の漏れを作って、点検が 1 を返すこと**を確かめる。

名前付きの誤りと、その期待する結果(moat-controls の形):

| 対照 | 仕込む誤り | 期待 |
|---|---|---|
| C1 | **散文の列**に作品名を1つ書く | 1、理由「散文に固有名」 |
| C2 | **散文の列**に完全パスを1つ書く | 1、理由「パス区切りを含む」 |
| C3 | **次元の列**に作品名を1つ書く | 1、理由「宣言外の値」 |
| C4 | **件数の列**に名前を1つ書く | 1、理由「数字でない値」 |
| C5 | 何も仕込まない(素の出力) | **0** |

**C5 だけが通って C1〜C4 が鳴らないなら、点検は何も見ていない。**
**C5 が鳴るなら、点検は偽陽性を出していて誰も読まなくなる**(2026-10-26 に実際そうなった:
`1` `2` `a` `x` `work` `tmp` という名のフォルダーが実在し、全部の件数が鳴った)。

    ./venv/bin/python tools/census/control_name_leak.py <公開した census ディレクトリ>
"""
import csv
import io
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402


def run_audit_quiet(outdir, vocab):
    """**戻り値の構造を見る。** 印字を読むと、印字を切った日に対照が黙る。"""
    return bc.audit_public(outdir, vocab, verbose=False)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = sys.argv[1]
    req3_rows, req3_pages, _ = bc.load_req3(bc.sources.by_key("req3_layer_kind")["path"])
    a33_rows, _ = bc.load_added33(bc.sources.by_key("added33_structure")["path"])
    vocab = bc.private_vocabulary(req3_pages, a33_rows)
    long_words, short_words = vocab
    leak_name = max(long_words, key=len)
    leak_path = sorted(req3_pages)[0]

    results = []
    for label, cell, target_col, want_reason in (
        ("C1 作品名を散文の列に1つ", leak_name, "caveat", "散文に固有名"),
        ("C2 完全パスを散文の列に1つ", leak_path, "caveat", "パス区切りを含む"),
        ("C3 作品名を次元の列に1つ", leak_name, "year_label", "宣言外の値"),
        ("C4 件数の列に名前を1つ", leak_name, "n_pages", "数字でない値"),
        ("C5 素の出力(何も仕込まない)", None, None, None),
    ):
        tmp = tempfile.mkdtemp(prefix="census_control_")
        try:
            for name in os.listdir(src):
                if name.endswith(".csv"):
                    shutil.copy(os.path.join(src, name), tmp)
            if cell is not None:
                placed = False
                for name in sorted(os.listdir(tmp)):
                    target = os.path.join(tmp, name)
                    rows = list(csv.reader(open(target, encoding="utf-8")))
                    if not rows or target_col not in rows[0]:
                        continue
                    rows[-1][rows[0].index(target_col)] = cell
                    with open(target, "w", encoding="utf-8", newline="") as fh:
                        csv.writer(fh).writerows(rows)
                    placed = True
                    break
                if not placed:
                    raise SystemExit(f"仕込む列 {target_col} が公開物に無い: {label}")
            rc, hits = run_audit_quiet(tmp, vocab)
        finally:
            shutil.rmtree(tmp)
        want_rc = 0 if cell is None else 1
        reason_seen = (any(want_reason in h[3] for h in hits)
                       if want_reason else not hits)
        ok = (rc == want_rc) and reason_seen
        results.append((label, rc, want_rc, reason_seen, ok))

    print("負の対照: 公開物の点検(`audit_public`)")
    print(f"  禁止語彙 {len(long_words)} 語(3文字以上)"
          f" / 盲点 {len(short_words)} 語(2文字以下)"
          f" / 仕込んだ名前・パスは伏せる\n")
    print(f"  {'対照':30s} {'出た':>4s} {'期待':>4s} {'理由も一致':>10s}  判定")
    for label, rc, want_rc, reason_seen, ok in results:
        print(f"  {label:30s} {rc:>4d} {want_rc:>4d} {str(reason_seen):>10s}  "
              f"{'PASS' if ok else '**FAIL**'}")
    bad = [r for r in results if not r[4]]
    print()
    if bad:
        print("**FAIL —— 点検は仕込んだ漏れを見つけていない。この点検を信じてはいけない**")
        return 1
    print("PASS —— 点検は C1〜C4 で鳴り、C5 で黙る。")
    print(f"  **届かないもの: 2文字以下の固有名 {len(short_words)} 語**"
          "(件数と見分けが付かない)、材料に出てこない名前、名前を含まない別名")
    return 0


if __name__ == "__main__":
    sys.exit(main())

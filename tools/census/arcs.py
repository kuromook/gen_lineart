#!/usr/bin/env python
"""編(arc)の公開表を**述語で書く。手で書かない。**

2026-10-30 に `census/arcs.csv` を**その場の走り書きで作った** —— 行は正しかったが
**器具が残らなかったので、便が返ってきたときに手で足すしかなくなった。**
**手で足すと点検を迂回する**(同日に一度踏んで負の対照 C6 が鳴っている)。だから器具にする。

材料は2つで、**出所が違う**:

- `data/census_full/arc_assignment_20261030.csv` —— **こちらが作者の回答から作って送った配分**
  (`.gitignore` 配下。作品フォルダーの名前を持つ)
- `inbox/work_arcs_20261030.csv` —— **抽出側が DB に入れて書き出したもの**(2026-10-30 19:15)

**この器具がするのは転記の点検である**(T132 の規則で、届く範囲を名前より狭く書く):

- **届く**: 送った配分と DB の表で**鍵の集合・編の値・組の数**が食い違っていること /
  先方が書いた `source_sha256` が**こちらの手元のファイルの hash と違う**こと
- **届かない**: **DB そのものの誤り。** **先方の照合**(42組が `pages` に当たる・頁数が
  一致する)は**こちらでは引き直していない** —— 作業木の外の断面(2026-10-26)に
  `work_arcs` は無く、D ドライブの行も無い。**先方の名を帯びたまま持つ**(条例5)。
  そして**中身が本当にその編かも届かない** —— **編は作者の裁定で、測定ではない**(条例6)

    ./venv/bin/python tools/census/arcs.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SENT = "data/census_full/arc_assignment_20261030.csv"
READBACK = "inbox/work_arcs_20261030.csv"

# **DB に入った形**(先方の便。**ここに書くのは名前だけで、数は下で数える**)
DB_TABLE = "work_arcs(year, work, arc, arc_no, decided_by, valid_from, source, source_sha256)"
DB_VIEW = "v_page_arc(page_path, year, work, arc, arc_no)"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load(path: str):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def reconcile(sent, back):
    """**送った配分と DB の表を突き合わせる。落ちたものを理由別に数えて返す**(規律9)。"""
    ks = {(r["year_label"], r["work"]): r["arc"] for r in sent}
    kb = {(r["year"], r["work"]): r["arc"] for r in back}
    only_sent = sorted(ks.keys() - kb.keys())
    only_back = sorted(kb.keys() - ks.keys())
    arc_differs = sorted(k for k in ks.keys() & kb.keys() if ks[k] != kb[k])
    # **先方が書いた `source_sha256` を、こちらの手元のファイルで引き直す**
    declared = {r["source_sha256"] for r in back}
    actual = sha256(os.path.join(ROOT, SENT))
    return dict(n_sent=len(sent), n_back=len(back),
                only_sent=only_sent, only_back=only_back, arc_differs=arc_differs,
                sha_declared=sorted(declared), sha_actual=actual,
                sha_ok=(declared == {actual}))


def arc_rows(sent):
    """**編 × 年の札の件数だけ。** 作品フォルダーの名前は1つも出さない(条例10)。"""
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in sent:
        a = agg[(r["arc"], r["year_label"])]
        a[0] += 1
        a[1] += int(r["n_pages"])
        a[2] += int(r["n_clip"])
        a[3] += int(r["n_lip"])
    return [[arc, yr, n, p, c, l, ""]
            for (arc, yr), (n, p, c, l) in sorted(agg.items())]


def numbering_rows(back):
    """**編の中の通し番号が入った組と、`不明` の組を数える。**

    **番号は名前から取った値である** —— 先方は「名前が連番の綴りそのものの時だけ
    その数を入れた。それ以外は推測しなかった」と書いた。**だから番号の有無は
    『作品に番号が無い』ではなく『名前から取れなかった』である**(T126 の家族)。
    """
    agg = collections.defaultdict(lambda: [0, 0])
    for r in back:
        a = agg[r["arc"]]
        a[0] += 1
        a[1] += int(r["arc_no"] == "不明")
    return [[arc, n, n - unk, unk, ""] for arc, (n, unk) in sorted(agg.items())]


NOTE_ARC = [
    "**編は作者の裁定(2026-10-30)。フォルダーの属性であって頁の属性ではない。** "
    "**鍵は(年の札, 作品フォルダー)の組** —— 同じフォルダー名が2つの編に在るため"
    "(1組が札 2017 で学祭編、札 2024b で fanbox編)。**名前だけでは割れない。** "
    "**編は時期では割れない** —— 学祭編は札 2020・2020b・2022b にも在り、"
    "fanbox編は札 2019b にも在る。**だから列が要る**(作者)",
    "**この表に無いもの**: **fanbox編の2組は D ドライブの札 2025 / 2025b に在り、"
    "構造 DB に入っていない**(DB は F ドライブだけを歩いている。断面 2026-10-26)。"
    "**C ドライブの札 2026 は作者が対象外とした**"
    "(「まだ製作中のものが多い。確定した過去作原稿としては対象としない」)",
    "**2026-10-30 19:15 に DB に入った**(抽出側): 表 "
    f"`{DB_TABLE}` と ビュー `{DB_VIEW}`。"
    "**`pages` には列を足していない** —— 頁は `(year, work)` で結ぶ。"
    "**`decided_by` は `作者の回答(2026-10-30)`、`valid_from` は `2026-10-30`** で、"
    "**版を重ねて残す作り**(ビューは最も新しい `valid_from` だけを使う)。"
    "**先方の照合**(42組が全部 `pages` に当たる / 頁数も 42組とも一致 / "
    "ビューの行 2,359)**は、こちらでは引き直していない** —— "
    "**作業木の外の断面(2026-10-26)に `work_arcs` は無い**(条例5)。"
    "**こちらで引き直したのは転記だけ** —— 鍵の集合・編の値・送ったファイルの sha256",
]


def main() -> int:
    sent, back = load(SENT), load(READBACK)
    rec = reconcile(sent, back)
    print("  [送った配分 と DB の表 の突き合わせ(**転記の点検。DB の正しさには届かない**)]")
    print(f"    送った組 {rec['n_sent']} / DB の組 {rec['n_back']}")
    for label, keys in (("送った側だけに在る組", rec["only_sent"]),
                        ("DB 側だけに在る組", rec["only_back"]),
                        ("編の値が食い違う組", rec["arc_differs"])):
        # **鍵は作品フォルダーの名前なので、数だけを印字する**(条例10)
        print(f"    {label}: {len(keys)}")
    print(f"    先方の `source_sha256` の異なり {len(rec['sha_declared'])} / "
          f"手元のファイルの hash と一致: {'はい' if rec['sha_ok'] else '**いいえ**'}")
    if rec["only_sent"] or rec["only_back"] or rec["arc_differs"] or not rec["sha_ok"]:
        print("  **食い違っている。表を書かずに止める。**")
        return 1

    outdir = os.path.join(ROOT, "census")
    rows = arc_rows(sent)
    rows += [["", "", "", "", "", "", n] for n in NOTE_ARC]
    n_arc = bc.write_csv(os.path.join(outdir, "arcs.csv"),
                         ["arc", "year_label", "n_work_folders", "n_pages",
                          "ext_clip", "ext_lip", "note"], rows)
    num = numbering_rows(back)
    num += [["", "", "", "", NOTE_NUM]]
    n_num = bc.write_csv(os.path.join(outdir, "arc_numbering.csv"),
                         ["arc", "n_work_folders", "arc_no_from_name",
                          "arc_no_unknown", "note"], num)
    print(f"  arcs.csv {n_arc}行 / arc_numbering.csv {n_num}行")
    return 0


NOTE_NUM = (
    "**番号は名前から取った値で、編の中の順番の確認ではない。** "
    "先方の言明: **「名前が連番の綴りそのものの時だけその数を入れ、それ以外は推測しなかった」。**"
    "**だから `不明` は『作品に番号が無い』ではなく『名前から取れなかった』である**(T126 の家族)"
    " —— 範囲や総集編の綴り、DL 版、連番でない呼び名が入る。"
    "**入った番号も、編の中で正しいかは未確認** —— **fanbox編の2組は連番の綴りを持つが、"
    "学祭編にも同じ綴りの組が在る**(編をまたいで番号が重なる)。"
    "**直すなら作者に問う**(条例9)。依頼作品は作者が新しく立てた名で、番号の体系が無い"
)

if __name__ == "__main__":
    raise SystemExit(main())

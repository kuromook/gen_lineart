#!/usr/bin/env python
"""頁の**但し書き**。原稿1枚ごとの事情を、国勢調査の行に残す。

**発議は作者**(2026-10-31、Track M 経由。逐語):

> **今回外した５点について、国勢調査にそれぞれの作品(頁)行に但し書きをつけるようにしてほしい**
> **(但し書きカラムがなければ追加するようにも依頼)**
> **セッションの記憶では今のような作品の事情はいずれ忘れてしまう**

そして翌日、**一度きりの欄ではないと作者が決めた**:

> **こればかりは仕方ない そういった作品は見つけ次第但し書きで判断できるようにしていこう**

**だから目的は「5頁に注を付ける」ではなく「常設の欄を持つ」である。**
この器具は**追記で育つ**ことを前提に書いてある ——
**1つの頁に複数の但し書きが付き得る**(別の時期に別の者が気づくため)。

**これは「悪い原稿」の印ではない**(作者:
**「工程が確定しない時期や忙しくて拙速になることはある。人間の作業は機械的にはなりきれない」**)。
**後から来た者が理由を辿れるようにするための欄である。**

材料と出力:

- 入力 `data/census_full/page_notes_<日付>.csv` —— **頁のパスと但し書きの文面**
  (`.gitignore` 配下。固有名を持つ)。**文面は依頼どおりそのまま載せる**
- 出力 `census/page_notes.csv` —— **公開。頁は `PN<n>` で指す**(段 A)
- 出力 `data/census_full/page_notes_subjects.csv` —— 鍵と頁のパスの対応(全量)

**この器具が届く範囲**(名前より狭く書く。T132):

- **届く**: 書かれた頁のパスが**母集団に実在するか**(2つの材料の両面で引く)/
  1つの頁に何件付いたか / 鍵が前の走行から動いていないか
- **届かない**: **但し書きの内容が正しいか。** 5件のうち4件は**作者の裁定**、
  1件は**Track M の測定**で、**どちらもここで引き直していない**(条例5・条例6)。
  そして**「工程のフローを守れているか」は列にならない** —— Track M が申告し、
  作者が「仕方ない」と裁定した。**この欄は、その計算できない判断の置き場である**

    ./venv/bin/python tools/census/page_notes.py
"""
import collections
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402
import db as census_db  # noqa: E402
import keys as census_keys  # noqa: E402
import sources  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# **日付の形で絞る。** この器具の出力は同じフォルダーに置くので、
# **緩い glob は自分の出力を入力として読み戻す** —— 2026-11-01 に実際そうなった
# (`page_notes_redactions.csv` を読んで `page_path` が無いと落ちた)。
# **落ちたのは幸運で、黙って混ざる形も在った**
INPUT_GLOB = "data/census_full/page_notes_20[0-9][0-9][0-9][0-9][0-9][0-9].csv"
NOT_INPUT = ("page_notes_subjects.csv", "page_notes_redactions.csv")
NEED_COLS = ("page_path", "note_class", "note_ja", "note_source", "note_date",
             "source_notice")


def input_files():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, INPUT_GLOB))):
        if os.path.basename(p) not in NOT_INPUT:
            out.append(p)
    return out


def load_notes(paths):
    """**列が揃っているかを先に見る。** 揃っていなければ、読まずに落ちる。"""
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            missing = [c for c in NEED_COLS if c not in (rd.fieldnames or [])]
            if missing:
                raise SystemExit(
                    f"**入力の列が足りない**: {os.path.basename(p)} —— "
                    f"{', '.join(missing)}。**この器具の出力を入力に混ぜていないか**")
            for r in rd:
                r["_file"] = os.path.basename(p)
                rows.append(r)
    return rows


def population_paths():
    """**母集団の頁のパスを2つの材料から取る。** 片面で済ませない(条例2 の形)。

    - 届いた sha256 の便 —— 全 9,653 ファイルの `page_path`(抽出側、2026-10-26)
    - 構造 DB の `pages` —— 同じ断面。**年の札と形式はこちらから取る**
    """
    sha_path = os.path.join(ROOT, sources.by_key("file_sha256")["path"])
    with open(sha_path, encoding="utf-8") as fh:
        delivered = {r["page_path"] for r in csv.DictReader(fh)}
    con = census_db.connect()
    try:
        in_db = {r[0]: (r[1], r[2]) for r in
                 con.execute("select page_path, year, format from pages").fetchall()}
    finally:
        con.close()
    print(f"    母集団: 便の sha256 {len(delivered)} 行 / DB の `pages` {len(in_db)} 行"
          f" / 両方に在る {len(delivered & in_db.keys())}")
    return delivered, in_db


def redactor():
    """**公開する文面から固有名を伏せる。** 点検と**同じ語彙**を使う。

    **別の語彙で伏せると、点検が後で鳴る** —— 2026-11-01 に実際そうなった:
    作者の裁定の文面に **`sketch set` フォルダー**と在り、
    **`sketch` という名の作品フォルダーが母集団に2つ実在する**ので、
    `audit_public` が「散文に固有名(6文字)」で鳴った。**鳴ったのは正しい。**

    **伏せ方は番号つきの差し替えである** ——
    `…` で消すと**文の構造が読めなくなる**(「作品本体が `[名前1]` フォルダーの中に
    入っている」は、名前を知らなくても工程の崩れが読める)。
    **番号と語の対応は `data/census_full/page_notes_redactions.csv`**(`.gitignore` 配下)。
    """
    req3_rows, req3_pages, _ = bc.load_req3(
        sources.by_key("req3_layer_kind")["path"])
    a33_rows, _ = bc.load_added33(sources.by_key("added33_structure")["path"])
    long_words, short_words = bc.private_vocabulary(req3_pages, a33_rows)
    matcher = bc.ProseMatcher(long_words)
    table = {}

    def run(text: str) -> tuple:
        out, n = text, 0
        for _ in range(64):          # **上限を置く** —— 差し替えた語が当たると回り続ける
            w = matcher.find(out)
            if w is None:
                break
            ph = table.setdefault(w, f"[名前{len(table) + 1}]")
            out = out.replace(w, ph)
            n += 1
        return out, n

    print(f"    伏せる語彙 {len(long_words)} 語(点検と同じもの)"
          f" / **盲点 2文字以下 {len(short_words)} 語**")
    return run, table


NOTE = [
    "**この欄は常設である**(作者、2026-10-31: "
    "**「そういった作品は見つけ次第但し書きで判断できるようにしていこう」**)。"
    "**一度きりの5件ではない。** **1つの頁に複数の但し書きが付き得る** —— "
    "別の時期に別の者が気づくので、**追記で育つ**",
    "**これは「悪い原稿」の印ではない。** 作者の言明が前提である: "
    "**「工程が確定しない時期や忙しくて拙速になることはある。"
    "人間の作業は機械的にはなりきれない」**(2026-10-31)。"
    "**後から来た者が理由を辿れるようにするための欄である**",
    "**内容はここで引き直していない。** `by` の列が出所である —— "
    "**作者の裁定**(条例6: 作者の言明は測定ではない)か、"
    "**ほかの track の測定**(条例5: 測った者の名を帯びる)。"
    "**census が確かめたのは、その頁が母集団に実在することだけ**",
    "**`page_key` は `data/census_full/page_keys.csv` に書き留めてある** —— "
    "**番号は走行をまたいで動かない。** 動くと、昨日の表の `PN1` と今日の `PN1` が"
    "別の頁になる。**頁のパスは公開物に出さない**(段 A)",
    "**「制作工程のフローを守れているか」は列にならない** —— Track M が申告し、"
    "**作者が「こればかりは仕方ない」と裁定した**(2026-10-31)。"
    "**この欄は、その計算できない判断の置き場である** —— "
    "**計算で判定せず、但し書きを読んで判断する**(作者の決めた運用)",
]


def main() -> int:
    paths = input_files()
    if not paths:
        print(f"  **入力が無い**: {INPUT_GLOB}")
        return 1
    print("  [頁の但し書き]")
    print("    入力: " + " / ".join(os.path.basename(p) for p in paths))
    notes = load_notes(paths)
    delivered, in_db = population_paths()

    # **実在しない頁に但し書きを付けていないか。** 落ちたものを理由別に数える(規律9)
    drops = collections.Counter()
    kept = []
    for r in notes:
        p = r["page_path"]
        if p not in in_db:
            drops["DB の `pages` に無い"] += 1
            continue
        if p not in delivered:
            drops["便の sha256 に無い"] += 1
            continue
        kept.append(r)
    print(f"    但し書き {len(notes)} 件 / 母集団に在る頁の分 {len(kept)}")
    for why, n in sorted(drops.items()):
        print(f"      落とした {n} 件 —— {why}")
    if drops:
        print("    **実在しない頁に但し書きが付いている。表を書かずに止める。**")
        return 1

    reg = census_keys.page_registry()
    redact, redactions = redactor()
    rows = []
    for i, r in enumerate(sorted(kept, key=lambda r: (r["page_path"], r["note_date"])),
                          start=1):
        year, fmt = in_db[r["page_path"]]
        key = reg.key_for(r["page_path"], where=r["_file"])
        content, n_red = redact(r["note_ja"])
        rows.append([i, key, year, fmt, r["note_class"], content, n_red,
                     r["note_source"], r["note_date"], r["source_notice"], ""])
    reg.report("頁の鍵")
    n_keys = reg.close()

    per_page = collections.Counter(r[1] for r in rows)
    print(f"    頁 {len(per_page)} / 但し書き {len(rows)}"
          f" / **2件以上付いた頁 {sum(1 for v in per_page.values() if v > 1)}**")
    by_src = collections.Counter(r[7] for r in rows)   # **`by` の列**(出所)
    for src, n in sorted(by_src.items()):
        print(f"      出所 {src}: {n} 件")

    n_red_total = sum(r[6] for r in rows)
    print(f"    **伏せた固有名 {n_red_total} 箇所 / 異なり {len(redactions)} 語**"
          "(**番号と語の対応は全量側だけ**)")
    with open(os.path.join(ROOT, "data/census_full",
                           "page_notes_redactions.csv"), "w",
              encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["placeholder", "word", "n_chars"])
        for word, ph in sorted(redactions.items(), key=lambda kv: kv[1]):
            w.writerow([ph, word, len(word)])

    rows += [["", "", "", "", "", "", "", "", "", "", n] for n in NOTE]
    header = ["note_id", "page_key", "year_label", "format", "category",
              "content", "n_redacted", "by", "as_of", "source", "note"]
    n_pub = bc.write_csv(os.path.join(ROOT, "census", "page_notes.csv"),
                         header, rows)

    # **全量側: 鍵と頁のパスの対応。** 公開側には1つも出さない
    full = os.path.join(ROOT, "data/census_full", "page_notes_subjects.csv")
    with open(full, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["page_key", "page_path", "year_label", "format", "n_notes"])
        for path, key in sorted(reg.map.items(), key=lambda kv: kv[1]):
            year, fmt = in_db[path]
            w.writerow([key, path, year, fmt, per_page[key]])
    print(f"    page_notes.csv {n_pub}行 / page_keys.csv {n_keys}行"
          f" / page_notes_subjects.csv {len(reg.map)}行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

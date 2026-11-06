#!/usr/bin/env python
"""**作者の版の答え**(2026-10-25〜27)を census の表に取る。

**ユーザーの指示(2026-11-07)**: 10-26 の版の答え(問い A〜O)を census の裁定の表に取り込む —— 「**やる**」。
**なぜ**: 答えは基盤の `doc/work_log.md` と Track M の札の表にしか無く、census は 11-06 に
`done*` の答えを拾えず「無い」と書いた(L に誤った返事)。**作者に同じことを二度聞かせない。**

**2つの表**:

1. **答えの登録**(`RULINGS`。公開): 問いごとに、日時・経路・**作者の言葉の逐語**(作品・
   フォルダー・ファイルの名前だけ〔 〕で伏せる)・それで付いた札。**出所は基盤の作業ログの節**
2. **ファイルごとの札**(非公開。公開は札 × 出所の数だけ): **Track M の札の表 `m78`
   (`version_roles_current.csv`、872行 = M の依頼3 の頁、10-27 23:59、器具 `8044d31`)を写した。**
   **census は札を作り直さない** —— M が答えを塊に当てた結果で、**決めたのは作者と M**。
   **`decided_by` が作者の言葉か作者の記入表のものだけを「作者の札」と数え、
   それ以外(`none` / `identical_metadata` / 塊の代表の規則)は「機械の札」として分ける**

**届かないもの**: **872頁の外**(M は依頼3 の頁にだけ当てた。**答えをフォルダーの外に広げない**
—— `done*` だけは `wip_pages.py` で全頁に当てた)/ **畳んだ後の代表**(M の `m86`。census の
畳みは sha256 で、版の代表は持たない)/ **構造 DB の `page_versions`**(10-26 17:57 の断面。古い)

    ./venv/bin/python tools/census/version_rulings.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
M_ROLES = "data/census_full/m_version_roles_current_m78_20261027.csv"
M_ROLES_SHA = "3f4dfcb4d443be86"           # 写した時(2026-11-07)の先頭16桁
WIP = "data/census_full/wip_pages_20261106.csv"
OUT_FULL = "data/census_full/version_rulings_20261107.csv"
OUT_PUB = "census/version_rulings.csv"
LOG = "基盤 `doc/work_log.md`"

# **作者の言葉は逐語。名前だけ〔 〕で伏せた。** `key` は M の `decided_by` に現れる印
RULINGS = [
    ("Vb", "2026-10-25 23:36", f"抽出側が中継({LOG} 10-25 23:36-23:50)",
     "「b付きが新しい　どちらも」", "札 b 付きの方を新しい側に(`final` / `superseded`)", "b 付きが新しい"),
    ("Vs", "2026-10-25 23:36", f"抽出側が中継(同上)",
     "「〔札とフォルダー〕のほうが新しい 作業もすすン出る頁がおおくある」", "2塊を反転", None),
    ("Vg", "2026-10-25 23:49", f"{LOG} 10-25 23:49",
     "「〔フォルダー〕は古い。〔フォルダー〕直下の clip が正」", "2塊を反転、1塊はそのまま", "直下が正"),
    ("VA", "2026-10-26 00:49", f"{LOG} 問い A", "「A は内容は同じだな」", "`duplicate_content`", "A は内容は同じ"),
    ("VB", "2026-10-26 00:53", f"{LOG} 問い B",
     "「B　一部内容が被ってそうだが別経路のブランチ　どちらが新しいはない」", "`branch`(順序なし)", None),
    ("VD", "2026-10-26 00:55", f"{LOG} 問い D",
     "「D わからない 同じにみえる ただのバックアップのようだ」", "`undecided`(作者も判定できない)", None),
    ("VE", "2026-10-26 00:57", f"{LOG} 問い E",
     "「E これも全くおなじに見える おそらくバックアップ」", "`undecided`(推定の語)", None),
    ("VF", "2026-10-26 00:59", f"{LOG} 問い F",
     "「F backup のほうにはゴミレイヤがのっている なぜこうなってるのか不明だが"
     "(この余分のレイヤを消したら一致するかもしれない)、backupのほうは使わない」", "`not_used`(使わない)", None),
    ("VG", "2026-10-26 01:00", f"{LOG} 問い G", "「G　まったくべつもの」", "`unrelated`(塊を解く)", None),
    ("VH", "2026-10-26 01:06", f"{LOG} 問い H",
     "「H　〔フォルダー〕下の二つは下絵のみ　〔ファイル〕と〔ファイル〕は中身はほぼ同じだが、"
     "〔片方〕のほうはテーブルがある　おそらく製作する際の依頼者により２種類おねがいされたもの」",
     "`rough`(下絵のみ)/ `variant`(両方が成果物)", None),
    ("VI", "2026-10-26 01:07", f"{LOG} 問い I",
     "「I 〔2つのフォルダー〕同様に done? はマージ前のバラバラページ」", "`wip`(`wip_pages` の R2)", None),
    ("VJ", "2026-10-26 01:13", f"{LOG} 問い J",
     "「J　内容同じ　〔作品〕は〔作品〕の続編のため 資料参照のために新しい作業フォルダにコピーしたようだ　"
     "塊24については不明」", "資料のためのコピー(内容同じ)", None),
    ("VK", "2026-10-26 01:18", f"{LOG} 問い K",
     "「K　単行本化のために原稿を束ねたフォルダ　改訂があるはず　〔フォルダー〕はこの単行本用の"
     "カラーカット(１枚イラスト)」", "`revision`(改訂のはず)/ 1枚絵の札", "K　単行本化"),
    ("VL", "2026-10-26 01:23", f"{LOG} 問い L", "「全部同じ運用だ　共通して背景がない」",
     "`wip`(`wip_pages` の R3)", "L 全部同じ運用だ"),
    ("VM", "2026-10-26 01:25", f"{LOG} 問い M'", "「M' 全部下絵」", "`rough`", "M' 全部下絵"),
    ("VO", "2026-10-26 01:30", f"{LOG} 問い O",
     "「O　イラストのみを作成して載せていたcmc管理ファイルからそれぞれの目的(納品のために)"
     "１ファイルとして取り出した」", "取り出した1枚絵", None),
    ("VN", "2026-10-26 01:45", f"{LOG} 問い N",
     "「N 資料用　現在はXnviewmpや pureref 等を使用しているが、当時都合の良い画像ビューアツールが"
     "見つからず原稿ファイルに画像をのこしていた　その名残」", "資料用(下絵ではない)", None),
    ("VW", "2026-10-27", "Track M の記入表(`author_workbook_20261027`。作者が塊ごとに記入)",
     "(記入表の行ごと。作者の備考は非公開の全量に在る)", "塊ごとの代表と採点の扱い", "author_workbook_20261027"),
]


def ruling_of(decided_by: str):
    for rid, *_, key in RULINGS:
        if key and key in decided_by:
            return rid
    return None


def main() -> int:
    print("[作者の版の答え(10-25〜27)]")
    h = hashlib.sha256(open(os.path.join(ROOT, M_ROLES), "rb").read()).hexdigest()
    print(f"    M の札の表 m78 sha256 {h[:16]} {'一致' if h.startswith(M_ROLES_SHA) else '**写した時と違う**'}")
    rows = list(csv.DictReader(open(os.path.join(ROOT, M_ROLES), encoding="utf-8-sig")))
    wip = {r["page_path"]: r["wip_status"] for r in csv.DictReader(open(os.path.join(ROOT, WIP), encoding="utf-8"))}
    con = db.connect()
    dbv = dict(con.execute("select page_path, version_role from page_versions").fetchall())
    full, agg, unmapped = [], collections.Counter(), collections.Counter()
    for r in rows:
        d = r["decided_by"]
        by_author = d.startswith("作者の") or d.startswith("author_workbook")
        rid = ruling_of(d) if by_author else ""
        if by_author and rid is None:
            unmapped[d[:40]] += 1
        full.append([r["page_path"], r["version_role"], r["use"], "作者" if by_author else "機械",
                     rid or "", d, r["dup_cluster_id"], r["superseded_by"]])
        agg[(r["version_role"], "作者" if by_author else "機械", rid or "")] += 1
    n_auth = sum(1 for x in full if x[3] == "作者")
    print(f"    行 {len(full)}(M の依頼3 の頁)/ **作者の札 {n_auth}** / 機械の札 {len(full) - n_auth}")
    print(f"    作者の札で問いに当てられなかった `decided_by` {sum(unmapped.values())}"
          + (f": {dict(unmapped)}" if unmapped else ""))
    # 既知セル: M の `wip` ⇄ census の `作業途中`(別の器具。census は名前の述語と裁定で当てた)
    m_wip = {x[0] for x in full if x[1] == "wip"}
    c_wip = {x[0] for x in full if wip.get(x[0]) == "作業途中"}
    print(f"    **既知セル(M の wip ⇄ census の作業途中)**: M の wip {len(m_wip)} / うち census も作業途中 "
          f"{len(m_wip & c_wip)} / census だけ作業途中 {len(c_wip - m_wip)}")
    # **census だけ作業途中の頁は、M では塊の代表の軸で札が付いている**(作者の記入表 10-27)。
    # **`use` がマージ前と書くか、作者が別の方を選んだと書くかを数える** —— それ以外が在れば食い違い
    only = [x for x in full if x[0] in c_wip - m_wip]
    merge = sum(1 for x in only if "マージ前" in x[2])
    chose = sum(1 for x in only if "作者が別の方を選んだ" in x[2])
    print(f"      census だけ作業途中 {len(only)} = M の `use` がマージ前 {merge} + 作者が別の方を選んだ {chose}"
          f" + **それ以外 {len(only) - merge - chose}**")
    bad_only = len(only) - merge - chose
    stale = sum(1 for x in full if x[0] in dbv and dbv[x[0]] != x[1])
    print(f"    構造 DB の `page_versions`(10-26 17:57)と札が違う行 {stale} / DB に行が無い {sum(1 for x in full if x[0] not in dbv)}")
    with open(os.path.join(ROOT, OUT_FULL), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["page_path", "version_role", "use", "decided_kind", "ruling", "decided_by",
                    "dup_cluster_id", "superseded_by"])
        w.writerows(full)
    pub = [["", rid, at, via, words, what, "", "", ""] for rid, at, via, words, what, _ in RULINGS]
    pub += [["札の数", rid, "", "", "", "", role, kind, n] for (role, kind, rid), n in sorted(agg.items())]
    pub += [[""] * 9 + [n] for n in NOTE]
    pub = [r + [""] if len(r) == 9 else r for r in pub]
    n_pub = bc.write_csv(os.path.join(ROOT, OUT_PUB),
                         ["row_kind", "ruling", "as_of", "source", "author_words", "what",
                          "version_role", "decided_kind", "n_pages", "note"], pub)
    print(f"    {OUT_PUB} {n_pub}行 / ファイルごと {len(full)}行(`{OUT_FULL}`)")
    return 1 if unmapped or len(m_wip - c_wip) or bad_only else 0


NOTE = [
    "**ユーザーの指示(2026-11-07)「やる」**: 作者の版の答えを census の裁定の表に取り込む。"
    "**答えは基盤の作業ログと Track M の札の表にしか無かった** —— census は 11-06 に `done*` の答えを拾えず、作者に聞き直しかけた",
    "**上の行は答えの登録**(問いごと。作者の言葉は逐語で、作品・フォルダー・ファイルの名前だけ〔 〕で伏せた)。"
    "**下の `札の数` の行は Track M の札の表(m78、M の依頼3 の 872頁、10-27)を写して数えたもの**",
    "**札を決めたのは作者と Track M** —— **census は札を作り直していない。** "
    "**`decided_kind` = 作者 は `decided_by` が作者の言葉か作者の記入表のもの、機械 はそれ以外**(名前の規則・塊の代表の規則・塊なし)",
    "**872頁の外には当てていない** —— 答えは塊(M が組んだ候補)について出たもので、**名前でフォルダーの外に広げない**(条例9)。"
    "**例外は `done*`**(作者がフォルダーの単位で答えた。`wip_pages.csv` が全頁に当てた)",
    "**ファイルごとの札は非公開**(`data/census_full/version_rulings_20261107.csv`。頁のパスを持つ)",
    "**この題材に版の順序はほとんど無い**(作者は 15問のうち順序を答えたのは b 付き・直下・K の「はず」だけ。基盤 10-26)。"
    "**札の名前が順序を含意して見えても、作者が答えたのは「どちらを使うか」と「どういう関係か」**",
    "**既知セル**: M の `wip` 65頁は 65頁とも `wip_pages.csv` の `作業途中`。**逆向きの 30頁**(census は作業途中、M は `final` 19 / `superseded` 11)は"
    "**全部 作者の記入表(10-27)で塊の代表を選んだ札**で、M の `use` が「マージ前のバラバラページ」19 /「作者が別の方を選んだ」11 —— "
    "**代表の軸と作業途中の軸は別で、矛盾ではない**",
    "**構造 DB の `page_versions`(10-26 17:57 の断面)は古い** —— 札の正本はこの表(M の m78 の写し)",
]

if __name__ == "__main__":
    raise SystemExit(main())

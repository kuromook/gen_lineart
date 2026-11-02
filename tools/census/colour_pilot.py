#!/usr/bin/env python
"""カラーイラスト頁の**候補**を選ぶ。**選別は測定である**(規律9)。

**発注は作者**(2026-10-31、Track M 経由): **「新たにカラーイラスト頁を５点発注する
ようにしてほしい」**。選定は census の領分(条例10)。

**この器具が出す主なものは候補ではなく、1つの否定である**:

> **`color_ratio` は、作者の「カラーか」の分類を再現しない。**
> 作者が**モノクロ漫画**と裁定した頁の `color_ratio` は **0.0646**、
> **カラーとして残っている**頁は **0.0274** —— **モノクロの方が高い。**
> **床をどこに置いても6頁を分けられない**(最良 5/6)。

**だから候補は出すが、最終の5点は作者に見ていただく** —— Track M の求めも同じ
(**「候補を 8〜10点に広げて出していただければ、こちらから作者に確認します」**)。
**これは規律13 そのものである**: **列で同定する前に、その列が作者の分類を再現すると
示す。示せなければ目視か census。**

**この器具が届く範囲**(名前より狭く書く。T132):

- **届く**: 門ごとに**残った数と落ちた数** / `color_ratio` が作者の分類を
  再現するかの**的中と取りこぼし** / `線画` を名前に含む可視の葉の数
- **届かない**:
  - **「カラーか」そのもの** —— `color_ratio` は**プレビューの彩度**で、
    **抽出側が測った列**(条例5)。**断面は 2026-10-26 17:57**
  - **「制作工程のフローを守れているか」** —— **列にならない**(Track M が申告し、
    **作者が「こればかりは仕方ない」と裁定した**)。**但し書きで判断する**(`page_notes.py`)
  - **`線画` を含む層が本当に線画か** —— **名前の述語である**(T106)。
    **`rendering_*` を外しているのは3D の線を避けるためで、中身を見たのではない**

    ./venv/bin/python tools/census/colour_pilot.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402
import db as census_db  # noqa: E402
import exclusions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# **11-01 の表は作者に送った8点の記録として残す**(上書きしない。規律10)。
# **11-02 からは除外の門を足したので、同じ走行でも選ばれる頁が変わる** ——
# **だから別の名前に書き、送った記録は前の表から読む**
OUT_FULL = "data/census_full/colour_pilot_candidates_20261102.csv"
SENT_RECORD = "data/census_full/colour_pilot_candidates_20261101.csv"

# **床は 0.30。** **これは「カラーの境」ではなく「候補を広げる篩」である** ——
# **下で示すとおり、この列は作者の分類を再現しない。**
# **床を宣言するのは、落ちた数を印字するため**(規律9)
COLOUR_FLOOR = 0.30
# **年の札ごとの上限。** Track M の求めは **8〜10点** —— **札が偏ると世代が1つに寄る**
PER_YEAR_CAP = 1

# **既存のパイロット8頁**(Track M の母集団。重複を避ける。条件4)
PILOTS = (
    r"F:\manuscripts\2018\skima10\skima10\page0064.clip",
    r"F:\manuscripts\2018\skima10\skima10\page0073.clip",
    r"F:\manuscripts\2018\illust201710\page0003.clip",
    r"F:\manuscripts\2018\illust201710\page0005.clip",
    r"F:\manuscripts\2017\illustration\sunshine2\page0002.clip",
    r"F:\manuscripts\2017\illustration\sunshine2\page0019.clip",
    r"F:\manuscripts\2018\skima11\skima12H.clip",
    r"F:\manuscripts\2018\skima11\skima12chara.clip",
)

# **作者が頁ごとに下した分類**(2026-10-31、Track M 経由)。
# **規律13 の点検の正解源である** —— **census が付けた分類ではない。**
# **`線画` が0枚の2頁には、カラーかどうかの裁定が無い**(母集団に入る前に外れた)
AUTHOR_CLASS = {
    r"F:\manuscripts\2018\illust201710\page0003.clip": "カラー",
    r"F:\manuscripts\2018\illust201710\page0005.clip": "カラー",
    r"F:\manuscripts\2018\skima11\skima12H.clip": "カラー",
    r"F:\manuscripts\2018\skima11\skima12chara.clip": "カラー",
    r"F:\manuscripts\2017\illustration\sunshine2\page0002.clip": "カラー",
    r"F:\manuscripts\2017\illustration\sunshine2\page0019.clip": "モノクロ漫画",
    r"F:\manuscripts\2018\skima10\skima10\page0064.clip": "**裁定なし**",
    r"F:\manuscripts\2018\skima10\skima10\page0073.clip": "**裁定なし**",
}
# **点検で「カラーでない」側に数える分類。** **混ぜて1つの値にしない** —— 表は分類ごとに出す
NOT_COLOUR = ("モノクロ漫画", "カラーだが色が入っていない")

# **作者が候補8点を目で見た裁定**(2026-11-01、Track M 経由)と **Track M が採用したか**。
# **表は `data/census_full/` の側に置く**(`.gitignore` 配下)—— **候補の頁のパスはこれまでも
# 全量側からしか読んでいない。** **公開のコードに候補のパスを書くと、裁定と採用の組から
# 作者の恒久の除外の頁が分かる**(2026-11-02 に一度書いて、commit の前に気づいて移した)。
# **3つめの分類「カラーだが色が入っていない」は作者の言葉から**:「カラーイラストの線画まで
# 入った状態で色がない」—— **制作の到達点の問題で、モノクロ原稿とは別の種類。**
# **作者の分類と Track M の採用は別の量である**(採用しなかった理由は彩度とも構造とも独立。
# **混ぜると規律13 の点検が曲がる**)。**採用しなかった理由はどこにも書かない**
VERDICTS = "data/census_full/author_verdicts_20261101.csv"


def _load_verdicts():
    path = os.path.join(ROOT, VERDICTS)
    if not os.path.isfile(path):
        raise SystemExit(f"**作者の8点の裁定の表が無い: {VERDICTS}** —— 点検を走らせない")
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    bad = [r["author_class"] for r in rows
           if r["author_class"] != "カラー" and r["author_class"] not in NOT_COLOUR]
    if len(rows) != 8 or bad:
        raise SystemExit(f"**裁定の表が8行でないか、宣言していない分類が在る**: {len(rows)} 行 / {bad}")
    return ({r["page_path"]: r["author_class"] for r in rows},
            tuple(r["page_path"] for r in rows if r["adopted_by_m"] == "yes"))


_CANDIDATE_CLASS, ADOPTED_BY_M = _load_verdicts()
AUTHOR_CLASS.update(_CANDIDATE_CLASS)
# **Track M の後知恵の仮説**(2026-11-01、依頼4):「`n_lineart_visible <= 5` が内容の裁定と一致する」。
# **候補8点を見た後に立てた** —— **だから8点の上の一致は検証ではない。** 独立なのはパイロットの6頁だけ
HYPOTHESIS_FORMED_ON = tuple(p for p, c in AUTHOR_CLASS.items()
                             if p not in PILOTS and c != "**裁定なし**")

LINEART_LIKE = "%線画%"       # **作者の裁定の語そのまま**(Track M の母集団の条件)
RENDER_LIKE = "rendering%"    # **3D から書き出した線**(作者の説明、2026-10-31)


def fetch(con, excluded):
    """門を1つずつ当てて、**残った数と落ちた数を両方**返す。"""
    where = []
    rows = []
    gates = bc.COLOUR_PILOT_GATES
    sql_by_gate = {
        gates[0]: "1=1",
        gates[1]: "p.format='clip'",
        gates[2]: "pc.status='ok'",
        gates[3]: f"pc.color_ratio >= {COLOUR_FLOOR}",
        gates[4]: ("exists(select 1 from layers l where l.page_path=p.page_path"
                   f" and not l.folder and l.name like '{LINEART_LIKE}'"
                   " and (l.visibility & 1)=1)"),
        gates[5]: ("not exists(select 1 from layers l where l.page_path=p.page_path"
                   f" and l.name like '{RENDER_LIKE}')"),
        gates[6]: "p.page_path not in (" + ",".join(
            "'" + x.replace("'", "''") + "'" for x in PILOTS) + ")",
        gates[7]: "p.page_path not in (" + ",".join(
            "'" + x.replace("'", "''") + "'" for x in excluded) + ")",
        # **母集団の中にも重複が在る。** **代表は「最も早い札」**(作者の裁定 2026-10-28:
        # **時期は最も早い札で数え、上限であると書く**)。**mtime はコピー時刻で使えない**
        gates[8]: ("p.page_path in (select page_path from ("
                   " select page_path, row_number() over ("
                   "   partition by f_sha256 order by year, page_path) rn"
                   " from file_hashes join pages using(page_path)) where rn = 1)"),
    }
    prev = None
    for g in gates:
        if g not in sql_by_gate:     # **ここから先は SQL では引けない門**
            break
        where.append(sql_by_gate[g])
        w = " and ".join(where)
        n_pages, n_works = con.execute(f"""
            select count(*), count(distinct p.year || '|' || p.work)
            from pages p left join page_colour pc using(page_path)
            where {w}""").fetchone()
        rows.append([g, n_pages, n_works, "" if prev is None else prev - n_pages])
        prev = n_pages
    # **最後の門は「作品ごとに1頁」** —— 年代が偏らないため(条件5)
    cands = con.execute(f"""
        with c as (
          select p.page_path, p.year, p.work, pc.color_ratio,
            (select count(*) from layers l where l.page_path=p.page_path
              and not l.folder and l.name like '{LINEART_LIKE}'
              and (l.visibility & 1)=1) n_lineart_visible,
            (select count(*) from layers l where l.page_path=p.page_path) n_layers,
            row_number() over (partition by p.year, p.work
                               order by pc.color_ratio desc, p.page_path) rn
          from pages p left join page_colour pc using(page_path)
          where {' and '.join(where)})
        select page_path, year, work, color_ratio, n_lineart_visible, n_layers
        from c where rn = 1 order by color_ratio desc""").fetchall()
    gi = len(rows)
    rows.append([gates[gi], len(cands), len({(r[1], r[2]) for r in cands}),
                 prev - len(cands)])
    prev = len(cands)

    # **パイロットと同じ作品フォルダーを外す**(条件5: できれば別々の作品から)
    pilot_works = {(y, w) for y, w in con.execute(
        "select year, work from pages where page_path in ("
        + ",".join("?" * len(PILOTS)) + ")", list(PILOTS)).fetchall()}
    pool = [c for c in cands if (c[1], c[2]) not in pilot_works]
    rows.append([gates[gi + 1], len(pool), len({(c[1], c[2]) for c in pool}),
                 prev - len(pool)])
    prev = len(pool)

    # **同じ作品名を2つの札から取らない。** **双子ではない**(hash が違う)が、
    # **同じ頁名・ほぼ同じ彩度の組が在り、8枠のうち2枠を同じ絵に使うことになる。**
    # **残すのは最も早い札**(作者の裁定の向き。2026-10-28)
    seen_work = set()
    deduped = []
    for c in sorted(pool, key=lambda c: (c[2], c[1], c[0])):
        if c[2] in seen_work:
            continue
        seen_work.add(c[2])
        deduped.append(c)
    pool = deduped
    rows.append([gates[gi + 2], len(pool), len({(c[1], c[2]) for c in pool}),
                 prev - len(pool)])
    prev = len(pool)

    # **年の札ごとに `PER_YEAR_CAP` 頁まで。** **年代が偏らないため**(条件5:
    # **世代もプールである** —— 制作工程は年で移り変わる)
    per_year = collections.Counter()
    sent = []
    for c in sorted(pool, key=lambda c: (-c[3], c[0])):
        if per_year[c[1]] >= PER_YEAR_CAP:
            continue
        per_year[c[1]] += 1
        sent.append(c)
    sent.sort(key=lambda c: (c[1], -c[3]))
    rows.append([gates[gi + 3], len(sent), len({(c[1], c[2]) for c in sent}),
                 prev - len(sent)])
    return rows, cands, pool, sent


def column_check(con):
    """**規律13 の点検**: `color_ratio` は作者の分類を再現するか。

    **反証条件はそのまま「的中率と取りこぼしを印字している」である。**
    床を動かしながら、**両方の誤りを別々に数える** ——
    引き算で片方を出すと、どちらの誤りか分からなくなる。
    """
    got = dict(con.execute(
        "select page_path, color_ratio from page_colour where page_path in ("
        + ",".join("?" * len(AUTHOR_CLASS)) + ")",
        list(AUTHOR_CLASS)).fetchall())
    judged = {p: c for p, c in AUTHOR_CLASS.items() if c == "カラー" or c in NOT_COLOUR}
    out = []
    floors = sorted({0.02, 0.05, 0.10, COLOUR_FLOOR}
                    | {round(v, 4) for p, v in got.items() if p in judged})
    for f in floors:
        hit = missed = false_pos = 0
        for p, c in judged.items():
            v = got.get(p)
            if v is None:
                continue
            passes = v >= f
            if c == "カラー":
                hit += int(passes)
                missed += int(not passes)
            else:
                false_pos += int(passes)
        out.append([f, len(judged), hit, missed, false_pos])
    return out, got


def lineart_check(con):
    """**Track M の依頼4**: 後知恵の仮説 `n_lineart_visible <= k` を、裁定の在る頁で当てる。

    **この列は名前で数えた列である**(`線画` を含む可視の葉。祖先の可視は見ていない)。
    **的中・取りこぼし・カラーでないものを通す数を別々に**、**全部の頁と、仮説を立てるのに
    使っていない頁(パイロット)とで別々に**印字する —— **仮説を立てた8点の上の一致は検証ではない。**
    """
    paths = [p for p, c in AUTHOR_CLASS.items() if c == "カラー" or c in NOT_COLOUR]
    n = dict(con.execute(
        "select l.page_path, count(*) from layers l where l.page_path in ("
        + ",".join("?" * len(paths)) + f") and not l.folder and l.name like '{LINEART_LIKE}'"
        " and (l.visibility & 1)=1 group by l.page_path", paths).fetchall())
    n = {p: n.get(p, 0) for p in paths}
    out = []
    for k in sorted(set(n.values())):
        for scope, ps in (("裁定の在る全頁", paths),
                          ("仮説を立てるのに使っていない頁",
                           [p for p in paths if p not in HYPOTHESIS_FORMED_ON])):
            hit = missed = fp = 0
            for p in ps:
                passes = n[p] <= k
                if AUTHOR_CLASS[p] == "カラー":
                    hit += int(passes)
                    missed += int(not passes)
                else:
                    fp += int(passes)
            out.append([k, scope, len(ps), hit, missed, fp])
    return out, n


def main() -> int:
    print("  [カラーイラスト頁の候補(Track M の依頼2)]")
    excluded = exclusions.load()
    print(f"    作者の恒久の除外の名簿: {len(excluded)} 頁(`{exclusions.REGISTER}`)")
    con = census_db.connect()
    try:
        gate_rows, cands, pool, sent = fetch(con, excluded)
        chk, got = column_check(con)
        lchk, n_line = lineart_check(con)
    finally:
        con.close()

    print("    門ごと(**残った数と落ちた数を両方**。規律9):")
    for g, n_pages, n_works, dropped in gate_rows:
        d = "" if dropped == "" else f" / 落ちた {dropped}"
        print(f"      頁 {n_pages:6d} / 作品 {n_works:4d}{d}  {g}")

    print(f"    **規律13 の点検 —— `color_ratio` は作者の分類を再現するか**"
          f"(裁定の在る {chk[0][1]} 頁):")
    print(f"      {'床':>8s} {'的中':>5s} {'取りこぼし':>11s} {'カラーでないものを通す':>15s}")
    best = None
    for f, n, hit, missed, fp in chk:
        print(f"      {f:>8.4f} {hit:>5d} {missed:>11d} {fp:>15d}")
        if best is None or (hit - fp) > best[0]:
            best = (hit - fp, f, hit, missed, fp)
    print(f"      → **最良でも {best[2]}/{chk[0][1]}**"
          f"(床 {best[1]:.4f}、取りこぼし {best[3]}、カラーでないものを通す {best[4]})"
          " —— **この列は分類を再現しない。だから最終は作者に見ていただく**")

    # **候補の名前つきは全量側だけ**(段 A に頁のパスを出さない)
    sent_paths = {c[0] for c in sent}
    pool_paths = {c[0] for c in pool}
    # **作者に送った記録は 11-01 の表から読む**(この走行の選びではない)
    with open(os.path.join(ROOT, SENT_RECORD), encoding="utf-8", newline="") as fh:
        sent_0101 = {r["page_path"] for r in csv.DictReader(fh) if r["to_author"] == "yes"}
    if len(sent_0101) != 8:
        raise SystemExit(f"**11-01 に送った記録が8点でない: {len(sent_0101)}**")
    with open(os.path.join(ROOT, OUT_FULL), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "selected_now", "sent_20261101", "author_class",
                    "adopted_by_m", "in_pool", "page_path", "year_label",
                    "work", "color_ratio", "n_lineart_visible", "n_layers"])
        for i, (pp, y, wk, cr, nl, nlay) in enumerate(cands, start=1):
            w.writerow([i, "yes" if pp in sent_paths else "no",
                        "yes" if pp in sent_0101 else "no",
                        AUTHOR_CLASS.get(pp, ""),
                        "yes" if pp in ADOPTED_BY_M else "no",
                        "yes" if pp in pool_paths else "no",
                        pp, y, wk, f"{cr:.4f}", nl, nlay])
    # **送った8点のうち、この走行で選ばれない頁**(除外の門のため)。**新しい候補は送らない**
    moved = sorted(sent_0101 - sent_paths)
    print(f"    11-01 に作者に送った8点のうち、今の走行で選ばれない頁 {len(moved)}"
          f" / 代わりに選ばれる頁 {len(sent_paths - sent_0101)}"
          " —— **発注は無いので送らない**(選びが動いた理由は除外の門だけか、下の門の表で確かめる)")
    years = collections.Counter(c[1] for c in sent)
    print(f"    候補の池 {len(cands)} 頁 / パイロットの作品を外して {len(pool)}"
          f" / **作者に見ていただく {len(sent)} 頁 / 年の札 {len(years)}**"
          f" —— 名前つきは `{OUT_FULL}`(`.gitignore` 配下)")
    print("      札ごと: " + " / ".join(f"{y} {n}" for y, n in sorted(years.items())))

    gates = [[g, n_pages, n_works, d if d != "" else "", ""]
             for g, n_pages, n_works, d in gate_rows]
    gates += [["", "", "", "", n] for n in NOTE_GATES]
    n1 = bc.write_csv(os.path.join(ROOT, "census", "colour_pilot_gates.csv"),
                      ["gate", "n_pages", "n_works", "n_dropped", "note"], gates)
    rows = [[f"{f:.4f}", n, hit, missed, fp, ""] for f, n, hit, missed, fp in chk]
    rows += [["", "", "", "", "", n] for n in NOTE_CHECK]
    n2 = bc.write_csv(os.path.join(ROOT, "census", "colour_ratio_vs_author.csv"),
                      ["floor_color_ratio", "n_author_class", "n_hit",
                       "n_missed", "n_false_positive", "note"], rows)

    # **作者の分類ごとの、列の実測の幅。** **これが「再現しない」の証拠である**
    by_class = collections.defaultdict(list)
    for p, c in AUTHOR_CLASS.items():
        if p in got:
            by_class[c].append(got[p])
    crows = [[c, len(v), f"{min(v):.4f}", f"{max(v):.4f}", ""]
             for c, v in sorted(by_class.items())]
    crows += [["", "", "", "", n] for n in NOTE_BY_CLASS]
    n3 = bc.write_csv(os.path.join(ROOT, "census",
                                   "colour_ratio_by_author_class.csv"),
                      ["author_class", "n_author_class", "color_ratio_min",
                       "color_ratio_max", "note"], crows)
    lrows = [[k, scope, n, hit, missed, fp, ""] for k, scope, n, hit, missed, fp in lchk]
    lrows += [["", "", "", "", "", "", x] for x in NOTE_LINEART]
    n4 = bc.write_csv(os.path.join(ROOT, "census", "lineart_count_vs_author.csv"),
                      ["max_n_lineart_visible", "judged_scope", "n_author_class", "n_hit",
                       "n_missed", "n_false_positive", "note"], lrows)
    print(f"    colour_pilot_gates.csv {n1}行 / colour_ratio_vs_author.csv {n2}行"
          f" / colour_ratio_by_author_class.csv {n3}行 / lineart_count_vs_author.csv {n4}行")
    print("    **Track M の依頼4 —— `n_lineart_visible <= k`(名前で数えた列)**:")
    print(f"      {'k':>3s} {'範囲':<32s} {'頁':>3s} {'的中':>4s} {'取りこぼし':>6s} {'カラーでないものを通す':>6s}")
    for k, scope, n, hit, missed, fp in lchk:
        print(f"      {k:>3d} {scope:<32s} {n:>3d} {hit:>4d} {missed:>6d} {fp:>6d}")
    for c in ("カラー",) + NOT_COLOUR:
        v = sorted(n_line[p] for p in n_line if AUTHOR_CLASS[p] == c)
        print(f"      作者の分類 {c}: {len(v)} 頁 / `n_lineart_visible` {v}")
    for c, v in sorted(by_class.items()):
        print(f"      作者の分類 {c}: {len(v)} 頁 / `color_ratio` "
              f"{min(v):.4f}〜{max(v):.4f}")
    return 0


NOTE_GATES = [
    "**選別は測定である**(規律9)—— **各行は前の行まで通ったものに次の条件を当てた数。**"
    "**落ちた数を隣に印字する。** **行を足し合わせない**(入れ子だから)",
    f"**床 {COLOUR_FLOOR} は「カラーの境」ではなく「候補を広げる篩」である** —— "
    "**`color_ratio` は作者の分類を再現しない**(`colour_ratio_vs_author.csv`)。"
    "**この床は、作者がカラーと裁定した頁のうち最も低いもの(0.0274)を落とす** —— "
    "**だから候補は「カラーの全部」ではない**",
    "**`color_ratio` はプレビューの彩度で、抽出側が測った列である**(条例5)。"
    "**断面は 2026-10-26 17:57** —— **`page_colour` は 7,634 行で、母集団 9,653 より少ない**"
    "(`.lip` と読めなかった頁が入らない)。**差を「彩度が無い頁」と読まない**",
    "**`線画` を含む可視の葉は名前の述語である**(T106)—— **中身の確認ではない。** "
    "**可視は `visibility & 1` で見た**(抽出側の実測、2026-10-31: "
    "**1の位が表示、2の位がマスクあり。`3` は表示である**)。"
    "**祖先のフォルダーの可視は見ていない** —— 葉だけの判定である",
    "**`rendering_*` を外したのは3D の線を避けるためである**(作者: "
    "**3D の線は「線」だが「キャラクターの線」ではなく、解明は別路線**)。"
    "**名前で外した** —— **形式では外せない**(抽出側、2026-10-31: "
    "**形式2 にも `rendering_*` が 22層在る**)",
    "**【2026-11-02】作者の恒久の除外の門を足した**(作者の裁定。**棒ではない** —— 点数と独立)。"
    "**落ちたのは1頁で、後ろの門の落ちる数は前と同じ**(双子 5 / 同じ作品フォルダー 3 / 同じ作品名 1 / 年の札 13。"
    "**作品フォルダーごとの門だけ 92 → 91** —— 除外した1頁の分)。**名簿と理由は公開物に出さない。** "
    "**11-01 に作者に送った8点の記録は前の表のまま残す**(名前つきは全量側)",
    "**この表に無い条件が1つ在る** —— **「制作工程のフローを守れているか」は列にならない。** "
    "Track M が申告し、**作者が「こればかりは仕方ない」と裁定した**(2026-10-31)。"
    "**但し書きで判断する**(`page_notes.csv`)。**だから最終の5点は作者に見ていただく**",
]

NOTE_CHECK = [
    "**規律13 の点検**: **同定したい種別の列が無いとき、近い列で代用しない。** "
    "**使う前にその列が作者の分類を再現すると示し、的中率と取りこぼしを印字する。** "
    "**この表はその印字である** —— **示せなかった。**",
    "**決定的な2頁**: 作者が**モノクロ漫画**と裁定した頁の `color_ratio` は **0.0646**、"
    "**カラーとして残っている**頁は **0.0274**。**モノクロの方が高い** —— "
    "**床をどこに置いても、この2頁は同じ側に落ちる。**",
    "**【2026-11-02】裁定の在る頁は 14 になった**(作者が候補8点を目で見た。Track M 経由)。"
    "**カラーでない側は「モノクロ漫画」3頁と「カラーだが色が入っていない」1頁** —— "
    "**分類としては別の種類だが、この点検ではどちらも「カラーでない」に数える**"
    "(分類ごとの幅は `colour_ratio_by_author_class.csv`)。"
    "**作者がカラーと分類したが Track M が採用しなかった頁は、ここでは「カラー」に数える** —— "
    "**採用しなかった理由は彩度と独立なので、点検を曲げない**(規律9)",
    "**【11-01 の読み、残す】母集団は6頁である**(裁定の在る頁)。**小さい。** "
    "**`線画` が0枚で外れた2頁には、カラーかどうかの裁定が無い** —— "
    "**「カラーでない」ではなく「聞かれていない」**(条例4 の家族)。"
    "**6頁で分離できないことは、母集団全体で分離できないことの証明ではない** —— "
    "**だが代用してよい根拠にもならない**(規律13 は再現を示す側に証明の責任を置く)",
    "**的中・取りこぼし・モノクロを通す数を別々に数えている** —— "
    "**引き算で片方を出すと、どちらの誤りか分からなくなる。**",
]

NOTE_LINEART = [
    "**Track M の依頼4**(2026-11-01): **後知恵の仮説「`n_lineart_visible <= 5` が内容の裁定と一致する」を、"
    "裁定の在る14頁で当てた。** **的中・取りこぼし・カラーでないものを通す数を別々に**印字する(規律13)",
    "**この列は名前で数えた列である** —— **`線画` を名前に含む可視の葉**(`visibility & 1`。"
    "**祖先のフォルダーの可視は見ていない**)。**中身が線画かは見ていない**(T106)",
    "**仮説は候補8点を見た後に立てた** —— **だから8点の上の一致は検証ではない。** "
    "**独立なのは「仮説を立てるのに使っていない頁」(パイロットの6頁)の行だけである**",
    "**読み**: **14頁では `<= 5` は 的中5 / 取りこぼし5 / カラーでないものを通す 0。** "
    "**パイロットの6頁では 的中1 / 取りこぼし4** —— **作者がカラーと裁定した頁の数は 1〜31 に散り、"
    "カラーでない4頁(13 / 16 / 18 / 27)はその幅の内側に在る。** "
    "**どの k でも両方の誤りを 0 にできない** —— **同定の列としては出さない**(`color_ratio` と同じ結論)",
    "**片側だけは言える形をしている**: **k が 12 以下なら、カラーでない頁を1つも通さない** —— "
    "**ただしそれは14頁での話で、カラー10頁のうち4頁を落とす。** **篩としても、作者に見ていただく前の"
    "絞り込みにしか使えない**",
]

NOTE_BY_CLASS = [
    "**`color_ratio` の幅が、作者の分類の間で重なっている** —— "
    "**これが「再現しない」の実体である。** 分けられる列なら、幅は重ならない",
    "**【2026-11-02、14頁】入れ子である** —— **カラーでない4頁(0.0646〜0.9509)は、"
    "カラー10頁の幅(0.0274〜0.9726)の内側に全部入る。** **上の床でも下の床でも1本の閾値では分けられない** —— "
    "**向きの問題ではないので、床の探索は閉じる**(Track M が測り、census が自分の鍵で引き直して一致)。"
    "**色が入っていない頁の値が 0.9337** —— **なぜ分けないかは測っていない**"
    "(列の定義の1行は抽出側の持ち物。**Track M の推測**: プレビューの圧縮の色ノイズで飽和しうる。**未測定**)",
    "**`**裁定なし**` の2頁は、`線画` を名前に含む可視の層が0枚で"
    "Track M の母集団に入る前に外れた頁である** —— "
    "**カラーかどうかを聞かれていない。** **「カラーでない」ではない**(条例4 の家族)。"
    "**しかもその2頁のうち1頁は `color_ratio` が最も高い群に入る** —— "
    "**列だけで候補を選べば、作者が線画の無さで外した頁を選び直すことになる**",
    "**値は抽出側がプレビューから測ったもの**(条例5)。**断面 2026-10-26 17:57**",
]

if __name__ == "__main__":
    raise SystemExit(main())

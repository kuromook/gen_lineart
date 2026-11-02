"""**作者の恒久の除外の名簿**を読む(基盤の依頼、2026-11-02)。

**名簿そのものは `data/census_full/author_exclusions.csv`(`.gitignore` 配下)に在る** ——
**頁のパスと除外を同じ公開物に並べない**(条例10)。**理由も公開物に書かない**
(作者の裁定: 内容を書かない。参照は番号とパスだけ)。

**名簿が無いときは止まる。** **黙って空の名簿で走ると、除外した頁が次の候補の便に戻る** ——
**その頁は彩度の篩を通るので、黙っていると再び上がってくる**(Track M、2026-11-01)。
**「名簿が無い」と「除外が 0件」を区別できない器具は、候補を作ってはいけない。**

**この名簿が覆うもの**(名前より狭く書く。T132): **census が名前つきの頁を出す器具**
(候補・発注の表)。**全数の表(年・形式・種別ごとの件数)からは外していない** ——
**全数の表は頁を名指ししないので出方の問題が無く、外すと総数 9,653 の保存が切れる。**
**外すかは作者の判断として基盤に聞いてある**(2026-11-02)。

**解除は作者の指示だけ。** Claude はこの名簿から行を消さない。
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTER = "data/census_full/author_exclusions.csv"
COLUMNS = ["page_path", "ruled_by", "ruled_on", "scope", "release_by",
           "verbatim", "source"]


def load() -> dict:
    """`page_path` → 行。**名簿が無い・列が違う・裁定者が作者でない行が在れば止まる。**"""
    path = os.path.join(ROOT, REGISTER)
    if not os.path.isfile(path):
        raise SystemExit(f"**作者の除外の名簿が無い: {REGISTER}** —— "
                         "**候補を作らない**(名簿が無いと除外した頁が戻る)")
    with open(path, encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        if r.fieldnames != COLUMNS:
            raise SystemExit(f"**除外の名簿の列が違う**: {r.fieldnames}")
        rows = list(r)
    bad = [x["page_path"] for x in rows if x["ruled_by"] != "作者"]
    if bad:
        raise SystemExit(f"**作者でない者の除外が {len(bad)} 行在る** —— "
                         "**この名簿は作者の裁定だけを持つ**")
    return {x["page_path"]: x for x in rows}

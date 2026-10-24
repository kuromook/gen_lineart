"""M15: **納品の CSV にはフォルダー行が `parent_id` と `order_in_parent` つきで
入っている。** manifest(パイロット)はフォルダー行を落としているので、
こちらは「フォルダを跨ぐ z 順は無い」と思って標準を作っていた。

効くところが2つある:

  1. **追補7 R3 の根拠**。重なりの解決を「面積の小さい成分が勝つ」にした理由は
     「フォルダを跨ぐ z 順がマニフェストに無い」だった。**納品には在る。**
     → 根拠は manifest については正しく、**納品については誤り**。
  2. **シルエット層の同定**。`m11.clip_bases` は manifest の層だけを兄弟にして
     クリップの連鎖のベースを取るので、**フォルダーや描画画素0の層がベースの
     場合を見落とす**。manifest で数えると **303のクリップ層のうち24は、
     同じ親の中に自分より下の `clip=0` 兄弟が無い**(= ベースが一覧に無い)。

このファイルは**納品の CSV だけから木を組み直して**、同定がどう変わるかを数える。
既知セル: **manifest 由来の 167層(追補6 §F2)が、この同定でどうなるかを層単位で
突き合わせる**(消えた層・増えた層を両方印字する)。

MEASURES ONLY。腕は採点しない。標準をどう変えるかは、この数を見て**登録として
別に書く**(検定集合の前に閉じる)。
"""
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m11_silhouette_components import clip_bases as manifest_clip_bases  # noqa

CSV = Path('/home/sh1/deepl/lineart/inbox/m_colour_page_layers_known8.csv')
MANIFEST = Path('/home/sh1/deepl/lineart/inbox/colour_layers_manifest_pilot.json')
PAPER_TYPE = '1584'


def rows_by_page():
    with open(CSV, encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r['page_path']].append(r)
    return byp


def tree_clip_bases(rows):
    """納品の全行(フォルダー込み)を兄弟にして、クリップの連鎖のベースを取る。

    規則は `m11.clip_bases` と同じ(同じ親の中で `order_in_parent` 昇順、
    `clip=0` が現れたらそれがベース、続く `clip=1` はそのベースに付く)。
    違いは**兄弟の集合にフォルダーと描画画素0の層が入る**ことだけである。
    用紙(`layer_type` 1584)は m11 と同じく兄弟から外す。
    """
    sib = collections.defaultdict(list)
    for r in rows:
        if r['layer_type'] == PAPER_TYPE or r['parent_id'] == '':
            continue
        sib[int(r['parent_id'])].append(r)
    bases, chains, orphan = {}, 0, 0
    for group in sib.values():
        group.sort(key=lambda r: int(r['order_in_parent']))
        base = None
        for r in group:
            if r['clip'] == '0':
                base = r
            else:
                chains += 1
                if base is None:
                    orphan += 1
                else:
                    bases[int(base['layer_id'])] = base
    return bases, chains, orphan


def main():
    byp = rows_by_page()
    man = {p['source']: p for p in json.load(
        open(MANIFEST, encoding='utf-8'))['pages']}
    tot = collections.Counter()
    print(f'{"page":30s} {"manifest":>9} {"納品の木":>9} {"共通":>5} '
          f'{"消":>4} {"増":>4} {"増のうちフォルダー":>10} {"増のうち描画0":>8} {"孤立":>5}')
    for src, rows in sorted(byp.items()):
        p = man[src]
        mb = manifest_clip_bases(p)
        tb, chains, orphan = tree_clip_bases(rows)
        drawn = {int(l['layer_id']) for l in p['layers']}
        folders = {int(r['layer_id']) for r in rows if r['folder'] == 'True'}
        lost = sorted(set(mb) - set(tb))
        gained = sorted(set(tb) - set(mb))
        g_fold = [i for i in gained if i in folders]
        g_zero = [i for i in gained if i not in folders and i not in drawn]
        tot.update({'manifest': len(mb), 'tree': len(tb),
                    'both': len(set(mb) & set(tb)), 'lost': len(lost),
                    'gained': len(gained), 'g_fold': len(g_fold),
                    'g_zero': len(g_zero), 'orphan': orphan, 'chains': chains})
        print(f'{p["page"][-28:]:30s} {len(mb):>9} {len(tb):>9} '
              f'{len(set(mb)&set(tb)):>5} {len(lost):>4} {len(gained):>4} '
              f'{len(g_fold):>10} {len(g_zero):>8} {orphan:>5}')
        if lost:
            print(f'    消えた層 {lost}')
        if g_fold:
            nm = {int(r["layer_id"]): r["name"] for r in rows}
            print(f'    フォルダーがベース: '
                  f'{[(i, nm[i]) for i in g_fold][:6]}')
    print(f'\n合計 manifest {tot["manifest"]} / 納品の木 {tot["tree"]} / '
          f'共通 {tot["both"]} / 消 {tot["lost"]} / 増 {tot["gained"]} '
          f'(フォルダー {tot["g_fold"]} / 描画0 {tot["g_zero"]}) / '
          f'クリップ層 {tot["chains"]} / 孤立 {tot["orphan"]}')
    print('\n腕は採点していない。標準の変更は登録として別に書く。')
    return 0


if __name__ == '__main__':
    sys.exit(main())

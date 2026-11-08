# -*- coding: utf-8 -*-
"""L38 **分解の台に溜まった裁定を読む**(記述。2026-11-08)。

**これは点数ではない。作者の裁定の集めである**(規律6)。**棒は登録していない** ——
腕W の採点は別の走行で、`r` を線の層から引いてから(登録 `doc/l38_decompose_registration_20261108.md` §6)。

- **最後の行が有効。前の行は消さない**(規律7)。**開き直して保存し直した行(筆 0)も数える**
- **母集団を必ず書く** —— ここに在るのは**作者が「分解しないといけない」と答えた面だけ**。
  **面プール全体の比ではない**(規律7: 分子・分母・母集団を並べる)
- **名前が札から来たか手入力かを分けて数える** —— 札は私が画面に出した語なので、**寄っている**
"""
import collections
import json
from pathlib import Path

D = Path('/home/sh1/deepl/lineart-fill-region/data/l38_decompose_20261108')


def main():
    raw = [json.loads(x) for x in open(D / 'l38_decompose_rulings.jsonl', encoding='utf-8') if x.strip()]
    last = {}
    for r in raw:
        last[r['item_id']] = r
    kmap = json.load(open(D / 'map.json', encoding='utf-8'))
    print(f'裁定の行 {len(raw)} / 単位 {len(last)}(**最後の行が有効。前の行は消していない**)')
    again = [r['item_id'] for r in raw if r.get('strokes') == 0 and r['value'] == 'split']
    print(f'  うち**開き直して保存し直した行 {len(again)}**(筆 0 = 戻した区分けをそのまま保存)')
    print(f'  値: {dict(collections.Counter(r["value"] for r in last.values()))}')
    # **時間と筆の数は「最初に区分けした行」から取る** ——
    # 最後の行は戻してから保存し直した行(筆 0・数秒)なので、**作業の手間を測れない**
    first = {}
    for r in raw:
        first.setdefault(r['item_id'], r)
    secs = sorted(first[k].get('seconds', 0) for k in last)
    print(f'  かけた時間(**最初に区分けした行から**。最後の行は戻した後の保存なので使えない): '
          f'合計 {sum(secs)} 秒 / 1単位の中央 {secs[len(secs) // 2]} 秒')
    print(f'  筆の数: 合計 {sum(first[k].get("strokes", 0) for k in last)} / '
          f'1単位の中央 {sorted(first[k].get("strokes", 0) for k in last)[len(last) // 2]}')

    print('\n--- 1単位が何個の部品になったか ---')
    for n, c in sorted(collections.Counter(r['n_parts'] for r in last.values()).items()):
        print(f'  {n} 部品   {c} 単位')
    tot = sum(r['n_parts'] for r in last.values())
    print(f'  **{len(last)} 面 -> {tot} 部品**(単位あたり {tot / len(last):.1f})')

    print('\n--- 名前(**個数と面積。札から来たか手入力かを分ける**)---')
    cnt, area, src = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    for r in last.values():
        for q in r.get('parts', []):
            nm = q.get('name') or '(名前なし)'
            cnt[nm] += 1
            area[nm] += q.get('area', 0)
            src[nm][q.get('name_src') or '(不明)'] += 1
    ta = sum(area.values())
    for nm, c in cnt.most_common():
        s = ' / '.join(f'{k} {v}' for k, v in src[nm].most_common())
        print(f'  {nm:<16} 部品 {c:>3}   面積 {area[nm]:>10,}({area[nm] / ta:5.1%})   [{s}]')
    print(f'  合計 面積 {ta:,}(**この{len(last)}面の中での比。面プール全体の比ではない**)')
    print('\n--- 名前ごとの**1部品の大きさ**(床 6,000 と比べる)---')
    import statistics
    sz = collections.defaultdict(list)
    for r in last.values():
        for q in r.get('parts', []):
            sz[q.get('name') or '(名前なし)'].append(q.get('area', 0))
    for nm, c in cnt.most_common():
        a = sorted(sz[nm])
        below = sum(1 for x in a if x < 6000)
        print(f'  {nm:<16} 中央 {statistics.median(a):>9,.0f}  最小 {a[0]:>8,}  最大 {a[-1]:>9,}'
              f'   **床 6,000 未満 {below}/{len(a)}**')
    print('\n--- 単位ごとの部品の名前 ---')
    for fid, r in sorted(last.items()):
        nm = [f'{q.get("name") or "(名前なし)"} {q.get("area", 0):,}' for q in r.get('parts', [])]
        print(f'  {fid}  面 {kmap[fid]["area"]:>9,}  帯の覆い {kmap[fid]["band_cov"]:.2f}  '
              f'{r["n_parts"]} 部品: ' + ' / '.join(nm))

    print('\n--- 画素の行き先(面積が合うか)---')
    lost = 0
    for fid, r in sorted(last.items()):
        pa = sum(q.get('area', 0) for q in r.get('parts', []))
        want = kmap[fid]['area']
        lost += want - pa
        print(f'  {fid}  面 {want:>9,} -> 部品の和 {pa:>9,}  '
              f'切り離した画素 {want - pa:>8,}({(want - pa) / want:5.1%})  筆 {r.get("strokes")}')
    print(f'  **切り離した画素の合計 {lost:,}**(= 作者が面から外した分。'
          f'**部品にも入らず、面にも残らない**)')

    print('\n--- 一言の欄 ---')
    for fid, r in sorted(last.items()):
        if r.get('comment'):
            print(f'  {fid}: {r["comment"]}')

    print('\n**次**: 腕W の `r` を**線の層から**引き、この裁定を相手に採点する'
          '(登録 §6。**まだ引いていない**)。**この表は棒ではない**')


if __name__ == '__main__':
    main()

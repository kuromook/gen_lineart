#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**追補96: 追加の行列(依頼10 の納品)。m56 をそのまま使い、母集団だけ差し替える。**

**m56 の既知セルの門は「45頁 / 315層」で、これは依頼5 の母集団への期待値である。**
**門を緩めない。** 別の母集団には**別の期待値を登録して**、別の器具として走らせる。

**登録する期待値(走らせる前に書く。実体は追補96 §A の取り込みの点検で確認済み)**:

| | 期待 |
|---|---:|
| 納品の頁 | **38** |
| 納品の層(書かれた PNG) | **16** |
| 用紙を外した問い | **11** |
| 問いの集合 | **追補94 で予測した11層と一致すること**(集合として) |

**絵・問い・選択肢・門は m56 のまま。何も変えない**(追補88 §E:
「問いの文面と UI は変えない。標本だけを直す」)。

  <venv>/python tools/ui/m71_build_layer_kind_queue_supp.py
"""
import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m56_build_layer_kind_queue as m56                      # noqa: E402

DELIV = ('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_request10')
OUT = pathlib.Path('results/m71_ui')
PRED = pathlib.Path('results/m70_order_set/order_set_v2.csv')
LEDGER = 'inbox/m_colour_page_layers_872.csv'


def predicted():
    """追補94 で予測した11層(用紙を外した表で `drawn_ratio > 0`)。"""
    out = set()
    for r in csv.DictReader(PRED.open(encoding='utf-8')):
        try:
            if float(r['drawn_ratio'] or 0) > 0:
                out.add((r['page_path'], str(r['layer_id'])))
        except ValueError:
            pass
    return out


def main():
    # **母集団の期待値を差し替える。緩めるのではなく、別の値を登録して置く**
    m56.EXPECT_PAGES = 38
    m56.EXPECT_LAYERS = 16
    m56.OUT = OUT
    (OUT / 'crops').mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], '--deliv', DELIV]
    rc = m56.main()
    if rc:
        return rc

    # ---- 用紙を**台帳の `layer_type` で**外す(追補96 §B の欠陥の直し)----
    # **m56 は `formats == 'paper'` で外している。台帳では用紙861層のうち118層
    # (13.7%)が `formats` 空**なので、**未納品の用紙は漏れる** ——
    # 追補92 の欠陥2(用紙を納品の印で同定した)と同じ族である。
    # **逆向きは 0**(`formats == 'paper'` で 1584 でない層は無い)ので、
    # **`layer_type == 1584` が健全な同定**である。
    led = {}
    for r in csv.DictReader(open(LEDGER, encoding='utf-8')):
        led[(r['page_path'], str(r['layer_id']))] = r
    q = json.loads((OUT / 'queue.json').read_text(encoding='utf-8'))
    items = q['queue']
    paper = [it for it in items
             if led.get((it['pid'], str(it['lid'])), {}).get('layer_type') == '1584']
    print(f'\n== 用紙を台帳の `layer_type` で外す ==')
    print(f'  行列 {len(items)} 問 → **用紙 {len(paper)} 問を外す** → {len(items)-len(paper)} 問')
    print('  (m56 の `formats` だけでは外れなかった層。**作者の裁定で母集団の外**)')
    items = [it for it in items if it not in paper]
    for i, it in enumerate(items):
        it['order'] = i
    q['queue'] = items
    for k in ('n', 'n_ask', 'n_all'):
        if k in q:
            q[k] = len(items)
    (OUT / 'queue.json').write_text(
        json.dumps(q, ensure_ascii=False), encoding='utf-8')
    print(f'  書き直した: {OUT / "queue.json"}  ({len(items)} 問)')

    got = {(it['pid'], str(it['lid'])) for it in items}
    pred = predicted()
    print(f'\n== 追補96 の門: 問いの集合が追補94 の予測と一致するか ==')
    print(f'  問い {len(items)} / 予測 {len(pred)}')
    print(f'  **集合として一致: {got == pred}**')
    if got != pred:
        print(f'  予測にしか無い {len(pred - got)} / 行列にしか無い {len(got - pred)}')
        print('  **一致しないなら、どちらが正しいかを先に調べる**')
        return 1
    print('  **PASS**')
    return 0


if __name__ == '__main__':
    sys.exit(main())

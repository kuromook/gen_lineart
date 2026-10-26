"""M49: **流用したサーバの JS が読む鍵と、こちらの行列の鍵を突き合わせる**(読み取り専用)。

2026-10-26、作者から「回答のボタンが無く、ショートカットキーも反応しない」。
原因は `m47` の行列に **`area` / `scale` / `ctx_scale` の3つが無かった**こと。
`it.area.toLocaleString()` が TypeError を投げて `go()` がそこで中断し、
**解答欄も「その他」も DOM に足されず、`keymap`/`okeys` が空のまま**になっていた。
**症状の両方(ボタンが無い・キーが効かない)が1つの原因から出る。**

**この罠の形**: **器具を流用したとき、行列の鍵の名前が唯一の接点**である。
「起動の門を通った / JS は構文検査を通った / HTTP 200」は、
**どれも `it.area` が undefined かどうかを知らない**。点検は鍵の名前に当てる。

```
./venv/bin/python tools/ui/m49_check_queue_keys.py results/m47_ui/queue.json
```

**絵の側も同じ便で点検する**(2026-10-26 に鍵とは別に3件見つかったため):
門(0) 窓が単色 / 門(1) 輪郭が窓の 10% 超 / 門(2) 窓の内側にマゼンタが無い。
**m47 の中の門と同じ述語を、出来上がった行列の PNG に対してもう一度当てる** ——
器具の中で通ったことと、配る物が通っていることは別である。

終了コード 1 = 足りない鍵がある / 読めない絵がある(配らない)。
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'colour'))
import m27_author_crops as m27                           # noqa: E402

MAXMG = 0.10

SERVER = Path('../lineart-fill-region/tools/ui/l19_fill_label_server.py')


def main():
    """`m49 <queue.json> [起動器のモジュール名] [絵の門: magenta|none]`

    **第2引数**: 起動器が `PAGE` を差し替えるなら、そのモジュール名
    (`m54_launch_page_kind`)。**差し替えた後の画面から鍵を拾う** ——
    **点検は「実際に配られる画面」に当てなければ意味が無い。**
    **第3引数**: 絵の門。`magenta` は m47 用(成分の輪郭を描く行列)。
    **`none` は「この行列には当たらない」ことを明示して飛ばす**
    (黙って呼ばないのが T110。**飛ばすときも理由を印字する**)。
    """
    qp = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/m47_ui/queue.json')
    launcher = sys.argv[2] if len(sys.argv) > 2 else None
    jssrc = None
    pics = sys.argv[3] if len(sys.argv) > 3 else 'magenta'
    src = SERVER.read_text(encoding='utf-8')
    if launcher:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        mod = __import__(launcher)
        if not hasattr(mod, 'patch'):
            raise SystemExit(f'**{launcher} に `patch()` が無い。点検しない**')
        import l19_fill_label_server as _l19
        jssrc = mod.patch(_l19.PAGE)
        print(f'**画面は {launcher} が差し替えた後のもので点検する**')
        # **原文を置換してはいけない**: `PAGE` の中の `\n` は原文では `\\n` と
        # 書かれているので、`src.replace(PAGE, page)` は**一致せず無言で何もしない**
        # (追補40 §F2。差し替えたつもりで元の鍵を点検していた)。
        # **組み立てた文字列そのものを読む。**
    # JS が読む鍵は**画面から**、`payload()` が足す鍵は**原文から**拾う
    js = sorted(set(re.findall(r'\bit\.([A-Za-z_][A-Za-z0-9_]*)',
                               jssrc if launcher else src)))
    # payload() が足す鍵(`r['x'] = ...` と `r['x'], r['y'] = ...`)
    body = src[src.index('def payload('):src.index('def flag(')]
    added = set()
    for m in re.finditer(r"r\[[\'\"]([A-Za-z_0-9]+)[\'\"]\]", body):
        added.add(m.group(1))
    q = json.loads(qp.read_text(encoding='utf-8'))
    items = q['queue']
    print(f'サーバ  {SERVER}')
    print(f'行列    {qp}  ({len(items)} 問)')
    print(f'\nJS が読む鍵 {len(js)}: {", ".join(js)}')
    print(f'payload() が足す鍵 {len(sorted(added))}: {", ".join(sorted(added))}')
    need = [k for k in js if k not in added]
    print(f'\n**行列が入れなければならない鍵 {len(need)}**: {", ".join(need)}')
    bad = 0
    for k in need:
        miss = [i for i, it in enumerate(items) if k not in it]
        none = [i for i, it in enumerate(items)
                if k in it and it[k] is None]
        if miss:
            bad += 1
            print(f'  **{k:12s} 無い {len(miss)}/{len(items)} 問** '
                  f'(例 order {miss[:5]})')
        elif none:
            bad += 1
            print(f'  **{k:12s} null {len(none)}/{len(items)} 問** '
                  f'(例 order {none[:5]})')
        else:
            print(f'  {k:12s} OK')
    # 型の点検: JS が数として使う鍵
    for k, why in (('area', '.toLocaleString()'), ('scale', 'sc<1 の比較'),
                   ('ctx_scale', 'sc<1 の比較')):
        if k in need:
            t = {type(it.get(k)).__name__ for it in items}
            if t - {'int', 'float'}:
                bad += 1
                print(f'  **{k} の型が数でない**: {t}({why})')
    # ---- 絵の点検(門(0)(1)(2) を、配る PNG に当て直す)----
    if pics == 'none':
        print('\n== 絵の点検: **飛ばす** ==')
        print('  門(0) 窓が単色 / 門(1) 輪郭 10%超 / 門(2) 内側にマゼンタ無し は、')
        print('  **成分の輪郭をマゼンタで描く行列のための門**である。')
        print('  この行列は頁ぜんたいを出すのでマゼンタが無く、紙が大半なので単色に見える。')
        print('  **当たらないので当てない。当てれば 44/44 が落ちる**'
              '(追補40 §C で実際に当てて確かめた)。')
        print(f'\n{"**FAIL** 配らない" if bad else "PASS"}  ({bad} 件)')
        return 1 if bad else 0
    print(f'\n== 絵の点検({len(items)} 枚)==')
    crops = qp.parent / 'crops'
    print(f'{"crop":58s} {"色":>3} {"輪郭":>7} {"内側":>8}')
    for it in items:
        f = crops / it['crop']
        im = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if im is None:
            bad += 1
            print(f'  **{it["crop"]} が無い**')
            continue
        h, w = im.shape[:2]
        nc = int((np.unique(m27.quant(im), return_counts=True)[1] / (h * w)
                  >= 0.05).sum())
        mg = (im[..., 2] > 200) & (im[..., 0] > 200) & (im[..., 1] < 60)
        inner = mg.copy()
        inner[:6, :] = False; inner[-6:, :] = False
        inner[:, :6] = False; inner[:, -6:] = False
        why = []
        if nc < 2:
            why.append('**窓が単色**')
        if float(mg.mean()) > MAXMG:
            why.append(f'**輪郭が {mg.mean():.1%}**')
        if inner.sum() == 0:
            why.append('**内側にマゼンタ無し**')
        if why:
            bad += 1
        print(f'  {it["crop"][:56]:56s} {nc:>3} {mg.mean():>7.2%} '
              f'{int(inner.sum()):>8}  {" / ".join(why)}')
    print(f'\n{"**FAIL** 配らない" if bad else "PASS"}  ({bad} 件)')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())

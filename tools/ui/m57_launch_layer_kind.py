"""M57: **「この層は何ですか」の UI を立ち上げる**(L のサーバは書き換えない)。

`m50` / `m54` と同じ形。**差し替えるのは画面の文5か所**で、
**置換元が見つからなければ起動しない。**
**`m54` は触らない**(別の行列のための起動器。T110: 器具を作り直すと前の門が黙って消える)。

**`m54` との違い**: この UI は **`C` で頁ぜんたいに切り替わる**(既定が層なので、向きが逆)。
`m54` は既定が頁で `C` が寄る絵だった —— **同じ文にできない。**

```
./venv/bin/python tools/ui/m57_launch_layer_kind.py \
    --host 0.0.0.0 --port 8769 --queue results/m56_ui/queue.json \
    --save labels/layer_kind.jsonl --extra ''
```
"""
import re
import sys
from pathlib import Path

MTREE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(MTREE.parent / 'lineart-fill-region/tools/ui'))
import l19_fill_label_server as l19                       # noqa: E402

for name in ('ROOT', 'LAB', 'load_state', 'main', 'S', 'PAGE'):
    if not hasattr(l19, name):
        raise SystemExit(f'**L のサーバに `{name}` が無い。起動しない** '
                         f'(流用の接点が変わった。m57 を読み直すこと)')

l19.ROOT = MTREE
l19.LAB = MTREE / 'labels'
l19.LAB.mkdir(exist_ok=True)

NEW_INFO = """function info(){
  if(!it) return;
  const sc=ctx?it.ctx_scale:it.scale;
  document.getElementById('info').textContent=
    (ctx?'[頁ぜんたい] ':'[いま聞いている層] ')+it.work+'   '+it.pid+
    (sc==null?'  (表示倍率 不明)':'  (元の原稿に対して '+(sc*100).toFixed(1)+'%)')+
    (it.saved?'  ← 記録済み: '+(it.saved.word||it.saved.free||it.saved.choice):'')+
    (it.flagged?'   ⚑ 異常として報告済み: '+it.flagged.note:'');
}"""


def patch(page):
    """5か所を差し替える。**1つでも見つからなければ SystemExit。**"""
    done = []
    m = re.search(r'function info\(\)\{.*?\n\}', page, re.S)
    if not m:
        raise SystemExit('**`info()` が見つからない。起動しない**')
    old = m.group(0)
    for mark in ('it.comp', 'it.region', 'it.area.toLocaleString'):
        if mark not in old:
            raise SystemExit(f'**`info()` の中身が変わっている(`{mark}` が無い)。'
                             f'起動しない** —— m57 を読み直すこと')
    page = page.replace(old, NEW_INFO); done.append('info()')

    pairs = [
        ("prompt('この黒は何ですか(語を入力)')",
         "prompt('この層は何ですか(一言で)')"),
        ('<b id="ask">この黒いベタは何?</b>',
         '<b id="ask">この層は何ですか。</b>'),
        ('<div class="meta" id="help">黒い塊が聞いている面。'
         '色は縁 4px に接している線のレイヤ(ヒント。答ではない)。\n'
         '工程の産物(隙間埋め)なら G / H。</div>',
         '<div class="meta" id="help">左端がこの頁の層で、'
         'いま聞いている層に枠が付いています。</div>'),
        # **`m54` とは向きが逆**: あちらは既定が頁で `C` が寄る絵、
        # こちらは**既定が層で `C` が頁ぜんたい**
        ('<button onclick="ctx=!ctx;showImg()">コマ全体 <span class="k">C</span>'
         '</button>',
         '<button onclick="ctx=!ctx;showImg()">頁ぜんたい / 戻る '
         '<span class="k">C</span></button>'),
    ]
    for a, b in pairs:
        if a not in page:
            raise SystemExit(f'**置換元が見つからない: {a[:40]}... 起動しない**')
        page = page.replace(a, b); done.append(a[:26])
    print('  画面の文を差し替えた: ' + ' / '.join(done), flush=True)
    return page


def install():
    l19.PAGE = patch(l19.PAGE)
    _orig = l19.load_state

    def load_state(save_path, *a, **k):
        """**門**: 読み書きする2つのファイルが M の木の下に在ることを確かめる。"""
        _orig(save_path, *a, **k)
        for key, what in (('save', '答の保存先'), ('flagfile', '異常の報告')):
            q = Path(l19.S[key]).resolve()
            if MTREE not in q.parents:
                raise SystemExit(f'**{what} が M の木の外: {q}。起動しない**')
            print(f'  {what}: {q}', flush=True)
        if l19.S['flag']:
            print(f'  **注意: 異常の既報が {len(l19.S["flag"])} 件ある。'
                  f'`order` で引かれるので、別の行列の注記が出うる**', flush=True)
        print('  **門: 保存先も異常の報告も M の木の下**(L の木には書かない)',
              flush=True)

    l19.load_state = load_state
    return l19.PAGE


if __name__ == '__main__':
    install()
    l19.main()

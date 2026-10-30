#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m94: **3案から選ぶ UI を立ち上げる**(追補144)。器具②-b。

`m57` と同じ形 —— **L のサーバ `l19` を書き換えず、画面の文と継ぎ目だけ差し替える。**
**置換元が1つでも見つからなければ起動しない**(T110: 器具を作り直すと前の門が黙って消える)。

**`m57` との違い**: この UI は**同じ窓の3つの分け方を同じ場所で差し替える**。
`l19` は絵の口が2つ(`/crop` と `/ctx`)しか無いので、**`/alt` を1つ足す**
(`H` を継いで `do_GET` を被せるだけ。`l19` の `H` は触らない)。

**キーの割り当て**(`b` = 前へ / `c` = 切替 / `n` = 次へ は `l19` が既に使っている):
  **スペース または → : A → B → C を巡回**(**同じ場所で差し替えるので見比べられる**)
  **C: 線なしの絵**(m27 の作法 —— 判定は線なしで、問いは線ありで示す)
  **1 / 2 / 3 / 4: A が近い / B が近い / C が近い / = 同じくらい**

**縮小しない**(NC3): `#img` の `max-width:100%` を **`none`** にする。
**画面が狭いときは縮小ではなく横スクロールになる。**

```
../lineart/venv/bin/python tools/ui/m94_launch_division.py \
    --host 0.0.0.0 --port 8771 --queue results/m93_ui/queue.json \
    --save labels/author_division.jsonl --extra ''
```
"""
import re
import sys
from pathlib import Path

MTREE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(MTREE.parent / 'lineart-fill-region/tools/ui'))
import l19_fill_label_server as l19                       # noqa: E402

for name in ('ROOT', 'LAB', 'load_state', 'main', 'S', 'PAGE', 'H', 'payload'):
    if not hasattr(l19, name):
        raise SystemExit(f'**L のサーバに `{name}` が無い。起動しない** '
                         f'(流用の接点が変わった。m94 を読み直すこと)')

l19.ROOT = MTREE
l19.LAB = MTREE / 'labels'
l19.LAB.mkdir(exist_ok=True)

NEW_SHOW = """function showImg(){
  const n=(it.alt||[]).length;
  if(ctx||!n){document.getElementById('img').src='/ctx?o='+cur+'&t='+Date.now();}
  else{document.getElementById('img').src='/alt?o='+cur+'&i='+slot+'&t='+Date.now();}
  info();
}"""

NEW_INFO = """function info(){
  if(!it) return;
  const nm=['A','B','C'];
  document.getElementById('info').textContent=
    (ctx?'[線なしの絵] ':'[分け方 '+(nm[slot]||slot)+'] ')+it.work+'  '+it.pid+
    '   スペース または → で A/B/C を差し替え、C で線なし'+
    '   (原寸 1:1。縮小していません)'+
    (it.saved?'  \\u2190 記録済み: '+(it.saved.word||it.saved.free||it.saved.choice):'')+
    (it.flagged?'   \\u2691 異常として報告済み: '+it.flagged.note:'');
}"""


def patch(page):
    """**1つでも見つからなければ SystemExit。** 数は印字する(規律9)。"""
    done = []

    m = re.search(r'function showImg\(\)\{.*?\n\}', page, re.S)
    if not m or "'/'+(ctx?'ctx':'crop')" not in m.group(0):
        raise SystemExit('**`showImg()` が見つからない/中身が変わっている。起動しない**')
    page = page.replace(m.group(0), NEW_SHOW); done.append('showImg()')

    m = re.search(r'function info\(\)\{.*?\n\}', page, re.S)
    if not m:
        raise SystemExit('**`info()` が見つからない。起動しない**')
    for mark in ('it.comp', 'it.region', 'it.area'):
        if mark not in m.group(0):
            raise SystemExit(f'**`info()` の中身が変わっている(`{mark}` が無い)。'
                             f'起動しない**')
    page = page.replace(m.group(0), NEW_INFO); done.append('info()')

    pairs = [
        # **絵を縮小しない**(NC3)。狭い画面では横スクロールになる
        ('#img{display:block;margin:8px auto;max-width:100%;height:auto;',
         '#img{display:block;margin:8px auto;max-width:none;height:auto;'),
        # `slot` を宣言し、問が変わるたび A に戻す
        ('let cur=0, it=null, t0=0, ctx=false, keymap={}, okeys={};',
         'let cur=0, it=null, t0=0, ctx=false, slot=0, keymap={}, okeys={};'),
        ('it=await r.json(); cur=o; t0=performance.now(); ctx=false;',
         'it=await r.json(); cur=o; t0=performance.now(); ctx=false; slot=0;'),
        # **スペース と → で巡回**(`b`/`c`/`n` は `l19` が使っている)
        ("  else if(k==='c'){ctx=!ctx;showImg();}",
         "  else if(k==='c'){ctx=!ctx;showImg();}\n"
         "  else if(k===' '||k==='arrowright'){"
         "e.preventDefault();ctx=false;"
         "slot=(slot+1)%Math.max(1,((it.alt||[]).length-1));showImg();}\n"
         "  else if(k==='arrowleft'){e.preventDefault();ctx=false;"
         "const m2=Math.max(1,((it.alt||[]).length-1));"
         "slot=(slot+m2-1)%m2;showImg();}"),
        ('<b id="ask">この黒いベタは何?</b>',
         '<b id="ask">この窓の絵を「面」に分けるとしたら、'
         'A・B・C のどれがあなたの分け方に最も近いですか。</b>'),
        ('<button onclick="ctx=!ctx;showImg()">コマ全体 <span class="k">C</span>'
         '</button>',
         '<button onclick="ctx=false;slot=(slot+1)%3;showImg()">'
         '分け方を差し替え <span class="k">space</span></button>'
         '<button onclick="ctx=!ctx;showImg()">線なしの絵 '
         '<span class="k">C</span></button>'),
        ("prompt('この黒は何ですか(語を入力)')",
         "prompt('A・B・C のどれでもないなら、どう分けたいかを一言で')"),
        ('<div class="meta" id="help">黒い塊が聞いている面。'
         '色は縁 4px に接している線のレイヤ(ヒント。答ではない)。\n'
         '工程の産物(隙間埋め)なら G / H。</div>',
         '<div class="meta" id="help">1画面 = 原稿の 1024x1024 画素そのまま'
         '(縮小していません)。マゼンタの線が「面の境目」の案です。'
         '水色は透明(その窓にその層が無い)。</div>'),
    ]
    for old, new in pairs:
        if old not in page:
            raise SystemExit(f'**置換元が見つからない。起動しない**\n  {old[:90]}')
        page = page.replace(old, new); done.append(old[:38])
    print(f'  画面の差し替え {len(done)} か所: ' + ' / '.join(done[:4]) + ' …',
          flush=True)
    return page


class H(l19.H):
    """`l19.H` を継いで **`/alt` を足すだけ**。他の口はそのまま親に渡す。"""

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        if u.path == '/alt':
            try:
                q = parse_qs(u.query)
                o, i = int(q['o'][0]), int(q['i'][0])
                if not 0 <= o < len(l19.S['q']):
                    return self._send(b'out of range', 'text/plain', 404)
                alt = l19.S['q'][o].get('alt') or []
                if not 0 <= i < len(alt):
                    return self._send(b'out of range', 'text/plain', 404)
                p = l19.S['dir'] / 'crops' / alt[i]
                return self._send(p.read_bytes(), 'image/png')
            except Exception as e:                                  # noqa: BLE001
                return self._send(str(e).encode(), 'text/plain', 500)
        return super().do_GET()


l19.PAGE = patch(l19.PAGE)
l19.H = H                       # **`main()` は呼ばれた時に `H` を引く**
if __name__ == '__main__':
    sys.exit(l19.main())

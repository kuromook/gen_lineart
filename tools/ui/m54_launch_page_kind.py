"""M54: **「この頁は何ですか」の UI を立ち上げる**(L のサーバは書き換えない)。

`m50` と同じ形(`ROOT`/`LAB` を M の木へ向け直し、門で確かめる)に加えて、
**画面の文を差し替える** —— `l19` の `info()` は

    [面の周り] <work> <pid> コマ0 面0  面積 ...px  (原寸)

と出す。**この UI は成分ではなく頁を聞いているので、「コマ0 面0」は意味を持たない上に、
`[面の周り]` は頁ぜんたいを出しているときに嘘になる**(縫い目2: 画面に出る量は測定値である)。
`act('free')` の文も「この黒は何ですか」のままでは問いと食い違う。
**`C` のボタンの「コマ全体」も同じ**(「コマ」が無く、しかも**既定が既に頁ぜんたいなので
「全体」は逆**。作者は全体を見ようとして押し、寄った絵が出る)。

**差し替えは文字列置換で行い、置換元が見つからなければ起動しない。**
**黙って L の文のまま出すのが T110 の形**(器具を作り直すと前の門が黙って消える)。
**`m50` は触らない**(8767 がそれで動いている)。

```
./venv/bin/python tools/ui/m54_launch_page_kind.py \
    --host 0.0.0.0 --port 8768 --queue results/m53_ui/queue.json \
    --save labels/page_kind.jsonl --extra ''
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
                         f'(流用の接点が変わった。m54 を読み直すこと)')

l19.ROOT = MTREE
l19.LAB = MTREE / 'labels'
l19.LAB.mkdir(exist_ok=True)

# ---- 画面の文の差し替え(見つからなければ起動しない) --------------------------
NEW_INFO = """function info(){
  if(!it) return;
  const sc=ctx?it.ctx_scale:it.scale;
  const iz=it.interp||1;
  document.getElementById('info').textContent=
    (ctx?'[絵に寄った] ':'[頁ぜんたい] ')+it.work+'   '+it.pid+
    (sc==null?'  (表示倍率 不明)':'  (元の原稿に対して '+(sc*100).toFixed(1)+'%)')+
    (ctx&&iz>1?'  ※プレビューを '+iz.toFixed(1)+'倍に拡大しています(ぼけます)':'')+
    (it.saved?'  ← 記録済み: '+(it.saved.word||it.saved.free||it.saved.choice):'')+
    (it.flagged?'   ⚑ 異常として報告済み: '+it.flagged.note:'');
}"""


def patch(page):
    """3か所を差し替える。**1つでも見つからなければ SystemExit。**"""
    done = []
    m = re.search(r'function info\(\)\{.*?\n\}', page, re.S)
    if not m:
        raise SystemExit('**`info()` が見つからない。起動しない**')
    old = m.group(0)
    for mark in ('it.comp', 'it.region', 'it.area.toLocaleString'):
        if mark not in old:
            raise SystemExit(f'**`info()` の中身が変わっている(`{mark}` が無い)。'
                             f'起動しない** —— m54 を読み直すこと')
    page = page.replace(old, NEW_INFO); done.append('info()')

    pairs = [
        ("prompt('この黒は何ですか(語を入力)')",
         "prompt('この頁は何ですか(一言で)')"),
        ('<b id="ask">この黒いベタは何?</b>',
         '<b id="ask">この頁は何ですか。</b>'),
        # **div ぜんたいを置く。** 最初は頭だけ置き換えて
        # **「色は縁 4px に接している線のレイヤ」「隙間埋めなら G / H」が残っていた** ——
        # **G / H はこの UI に無いキー**で、`it.help` が来るまでの一瞬、画面に出る。
        # **fetch が転べば出たままになる。**
        ('<div class="meta" id="help">黒い塊が聞いている面。'
         '色は縁 4px に接している線のレイヤ(ヒント。答ではない)。\n'
         '工程の産物(隙間埋め)なら G / H。</div>',
         '<div class="meta" id="help">頁ぜんたいを縮小して出しています'
         '(倍率は上の行に出ます)。当てはまる言葉が無ければ T に一言、'
         '迷ったら S。C で墨に寄れます。B で前へ、N で次へ。</div>'),
        # **5か所目(基盤が配信中の HTML を見て見つけた)。`info` 行だけ直して
        # ボタンを残すと、文言が二重に嘘になる**:
        #   「コマ」が無い(問いは頁について)/ 「全体」が逆(**既定が頁ぜんたいで、
        #   `C` は墨に寄る**)—— 作者は全体を見ようとして押し、寄った絵が出る
        ('<button onclick="ctx=!ctx;showImg()">コマ全体 <span class="k">C</span>'
         '</button>',
         '<button onclick="ctx=!ctx;showImg()">墨に寄る / 戻る '
         '<span class="k">C</span></button>'),
    ]
    for a, b in pairs:
        if a not in page:
            raise SystemExit(f'**置換元が見つからない: {a[:40]}... 起動しない**')
        page = page.replace(a, b); done.append(a[:28])
    print('  画面の文を差し替えた: ' + ' / '.join(done), flush=True)
    return page


def install():
    """**画面の文を差し替え、保存先を M の木へ向ける。** `m49` が点検のために呼ぶ。"""
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
        # **L の既報が M の問に出る件**(m50 で見つけた副作用(2))をここでも確かめる
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

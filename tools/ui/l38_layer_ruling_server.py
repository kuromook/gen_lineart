# -*- coding: utf-8 -*-
"""L38: **面1つに意味がいくつ入っているか**を作者が答える UI(2026-11-07)。

    /home/sh1/deepl/lineart/venv/bin/python tools/ui/l38_layer_ruling_server.py --host 0.0.0.0
    -> http://<host>:8477

**この UI は測定ではない。** 答を集める器具である。**棒も帰無値もまだ登録していない**
(規律9)。集めてから、**層の軸が作者の裁定を再現するか**を見る(規律13)。

## なぜ要るか

2026-11-07 の裁定2件で、**首の半径は両方 1 なのに裁定が逆**だった ——
分けたのは **寄与した塗り層の数と支配率**である。**n=2 では軸と言えない**ので、
**2x2 の各セルから3面ずつ**出した(`l38_neck_crops_20261107.py --cell`)。

## **群を画面に出さない**(目隠し)

**こちらの予想を作者に見せたら、それは作者の判断ではなくなる。** だから:

- **並び順は固定の種で混ぜる**(`BLIND_SEED`)。読み込み直しても同じ順
- **ファイル名を画面にも URL にも出さない**(名前に `L2_neck1` と書いてある)。
  画像は `/img?i=<順番>` で出す
- 答が入ってから、**保存に書いてある素性で群に分ける**

## 画面に何が出るか(**何について答えるのかを画面に書く**。2026-10-26 の指摘)

- **原寸 1:1** の切り出し。**水色が問う面**、淡い灰が同じコマの他の面、黒が原稿の線
- 「全体を縮小して見る」は**目印であって判断の場ではない**と画面に書く(規律6)
- 首の窓(在るものだけ)。**無い面では出さない** —— 無いものの場所は出せない
- 答: **1つのもの / 2つ以上(何と何かを書ける) / 分からない / 保留して次へ**

## 保存

`labels/face_meaning_count.jsonl` に**追記のみ**。1行が1つの答。取り消しは
`kind=undo` の行で、**過去の行は消さない**(規律7)。
**`labels/region_cluster.jsonl`(まとまり)には触らない** —— 別の単位の別の答である。

## 起動の門(通らなければ立ち上がらない)

1. 切り出し 9 枚が読め、**`(work,pid,comp,region,area)` で素性にちょうど1行当たる**
   (**面積だけでは同定にならない**: 1,130面に重複する面積が 55 在る。
   **`comp` を鍵にしない**という 2026-11-07 の結論)
2. **どの画像にも線の墨が在る**(0% ならシルエットで、作者は答えられない ——
   同じ日に一度失敗している)
3. 出す JS が構文検査を通る(`node --check`)
4. **既知セル**(規律2): 既に裁定の在る2面がこの 9 枚に在り、素性が裁定の日と同じであること
"""
import argparse, json, os, random, re, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

import cv2

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
CROPS = ROOT / 'results' / 'l38_neck_crops_20261107'
FEAT = ROOT / 'results' / 'l38_merge_cause_20261106.json'
SAVE = ROOT / 'labels' / 'face_meaning_count.jsonl'
BLIND_SEED = 20261107
# **既知セル**(2026-11-07 の裁定。`doc/work_log.md` に逐語)
KNOWN = {2503898: ('背景のベタのみ', 1, 1), 2232129: ('髪と衣類のベタが繋がった', 2, 1)}

NAME = re.compile(r'^(?P<work>[^_]+)_(?P<page>.+?)_c(?P<comp>\d+)_r(?P<region>\d+)'
                  r'_area(?P<area>\d+)_L(?P<nl>\d+)_neck(?P<neck>\d+|None)_a_face\.png$')
S, LOCK = {}, Lock()


def load():
    """切り出しを読み、**素性にちょうど1行**当てる(門1)。当たらなければ落ちる。"""
    feats = json.load(open(FEAT, encoding='utf-8'))
    items = []
    for cell in sorted(os.listdir(CROPS)):
        d = CROPS / cell
        if not d.is_dir():
            continue
        for fn in sorted(os.listdir(d)):
            m = NAME.match(fn)
            if not m:
                continue
            g = m.groupdict()
            hit = [r for r in feats
                   if r['work'] == g['work'] and r['pid'].endswith(g['page'])
                   and r['comp'] == int(g['comp']) and r['region'] == int(g['region'])
                   and r['area'] == int(g['area'])]
            if len(hit) != 1:
                raise SystemExit(f'門1: 素性が {len(hit)} 行当たった(1行でない): {fn}')
            r = hit[0]
            nk = d / fn.replace('_a_face.png', '_b_neck.png')
            im = cv2.imread(str(d / fn))
            if im is None:
                raise SystemExit(f'門1: 画像が読めない: {fn}')
            ink = float((im.max(axis=2) < 40).mean())
            if ink <= 0:
                raise SystemExit(f'門2: 線の墨が 0%(シルエットは答えられない): {fn}')
            items.append(dict(cell=cell, face=str(d / fn),
                              neck_img=str(nk) if nk.exists() else '',
                              w=int(im.shape[1]), h=int(im.shape[0]), ink=ink,
                              work=r['work'], pid=r['pid'], comp=r['comp'],
                              region=r['region'], area=r['area'],
                              n_layer=r['n_layer'], neck=r['neck'],
                              n_comp_4=r['n_comp_4'],
                              known=KNOWN.get(r['area'], ('', 0, 0))[0]))
    # 門4: 既知セルが在り、素性が裁定と同じか
    for area, (word, nl, neck) in KNOWN.items():
        k = [r for r in items if r['area'] == area]
        if len(k) != 1:
            raise SystemExit(f'門4: 既知セル(面積 {area})が {len(k)} 枚。1枚でない')
        if (k[0]['n_layer'], k[0]['neck']) != (nl, neck):
            raise SystemExit(f'門4: 既知セル(面積 {area})の素性が裁定の日と違う: '
                             f"{k[0]['n_layer']}層/首{k[0]['neck']} != {nl}層/首{neck}")
    random.Random(BLIND_SEED).shuffle(items)       # **目隠し**。種は固定
    return items


def answers():
    """保存を**ファイルの順に**畳む。`undo` はその時点の答を消す(行は残る。規律7)。"""
    cur = {}
    if SAVE.exists():
        for ln in open(SAVE, encoding='utf-8'):
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            k = (r.get('work'), r.get('pid'), r.get('comp'), r.get('region'), r.get('area'))
            if r.get('kind') == 'undo':
                cur.pop(k, None)
            else:
                cur[k] = r
    return cur


def key(it):
    return (it['work'], it['pid'], it['comp'], it['region'], it['area'])


JS = r'''
var ST={i:0};
function get(u){return fetch(u).then(function(r){return r.json();});}
function post(b){return fetch('/save',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})
  .then(function(r){return r.json();});}
function esc(s){return String(s).replace(/[&<>]/g,function(c){
  return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}
function load(i){
  return get('/state?i='+i).then(function(d){
    ST=d;
    document.getElementById('hd').textContent=
      '面 '+(d.i+1)+' / '+d.n+'   答えた面 '+d.n_done+' / '+d.n+
      '   原寸 '+d.w+'x'+d.h;
    var im=document.getElementById('im');
    im.src='/img?i='+d.i+'&kind=face&t='+Date.now();
    var nb=document.getElementById('neckbox');
    if(d.has_neck){nb.style.display='';
      document.getElementById('nk').src='/img?i='+d.i+'&kind=neck&t='+Date.now();}
    else{nb.style.display='none';document.getElementById('nk').removeAttribute('src');}
    document.getElementById('w').value=(d.answer&&d.answer.word)||'';
    var a=d.answer?('<b>答: '+esc(d.answer.label)+'</b>'+
        (d.answer.word?('  「'+esc(d.answer.word)+'」'):'')+
        '  <span class=t>'+esc(d.answer.t)+'</span>'):
        '<span class=t>まだ答えていません</span>';
    document.getElementById('ans').innerHTML=a;
    document.getElementById('undo').style.display=d.answer?'':'none';
    var L=document.getElementById('list');L.innerHTML='';
    d.list.forEach(function(x,j){
      var b=document.createElement('button');
      b.className='nav'+(j===d.i?' cur':'')+(x.done?' done':'');
      b.textContent=(j+1)+(x.done?' ✓':' ·');
      b.onclick=function(){load(j);};L.appendChild(b);});
    fit(document.getElementById('fit').checked);
  });
}
function fit(on){
  ['im','nk'].forEach(function(id){
    var e=document.getElementById(id);
    e.style.maxWidth=on?'100%':'none';});
  document.getElementById('fitnote').textContent=on?
    '**いまは縮小しています。** 判断は原寸で(この表示は目印です)':'原寸 1:1 で出しています';
}
function save(choice,label){
  var w=document.getElementById('w').value.trim();
  if(choice==='many'&&w===''){
    if(!confirm('何と何が繋がっているかを書かずに「2つ以上」で保存します。よろしいですか'))
      {return;}
  }
  post({i:ST.i,choice:choice,label:label,word:w}).then(function(d){
    if(d.next!==undefined&&d.next!==ST.i){load(d.next);}else{load(ST.i);}});
}
function undo(){
  if(!confirm('この面の答を取り消します(行は消えません。取り消しの行を足します)')){return;}
  post({i:ST.i,kind:'undo'}).then(function(){load(ST.i);});
}
window.addEventListener('keydown',function(e){
  if(e.target.tagName==='INPUT'){return;}
  if(e.key==='ArrowRight'){load(Math.min(ST.i+1,ST.n-1));}
  if(e.key==='ArrowLeft'){load(Math.max(ST.i-1,0));}
});
document.getElementById('fit').onchange=function(){fit(this.checked);};
[['1つのもの','one','1つのもの'],
 ['2つ以上が繋がっている','many','2つ以上'],
 ['分からない','unknown','分からない'],
 ['保留して次へ','skip','保留']].forEach(function(t){
  var b=document.createElement('button');b.className='big '+t[1];
  b.textContent=t[0];b.onclick=function(){save(t[1],t[2]);};
  document.getElementById('btns').appendChild(b);});
document.getElementById('undo').onclick=undo;
document.getElementById('prev').onclick=function(){load(Math.max(ST.i-1,0));};
document.getElementById('next').onclick=function(){load(Math.min(ST.i+1,ST.n-1));};
load(0);
'''

HTML = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<title>面の意味はいくつ入っているか</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#fafafa;color:#111}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;
   padding:8px 12px;z-index:9}
 h1{font-size:15px;margin:0 0 4px}
 .q{font-size:13px;color:#333;margin:2px 0}
 .t{color:#888;font-size:12px}
 #hd{font-size:13px;color:#444;margin:4px 0}
 #btns button{font-size:15px;padding:8px 14px;margin:4px 6px 4px 0;cursor:pointer}
 .one{background:#eef7ee}.many{background:#fdf0ee}
 #w{font-size:15px;padding:6px;width:min(520px,86vw)}
 .nav{font-size:12px;padding:3px 6px;margin:2px;cursor:pointer}
 .nav.cur{outline:2px solid #333;font-weight:bold}
 .nav.done{background:#e8f0ff}
 .pane{overflow:auto;background:#fff;border:1px solid #ddd;margin:10px 12px;
   padding:6px;max-height:78vh}
 img{image-rendering:pixelated;display:block}
 .leg{font-size:12px;color:#555;margin:4px 12px}
</style>
<header>
 <h1>この面には、意味がいくつ入っていますか</h1>
 <div class="q"><b>水色</b>が問う面、淡い灰は同じコマの別の面、黒は原稿の線です。
  「2つ以上」のときは<b>何と何か</b>を書いてください(例: 髪と衣類)。</div>
 <div class="q">どの群の面かは<b>画面に出していません</b> ——
  こちらの予想に引きずられないようにするためです。</div>
 <div id="hd"></div>
 <div id="ans"></div>
 <div id="btns"></div>
 <div>何と何か: <input id="w" placeholder="2つ以上のときだけ(空でも保存できます)">
  <button id="undo">この面の答を取り消す</button></div>
 <div class="q"><button id="prev">← 前</button> <button id="next">次 →</button>
  <label><input type="checkbox" id="fit"> 全体を縮小して見る</label>
  <span class="t" id="fitnote"></span></div>
 <div id="list"></div>
</header>
<div class="leg">面の切り出し(原寸 1:1)</div>
<div class="pane"><img id="im" alt=""></div>
<div id="neckbox"><div class="leg">細い首の周り(在る面だけ。原寸 1:1)</div>
 <div class="pane"><img id="nk" alt=""></div></div>
<script src="/app.js"></script></html>'''


def payload(i):
    it = S['items'][i]
    a = S['ans'].get(key(it))
    return dict(i=i, n=len(S['items']), w=it['w'], h=it['h'],
                has_neck=bool(it['neck_img']),
                n_done=sum(1 for x in S['items'] if key(x) in S['ans']),
                answer=(dict(label=a['label'], word=a.get('word', ''), t=a['t'])
                        if a else None),
                list=[dict(done=key(x) in S['ans']) for x in S['items']])


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        if u.path == '/':
            return self._send(200, 'text/html; charset=utf-8', HTML.encode('utf-8'))
        if u.path == '/app.js':
            return self._send(200, 'application/javascript', JS.encode('utf-8'))
        if u.path == '/state':
            i = max(0, min(len(S['items']) - 1, int(qs.get('i', ['0'])[0])))
            return self._send(200, 'application/json',
                              json.dumps(payload(i), ensure_ascii=False).encode('utf-8'))
        if u.path == '/img':
            i = int(qs.get('i', ['-1'])[0])
            kind = qs.get('kind', ['face'])[0]
            if not (0 <= i < len(S['items'])) or kind not in ('face', 'neck'):
                return self._send(400, 'text/plain', b'bad')
            # **名前を出さない**(名前に群が書いてある)。番号でしか出せない
            p = S['items'][i]['face'] if kind == 'face' else S['items'][i]['neck_img']
            if not p:
                return self._send(404, 'text/plain', b'no')
            return self._send(200, 'image/png', Path(p).read_bytes())
        return self._send(404, 'text/plain', b'no')

    def do_POST(self):
        if urlparse(self.path).path != '/save':
            return self._send(404, 'text/plain', b'no')
        n = int(self.headers.get('Content-Length', 0))
        d = json.loads(self.rfile.read(n) or b'{}')
        with LOCK:
            i = max(0, min(len(S['items']) - 1, int(d.get('i', 0))))
            it = S['items'][i]
            base = dict(t=time.strftime('%Y-%m-%dT%H:%M:%S'), work=it['work'],
                        pid=it['pid'], comp=it['comp'], region=it['region'],
                        area=it['area'])
            if d.get('kind') == 'undo':
                row = dict(base, kind='undo')
                S['ans'].pop(key(it), None)
                nxt = i
            else:
                ch = d.get('choice', 'skip')
                row = dict(base, kind='meaning_count', choice=ch,
                           label=d.get('label', ''), word=d.get('word', ''),
                           blind_order=i, seed=BLIND_SEED,
                           # **素性は保存に書く**(画面には出さない)。
                           # 後で群に分けるのはこの値で、名前ではない
                           cell=it['cell'], n_layer=it['n_layer'], neck=it['neck'],
                           n_comp_4=it['n_comp_4'], known=it['known'],
                           annotator='author')
                if ch != 'skip':
                    S['ans'][key(it)] = row
                nxt = min(i + 1, len(S['items']) - 1)
            with open(SAVE, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
        return self._send(200, 'application/json',
                          json.dumps(dict(next=nxt)).encode('utf-8'))


def js_check():
    """**門3**: 出す JS を構文検査する(`node --check`)。"""
    import shutil, subprocess, tempfile
    node = shutil.which('node')
    if not node:
        print('門3: node が無いので構文検査を飛ばした(**検査していない**)')
        return
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'ui.js')
        open(p, 'w', encoding='utf-8').write(JS)
        r = subprocess.run([node, '--check', p], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit('門3: JS が構文検査に落ちた\n' + r.stderr)


def main():
    global SAVE
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8477)
    # **試験で本番の保存を上書きしない**(2026-11-07、`--limit` の煙試験が
    # 本番の `queue.json` を潰した。器具には必ず出口を切り替える口を付ける)
    ap.add_argument('--save', default=str(SAVE), help='保存先(既定は本番。追記のみ)')
    a = ap.parse_args()
    SAVE = Path(a.save)
    S['items'] = load()
    S['ans'] = answers()
    js_check()
    # **並び順と結び付く形で印字しない**(端末の出力から群が読めたら目隠しが壊れる)。
    # 出すのは**群ごとの数と範囲**だけで、どの番号がどれかは出さない
    import collections as _c
    g = _c.Counter(it['cell'] for it in S['items'])
    ink = [it['ink'] for it in S['items']]
    print(f'門を通った。面 {len(S["items"])} 枚 / 既に答の在る面 {len(S["ans"])} / '
          f'既知セル {sum(1 for it in S["items"] if it["known"])} 枚'
          f'(**どれかは出さない**)')
    print('  群ごとの枚数(並び順とは結び付けない): '
          + ' / '.join(f'{k} {v}' for k, v in sorted(g.items())))
    print(f'  線の墨 {min(ink):.2%} 〜 {max(ink):.2%}(**0% は門2 で止まる**)/ '
          f'首の窓が在る面 {sum(1 for it in S["items"] if it["neck_img"])}')
    print(f'保存 {SAVE}(追記のみ)  -> http://{a.host}:{a.port}')
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""L38: **面のまとまり**を作者が選んで語を当てる UI(2026-10-28)。

作者の指示: **「これはUIの出番だ 複数の面を表示し、ユーザがまとまりを選択して語をわりあてる」**

  python tools/ui/l38_cluster_label_server.py --host 0.0.0.0
  -> http://<host>:8476

**この UI は測定ではない。単位を作る器具である。**
L18 が閉じたのは「面1つに語1つ、連結した墨で運ぶ」形で、**まとまりを単位にした形は
一度も測っていない**。測るにはまず単位が要り、**その単位は作者が選ぶことでしか作れない**
(規律6。作者の言明「複数の面のまとまりとなってはじめて意味を持たせられるものも多い」)。

## 画面に何が出るか(**何について答えるのかを画面に書く**。2026-10-26 の指摘)

- コマ1つ。**その中の面を全部**、淡い色で塗り分けて出す
- **面ごとの答(628件)が在る面には、その語が札に出る** —— 作者が前に何と答えたかを
  見ながら束ねられる。**無い面は「—」**(答えていないことを空白で隠さない)
- 選ぶ: 画像の上を押す / 右の一覧を押す。**選択中の面は濃い縁と太字の札**
- 答える: 語のボタン / 自由入力 / **まとまりでない** / 保留
- **1コマに複数のまとまりを作れる。** 作り終えたら「このコマは終わり」で次へ

## 保存

`labels/region_cluster.jsonl` に追記。**`labels/fill_words.jsonl`(面ごとの 628件)には
一切触らない** —— 別の単位の別の答である。1行が1つのまとまり。
取り消しは `kind=undo` の行で、**過去の行は消さない**(規律7)。

## 起動の門(通らなければ立ち上がらない)

1. 行列の全コマで記録を組み立てられる
2. 出す JS が構文検査を通る(`node --check`)
3. **当たり判定の既知セル**: 面ごとの代表点を地図に引いて、その面が返ること。
   **外した面を理由別に数える。** 1つでも外したら立ち上がらない ——
   **押した面と違う面が選ばれる UI は、作者の答を黙って壊す**
"""
import argparse, collections, json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

import numpy as np, cv2

ROOT = Path(__file__).resolve().parent.parent.parent
UI = ROOT / 'results/l38_cluster_ui'
LAB = ROOT / 'labels'
WORDS = ['髪', '肌', '顔(目鼻口)', '服', '影', '背景・3D', '効果',
         '描き文字', 'まつ毛', '眉', 'フキダシ', '黒消し']
S = {}
LOCK = Lock()


def load_state(save, qpath, start):
    q = json.load(open(qpath or (UI / 'queue.json'), encoding='utf-8'))
    S['q'] = q['queue']
    S['meta'] = {k: v for k, v in q.items() if k != 'queue'}
    S['save'] = Path(save)
    S['maps'] = {}
    # ---- 保存を読み直す。**鍵は `tag`(コマの名前)で、`order` では引かない**(T144)
    done = collections.defaultdict(list)
    why = collections.Counter()
    tags = {r['tag'] for r in S['q']}
    if S['save'].exists():
        for ln in open(S['save'], encoding='utf-8'):
            if not ln.strip():
                continue
            r = json.loads(ln)
            t = r.get('tag')
            if t not in tags:
                why['この行列に無いコマ (別の行列の保存先か、行列が変わった)'] += 1
                continue
            if r.get('kind') == 'undo':
                done[t] = [x for x in done[t] if x.get('cid') != r.get('cid')]
                continue
            done[t].append(r)
    S['done'] = done
    S['why'] = why
    S['cur'] = start if start is not None else next(
        (i for i, r in enumerate(S['q'])
         if not any(x.get('kind') == 'panel_done' for x in done[r['tag']])),
        0)


def fmap(tag):
    """コマの地図(画素 -> 面の番号)。**当たり判定の正本**。"""
    if tag not in S['maps']:
        m = cv2.imread(str(UI / 'panels' / f'{tag}_map.png'), cv2.IMREAD_UNCHANGED)
        S['maps'][tag] = m
    return S['maps'][tag]


def kc_hit():
    """**起動の門その3**: 面の代表点を画面の座標へ移し、**原寸へ戻して**地図を引く。

    **押した面と違う面が選ばれる UI は、作者の答を黙って壊す。**
    だから起動のたびに全コマで引き直す(作った時だけでなく)。

    **2つを分ける**(機序が違うから。直すために分けたのではない):

    - **別の面を返す = 致命。立ち上げない**
    - **0 を返す = 画像からは押せないだけ。** 一覧からは選べるので、
      **印を付けて数え、画面にも出す**(規律9。黙って押せない面を作らない)
    """
    bad, unreach = collections.Counter(), 0
    n = 0
    for r in S['q']:
        m = fmap(r['tag'])
        if m is None:
            bad['地図が読めない'] += len(r['faces']); continue
        for i, f in enumerate(r['faces'], 1):
            n += 1
            y, x = int(round(f['sy'] / r['scale'])), int(round(f['sx'] / r['scale']))
            v = (int(m[y, x]) if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] else -1)
            f['clickable'] = bool(v == i)
            if v == i:
                continue
            if v in (0, -1):
                unreach += 1
            else:
                bad['**別の面を返す(致命)**'] += 1
    print(f'  起動の門その3 当たり判定の往復: 面 {n}  '
          f'**別の面を返した {sum(bad.values())}**  画像から押せない {unreach}'
          f'(一覧からは選べる)')
    for w, c in bad.items():
        print(f'     {w}: {c}')
    if not bad:
        print('     (致命の理由は発火していない)')
    return sum(bad.values())


def png_size(path):
    """PNG の IHDR だけ読んで (幅, 高さ)。**画像を展開しない**(246枚を起動のたびに読む)。"""
    with open(path, 'rb') as fh:
        d = fh.read(26)
    if len(d) < 24 or d[:8] != b'\x89PNG\r\n\x1a\n':
        return None
    return int.from_bytes(d[16:20], 'big'), int.from_bytes(d[20:24], 'big')


def kc_dims():
    """**起動の門その4**: 行列が配る画面の寸法が、実際の選択画像と一致するか。

    2026-10-28 に踏んだ: 地図を原寸にしたとき、`sel_w/sel_h` が**地図の寸法**から
    取られたままだった。**原寸の寸法が画面の寸法として配られ**、ブラウザが画像を
    引き伸ばし、**札の位置も押した座標もずれた**。
    **当たり判定の既知セルはこれを映さない** —— あちらは `sy/sx` と `scale` しか
    見ないので、その2つが揃っていれば通ってしまう。

    > **当たり判定が正しいことと、画面が正しいことは別である。**
    > **器具の中で閉じた点検は、画面に出る形を見ていない。**
    """
    bad = 0
    for r in S['q']:
        sz = png_size(UI / 'panels' / f'{r["tag"]}_sel.png')
        if sz is None or sz != (r['sel_w'], r['sel_h']):
            bad += 1
    print(f'  起動の門その4 画面の寸法: 行列と選択画像が食い違うコマ **{bad}**')
    return bad


def payload(i):
    r = S['q'][i]
    d = S['done'][r['tag']]
    cl = [x for x in d if x.get('kind') == 'cluster']
    return dict(order=i, n=len(S['q']), tag=r['tag'], work=r['work'], pid=r['pid'],
                pan=r['pan'], comp=r['comp'], scale=r['scale'],
                sel_w=r['sel_w'], sel_h=r['sel_h'], nat_w=r['nat_w'], nat_h=r['nat_h'],
                faces=[dict(i=k, region=f['region'], area=f['area'], adj=f['adj'],
                            word=f['word'], choice=f['choice'],
                            clickable=bool(f.get('clickable', True)),
                            sy=f['sy'], sx=f['sx'], crop=f['crop'])
                       for k, f in enumerate(r['faces'], 1)],
                clusters=[dict(cid=x['cid'], faces=x['faces'], word=x.get('word', ''),
                               choice=x['choice']) for x in cl],
                done=any(x.get('kind') == 'panel_done' for x in d),
                words=WORDS)


JS = r'''
let ST=null, SEL=new Set(), NAT=false;
function $(id){return document.getElementById(id);}
function load(o){
  fetch('/state?order='+o).then(r=>r.json()).then(d=>{ST=d;SEL=new Set();draw();});
}
function draw(){
  const it=ST;
  $('hdr').textContent=it.work+' '+it.pid+' コマ '+it.pan+' ('+it.order+'/'+it.n+')';
  $('ask').textContent='このコマの中で、**1つの意味になる面**を選んでください ('
    +it.faces.length+' 面、うち語が付いている面 '
    +it.faces.filter(f=>f.word).length+')';
  const img=$('im');
  img.src='/img?tag='+encodeURIComponent(it.tag)+'&kind='+(NAT?'nat':'sel')+'&v='+it.order;
  img.width=NAT?it.nat_w:it.sel_w;
  const bl=$('badges'); bl.innerHTML='';
  const k=NAT?(1/it.scale):1;
  it.faces.forEach(f=>{
    const b=document.createElement('div');
    b.className='bg'+(SEL.has(f.i)?' on':'');
    b.style.left=(f.sx*k-11)+'px'; b.style.top=(f.sy*k-11)+'px';
    b.textContent=f.i; b.title='面'+f.i+' 面積'+f.area+(f.word?(' 語:'+f.word):'');
    b.onclick=(e)=>{e.stopPropagation();toggle(f.i);};
    bl.appendChild(b);
  });
  const ul=$('list'); ul.innerHTML='';
  it.faces.forEach(f=>{
    const li=document.createElement('div');
    li.className='row'+(SEL.has(f.i)?' on':'');
    li.textContent=f.i+'. 面積 '+f.area.toLocaleString()+'  語 '+(f.word||'—')
      +(f.adj?'':'  [縁に線なし]')
      +(f.clickable?'':'  [画像からは押せない。ここから選ぶ]');
    li.onclick=()=>toggle(f.i);
    ul.appendChild(li);
  });
  const cl=$('cl'); cl.innerHTML='';
  it.clusters.forEach(c=>{
    const li=document.createElement('div');
    li.className='crow';
    li.textContent='['+c.cid+'] '+(c.choice==='not_cluster'?'(まとまりでない)':c.word)
      +'  面 '+c.faces.join(',');
    const b=document.createElement('button');
    b.textContent='取消'; b.onclick=()=>undo(c.cid);
    li.appendChild(b); cl.appendChild(li);
  });
  $('sel').textContent=SEL.size?('選択中: '+[...SEL].sort((a,b)=>a-b).join(', ')):'未選択';
  $('dn').textContent=it.done?'このコマは「終わり」と記録済み':'';
}
function toggle(i){ if(SEL.has(i))SEL.delete(i); else SEL.add(i); draw(); }
function hit(e){
  const im=$('im'), r=im.getBoundingClientRect();
  const k=NAT?ST.scale:1;
  const x=Math.round((e.clientX-r.left)*k), y=Math.round((e.clientY-r.top)*k);
  fetch('/hit?tag='+encodeURIComponent(ST.tag)+'&y='+y+'&x='+x)
    .then(r=>r.json()).then(d=>{ if(d.i>0) toggle(d.i); });
}
function save(choice,word){
  if(choice!=='panel_done' && SEL.size===0){ alert('面が選ばれていません'); return; }
  fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({order:ST.order,faces:[...SEL].sort((a,b)=>a-b),
                         choice:choice,word:word||''})})
    .then(r=>r.json()).then(d=>{ if(d.next!==undefined) load(d.next); else load(ST.order); });
}
function undo(cid){
  fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({order:ST.order,kind:'undo',cid:cid})})
    .then(()=>load(ST.order));
}
function free(){ const w=prompt('語を入力'); if(w) save('free',w); }
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  if(e.key==='Enter')save('panel_done','');
  else if(e.key==='c'||e.key==='C'){NAT=!NAT;draw();}
  else if(e.key==='n')load(Math.min(ST.order+1,ST.n-1));
  else if(e.key==='b')load(Math.max(ST.order-1,0));
  else if(e.key==='Escape'){SEL=new Set();draw();}
});
'''

HTML = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<title>L38 面のまとまり</title><style>
body{font-family:system-ui,sans-serif;margin:12px;background:#fafafa}
#hdr{font-weight:700}#ask{margin:6px 0;color:#044}
#wrap{display:flex;gap:14px;align-items:flex-start}
#pane{position:relative;display:inline-block;border:1px solid #ccc;background:#fff}
#im{display:block;cursor:crosshair}
.bg{position:absolute;width:22px;height:22px;border-radius:11px;background:#fff;
    border:2px solid #555;font:700 12px/18px system-ui;text-align:center;cursor:pointer;
    opacity:.85}
.bg.on{background:#c00;color:#fff;border-color:#700;opacity:1}
#side{width:330px;max-height:86vh;overflow:auto}
.row{padding:3px 6px;border-bottom:1px solid #eee;cursor:pointer;font-size:13px}
.row.on{background:#fdd;font-weight:700}
.crow{padding:3px 6px;background:#eef;margin:2px 0;font-size:13px}
button{margin:2px;padding:6px 10px;font-size:14px}
#sel{margin:6px 0;font-weight:700}#dn{color:#a00}
</style>
<div id="hdr"></div><div id="ask"></div>
<div id="wrap">
 <div id="pane"><img id="im" onclick="hit(event)"><div id="badges"></div></div>
 <div id="side">
  <div id="sel"></div><div id="dn"></div>
  <div id="btns"></div>
  <h4>このコマの面</h4><div id="list"></div>
  <h4>作ったまとまり</h4><div id="cl"></div>
  <p style="font-size:12px;color:#666">Enter = このコマは終わり / C = 原寸 /
   Esc = 選択解除 / n,b = 次,前のコマ</p>
 </div></div>
<script>%s
const B=$('btns');
%s.forEach(w=>{const b=document.createElement('button');b.textContent=w;
  b.onclick=()=>save('word',w);B.appendChild(b);});
[['自由入力',free],['まとまりでない',()=>save('not_cluster','')],
 ['このコマは終わり',()=>save('panel_done','')]].forEach(([t,fn])=>{
  const b=document.createElement('button');b.textContent=t;b.onclick=fn;B.appendChild(b);});
load(%d);
</script></html>'''


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
            body = (HTML % (JS, json.dumps(WORDS, ensure_ascii=False),
                            S['cur'])).encode('utf-8')
            return self._send(200, 'text/html; charset=utf-8', body)
        if u.path == '/app.js':
            return self._send(200, 'application/javascript', JS.encode('utf-8'))
        if u.path == '/state':
            i = max(0, min(len(S['q']) - 1, int(qs.get('order', ['0'])[0])))
            S['cur'] = i
            return self._send(200, 'application/json',
                              json.dumps(payload(i), ensure_ascii=False).encode('utf-8'))
        if u.path == '/img':
            tag = qs['tag'][0]
            kind = qs.get('kind', ['sel'])[0]
            if kind not in ('sel', 'nat') or tag not in {r['tag'] for r in S['q']}:
                return self._send(400, 'text/plain', b'bad')
            p = UI / 'panels' / f'{tag}_{kind}.png'
            return self._send(200, 'image/png', p.read_bytes())
        if u.path == '/hit':
            tag = qs['tag'][0]
            m = fmap(tag)
            # **地図は原寸。** 画面の座標を `1/scale` で戻して引く ——
            # 縮めた地図で引くと 1画素より細い面が消える(2026-10-28 に 2面で実測)
            sc = next((r['scale'] for r in S['q'] if r['tag'] == tag), 1.0)
            y = int(round(int(qs['y'][0]) / sc))
            x = int(round(int(qs['x'][0]) / sc))
            i = 0
            if m is not None and 0 <= y < m.shape[0] and 0 <= x < m.shape[1]:
                i = int(m[y, x])
            return self._send(200, 'application/json',
                              json.dumps(dict(i=i)).encode('utf-8'))
        return self._send(404, 'text/plain', b'no')

    def do_POST(self):
        if urlparse(self.path).path != '/save':
            return self._send(404, 'text/plain', b'no')
        n = int(self.headers.get('Content-Length', 0))
        d = json.loads(self.rfile.read(n) or b'{}')
        with LOCK:
            r = S['q'][int(d['order'])]
            cur = S['done'][r['tag']]
            if d.get('kind') == 'undo':
                row = dict(kind='undo', tag=r['tag'], cid=int(d['cid']))
                S['done'][r['tag']] = [x for x in cur if x.get('cid') != int(d['cid'])]
            else:
                ch = d['choice']
                kind = 'panel_done' if ch == 'panel_done' else 'cluster'
                cid = 1 + max([x.get('cid', 0) for x in cur] or [0])
                idx = {k: f for k, f in enumerate(r['faces'], 1)}
                fs = [int(i) for i in d.get('faces', [])]
                row = dict(t=time.strftime('%Y-%m-%dT%H:%M:%S'), queue=S['meta'].get(
                    'queue_id', ''), kind=kind, cid=cid, order=int(d['order']),
                    work=r['work'], pid=r['pid'], pan=r['pan'], comp=r['comp'],
                    tag=r['tag'], choice=ch, word=d.get('word', ''),
                    faces=fs, regions=[idx[i]['region'] for i in fs],
                    crops=[idx[i]['crop'] for i in fs],
                    n_faces=len(fs), area=sum(idx[i]['area'] for i in fs),
                    face_words=[idx[i]['word'] for i in fs],
                    annotator='author')
                S['done'][r['tag']].append(row)
            with open(S['save'], 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            nxt = (min(int(d['order']) + 1, len(S['q']) - 1)
                   if d.get('choice') == 'panel_done' else int(d['order']))
        return self._send(200, 'application/json',
                          json.dumps(dict(next=nxt)).encode('utf-8'))


def js_check():
    """**起動の門その2**: 出す JS を構文検査する(`node --check`)。"""
    import shutil, subprocess, tempfile
    node = shutil.which('node') or shutil.which('nodejs')
    if not node:
        print('  (node が無いので JS の構文検査は飛ばす —— **渡す前に画面を見ること**)')
        return
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False,
                                     encoding='utf-8') as fh:
        fh.write(JS)
        p = fh.name
    r = subprocess.run([node, '--check', p], capture_output=True, text=True)
    os.unlink(p)
    if r.returncode:
        print('**JS が構文検査を通らない。立ち上げない**\n' + r.stderr)
        sys.exit(2)
    print('  起動の門その2: JS は構文検査を通った')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8476)
    ap.add_argument('--queue', default='')
    ap.add_argument('--save', default=str(LAB / 'region_cluster.jsonl'))
    ap.add_argument('--from', dest='start', type=int, default=None)
    a = ap.parse_args()
    load_state(a.save, a.queue, a.start)
    m = S['meta']
    print(f'行列 {len(S["q"])} コマ  面 {m.get("n_face")}  '
          f'**面ごとの答が在る面 {m.get("n_face_answered")}**  '
          f'保存先 {Path(a.save).name}  -> http://{a.host}:{a.port}')
    print(f'  単位: {m.get("unit")}  標準: {m.get("standard")}  '
          f'**まとまりはコマを跨がない**: {not m.get("cross_panel")}')
    nd = sum(1 for r in S['q']
             if any(x.get('kind') == 'panel_done' for x in S['done'][r['tag']]))
    ncl = sum(len([x for x in S['done'][r['tag']] if x.get('kind') == 'cluster'])
              for r in S['q'])
    print(f'  保存の読み込み: **終わったコマ {nd} / まとまり {ncl}**'
          f'(鍵は `tag`。`order` では引かない。T144)')
    for w, c in S['why'].items():
        print(f'    {c}  {w}')
    if not S['why']:
        print('    0  (別の行列の行は無い)')
    for i in range(min(3, len(S['q']))):
        payload(i)
    print(f'  起動の門その1: 行列 {len(S["q"])} コマすべてで記録を組み立てられた')
    js_check()
    if kc_dims():
        print('**ブラウザが画像を引き伸ばす。押した座標がずれる。立ち上げない**')
        sys.exit(4)
    if kc_hit():
        print('**当たり判定が外れている。立ち上げない** —— '
              '押した面と違う面が選ばれる UI は、作者の答を黙って壊す')
        sys.exit(3)
    print(f'  最初に出すコマ: order={S["cur"]}')
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == '__main__':
    main()

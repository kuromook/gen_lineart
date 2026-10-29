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

## **単位は上にも下にも壊れる**(2026-10-29、作者の指摘)

> **線の判断でも同じ問題があったが、ベタが過剰に結合していて複数の意味を内容している
> 場合どうかくか**

**まとまりは「面1つでは足りない」側の壊れ方で、その逆向きが在る** ——
**面1つの中に意味が複数入っている**(過剰結合)。面は墨の連結成分なので、
**どちらも作者が決める単位と、墨の連結が一致していない形**である。
Track K が線で踏んだ **過接続と過分割**と同じ対(K では**どちらも修理が届かなかった**)。

- 印は**面1つ**に付く(`kind=overmerged`)。まとまりと**同じ行列に混ぜない**
- 中に在る意味を書ける(**空でもよい。印だけ残る**)。`words_given` で区別する
- **この印を付けた面は、どのまとまりに入れても正しくない。** だから
  **面を単位にした器具の天井**になる —— 器具の良し悪しではなく、単位の限界である
- **UI は禁止しない**(規律6。作者が束ねたいなら束ねられる。両方記録して測る側で分ける)

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
import argparse, collections, json, os, re, sys, time
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


def render_page():
    """頁の全文。**`%` 書式は使わない** —— CSS の `max-width:100%;` が書式指定として
    読まれ、**起動の門を全部通った後で、頁を出す所だけが 500 で落ちた**(2026-10-28)。
    置換なら雛形に何を書いても壊れない。**起動時にもこの関数を通す**(下の `page_check`)。
    """
    return (HTML.replace('__JS__', JS)
                .replace('__WORDS__', json.dumps(WORDS, ensure_ascii=False))
                .replace('__CUR__', str(S['cur'])))


def page_check():
    """**起動の門その5**: 頁を組み立て、**その JS を実際に走らせて選べるかを見る**。

    2026-10-28、作者から「**選択ができない どういう状況かも不明**」。
    それまで通っていたのは起動の門(行列・JS の構文・寸法・当たり判定)と HTTP 200 だけで、
    **どれも画面の JS を1行も走らせていない**。L19 が 2026-10-26 に同じ形で踏んでいる ——
    **同じ罠を、同じ器具の族で2度踏んだ。**

    `l38_ui_smoke.js` は **画像が縮んで描画されている状況**(窓が狭いと flex が縮める)で
    札と一覧と画像の押下を試す。**縮んだ画像の上の座標を縮んでいない前提で送る実装は、
    ここで落ちる**(実測のずれ 462 / 1,911 画素)。
    """
    import re as _re, shutil, subprocess, tempfile
    node = shutil.which('node') or shutil.which('nodejs')
    if not node:
        print('  (node が無いので画面の試験は飛ばす —— **渡す前に画面を見ること**)')
        return 0
    try:
        page = render_page()
    except Exception as e:
        print(f'**頁が組み立てられない: {e}。立ち上げない**')
        return 1
    m = _re.search(r'<script>(.*)</script>', page, _re.S)
    if not m:
        print('**頁に <script> が無い。立ち上げない**')
        return 1
    i = next((k for k, r in enumerate(S['q']) if len(r['faces']) >= 2), 0)
    d = tempfile.mkdtemp()
    open(f'{d}/ui.js', 'w', encoding='utf-8').write(m.group(1))
    open(f'{d}/pay.json', 'w', encoding='utf-8').write(
        json.dumps(payload(i), ensure_ascii=False))
    smoke = str(Path(__file__).resolve().parent / 'l38_ui_smoke.js')
    r = subprocess.run([node, smoke, f'{d}/pay.json', f'{d}/ui.js'],
                       capture_output=True, text=True)
    print('  起動の門その5 画面: ' + (r.stdout.strip().splitlines() or ['(出力なし)'])[-1]
          + f'  (コマ order={i})')
    if r.returncode:
        print(r.stdout)
        print(r.stderr)
        return 1
    return 0


MASKS = {}


def mask_png(tag, i, c):
    """面 i の形を、**画面の大きさの半透明 PNG**(縁は不透明)で返す。

    2026-10-28、作者の報告: **「選択している面がどれか画面上でわからない。
    右にリストはあるが対応がわからない」**。
    番号の札だけでは**どの面を選んだかが絵の上に出ない** ——
    **札は面を指しているが、面を見せていない。**
    だから**面そのものを塗る**。

    縮小は `INTER_AREA` を 0/255 に当てて **> 0 で拾う** ——
    `INTER_NEAREST` だと**細い面が画面から消える**(地図でも同じ所を踏んだ)。
    """
    k = (tag, i, c)
    if k in MASKS:
        return MASKS[k]
    r = next(x for x in S['q'] if x['tag'] == tag)
    m = fmap(tag)
    m8 = ((m == i).astype(np.uint8) * 255)
    w, h = r['sel_w'], r['sel_h']
    if (m8.shape[1], m8.shape[0]) != (w, h):
        m8 = cv2.resize(m8, (w, h), interpolation=cv2.INTER_AREA)
    sel = m8 > 0
    edge = sel & ~cv2.erode(sel.astype(np.uint8),
                            np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    # **OpenCV は BGR。** 最初ここを RGB のつもりで書いて、
    # **札は青・塗りは赤**という食い違いになった(2026-10-28、合成して目で見て気付いた)。
    # 選択 = 青(札 `.bg.on` の `#00c` と同じ) / 一覧に触れた = 橙
    bgr = (230, 60, 60) if c == 'sel' else (40, 150, 230)
    img = np.zeros((h, w, 4), np.uint8)
    img[sel, :3] = bgr
    img[sel, 3] = 90                      # 面の中は薄く
    img[edge, 3] = 255                     # 縁ははっきり
    ok, buf = cv2.imencode('.png', img)
    MASKS[k] = buf.tobytes()
    return MASKS[k]


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
    om = [x for x in d if x.get('kind') == 'overmerged']
    omw = collections.defaultdict(list)
    for x in om:
        for k in x.get('faces', []):
            omw[k] += x.get('words', [])
    return dict(order=i, n=len(S['q']), tag=r['tag'], work=r['work'], pid=r['pid'],
                pan=r['pan'], comp=r['comp'], scale=r['scale'],
                sel_w=r['sel_w'], sel_h=r['sel_h'], nat_w=r['nat_w'], nat_h=r['nat_h'],
                faces=[dict(i=k, region=f['region'], area=f['area'], adj=f['adj'],
                            word=f['word'], choice=f['choice'],
                            clickable=bool(f.get('clickable', True)),
                            om=bool(k in omw), om_words=omw.get(k, []),
                            sy=f['sy'], sx=f['sx'], crop=f['crop'])
                       for k, f in enumerate(r['faces'], 1)],
                clusters=[dict(cid=x['cid'], faces=x['faces'], word=x.get('word', ''),
                               choice=x['choice']) for x in cl],
                oms=[dict(cid=x['cid'], faces=x.get('faces', []),
                          words=x.get('words', [])) for x in om],
                done=any(x.get('kind') == 'panel_done' for x in d),
                words=WORDS)


JS = r'''
let ST=null, SEL=new Set(), NAT=false, HOV=0;
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
  // **選んだ面を絵の上に塗る。** 札は面を指すだけで、面を見せていない
  // (2026-10-28、作者の「どれを選んでるか不明」)。
  const ov=$('ovs'); ov.innerHTML='';
  SEL.forEach(i=>{
    const o=document.createElement('img');
    o.className='ov';
    o.src='/mask?tag='+encodeURIComponent(it.tag)+'&i='+i+'&c=sel';
    ov.appendChild(o);
  });
  if(HOV && !SEL.has(HOV)){
    const o=document.createElement('img');
    o.className='ov';
    o.src='/mask?tag='+encodeURIComponent(it.tag)+'&i='+HOV+'&c=hov';
    ov.appendChild(o);
  }
  const bl=$('badges'); bl.innerHTML='';
  // **札は百分率で置く。** px で置くと、窓が狭くて画像が縮んだとき札だけ元の位置に残り、
  // **面から外れる**(2026-10-28、作者の「選択ができない」の正体)。
  // 百分率なら `#pane` が画像とぴたり同じ大きさなので、どんな倍率でも面に乗る。
  const W=NAT?it.nat_w:it.sel_w, H=NAT?it.nat_h:it.sel_h;
  const kx=NAT?(1/it.scale):1;
  it.faces.forEach(f=>{
    const b=document.createElement('div');
    b.className='bg'+(SEL.has(f.i)?' on':'')+(f.om?' om':'');
    b.style.left='calc('+(f.sx*kx/W*100)+'% - 11px)';
    b.style.top='calc('+(f.sy*kx/H*100)+'% - 11px)';
    b.textContent=f.i; b.title='面'+f.i+' 面積'+f.area+(f.word?(' 語:'+f.word):'')
      +(f.om?' 【複数の意味】':'');
    b.onclick=(e)=>{e.stopPropagation();toggle(f.i);};
    bl.appendChild(b);
  });
  const ul=$('list'); ul.innerHTML='';
  it.faces.forEach(f=>{
    const li=document.createElement('div');
    li.className='row'+(SEL.has(f.i)?' on':'');
    li.textContent='［'+f.i+'］ 面積 '+f.area.toLocaleString()+'  語 '+(f.word||'—')
      +(f.om?('  【複数の意味'+(f.om_words.length?': '+f.om_words.join('/'):'')+'】'):'')
      +(f.adj?'':'  [縁に線なし]')
      +(f.clickable?'':'  [画像からは押せない。ここから選ぶ]');
    li.onclick=()=>toggle(f.i);
    // **一覧と絵の対応**: 行に触れると、その面が絵の上で橙に光る
    li.onmouseenter=()=>{HOV=f.i;draw();};
    li.onmouseleave=()=>{if(HOV===f.i){HOV=0;draw();}};
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
  const ol=$('om'); ol.innerHTML='';
  it.oms.forEach(c=>{
    const li=document.createElement('div');
    li.className='orow';
    li.textContent='['+c.cid+'] 面 '+c.faces.join(',')+' の中に複数の意味'
      +(c.words.length?(': '+c.words.join(' / ')):'(中身は未記入)');
    const b=document.createElement('button');
    b.textContent='取消'; b.onclick=()=>undo(c.cid);
    li.appendChild(b); ol.appendChild(li);
  });
  $('sel').textContent=SEL.size?('選択中: '+[...SEL].sort((a,b)=>a-b).join(', ')):'未選択';
  const im2=$('im').getBoundingClientRect?$('im').getBoundingClientRect():{width:0};
  $('diag').textContent='状態: 面'+it.faces.length+' 札'+bl.children.length
    +' 画像'+(NAT?'原寸':'縮小')+' '+W+'x'+H+' 実表示'+Math.round(im2.width||0)+'px';
  $('dn').textContent=it.done?'このコマは「終わり」と記録済み':'';
}
function toggle(i){ if(SEL.has(i))SEL.delete(i); else SEL.add(i); draw(); }
function hit(e){
  // **描画された大きさから原寸へ直す。** `r.width` は実際に画面に出ている幅で、
  // 窓が狭ければ画像は縮む(flex は既定で縮む)。
  // **縮んだ画像の上の座標を、縮んでいない前提で送っていたのが「選べない」の正体**
  // (2026-10-28)。ここで出すのは**原寸の座標**で、サーバはそのまま地図を引く。
  const im=$('im'), r=im.getBoundingClientRect();
  if(!r.width||!r.height) return;
  const fx=(e.clientX-r.left)/r.width, fy=(e.clientY-r.top)/r.height;
  const nx=Math.round(fx*ST.nat_w), ny=Math.round(fy*ST.nat_h);
  fetch('/hit?tag='+encodeURIComponent(ST.tag)+'&y='+ny+'&x='+nx)
    .then(r=>r.json()).then(d=>{ if(d.i>0) toggle(d.i); else
      $('dn').textContent='そこには面がありません(面の外を押した)'; });
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
// **面1つの中に意味が複数在る**(過剰結合)。まとまりの逆向きの壊れ方で、
// **まとまりとは別の種類の行**になる(2026-10-29、作者の指摘)。
function omerge(){
  if(SEL.size!==1){ alert('印は面1つに付きます。**面を1つだけ**選んでください'
    +(SEL.size?('(いま '+SEL.size+' 面)'):'')); return; }
  const w=prompt('この面の中に入っている意味を、読点か空白で区切って入力。'
    +'\n分からなければ空のまま OK —— **印だけ残します**');
  if(w===null) return;      // 取り消しは何も書かない
  save('overmerged',w);
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  if(e.key==='Enter')save('panel_done','');
  else if(e.key==='c'||e.key==='C'){NAT=!NAT;draw();}
  else if(e.key==='n')load(Math.min(ST.order+1,ST.n-1));
  else if(e.key==='b')load(Math.max(ST.order-1,0));
  else if(e.key==='m'||e.key==='M')omerge();
  else if(e.key==='Escape'){SEL=new Set();draw();}
});
'''

HTML = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<title>L38 面のまとまり</title><style>
body{font-family:system-ui,sans-serif;margin:12px;background:#fafafa}
#hdr{font-weight:700}#ask{margin:6px 0;color:#044}
#wrap{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}
#pane{position:relative;display:inline-block;border:1px solid #ccc;background:#fff;
      max-width:calc(100vw - 380px);min-width:280px}
#im{display:block;cursor:crosshair;max-width:100%;height:auto}
.bg{position:absolute;width:22px;height:22px;border-radius:11px;background:#fff;
    border:2px solid #555;font:700 12px/18px system-ui;text-align:center;cursor:pointer;
    opacity:.85;z-index:3}
.bg.on{background:#00c;color:#fff;border-color:#008;opacity:1;
       width:28px;height:28px;border-radius:14px;font:700 15px/24px system-ui;
       box-shadow:0 0 0 3px rgba(0,0,200,.35)}
/* 選んだ面を塗る層。**押下は通す**(pointer-events:none) */
.ov{position:absolute;left:0;top:0;width:100%;height:auto;pointer-events:none;z-index:2}
#side{width:330px;max-height:86vh;overflow:auto}
.row{padding:3px 6px;border-bottom:1px solid #eee;cursor:pointer;font-size:13px}
.row.on{background:#ccf;font-weight:700;border-left:6px solid #00c}
.row:hover{background:#fe8}
.crow{padding:3px 6px;background:#eef;margin:2px 0;font-size:13px}
.orow{padding:3px 6px;background:#fde8cf;margin:2px 0;font-size:13px}
/* 印の付いた面の札は**破線の橙**。束ねる前に、画面で分かる */
.bg.om{border-color:#c50;border-style:dashed;border-width:3px}
button{margin:2px;padding:6px 10px;font-size:14px}
#sel{margin:6px 0;font-weight:700}#dn{color:#a00}
</style>
<div id="hdr"></div><div id="ask"></div>
<div id="wrap">
 <div id="pane"><img id="im" onclick="hit(event)"><div id="ovs"></div>
  <div id="badges"></div></div>
 <div id="side">
  <div id="sel"></div><div id="dn"></div>
  <div id="diag" style="font-size:11px;color:#777"></div>
  <div id="btns"></div>
  <h4>このコマの面</h4><div id="list"></div>
  <h4>作ったまとまり</h4><div id="cl"></div>
  <h4>複数の意味が入っている面</h4><div id="om"></div>
  <p style="font-size:12px;color:#666">Enter = このコマは終わり / C = 原寸 /
   Esc = 選択解除 / n,b = 次,前のコマ / m = この面は複数の意味</p>
 </div></div>
<script>__JS__
const B=$('btns');
__WORDS__.forEach(w=>{const b=document.createElement('button');b.textContent=w;
  b.onclick=()=>save('word',w);B.appendChild(b);});
[['自由入力',free],['まとまりでない',()=>save('not_cluster','')],
 ['この面は複数の意味(m)',omerge],
 ['このコマは終わり',()=>save('panel_done','')]].forEach(([t,fn])=>{
  const b=document.createElement('button');b.textContent=t;b.onclick=fn;B.appendChild(b);});
load(__CUR__);
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
            body = render_page().encode('utf-8')
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
        if u.path == '/mask':
            tag, i = qs['tag'][0], int(qs['i'][0])
            c = qs.get('c', ['sel'])[0]
            if tag not in {r['tag'] for r in S['q']} or c not in ('sel', 'hov'):
                return self._send(400, 'text/plain', b'bad')
            return self._send(200, 'image/png', mask_png(tag, i, c))
        if u.path == '/hit':
            tag = qs['tag'][0]
            m = fmap(tag)
            # **受け取るのは原寸の座標**(2026-10-28 に画面側へ移した)。
            # 画面側は `getBoundingClientRect()` の**実際に描画された大きさ**から
            # 割合を出して原寸へ直す —— 窓の幅に依らない唯一の形である。
            y, x = int(qs['y'][0]), int(qs['x'][0])
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
                kind = ('panel_done' if ch == 'panel_done' else
                        'overmerged' if ch == 'overmerged' else 'cluster')
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
                if kind == 'overmerged':
                    # **空は「意味が無い」ではなく「書かなかった」。** 分けて記録する
                    ws = [w for w in re.split(r'[、,\s/／・]+', d.get('word', '')) if w]
                    row.update(words=ws, n_words=len(ws), words_given=bool(ws))
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
    nom = sum(len([x for x in S['done'][r['tag']] if x.get('kind') == 'overmerged'])
              for r in S['q'])
    nomw = sum(len([x for x in S['done'][r['tag']]
                    if x.get('kind') == 'overmerged' and x.get('words_given')])
               for r in S['q'])
    print(f'  保存の読み込み: **終わったコマ {nd} / まとまり {ncl}**'
          f'(鍵は `tag`。`order` では引かない。T144)')
    # **両方の壊れ方を、名前を付けて数える**(規律9)。印だけの行も隠さない
    print(f'  **複数の意味が入っている面の印 {nom}**'
          f'(うち中身が書かれている {nomw} / 印だけ {nom - nomw})')
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
    if page_check():
        print('**画面で選べない。立ち上げない**')
        sys.exit(5)
    if kc_hit():
        print('**当たり判定が外れている。立ち上げない** —— '
              '押した面と違う面が選ばれる UI は、作者の答を黙って壊す')
        sys.exit(3)
    print(f'  最初に出すコマ: order={S["cur"]}')
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == '__main__':
    main()

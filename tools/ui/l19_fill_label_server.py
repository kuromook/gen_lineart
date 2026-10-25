#!/usr/bin/env python
"""L19: ベタ面から単語を手で決める UI (Track L、2026-10-25)。

L17 の結論「ベタ面は線を1層に絞らない(過半の面では)。ただし迷うのは1頁 6面、その8割が2択」
の上に立つ。**候補を出して作者が選ぶ**。purity は曖昧でないことの測度で正しさの測度では
ないので、**正しさは作者の答からしか作れない**。

  python tools/ui/l19_fill_label_server.py --host 0.0.0.0          試し (labels/fill_words_trial.jsonl)
  python tools/ui/l19_fill_label_server.py --host 0.0.0.0 --final  本番 (labels/fill_words.jsonl)
  python ... --final --reask results/l19_ui/reask.json              出し直し (語彙が増える前に答えた分)
  -> http://<host>:8475

**自動で決まる面も 1割が同じ見た目で混ざっている**(`why` はクライアントに送らない)。
"""
import argparse, json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent.parent
UI = ROOT / 'results/l19_ui'
LAB = ROOT / 'labels'
PORT = 8475
COLS = ['#FF3C3C', '#3CC83C', '#008CFF', '#DC3CC8', '#DCC800']
EXTRA_WORDS = ['まつ毛', '眉', 'フキダシ', '黒消し']   # 黒消しは自由入力から昇格(4件)        # 層に対応が無い語(縁に接していなくても押せる)

lock = Lock()
S = {}


def load_state(save_path, reask=None):
    S['save'] = save_path
    d = json.load(open(UI / 'queue.json'))
    S['q'] = d['queue']
    S['meta'] = {k: d[k] for k in ('n', 'n_ask', 'n_check', 'n_auto', 'n_all',
                                   'words')}
    for w in EXTRA_WORDS:                 # 作り直さずに語を足せるようにしておく
        if w not in S['meta']['words']:
            S['meta']['words'].append(w)
    S['done'] = {}
    S['only'] = None                      # 出し直しのときだけ、行列を部分集合に絞る
    S['re'] = set()                       # この走りで答え直した問
    if reask:
        got = {int(o) for o in json.loads(Path(reask).read_text())}
        S['only'] = sorted(o for o in got if 0 <= o < len(S['q']))
    LAB.mkdir(exist_ok=True)
    if save_path.exists():
        for ln in open(save_path):
            if ln.strip():
                r = json.loads(ln)
                S['done'][int(r['order'])] = r


def payload(o):
    r = dict(S['q'][o])
    r.pop('why', None)                      # **出自は出さない**
    r['cols'] = COLS[:len(r['cands'])]
    r['words'] = S['meta']['words']
    r['saved'] = S['done'].get(o)
    only = S['only']
    if only is None:
        r['n'], r['pos'] = len(S['q']), o + 1
        r['n_done'] = len(S['done'])
        r['nav_prev'], r['nav_next'] = max(0, o - 1), min(len(S['q']) - 1, o + 1)
    else:
        i = only.index(o) if o in only else 0
        r['n'], r['pos'] = len(only), i + 1
        r['n_done'] = len(S['re'])
        r['nav_prev'], r['nav_next'] = only[max(0, i - 1)], only[min(len(only) - 1, i + 1)]
    return r


def first_todo():
    if S['only'] is not None:               # 出し直しでは「この走りで答え直したか」を見る
        for o in S['only']:
            if o not in S['re']:
                return o
        return S['only'][-1]
    for i in range(len(S['q'])):
        if i not in S['done']:
            return i
    return len(S['q']) - 1


def save(d, annotator):
    o = int(d['order'])
    q = S['q'][o]
    rec = dict(order=o, work=q['work'], pid=q['pid'], comp=q['comp'],
               region=q['region'], why=q.get('why'), auto=q['auto'],
               purity=q['purity'], n_lay=q['n_lay'], n_lay90=q['n_lay90'],
               top_lid=q['cands'][0]['lid'], top_name=q['cands'][0]['name'],
               top_word=q.get('top_word', ''), word_share=q.get('word_share', {}),
               choice=d.get('choice'), word=d.get('word'), free=d.get('free'),
               ms=int(d.get('ms', 0)), annotator=annotator,
               ts=time.strftime('%Y-%m-%dT%H:%M:%S'))
    rec['agree'] = int(rec['choice'] == 'word' and rec['word'] == rec['top_word']
                       and bool(rec['top_word']))
    with lock:
        S['done'][o] = rec
        S['re'].add(o)
        with open(S['save'], 'a') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return rec


PAGE = """<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>このベタは何</title>
<style>
:root{color-scheme:light dark}
body{margin:0;font:15px/1.5 system-ui,sans-serif;background:#fff;color:#111}
@media (prefers-color-scheme:dark){body{background:#151515;color:#eee}}
#top{position:sticky;top:0;background:inherit;z-index:3;max-height:55vh;overflow:auto;
  border-bottom:1px solid #8886;box-shadow:0 2px 6px #0002}
header{display:flex;gap:12px;align-items:baseline;padding:8px 12px}
#img{display:block;margin:8px auto;max-width:100%;height:auto;
  image-rendering:pixelated;border:1px solid #8884}
#words{display:flex;flex-wrap:wrap;gap:6px;padding:0 12px 4px}
#other{display:flex;flex-wrap:wrap;gap:6px;padding:0 12px 8px}
button{padding:9px 11px}
button{font:inherit;padding:12px 14px;border:1px solid #8886;border-radius:8px;
  background:#8881;color:inherit;cursor:pointer;display:flex;gap:7px;align-items:center}
button:hover{background:#8883}
button.hit{border-color:#888;background:#8882;font-weight:600}
.sw{width:14px;height:14px;border-radius:3px;display:inline-block;flex:none}
.k{opacity:.55;font-size:12px}
.meta{opacity:.75;font-size:13px;padding:2px 12px}
#hint{display:flex;flex-wrap:wrap;gap:10px;padding:2px 12px 8px;font-size:13px;opacity:.85}
#done{color:#2a2;font-weight:600}
</style>
<div id="top">
<header>
  <b>この黒いベタは何?</b>
  <span id="pos" class="meta"></span><span id="done"></span>
  <span style="flex:1"></span>
  <button onclick="prv()">← <span class="k">B</span></button>
  <button onclick="nxt()">→ <span class="k">N</span></button>
  <button onclick="ctx=!ctx;showImg()">コマ全体 <span class="k">C</span></button>
</header>
<div id="words"></div>
<div id="other"></div>
</div>
<img id="img" alt="">
<div class="meta" id="info"></div>
<div id="hint"></div>
<div class="meta">黒い塊が聞いている面。色は縁 4px に接している線のレイヤ(ヒント。答ではない)。
工程の産物(隙間埋め)なら G / H。</div>
<script>
let cur=0, it=null, t0=0, ctx=false, keymap={};
async function go(o){
  if(o<0) o=0;
  const r=await fetch('/api/item?o='+o); if(!r.ok) return;
  it=await r.json(); cur=o; t0=performance.now(); ctx=false; showImg();
  document.getElementById('pos').textContent=it.pos+' / '+it.n;
  document.getElementById('done').textContent=it.n_done+' 済み';
  // 縁に接している層(色のヒント)
  const h=document.getElementById('hint'); h.innerHTML='';
  it.cands.forEach((x,i)=>{
    const d=document.createElement('span');
    d.innerHTML='<span class="sw" style="background:'+it.cols[i]+'"></span> '+
      (x.name||'')+(x.word?'':' <span class="k">(語に対応しない層)</span>')+
      ' <span class="k">'+(x.share*100).toFixed(0)+'%</span>';
    d.style.display='flex'; d.style.gap='6px'; d.style.alignItems='center';
    h.appendChild(d);
  });
  // 解答欄 = 語。**並びは固定**(位置が毎回同じになるように。2026-10-25 作者の指摘)
  const ws=it.words;
  const w=document.getElementById('words'); w.innerHTML=''; keymap={};
  ws.forEach((word,i)=>{
    const sh=it.word_share[word];
    const ci=it.cands.findIndex(c=>c.word===word);
    const b=document.createElement('button');
    b.innerHTML=(ci>=0?'<span class="sw" style="background:'+it.cols[ci]+'"></span>':'')+
      word+(sh?' <span class="k">'+(sh*100).toFixed(0)+'%</span>':'')+
      ' <span class="k">'+(i+1)+'</span>';
    if(sh) b.className='hit';
    b.onclick=()=>pick('word',word,null); w.appendChild(b);
    if(i<9) keymap[String(i+1)]=word;
  });
  const o2=document.getElementById('other'); o2.innerHTML='';
  [['gapfill_line','隙間埋め(線画の抜けを足した)','G'],
   ['gapfill_fill','隙間埋め(別レイヤのベタ用)','H'],
   ['free','その他(語を入力)','T'],['mixed','混在(1語にできない)','M'],
   ['skip','分からない','S']].forEach(([k,t,key])=>{
    const b=document.createElement('button');
    b.innerHTML=t+' <span class="k">'+key+'</span>';
    b.onclick=()=>act(k); o2.appendChild(b);
  });
  info();
}
function nxt(){ if(it) go(it.nav_next); }
function prv(){ if(it) go(it.nav_prev); }
function act(k){
  if(k==='free'){const v=prompt('この黒は何ですか(語を入力)'); if(v) pick('free',null,v);}
  else pick(k,null,null);
}
function showImg(){
  document.getElementById('img').src='/'+(ctx?'ctx':'crop')+'?o='+cur+'&t='+Date.now();
  info();
}
function info(){
  if(!it) return;
  const sc=ctx?it.ctx_scale:it.scale;
  document.getElementById('info').textContent=
    (ctx?'[コマ全体] ':'[面の周り] ')+it.work+' '+it.pid+' コマ'+it.comp+' 面'+it.region+
    '  面積 '+it.area.toLocaleString()+'px'+
    (sc<1?'  (表示は '+(sc*100).toFixed(0)+'%)':'  (原寸)')+
    (it.saved?'  ← 記録済み: '+(it.saved.word||it.saved.free||it.saved.choice):'');
}
async function pick(choice,word,free){
  await fetch('/api/save',{method:'POST',body:JSON.stringify({
    order:cur,choice:choice,word:word,free:free,
    ms:Math.round(performance.now()-t0)})});
  nxt();
}
addEventListener('keydown',e=>{
  if(!it||e.metaKey||e.ctrlKey) return;
  const k=e.key.toLowerCase();
  if(keymap[k]) pick('word',keymap[k],null);
  else if(k==='g') act('gapfill_line');
  else if(k==='h') act('gapfill_fill');
  else if(k==='t') act('free');
  else if(k==='m') act('mixed');
  else if(k==='s') act('skip');
  else if(k==='c'){ctx=!ctx;showImg();}
  else if(k==='b') prv();
  else if(k==='n') nxt();
});
fetch('/api/todo').then(r=>r.json()).then(d=>go(d.o));
</script>
</html>"""


class H(BaseHTTPRequestHandler):
    annotator = 'user'

    def log_message(self, *a):
        pass

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            if u.path == '/':
                return self._send(PAGE.encode(), 'text/html; charset=utf-8')
            if u.path == '/api/todo':
                return self._send(json.dumps(dict(o=first_todo())).encode(),
                                  'application/json')
            o = int(q['o'][0])
            if not 0 <= o < len(S['q']):
                return self._send(b'out of range', 'text/plain', 404)
            if u.path == '/api/item':
                return self._send(json.dumps(payload(o), ensure_ascii=False).encode(),
                                  'application/json')
            if u.path in ('/crop', '/ctx'):
                key = 'crop' if u.path == '/crop' else 'ctx'
                p = UI / 'crops' / S['q'][o][key]
                return self._send(p.read_bytes(), 'image/png')
            self._send(b'not found', 'text/plain', 404)
        except Exception as e:                                      # noqa: BLE001
            self._send(str(e).encode(), 'text/plain', 500)

    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            save(d, self.annotator)
            self._send(b'ok', 'text/plain')
        except Exception as e:                                      # noqa: BLE001
            self._send(str(e).encode(), 'text/plain', 400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--annotator', default='user')
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--reask', help='出し直す order の JSON 配列')
    a = ap.parse_args()
    load_state(LAB / ('fill_words.jsonl' if a.final else 'fill_words_trial.jsonl'),
               a.reask)
    m = S['meta']
    if S['only'] is None:
        print(f"行列 {m['n']} (要判断 {m['n_ask']} + 自動の確認 {m['n_check']})  "
              f"保存済み {len(S['done'])}  保存先 {S['save'].name}  "
              f"-> http://{a.host}:{a.port}", flush=True)
    else:
        print(f"出し直し {len(S['only'])} 問 (全 {m['n']} 問のうち)  "
              f"保存先 {S['save'].name} に追記  "
              f"-> http://{a.host}:{a.port}", flush=True)
    H.annotator = a.annotator
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""線単位のまとめ直し UI(Track H、2026-10-09 事前登録)。

labels/face_queue_v1.csv の順に顔を 1 つずつ出し、線を選んで「1 つのパーツを描く線の集まり」= グループにする。
背景は線画 PNG(外部の基準)。選べる線は panel_pack strokes.npy の線((y, x) 順 → 画像の (x, y) に直す)。

Run:  python tools/face/group_server.py --host 0.0.0.0          (試し。labels/face_groups_trial.jsonl)
      python tools/face/group_server.py --host 0.0.0.0 --final  (本番。labels/face_groups.jsonl)
      → http://<host>:8473
"""
import argparse, csv, json, time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
TRACKF = Path("/home/sh1/deepl/lineart-stroke-grammar")
PACK = TRACKF / "results/panel_pack_20260919"
DATASET = Path("/home/sh1/deepl/lineart/dataset")
LAB = ROOT / "labels"
QUEUE = LAB / "face_queue_v1.csv"
PORT = 8473
LABELS = ["face.eye.open", "face.eye.closed", "face.brow", "face.nose", "face.mouth", "face.ear", "face.other"]
VIEW_K, PICK_K, IMG_PX = 1.9, 1.5, 1500      # 表示の窓 / 選べる線の範囲(枠の何倍か)/ 背景の解像度

lock = Lock()
S = {}


def load_state(save_path):
    S["save"] = save_path
    S["queue"] = list(csv.DictReader(open(QUEUE)))
    S["panels"] = list(csv.DictReader(open(PACK / "panels.csv")))
    S["strokes"] = np.load(PACK / "strokes.npy", mmap_mode="r")
    S["clab"] = np.load(PACK / "cluster_labels_noend.npy", mmap_mode="r")
    S["png"] = OrderedDict()
    S["faces"] = {}
    if save_path.exists():
        for ln in open(save_path):
            if ln.strip():
                d = json.loads(ln); S["faces"][int(d["order"])] = d


def panel_img(pid):
    c = S["png"]
    if pid not in c:
        r = S["panels"][pid]
        c[pid] = cv2.imread(str(DATASET / r["source"] / "line" / r["name"]))
        while len(c) > 6:
            c.popitem(last=False)
    c.move_to_end(pid)
    return c[pid]


def window(q):
    x0, y0, x1, y1 = (float(q[k]) for k in ("x0", "y0", "x1", "y1"))
    side = VIEW_K * max(x1 - x0, y1 - y0)
    return (x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2, side


def panel_strokes(pid):
    """返る: pts (n, 16, 2) 画像の (x, y)、太さ (n,)、まとまり番号 (n,)"""
    r = S["panels"][pid]
    s, n = int(r["start"]), int(r["n"])
    block = np.asarray(S["strokes"][s:s + n])
    pts = block[:, :32].reshape(n, 16, 2)[..., ::-1].astype(np.float64)      # (y, x) → (x, y)
    return pts, block[:, 33], np.asarray(S["clab"][s:s + n])


def face_payload(order):
    q = S["queue"][order]
    pid = int(q["panel"])
    x0, y0, x1, y1 = (float(q[k]) for k in ("x0", "y0", "x1", "y1"))
    pts, wid, cl = panel_strokes(pid)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = PICK_K * (x1 - x0) / 2, PICK_K * (y1 - y0) / 2
    inside = ((np.abs(pts[..., 0] - cx) <= hw) & (np.abs(pts[..., 1] - cy) <= hh)).any(1)
    csize = np.bincount(cl[cl >= 0], minlength=int(cl.max()) + 1) if len(cl) else np.zeros(1, int)
    strokes = [dict(id=int(i), p=np.round(pts[i], 1).reshape(-1).tolist(), w=round(float(wid[i]), 1),
                    c=int(cl[i]), cs=int(csize[cl[i]]) if cl[i] >= 0 else 0) for i in np.flatnonzero(inside)]
    wx, wy, side = window(q)
    saved = S["faces"].get(order)
    return dict(order=order, panel=pid, box=[x0, y0, x1, y1], win=[wx, wy, side], work=q["work"], conf=float(q["conf"]),
                strokes=strokes, saved=saved, n_queue=len(S["queue"]))


def face_png(order):
    q = S["queue"][order]
    img = panel_img(int(q["panel"]))
    wx, wy, side = window(q)
    f = IMG_PX / side
    A = np.array([[f, 0, -wx * f], [0, f, -wy * f]])
    out = cv2.warpAffine(img, A, (IMG_PX, IMG_PX), flags=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(236, 236, 236))
    return cv2.imencode(".png", out)[1].tobytes()


def stats():
    f = S["faces"]
    done = sum(1 for d in f.values() if d["status"] == "done")
    skip = sum(1 for d in f.values() if d["status"] == "skip")
    nxt = next((o for o in range(len(S["queue"])) if o not in f), len(S["queue"]))
    return dict(done=done, skip=skip, next=nxt, groups=sum(len(d["groups"]) for d in f.values() if d["status"] == "done"),
                mode="final" if S["save"].name == "face_groups.jsonl" else "trial")


def save(d, annotator):
    order = int(d["order"])
    q = S["queue"][order]
    assert d["status"] in ("done", "skip")
    valid = {s["id"] for s in face_payload(order)["strokes"]}
    groups, used = [], set()
    for g in d.get("groups", []) if d["status"] == "done" else []:
        ids = sorted({int(i) for i in g["strokes"]})
        assert g["label"] in LABELS, f"目印が不正: {g['label']}"
        assert ids and set(ids) <= valid, "この顔で選べない線が入っている"
        assert not (set(ids) & used), "同じ線が 2 つのグループに入っている"
        used |= set(ids)
        groups.append(dict(label=g["label"], strokes=ids, unsplit=bool(g.get("unsplit", False))))
    rec = dict(order=order, panel=int(q["panel"]), box=int(q["box"]), usage=q["usage"], status=d["status"],
               groups=groups, n_strokes=len(valid), annotator=annotator, datetime=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with lock:
        S["faces"][order] = rec
        with open(S["save"], "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no"><title>Track H まとめ直し</title>
<style>
:root{--bg:#f4f4f2;--fg:#1c1c1c;--mut:#6b6b6b;--line:#cfcfcb;--card:#fff;--acc:#b3261e}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,"Noto Sans JP",sans-serif;overflow:hidden;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none;overscroll-behavior:none}
.app{display:flex;gap:12px;height:100%;padding:10px}
.stage{position:relative;flex:0 0 auto}
canvas{background:#fff;border:1px solid var(--line);display:block;cursor:crosshair;touch-action:none}
.side{flex:1 1 300px;min-width:290px;max-width:430px;display:flex;flex-direction:column;gap:8px;overflow:auto;min-height:0}
.app.port{flex-direction:column}.app.port .side{max-width:none;flex:1 1 0}
@media (pointer:coarse){button{padding:10px 12px}.g{padding:8px}.g .x{padding:6px 12px}}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 10px}
h1{font-size:15px;margin:0 0 2px}.mut{color:var(--mut)}.small{font-size:12px}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
button{font:inherit;padding:6px 9px;border:1px solid var(--line);background:#fafaf8;border-radius:5px;cursor:pointer}
button.on{background:#1c1c1c;color:#fff;border-color:#1c1c1c}
button.prim{background:var(--acc);color:#fff;border-color:var(--acc)}
kbd{display:inline-block;min-width:1.3em;text-align:center;padding:0 4px;border:1px solid currentColor;border-radius:4px;font:11px ui-monospace,monospace;opacity:.75;margin-right:4px}
#groups{display:flex;flex-direction:column;gap:4px}
.g{display:flex;align-items:center;gap:6px;padding:4px 6px;border:1px solid var(--line);border-radius:5px;cursor:pointer;background:#fff}
.g.cur{outline:2px solid #1c1c1c}.chip{width:14px;height:14px;border-radius:3px;flex:0 0 auto}
.g .x{margin-left:auto;padding:1px 7px}.g .u{padding:1px 6px;font-size:12px}
#msg{color:var(--acc);min-height:1.3em}
ul{margin:4px 0 0;padding-left:18px}li{margin:1px 0}
#mode{font-weight:600}
</style></head><body><div class="app">
<div class="stage"><canvas id="cv"></canvas></div>
<div class="side">
 <div class="card"><h1>まとめ直し <span id="mode"></span></h1>
  <div id="pos"></div><div id="info" class="mut small"></div><div id="prog" class="mut small"></div><div id="msg"></div></div>
 <div class="card"><div class="row">
   <button id="tLasso" onclick="setTool('lasso')"><kbd>L</kbd>投げ縄</button>
   <button id="tRect" onclick="setTool('rect')"><kbd>R</kbd>矩形</button>
   <button id="tPan" onclick="setTool('pan')"><kbd>H</kbd>移動</button>
   <button onclick="undo()"><kbd>Ctrl+Z</kbd>取り消し</button>
   <button onclick="cycleBg()"><kbd>B</kbd>原画の濃さ</button>
   <button onclick="resetView()"><kbd>F</kbd>全体表示</button></div>
  <div class="row" style="margin-top:6px"><span class="mut small">選び方</span>
   <button id="mNew" onclick="setMode('new')">新しく</button>
   <button id="mAdd" onclick="setMode('add')">追加</button>
   <button id="mSub" onclick="setMode('sub')">除外</button>
   <button onclick="clearSel()"><kbd>Esc</kbd>選択を解除</button></div>
  <div class="row" style="margin-top:6px" id="labBtns"></div>
  <div class="row" style="margin-top:6px">
   <button onclick="ungroup()"><kbd>X</kbd>選択をグループから外す</button>
   <button onclick="toggleUnsplit()"><kbd>M</kbd>分けられない線あり</button></div></div>
 <div class="card"><div class="row" style="justify-content:space-between"><b>グループ</b><span id="selInfo" class="mut small"></span></div>
  <div id="groups" style="margin-top:6px"></div></div>
 <div class="card"><div class="row">
   <button class="prim" onclick="finish('done')"><kbd>Enter</kbd>この顔は完了 → 次へ</button>
   <button onclick="finish('skip')"><kbd>K</kbd>飛ばす(顔でない・小さすぎ)</button></div>
  <div class="row" style="margin-top:6px"><button onclick="nav(-1)"><kbd>←</kbd>前の顔</button><button onclick="nav(1)"><kbd>→</kbd>次の顔</button>
   <button onclick="go(st.next)"><kbd>N</kbd>未完了の先頭へ</button></div></div>
 <div class="card small mut"><b>使い方</b><ul>
  <li>クリック(タップ): 線を 1 本選ぶ / ダブルクリック(2 回タップ): いまのまとまりを丸ごと選ぶ</li>
  <li>タブレット: 指 1 本またはペンで選ぶ / 指 2 本で拡大・縮小と移動 / 追加・除外は「選び方」のボタン</li>
  <li>ドラッグ: 投げ縄(または矩形)で囲んだ線を選ぶ</li>
  <li>Shift を押しながら: 追加 / Alt を押しながら: 除外 / Esc: 選択を解除</li>
  <li>線を選んで 1〜7: その目印のグループにする(左目と右目は別々に)</li>
  <li>ホイール: 拡大・縮小 / 右ドラッグ(または Space+ドラッグ): 移動</li>
  <li>グループにしなかった線は「それ以外」(髪・輪郭の外・背景)として扱います</li>
  <li>青い点線の枠 = 検出された顔。表示されている線はすべて選べます</li></ul></div>
</div></div>
<script>
const DEF=[["1","face.eye.open","目(開)"],["2","face.eye.closed","目(閉)"],["3","face.brow","眉"],["4","face.nose","鼻"],["5","face.mouth","口"],["6","face.ear","耳"],["7","face.other","その他の顔"]];
const PAL=["#d1495b","#1f7a8c","#e09f3e","#5b8e3f","#7b4fa3","#c2577a","#2e6fb7","#8a6d3b","#3aa39a","#a03f2d","#5d6bd0","#6b8f23"];
const $=id=>document.getElementById(id), cv=$("cv"), ctx=cv.getContext("2d");
let F=null, bg=new Image(), bgOK=false, bgMode=0, st={done:0,skip:0,next:0,mode:""};
let strokes=[], byId=new Map(), sel=new Set(), groups=[], gseq=0, hist=[], dirty=false, cur=0;
let view={s:1,ox:0,oy:0}, tool="lasso", drag=null, hover=null, spaceDown=false, SIZE=900, selMode="new", pinch=null, lastTap=null; const pts=new Map();

function fit(){ const W=window.innerWidth,H=window.innerHeight,port=H>W*1.05; document.querySelector(".app").classList.toggle("port",port);
  SIZE=port?Math.max(320,Math.min(W-22,Math.round(H*0.62))):Math.max(320,Math.min(H-22,W-330)); const d=window.devicePixelRatio||1;
  cv.style.width=cv.style.height=SIZE+"px"; cv.width=cv.height=Math.round(SIZE*d); ctx.setTransform(d,0,0,d,0,0); draw(); }
function base(){ return SIZE/F.win[2]; }
function w2s(x,y){ const b=base()*view.s; return [(x-F.win[0])*b+view.ox,(y-F.win[1])*b+view.oy]; }
function s2w(px,py){ const b=base()*view.s; return [(px-view.ox)/b+F.win[0],(py-view.oy)/b+F.win[1]]; }
function resetView(){ view={s:1,ox:0,oy:0}; draw(); }
function groupOf(id){ for(const g of groups) if(g.strokes.has(id)) return g; return null; }
function snapshot(){ hist.push(JSON.stringify({g:groups.map(g=>({id:g.id,label:g.label,unsplit:g.unsplit,strokes:[...g.strokes]})),s:[...sel]})); if(hist.length>200)hist.shift(); }
function undo(){ if(!hist.length)return; const h=JSON.parse(hist.pop()); groups=h.g.map(g=>({...g,strokes:new Set(g.strokes)})); sel=new Set(h.s); dirty=true; refresh(); }

function path(s){ const p=s.p; ctx.beginPath(); let a=w2s(p[0],p[1]); ctx.moveTo(a[0],a[1]); for(let i=2;i<32;i+=2){ a=w2s(p[i],p[i+1]); ctx.lineTo(a[0],a[1]); } }
function draw(){ if(!F)return; ctx.clearRect(0,0,SIZE,SIZE); ctx.fillStyle="#fff"; ctx.fillRect(0,0,SIZE,SIZE);
  const al=[0.28,0.85,0][bgMode];
  if(bgOK&&al>0){ ctx.globalAlpha=al; const b=SIZE*view.s; ctx.drawImage(bg,view.ox,view.oy,b,b); ctx.globalAlpha=1; }
  const a=w2s(F.box[0],F.box[1]), c=w2s(F.box[2],F.box[3]); ctx.strokeStyle="#3c8dde"; ctx.lineWidth=1; ctx.setLineDash([5,4]); ctx.strokeRect(a[0],a[1],c[0]-a[0],c[1]-a[1]); ctx.setLineDash([]);
  ctx.lineCap="round"; ctx.lineJoin="round";
  const gcol=new Map(); groups.forEach((g,i)=>g.strokes.forEach(id=>gcol.set(id,PAL[g.color%PAL.length])));
  for(const s of strokes){ if(gcol.has(s.id)||sel.has(s.id))continue; path(s); ctx.strokeStyle=(hover&&hover.c>=0&&s.c===hover.c)?"#111":"#6f6f6f"; ctx.lineWidth=(hover&&hover.c>=0&&s.c===hover.c)?2.2:1.2; ctx.stroke(); }
  for(const s of strokes){ if(!gcol.has(s.id)||sel.has(s.id))continue; path(s); ctx.strokeStyle=gcol.get(s.id); ctx.lineWidth=3.2; ctx.stroke(); }
  for(const s of strokes){ if(!sel.has(s.id))continue; path(s); ctx.strokeStyle="#111"; ctx.lineWidth=5.5; ctx.stroke(); path(s); ctx.strokeStyle=gcol.has(s.id)?gcol.get(s.id):"#ffb000"; ctx.lineWidth=3; ctx.stroke(); }
  ctx.font="12px system-ui"; ctx.textBaseline="bottom";
  for(const g of groups){ let x=0,y=1e9,n=0; g.strokes.forEach(id=>{const s=byId.get(id); for(let i=0;i<32;i+=2){const q=w2s(s.p[i],s.p[i+1]); x+=q[0]; n++; if(q[1]<y)y=q[1];}});
    if(!n)continue; const t=DEF.find(d=>d[1]==g.label)[2]+(g.unsplit?" *":""); const tw=ctx.measureText(t).width; x=x/n-tw/2; y-=4;
    ctx.fillStyle="rgba(255,255,255,.88)"; ctx.fillRect(x-3,y-15,tw+6,16); ctx.fillStyle="#111"; ctx.fillText(t,x,y); }
  if(drag&&drag.moved&&drag.kind=="select"){ ctx.strokeStyle="#111"; ctx.lineWidth=1; ctx.setLineDash([4,3]); ctx.beginPath();
    if(tool=="rect"){ ctx.rect(drag.x0,drag.y0,drag.x-drag.x0,drag.y-drag.y0); } else { drag.poly.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1])); ctx.closePath(); }
    ctx.stroke(); ctx.setLineDash([]); }
}
function segDist(px,py,ax,ay,bx,by){ const dx=bx-ax,dy=by-ay,l=dx*dx+dy*dy; let t=l?((px-ax)*dx+(py-ay)*dy)/l:0; t=Math.max(0,Math.min(1,t)); return Math.hypot(px-ax-t*dx,py-ay-t*dy); }
function nearest(px,py,tol){ let best=null,bd=tol||8; for(const s of strokes){ let a=w2s(s.p[0],s.p[1]); for(let i=2;i<32;i+=2){ const b=w2s(s.p[i],s.p[i+1]); const d=segDist(px,py,a[0],a[1],b[0],b[1]); if(d<bd){bd=d;best=s;} a=b; } } return best; }
function inPoly(x,y,poly){ let c=false; for(let i=0,j=poly.length-1;i<poly.length;j=i++){ const a=poly[i],b=poly[j]; if((a[1]>y)!=(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])c=!c; } return c; }
function pickArea(test){ const out=[]; for(const s of strokes){ let n=0; for(let i=0;i<32;i+=2){ const q=w2s(s.p[i],s.p[i+1]); if(test(q[0],q[1]))n++; } if(n>=8)out.push(s.id); } return out; }
function modeOf(e){ return e.altKey?"sub":e.shiftKey?"add":selMode; }
function applySel(ids,m){ if(!ids.length&&m!="new")return; snapshot(); if(m=="sub"){ ids.forEach(i=>sel.delete(i)); } else if(m=="add"){ ids.forEach(i=>sel.add(i)); } else { sel=new Set(ids); } refresh(); }
function pos(e){ const r=cv.getBoundingClientRect(); return [e.clientX-r.left,e.clientY-r.top]; }

cv.addEventListener("contextmenu",e=>e.preventDefault());
function clampS(v){ return Math.max(0.6,Math.min(12,v)); }
function two(){ const [a,b]=[...pts.values()]; return {d:Math.hypot(a[0]-b[0],a[1]-b[1])||1,cx:(a[0]+b[0])/2,cy:(a[1]+b[1])/2}; }
cv.addEventListener("pointerdown",e=>{ const [x,y]=pos(e); e.preventDefault(); try{cv.setPointerCapture(e.pointerId);}catch(_){}
  pts.set(e.pointerId,[x,y]);
  if(pts.size==2){ drag=null; const t=two(); pinch={d:t.d,cx:t.cx,cy:t.cy,s:view.s,ox:view.ox,oy:view.oy}; draw(); return; }
  if(pts.size>2)return;
  const mouse=e.pointerType=="mouse", pan=e.button==2||e.button==1||spaceDown||tool=="pan";
  drag={kind:pan?"pan":"select",pid:e.pointerId,x0:x,y0:y,x,y,poly:[[x,y]],moved:false,ox:view.ox,oy:view.oy,tol:mouse?8:20,slop:mouse?4:10}; });
cv.addEventListener("pointermove",e=>{ const [x,y]=pos(e);
  if(pinch){ if(!pts.has(e.pointerId))return; pts.set(e.pointerId,[x,y]); if(pts.size<2)return; const t=two(), ns=clampS(pinch.s*t.d/pinch.d), r=ns/pinch.s;
    view.s=ns; view.ox=t.cx-(pinch.cx-pinch.ox)*r; view.oy=t.cy-(pinch.cy-pinch.oy)*r; draw(); return; }
  if(!drag){ if(e.pointerType=="mouse"){ const h=nearest(x,y); if(h!==hover){hover=h;draw();} } return; }
  if(e.pointerId!=drag.pid)return; pts.set(e.pointerId,[x,y]);
  drag.x=x; drag.y=y; if(Math.hypot(x-drag.x0,y-drag.y0)>drag.slop)drag.moved=true;
  if(drag.kind=="pan"){ view.ox=drag.ox+x-drag.x0; view.oy=drag.oy+y-drag.y0; } else drag.poly.push([x,y]); draw(); });
function pointerEnd(e,cancel){ pts.delete(e.pointerId);
  if(pinch){ if(pts.size<2)pinch=null; return; }
  if(!drag||e.pointerId!=drag.pid)return; const d=drag; drag=null; const m=modeOf(e);
  if(!cancel&&d.kind=="select"){
    if(!d.moved){ const s=nearest(d.x0,d.y0,d.tol), now=performance.now();
      if(s&&s.c>=0&&lastTap&&now-lastTap.t<380&&Math.hypot(d.x0-lastTap.x,d.y0-lastTap.y)<28){ lastTap=null; applySel(strokes.filter(t=>t.c===s.c).map(t=>t.id),m=="sub"?"sub":(m=="add"?"add":"new")); }
      else { lastTap={t:now,x:d.x0,y:d.y0}; applySel(s?[s.id]:[],m); } }
    else if(tool=="rect"){ const xa=Math.min(d.x0,d.x),xb=Math.max(d.x0,d.x),ya=Math.min(d.y0,d.y),yb=Math.max(d.y0,d.y); applySel(pickArea((x,y)=>x>=xa&&x<=xb&&y>=ya&&y<=yb),m); }
    else if(d.poly.length>2) applySel(pickArea((x,y)=>inPoly(x,y,d.poly)),m); }
  draw(); }
cv.addEventListener("pointerup",e=>pointerEnd(e,false));
cv.addEventListener("pointercancel",e=>pointerEnd(e,true));
cv.addEventListener("pointerleave",e=>{ if(!drag&&hover){hover=null;draw();} });
cv.addEventListener("wheel",e=>{ e.preventDefault(); const [x,y]=pos(e); const k=Math.exp(-e.deltaY*0.0015); const ns=clampS(view.s*k); const r=ns/view.s;
  view.ox=x-(x-view.ox)*r; view.oy=y-(y-view.oy)*r; view.s=ns; draw(); },{passive:false});

function makeGroup(label){ if(!sel.size){ $("msg").textContent="先に線を選んでください"; return; } snapshot();
  for(const g of groups) sel.forEach(i=>g.strokes.delete(i)); groups=groups.filter(g=>g.strokes.size);
  groups.push({id:++gseq,label,unsplit:false,strokes:new Set(sel),color:nextColor()}); sel=new Set(); dirty=true; $("msg").textContent=""; refresh(); }
function nextColor(){ const used=new Set(groups.map(g=>g.color)); for(let i=0;i<PAL.length;i++) if(!used.has(i)) return i; return groups.length%PAL.length; }
function ungroup(){ if(!sel.size)return; snapshot(); for(const g of groups) sel.forEach(i=>g.strokes.delete(i)); groups=groups.filter(g=>g.strokes.size); dirty=true; refresh(); }
function touched(){ return groups.filter(g=>[...sel].some(i=>g.strokes.has(i))); }
function toggleUnsplit(){ const t=touched(); if(!t.length){ $("msg").textContent="印を付けるグループの線を選んでください"; return; } snapshot(); t.forEach(g=>g.unsplit=!g.unsplit); dirty=true; refresh(); }
function delGroup(id){ snapshot(); groups=groups.filter(g=>g.id!=id); dirty=true; refresh(); }
function selGroup(id){ const g=groups.find(g=>g.id==id); if(g){ sel=new Set(g.strokes); refresh(); } }
function setTool(t){ tool=t; $("tLasso").className=t=="lasso"?"on":""; $("tRect").className=t=="rect"?"on":""; $("tPan").className=t=="pan"?"on":""; cv.style.cursor=t=="pan"?"grab":"crosshair"; }
function setMode(m){ selMode=m; $("mNew").className=m=="new"?"on":""; $("mAdd").className=m=="add"?"on":""; $("mSub").className=m=="sub"?"on":""; }
function clearSel(){ sel=new Set(); refresh(); }
function cycleBg(){ bgMode=(bgMode+1)%3; draw(); }
function refresh(){ const t=new Set(touched().map(g=>g.id));
  $("groups").innerHTML=groups.length?groups.map(g=>`<div class="g ${t.has(g.id)?"cur":""}" onclick="selGroup(${g.id})"><span class="chip" style="background:${PAL[g.color%PAL.length]}"></span>${DEF.find(d=>d[1]==g.label)[2]}<span class="mut small">${g.strokes.size} 本${g.unsplit?"・分けられない線あり":""}</span><button class="x" onclick="event.stopPropagation();delGroup(${g.id})">×</button></div>`).join(""):'<span class="mut small">まだありません</span>';
  const un=[...sel].filter(i=>!groupOf(i)).length; $("selInfo").textContent=sel.size?`選択 ${sel.size} 本(うち未グループ ${un})`:""; draw(); }
async function loadStats(){ st=await (await fetch("/api/stats")).json(); $("mode").textContent=st.mode=="final"?"":"(試し)"; $("prog").textContent=`完了 ${st.done} 顔 / 飛ばし ${st.skip} / グループ ${st.groups}`; }
async function go(k){ if(k<0)return; if(dirty&&!confirm("この顔の変更は保存されていません。移動しますか?"))return;
  const r=await fetch(`/api/face?o=${k}`); if(!r.ok){ $("msg").textContent="これ以上ありません"; return; }
  F=await r.json(); cur=F.order; strokes=F.strokes; byId=new Map(strokes.map(s=>[s.id,s])); sel=new Set(); groups=[]; gseq=0; hist=[]; dirty=false; hover=null; view={s:1,ox:0,oy:0};
  if(F.saved&&F.saved.status=="done") groups=F.saved.groups.map((g,i)=>({id:++gseq,label:g.label,unsplit:g.unsplit,strokes:new Set(g.strokes.filter(id=>byId.has(id))),color:i%PAL.length}));
  $("pos").textContent=`${cur+1} 番目の顔`+(F.saved?(F.saved.status=="done"?"(完了済み)":"(飛ばし済み)"):"");
  $("info").textContent=`コマ p${F.panel} / ${F.work} / 線 ${strokes.length} 本 / 検出の信頼度 ${F.conf.toFixed(2)}`; $("msg").textContent="";
  bgOK=false; bg=new Image(); bg.onload=()=>{bgOK=true;draw();}; bg.src=`/img?o=${cur}`; refresh(); loadStats(); }
function nav(d){ go(cur+d); }
async function finish(status){ const body={order:cur,status,groups:groups.map(g=>({label:g.label,unsplit:g.unsplit,strokes:[...g.strokes]}))};
  if(status=="done"&&!groups.length&&!confirm("グループが 1 つもありません。パーツが 1 つも描かれていない顔として完了にしますか?"))return;
  const r=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  if(!r.ok){ $("msg").textContent="保存に失敗: "+await r.text(); return; } dirty=false; go(cur+1); }
document.addEventListener("keydown",e=>{ if(e.key==" "){spaceDown=true;e.preventDefault();return;}
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()=="z"){undo();e.preventDefault();return;} if(e.ctrlKey||e.metaKey)return;
  const k=e.key.toLowerCase(); const d=DEF.find(d=>d[0]==k); if(d){makeGroup(d[1]);e.preventDefault();return;}
  if(k=="enter"){finish("done");e.preventDefault();} else if(k=="k")finish("skip"); else if(k=="x")ungroup(); else if(k=="m")toggleUnsplit();
  else if(k=="l")setTool("lasso"); else if(k=="r")setTool("rect"); else if(k=="h")setTool("pan"); else if(k=="b")cycleBg(); else if(k=="f")resetView();
  else if(k=="escape")clearSel(); else if(k=="arrowleft")nav(-1); else if(k=="arrowright")nav(1); else if(k=="n")go(st.next); });
document.addEventListener("keyup",e=>{ if(e.key==" ")spaceDown=false; });
document.addEventListener("click",e=>{ const b=e.target.closest&&e.target.closest("button"); if(b)b.blur(); });
window.addEventListener("resize",fit);
window.addEventListener("beforeunload",e=>{ if(dirty){e.preventDefault();e.returnValue="";} });
$("labBtns").innerHTML=DEF.map(d=>`<button onclick="makeGroup('${d[1]}')"><kbd>${d[0]}</kbd>${d[2]}</button>`).join("");
setTool("lasso"); setMode("new"); fetch("/api/stats").then(r=>r.json()).then(s=>{st=s; fit(); go(s.next);});
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    annotator = "user"

    def log_message(self, *a):
        pass

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            if u.path == "/":
                return self._send(PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/api/stats":
                return self._send(json.dumps(stats()).encode(), "application/json")
            o = int(q["o"][0])
            if not 0 <= o < len(S["queue"]):
                return self._send(b"out of range", "text/plain", 404)
            if u.path == "/api/face":
                return self._send(json.dumps(face_payload(o)).encode(), "application/json")
            if u.path == "/img":
                with lock:
                    png = face_png(o)
                return self._send(png, "image/png")
            self._send(b"not found", "text/plain", 404)
        except Exception as e:                                     # noqa: BLE001
            self._send(str(e).encode(), "text/plain", 500)

    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            save(d, self.annotator)
            self._send(b"ok", "text/plain")
        except Exception as e:                                     # noqa: BLE001
            self._send(str(e).encode(), "text/plain", 400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--annotator", default="user")
    ap.add_argument("--final", action="store_true", help="本番(labels/face_groups.jsonl)。なければ試し")
    a = ap.parse_args()
    load_state(LAB / ("face_groups.jsonl" if a.final else "face_groups_trial.jsonl"))
    H.annotator = a.annotator
    print(f"顔 {len(S['queue'])}  保存済み {len(S['faces'])}  保存先 {S['save'].name}  → http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()

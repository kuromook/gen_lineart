#!/usr/bin/env python
"""戦略ラベルの目印付け UI(Track H、2026-10-04 事前登録の手順)。

labels/queue_v1.csv の順に、顔の枠に入るまとまりを 1 個ずつ出す。背景は線画 PNG(外部の基準)、まとまりは赤。
目印は labels/strategic_labels.csv(現在の状態)と labels/label_log.jsonl(操作の追記記録)に保存。

Run:  python tools/face/label_server.py          → http://localhost:8472
      --host 0.0.0.0 で他の機械からも開ける
"""
import argparse, csv, json, os, time
from collections import Counter, OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
TRACKF = Path("/home/sh1/deepl/lineart-stroke-grammar")
CLUSTERS = TRACKF / "results/cluster_set_20260919"
PACK = TRACKF / "results/panel_pack_20260919"
DATASET = Path("/home/sh1/deepl/lineart/dataset")
LAB = ROOT / "labels"
QUEUE, SAVE, LOG = LAB / "queue_v1.csv", LAB / "strategic_labels.csv", LAB / "label_log.jsonl"
PORT = 8472
LABELS = ["face.eye.open", "face.eye.closed", "face.brow", "face.nose", "face.mouth", "face.ear", "face.other",
          "not_face", "unsure"]
EXCLUSIVE = {"unsure"}            # not_face は他と併用可(= 顔以外の線も混ざっている)
COLS = ["row", "panel", "box", "labels", "annotator", "datetime", "usage", "order"]

lock = Lock()
S = {}


def load_state():
    S["queue"] = list(csv.DictReader(open(QUEUE)))
    S["meta"] = list(csv.DictReader(open(CLUSTERS / "meta.csv")))
    S["P"] = np.load(CLUSTERS / "pts.npy", mmap_mode="r")
    S["M"] = np.load(CLUSTERS / "mask.npy", mmap_mode="r")
    S["panels"] = list(csv.DictReader(open(PACK / "panels.csv")))
    S["labels"] = {}
    if SAVE.exists():
        for t in csv.DictReader(open(SAVE)):
            S["labels"][int(t["order"])] = t
    S["png"] = OrderedDict()


def panel_img(pid):
    c = S["png"]
    if pid not in c:
        r = S["panels"][pid]
        c[pid] = cv2.imread(str(DATASET / r["source"] / "line" / r["name"]))
        while len(c) > 6:
            c.popitem(last=False)
    c.move_to_end(pid)
    return c[pid]


def inst_lines(row):
    """まとまりの線(画像の (x, y))。pts は (y, x) 順、meta cx = x・cy = y"""
    m = S["meta"][row]
    sc = float(m["scale"])
    xy = np.array([float(m["cx"]), float(m["cy"])])
    mk = np.asarray(S["M"][row])
    pts = np.asarray(S["P"][row], np.float64)[..., ::-1] / sc + xy
    return [pts[s] for s in range(len(mk)) if mk[s]]


def render(order, view, overlay, size=640):
    q = S["queue"][order]
    pid, row = int(q["panel"]), int(q["row"])
    img = panel_img(pid)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (float(q[k]) for k in ("x0", "y0", "x1", "y1"))
    lines = inst_lines(row)
    allp = np.concatenate(lines)
    if view == "face":
        ctr = np.array([(x0 + x1) / 2, (y0 + y1) / 2]); side = 1.7 * max(x1 - x0, y1 - y0)
    elif view == "zoom":
        ctr = (allp.min(0) + allp.max(0)) / 2
        side = max(3.0 * float((allp.max(0) - allp.min(0)).max()), 0.35 * max(x1 - x0, y1 - y0))
    else:
        ctr = np.array([W / 2, H / 2]); side = float(max(H, W))
    f = size / side
    o = ctr - side / 2
    A = np.array([[f, 0, -o[0] * f], [0, f, -o[1] * f]])
    out = cv2.warpAffine(img, A, (size, size), flags=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(236, 236, 236))
    p0 = tuple(np.round((np.array([x0, y0]) - o) * f).astype(int)); p1 = tuple(np.round((np.array([x1, y1]) - o) * f).astype(int))
    cv2.rectangle(out, p0, p1, (220, 160, 60), 1 if view != "panel" else 2)
    if overlay:
        th = 3 if view != "panel" else 2
        lay = out.copy()
        for ln in lines:
            cv2.polylines(lay, [np.round((ln - o) * f * 4).astype(np.int32)], False, (0, 0, 235), th, cv2.LINE_AA, 2)
        out = cv2.addWeighted(lay, 0.75, out, 0.25, 0)
        if view == "panel":
            c = np.round((((allp.min(0) + allp.max(0)) / 2) - o) * f).astype(int)
            cv2.circle(out, tuple(c), 14, (0, 0, 235), 2, cv2.LINE_AA)
    return cv2.imencode(".png", out)[1].tobytes()


def item(order):
    q = S["queue"][order]
    cur = S["labels"].get(order)
    return dict(order=order, row=int(q["row"]), panel=int(q["panel"]), r=float(q["r"]), work=q["work"],
                labels=cur["labels"].split(";") if cur else [], n_queue=len(S["queue"]))


def stats():
    cnt = Counter()
    for t in S["labels"].values():
        for lb in t["labels"].split(";"):
            cnt[lb] += 1
    done = sorted(S["labels"])
    nxt = next((o for o in range(len(S["queue"])) if o not in S["labels"]), len(S["queue"]))
    return dict(done=len(done), counts={lb: cnt.get(lb, 0) for lb in LABELS}, next=nxt)


def save(order, labels, annotator):
    labels = [lb for lb in LABELS if lb in labels]
    assert labels, "目印が空"
    assert not (set(labels) & EXCLUSIVE and len(labels) > 1), "unsure は単独で選ぶ"
    q = S["queue"][order]
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rec = dict(row=q["row"], panel=q["panel"], box=q["box"], labels=";".join(labels), annotator=annotator,
               datetime=now, usage=q["usage"], order=str(order))
    with lock:
        S["labels"][order] = rec
        with open(LOG, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp = SAVE.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=COLS); wr.writeheader()
            wr.writerows(S["labels"][o] for o in sorted(S["labels"]))
        os.replace(tmp, SAVE)


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>Track H 目印付け</title>
<style>
:root{--bg:#f4f4f2;--fg:#1c1c1c;--mut:#6b6b6b;--line:#cfcfcb;--on:#b3261e;--card:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,"Noto Sans JP",sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:12px 16px}
.top{display:flex;gap:18px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px}
.top b{font-size:17px}.mut{color:var(--mut)}
.views{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start}
.views figure{margin:0;background:var(--card);border:1px solid var(--line);padding:6px}
.views figcaption{font-size:12px;color:var(--mut);margin-bottom:4px}
#face{width:640px;height:640px;display:block}#zoom{width:400px;height:400px;display:block}#panel{width:300px;height:300px;display:block}
.side{display:flex;flex-direction:column;gap:10px}
.btns{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 6px}
button{font:inherit;padding:9px 14px;border:1px solid var(--line);background:var(--card);border-radius:6px;cursor:pointer}
button.on{background:var(--on);color:#fff;border-color:var(--on)}
button kbd{display:inline-block;min-width:1.4em;margin-right:6px;padding:0 4px;border:1px solid currentColor;border-radius:4px;font:12px ui-monospace,monospace;opacity:.8}
.nav button{background:#e9e9e5}.help{font-size:13px;color:var(--mut)}
.counts{font-size:13px;color:var(--mut);font-variant-numeric:tabular-nums}
#msg{color:var(--on);min-height:1.4em}
</style></head><body><div class="wrap">
<div class="top"><b>目印付け</b><span id="pos"></span><span id="info" class="mut"></span><span id="cur" class="mut"></span></div>
<div class="views">
 <figure><figcaption>顔(青い枠 = 検出された顔、赤 = 判定するまとまり)</figcaption><img id="face"></figure>
 <div class="side">
  <figure><figcaption>拡大</figcaption><img id="zoom"></figure>
  <figure><figcaption>コマ全体</figcaption><img id="panel"></figure>
 </div>
</div>
<div class="btns" id="btns"></div>
<div class="btns nav">
 <button onclick="confirmNext()"><kbd>Enter</kbd>確定して次へ</button>
 <button onclick="go(cur-1)"><kbd>←</kbd>1 つ戻る</button>
 <button onclick="go(cur+1)"><kbd>→</kbd>付けずに進む</button>
 <button onclick="toggleOv()"><kbd>H</kbd>赤を消す/出す</button>
 <button onclick="go(st.next)"><kbd>N</kbd>未判定の先頭へ</button>
</div>
<div id="msg"></div>
<div class="help">1〜7 は複数選べます(押すたびに入/切)→ Enter で確定。<b>0(顔パーツではない)だけのときと S(わからない)は、押した時点で確定して次へ</b>進みます。顔のものと顔以外の線が混ざっているときは、先に 1〜7 を選んでから 0 を足して Enter。付け直しは戻って選び直し、Enter。</div>
<div class="counts" id="counts"></div>
</div><script>
const DEF=[["1","face.eye.open","目(開)"],["2","face.eye.closed","目(閉)"],["3","face.brow","眉"],["4","face.nose","鼻"],["5","face.mouth","口"],["6","face.ear","耳"],["7","face.other","その他の顔(シワなど)"],["0","not_face","顔パーツではない"],["s","unsure","わからない"]];

let cur=0,sel=new Set(),ov=1,st={done:0,next:0,counts:{}},it=null,TARGET=300;
const $=id=>document.getElementById(id);
function drawBtns(){$("btns").innerHTML=DEF.map(d=>`<button class="${sel.has(d[1])?"on":""}" onclick="pick('${d[1]}')"><kbd>${d[0].toUpperCase()}</kbd>${d[2]}</button>`).join("")}
function imgs(){for(const v of ["face","zoom","panel"])$(v).src=`/img?o=${cur}&v=${v}&ov=${ov}`}
async function go(k){
  if(k<0)return; const r=await fetch(`/api/item?o=${k}`); if(!r.ok){$("msg").textContent="これ以上ありません";return}
  it=await r.json(); cur=it.order; sel=new Set(it.labels); $("msg").textContent="";
  $("pos").textContent=`${cur+1} 番目`; $("info").textContent=`行 r${it.row} / コマ p${it.panel} / 大きさ r=${it.r.toFixed(2)} / ${it.work}`;
  $("cur").textContent=it.labels.length?`保存済み: ${it.labels.map(l=>DEF.find(d=>d[1]==l)[2]).join(" + ")}`:"未判定";
  drawBtns(); imgs(); refresh();
}
async function refresh(){st=await (await fetch("/api/stats")).json();
  $("counts").textContent=`済み ${st.done} / 目安 ${TARGET}   |   `+DEF.map(d=>`${d[2]} ${st.counts[d[1]]||0}`).join("   ");
  if(st.done==TARGET)$("msg").textContent=`${TARGET} 個に達しました。ここで一度止めて、集まり具合を確認します。`}
async function store(labels){
  const r=await fetch("/api/label",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({order:cur,labels})});
  if(!r.ok){$("msg").textContent="保存に失敗: "+await r.text();return false} return true}
async function pick(l){
  const others=[...sel].filter(x=>x!="not_face"&&x!="unsure");
  if(l=="unsure"||(l=="not_face"&&!others.length)){ if(await store([l])) go(cur+1); return }
  sel.delete("unsure"); sel.has(l)?sel.delete(l):sel.add(l); drawBtns()}
async function confirmNext(){ if(!sel.size){$("msg").textContent="目印を選んでから Enter(付けずに進むなら →)";return}
  if(await store([...sel])) go(cur+1)}
function toggleOv(){ov=1-ov;imgs()}
document.addEventListener("keydown",e=>{ if(e.metaKey||e.ctrlKey||e.altKey)return; const k=e.key.toLowerCase();
  const d=DEF.find(d=>d[0]==k); if(d){pick(d[1]);e.preventDefault();return}
  if(k=="enter"||k==" "){confirmNext();e.preventDefault()} else if(k=="arrowleft"||k=="backspace"){go(cur-1);e.preventDefault()}
  else if(k=="arrowright"){go(cur+1)} else if(k=="h"){toggleOv()} else if(k=="n"){go(st.next)}});
fetch("/api/stats").then(r=>r.json()).then(s=>{st=s;go(s.next)});
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
            if u.path == "/api/item":
                return self._send(json.dumps(item(o)).encode(), "application/json")
            if u.path == "/img":
                v = q.get("v", ["face"])[0]
                size = dict(face=640, zoom=400, panel=300)[v]
                with lock:
                    png = render(o, v, q.get("ov", ["1"])[0] == "1", size)
                return self._send(png, "image/png")
            self._send(b"not found", "text/plain", 404)
        except Exception as e:                                     # noqa: BLE001
            self._send(str(e).encode(), "text/plain", 500)

    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            save(int(d["order"]), list(d["labels"]), self.annotator)
            self._send(b"ok", "text/plain")
        except Exception as e:                                     # noqa: BLE001
            self._send(str(e).encode(), "text/plain", 400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--annotator", default="user")
    a = ap.parse_args()
    load_state()
    H.annotator = a.annotator
    print(f"候補 {len(S['queue'])}  保存済み {len(S['labels'])}  → http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()

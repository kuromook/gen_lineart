#!/usr/bin/env python
"""L19: ベタ面から単語を手で決める UI (Track L、2026-10-25)。

L17 の結論「ベタ面は線を1層に絞らない(過半の面では)。ただし迷うのは1頁 6面、その8割が2択」
の上に立つ。**候補を出して作者が選ぶ**。purity は曖昧でないことの測度で正しさの測度では
ないので、**正しさは作者の答からしか作れない**。

  python tools/ui/l19_fill_label_server.py --host 0.0.0.0          試し (labels/fill_words_trial.jsonl)
  python tools/ui/l19_fill_label_server.py --host 0.0.0.0 --final  本番 (labels/fill_words.jsonl)
  python ... --final --reask results/l19_ui/reask.json              出し直し (語彙が増える前に答えた分)
  python ... --final --from 0                                       先頭から見直す (異常の報告用)
  python ... --queue results/l25_layer_ui/queue.json --save labels/layer_kind.jsonl --extra ''
                                                                    **層ごとの正解** (L25)
  -> http://<host>:8475

**異常の報告** (2026-10-25 作者の指示): 語の答とは別の流れ。`X` または「異常」ボタンで
`labels/fill_anomaly.jsonl` に追記する。**語の答を上書きしない / 次の問に進まない** ——
見直しながら印を付ける道具なので、進むのは B / N で作者が決める。

**自動で決まる面も 1割が同じ見た目で混ざっている**(`why` はクライアントに送らない)。
"""
import collections
import argparse, json, os, time
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


IDENT = ('work', 'pid', 'comp', 'region', 'why', 'lid', 'crop', 'area')


def match_row(r, q, proj):
    """保存された1行を**中身で**行列の位置に結ぶ。返すのは (位置, 落とした理由)。

    **T144**(2026-10-27、基盤から回送。M が答を失う前に捕らえた):
    ここは `S['done'][int(r['order'])]` だった ——
    **`order` は行列の中の番号で、行列を振り直すと別の問を指す。**
    M は補充の行列で `order` を 0 から振り直していて、1巡目と同じ保存先に向けると
    **11問すべてが「もう答えた」として立ち上がる**ところだった
    (1巡目 339記録のうち 14記録が `order` 0〜10 を占めている)。

    > **症状は「やることが無い」と出る。** 何も測らずに通る点検と同じ署名である。
    > **番号で鍵を作る器具は、振り直した行列を黙って「済み」と読む。**

    **鍵は1つに決まらない。** 2026-10-27 に3つの行列で測った:

    ```
    (work,pid,comp,region)        l19 で重複 45鍵/105行   <- `region` の意味が ask と check で違う
    + crop                        l19 で 0、**l25 で 5**(1つの切り出しに複数の層を問う)
    + why                         l25/l29 で 0、**l19 で 37**
    IDENT 全部(在る欄だけ射影)   3つの行列すべてで **0**
    ```

    だから**記録が持っている欄だけで射影し、合致が1つのときだけ結ぶ**。
    **0個でも2個以上でも結ばない** —— **結ばなければ、その問はもう一度出る。**
    向きが安全な側である(黙って「済み」にするのは答を失う)。
    **理由別に数えて起動時に印字する**(規律9)。
    """
    fields = [k for k in IDENT if k in r]
    if not fields:
        return None, '記録に同定できる欄が1つも無い'
    key = tuple(r[k] for k in fields)
    hits = proj.setdefault(tuple(fields), None)
    if hits is None:
        hits = {}
        for o, it in enumerate(q):
            hits.setdefault(tuple(it.get(k) for k in fields), []).append(o)
        proj[tuple(fields)] = hits
    got = hits.get(key, [])
    if len(got) == 1:
        return got[0], ''
    if not got:
        return None, 'この行列に無い記録 (別の行列の保存先か、行列が変わった)'
    # **中身で絞ったうえで `order` を同点決着に使う。** `order` だけを鍵にするのが T144 で、
    # **中身の合致の中から選ぶのは別の形**である —— 振り直した行列が、同じ中身の行の上に
    # 同じ番号を落とす確率は低い。**それでも何件そうしたかを数える**(規律9)。
    o = r.get('order')
    if o is not None and int(o) in got:
        return int(o), f'中身が {len(got)}行に合致。**order で同点決着した**'
    return None, f'**鍵が1つに決まらない({len(got)}行が同じ鍵。order も合わない)。結ばない**'


def _read_jsonl(path, q, counts, undo_key=None):
    """保存された行を**中身で**行列の位置に結び直す。`order` は当てにしない(T144)。"""
    out, proj = {}, {}
    for ln in open(path):
        if not ln.strip():
            continue
        r = json.loads(ln)
        o, why = match_row(r, q, proj)
        if why:
            counts[why] += 1      # **結べた場合の但し書きも数える**(同点決着を黙らせない)
        if o is None:
            continue
        if undo_key and r.get(undo_key[0]) == undo_key[1]:
            out.pop(o, None)
            continue
        out[o] = r
    return out


def load_state(save_path, reask=None, queue=None, extra=None):
    S['save'] = save_path
    # **保存先ごとに異常の報告先を分ける**(2026-10-26)。
    # それまで `labels/fill_anomaly.jsonl` 固定で、**`--save` を無視していた** ——
    # M がこのサーバーを別の行列で再利用すると、**M の異常の報告が L のファイルに入る。**
    # いまの `labels/fill_anomaly.jsonl` 15行は**全部 L の面**だった
    # (`crop` が L の名前の形)ので**実害は出ていない。潜在していただけ**。
    # **既存のファイルは触らない**(行の出所を後から振り分けない)。
    S['flagfile'] = save_path.with_name(save_path.stem + '_anomaly.jsonl')
    S['flag'] = {}
    d = json.load(open(queue or (UI / 'queue.json')))
    S['q'] = d['queue']
    S['dir'] = Path(queue).parent if queue else UI
    S['meta'] = {k: d[k] for k in ('n', 'n_ask', 'n_check', 'n_auto', 'n_all',
                                   'words')}
    S['meta']['ask'] = d.get('ask')
    S['meta']['help'] = d.get('help')
    # 「その他」の行は行列が決める(層の正解では隙間埋けの選択肢が要らない)
    S['meta']['others'] = d.get('others') or [
        ['gapfill_line', '隙間埋め(線画の抜けを足した)', 'G'],
        ['gapfill_fill', '隙間埋め(別レイヤのベタ用)', 'H'],
        ['free', 'その他(語を入力)', 'T'], ['mixed', '混在(1語にできない)', 'M'],
        ['skip', '分からない', 'S'],
        ['anomaly', '⚑ 異常を報告(別のコマ・別の画像が混じっている)', 'X']]
    for w in (EXTRA_WORDS if extra is None else extra):
        if w not in S['meta']['words']:
            S['meta']['words'].append(w)
    S['done'] = {}
    S['only'] = None                      # 出し直しのときだけ、行列を部分集合に絞る
    S['re'] = set()                       # この走りで答え直した問
    if reask:
        got = {int(o) for o in json.loads(Path(reask).read_text())}
        S['only'] = sorted(o for o in got if 0 <= o < len(S['q']))
    LAB.mkdir(exist_ok=True)
    # **行列の位置は、保存された行の中身から引き直す**(T144。`order` では引かない)
    S['load'] = collections.Counter()
    if save_path.exists():
        S['done'] = _read_jsonl(save_path, S['q'], S['load'])
    if S['flagfile'].exists():
        S['flag'] = _read_jsonl(S['flagfile'], S['q'], S['load'],
                                undo_key=('note', '__undo__'))


def payload(o):
    r = dict(S['q'][o])
    r.pop('why', None)                      # **出自は出さない**
    r['cols'] = COLS[:len(r['cands'])]
    r['words'] = S['meta']['words']
    r['others'] = S['meta']['others']
    r['ask'] = S['meta'].get('ask')
    r['help'] = S['meta'].get('help')
    r['saved'] = S['done'].get(o)
    r['flagged'] = S['flag'].get(o)
    r['n_flag'] = len(S['flag'])
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


def flag(d, annotator):
    """異常の報告。**語の答とは別のファイル**に追記する(上書きしない)。"""
    o = int(d['order'])
    q = S['q'][o]
    note = (d.get('note') or '').strip()
    rec = dict(order=o, work=q['work'], pid=q['pid'], comp=q['comp'],
               region=q['region'], crop=q['crop'], area=q.get('area'),
               note=note or '(記述なし)', annotator=annotator,
               ts=time.strftime('%Y-%m-%dT%H:%M:%S'))
    if d.get('undo'):
        rec['note'] = '__undo__'
    with lock:
        if d.get('undo'):
            S['flag'].pop(o, None)
        else:
            S['flag'][o] = rec
        with open(S['flagfile'], 'a') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return rec


def first_todo():
    if S.get('start') is not None:
        return max(0, min(len(S['q']) - 1, S['start']))
    if S['only'] is not None:               # 出し直しでは「この走りで答え直したか」を見る
        for o in S['only']:
            if o not in S['re']:
                return o
        return S['only'][-1]
    for i in range(len(S['q'])):
        if i not in S['done']:
            return i
    return len(S['q']) - 1


def make_rec(o, d, annotator):
    """保存する記録を組み立てる。**行列に候補が無くても落ちないこと。**

    2026-10-25: ここが `q['cands'][0]` で落ち、POST が 400 を返していた。
    クライアントが戻り値を見ずに次へ進んでいたので、**72問が無言で捨てられた。**
    起動時に全行でこれを空打ちして、落ちるなら**サーバを起動しない**(下の self_check)。
    """
    q = S['q'][o]
    cands = q.get('cands') or []
    top = cands[0] if cands else {}
    # **`crop` も残す**(2026-10-27、T144)。これが在れば行列を振り直しても
    # **中身だけで結べる**。無かった 2026-10-26 までの記録は、
    # 中身で絞った上で `order` を同点決着に使って結んでいる(起動時に件数を印字する)。
    rec = dict(order=o, work=q['work'], pid=q['pid'], comp=q['comp'],
               region=q['region'], crop=q.get('crop'),
               why=q.get('why'), auto=q.get('auto', 0),
               purity=q.get('purity'), n_lay=q.get('n_lay'),
               n_lay90=q.get('n_lay90'), lid=q.get('lid'), area=q.get('area'),
               top_lid=top.get('lid'), top_name=top.get('name'),
               top_word=q.get('top_word', ''), word_share=q.get('word_share', {}),
               choice=d.get('choice'), word=d.get('word'), free=d.get('free'),
               ms=int(d.get('ms', 0)), annotator=annotator,
               ts=time.strftime('%Y-%m-%dT%H:%M:%S'))
    rec['agree'] = int(rec['choice'] == 'word' and rec['word'] == rec['top_word']
                       and bool(rec['top_word']))
    return rec


def self_check():
    """**起動の門**: 行列の全行で記録を組み立ててみる。1行でも落ちたら起動しない。"""
    probe = dict(choice='word', word='_probe_', free=None, ms=0)
    for o in range(len(S['q'])):
        make_rec(o, probe, 'probe')
    return True


def js_check():
    """**起動の門その2**: 出す JS を構文検査する。

    2026-10-25: `PAGE` の中に `\\n` を `\n` と書いてしまい、Python が本物の改行にした。
    **JS の文字列が閉じず、スクリプト全体が死んだ** —— 画面には静的な HTML だけが残り、
    ボタンは見えるのに何も動かない、という**最も気付きにくい壊れ方**をした。
    """
    import re as _re, shutil, subprocess, tempfile
    m = _re.search(r'<script>(.*?)</script>', PAGE, _re.S)
    if not m:
        return True
    node = shutil.which('node') or shutil.which('nodejs')
    if not node:
        print('  (node が無いので JS の構文検査は飛ばす)', flush=True)
        return True
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as fh:
        fh.write(m.group(1)); path = fh.name
    r = subprocess.run([node, '--check', path], capture_output=True, text=True)
    os.unlink(path)
    if r.returncode != 0:
        raise SystemExit('**JS が構文エラー。起動しない**\n' + r.stderr[:800])
    return True


# **JS 側で「無ければ出さない」と書いてある鍵**だけをここに置く。
# 置くことは「欠けても壊れない」の宣言であって、「欠けてよい」ではない ——
# 欠けたら起動時に印字する。黙って 0 や「原寸」と出す形に書き換えたら、ここから外すこと
OPTIONAL = frozenset(('area', 'scale', 'ctx_scale'))


def key_check():
    """**起動の門その3**: JS が読む `it.<鍵>` が、行列と `payload()` にあるか。

    2026-10-26 (基盤の報告): M が行列を流用したとき `area` が `area_px` という名で
    入っていた。**その1鍵の欠けで解答欄とキー割り当てが丸ごと消え**、
    作者に渡してから発覚した。門その1(記録を組み立てられるか)と
    門その2(JS の構文)は**どちらも通る**種類の欠陥だった ——
    器具の流用の接点は**行列の鍵の名前**なので、点検はそこに当てる。
    """
    import re as _re
    m = _re.search(r'<script>(.*?)</script>', PAGE, _re.S)
    if not m:
        return True
    want = set(_re.findall(r'\bit\.([A-Za-z_][A-Za-z0-9_]*)', m.group(1)))
    have = set(payload(0))
    soft = sorted((want - have) & OPTIONAL)
    miss = sorted(want - have - OPTIONAL)
    print(f'  起動の門その3: JS が読む鍵 {len(want)} / 行列と payload にある鍵 '
          f'{len(have)}  欠け {len(miss)}', flush=True)
    for k in soft:
        print(f'    (欠けを許した鍵) it.{k} が行列に無い —— '
              f'**その数は画面に出ない**(0 や「原寸」と出さない)', flush=True)
    if miss:
        lines = []
        for k in miss:
            near = [h for h in sorted(have)
                    if h.startswith(k[:3]) or k in h or h in k]
            lines.append(f'  it.{k}  <- 行列に無い。近い名前: '
                         f'{", ".join(near) if near else "無し"}')
        raise SystemExit('**JS が読む鍵が行列に無い。起動しない**\n'
                         + '\n'.join(lines)
                         + '\n(名前が違うだけなら行列側を直すか、JS に鍵を足すこと。'
                           '**出さない方がよい数は JS 側で欠けを許す形に書く**)')
    return True


def go_check():
    """**起動の門その4**: `go()` を実際に走らせて、解答欄が出るかを見る。

    門その1〜3 は**どれも go() を走らせていない。** 2026-10-26 の欠陥
    (流用した行列に `area` が無く、解答欄とキー割り当てが丸ごと消えた)は
    門その1・その2 を通り、HTTP も 200 を返していた。**作者に渡ってから発覚した。**
    ここでは最小の DOM を当てて `go(0)` を走らせ、語ボタン・その他ボタン・
    `keymap`・`okeys` が揃っているかを数える。
    """
    import re as _re, shutil, subprocess, tempfile, json as _j
    node = shutil.which('node') or shutil.which('nodejs')
    if node is None:
        print('  (node が無いので go() の点検は飛ばす —— **渡す前に画面を見ること**)',
              flush=True)
        return True
    smoke = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'l19_ui_smoke.js')
    if not os.path.exists(smoke):
        raise SystemExit(f'**{smoke} が無い。起動しない**(門を消して走らせない)')
    m = _re.search(r'<script>(.*?)</script>', PAGE, _re.S)
    if not m:
        return True
    n = len(S['q'])
    # **行ごとに形が違う**(候補の無い面、語に当たらない面)。端と真ん中を見る
    orders = sorted({0, n // 2, n - 1})
    pl = {f'order={o}': payload(o) for o in orders}
    d = tempfile.mkdtemp(prefix='l19_ui_smoke_')
    pj, uj = os.path.join(d, 'payloads.json'), os.path.join(d, 'ui.js')
    with open(pj, 'w', encoding='utf-8') as fh:
        _j.dump(pl, fh, ensure_ascii=False, default=str)
    with open(uj, 'w', encoding='utf-8') as fh:
        fh.write(m.group(1))
    r = subprocess.run([node, smoke, pj, uj], capture_output=True, text=True)
    print(r.stdout.rstrip(), flush=True)
    if r.returncode != 0:
        raise SystemExit('**go() を走らせたら解答欄が出なかった。起動しない**\n'
                         + (r.stderr[:800] or '(上の行を見ること)'))
    return True


def save(d, annotator):
    o = int(d['order'])
    rec = make_rec(o, d, annotator)
    with lock:
        S['done'][o] = rec
        S['re'].add(o)
        with open(S['save'], 'a') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return rec


PAGE = """<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>何を聞いているか</title>
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
  <b id="ask">この黒いベタは何?</b>
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
<div class="meta" id="help">黒い塊が聞いている面。色は縁 4px に接している線のレイヤ(ヒント。答ではない)。
工程の産物(隙間埋め)なら G / H。</div>
<script>
let cur=0, it=null, t0=0, ctx=false, keymap={}, okeys={};
async function go(o){
  if(o<0) o=0;
  const r=await fetch('/api/item?o='+o); if(!r.ok) return;
  it=await r.json(); cur=o; t0=performance.now(); ctx=false;
  // **解答欄を先に組む。** 2026-10-26 (基盤の報告): ここは `showImg()` が先頭にあり、
  // 表示用の1鍵 (`area`) が行列に無いだけで info() が投げ、**解答欄と keymap/okeys が
  // 丸ごと消えた** —— ボタンが出ないのとキーが死ぬのは同じ1つの例外だった。
  // 器具の流用で2回目で、作者の時間に届いた。**表示が転んでも答えは入れられる順にする**
  buildAnswers();
  if(it.ask) document.getElementById('ask').textContent=it.ask;
  if(it.help) document.getElementById('help').textContent=it.help;
  document.getElementById('pos').textContent=it.pos+' / '+it.n;
  document.getElementById('done').textContent=
    it.n_done+' 済み'+(it.n_flag?'   ⚑ 異常 '+it.n_flag+' 件':'');
  // 縁に接している層(色のヒント)
  const h=document.getElementById('hint'); h.innerHTML='';
  (it.cands||[]).forEach((x,i)=>{
    const d=document.createElement('span');
    d.innerHTML='<span class="sw" style="background:'+(it.cols||[])[i]+'"></span> '+
      (x.name||'')+(x.word?'':' <span class="k">(語に対応しない層)</span>')+
      ' <span class="k">'+(x.share*100).toFixed(0)+'%</span>';
    d.style.display='flex'; d.style.gap='6px'; d.style.alignItems='center';
    h.appendChild(d);
  });
  showImg();
}
// 解答欄 = 語。**並びは固定**(位置が毎回同じになるように。2026-10-25 作者の指摘)
// **ここは go() の先頭で呼ぶ。** 表示用の鍵が欠けても答は入れられるようにするため
function buildAnswers(){
  const ws=it.words||[];
  const shm=it.word_share||{}; const cands=it.cands||[]; const cols=it.cols||[];
  const w=document.getElementById('words'); w.innerHTML=''; keymap={};
  ws.forEach((word,i)=>{
    const sh=shm[word];
    const ci=cands.findIndex(c=>c.word===word);
    const b=document.createElement('button');
    b.innerHTML=(ci>=0?'<span class="sw" style="background:'+cols[ci]+'"></span>':'')+
      word+(sh?' <span class="k">'+(sh*100).toFixed(0)+'%</span>':'')+
      ' <span class="k">'+(i+1)+'</span>';
    if(sh) b.className='hit';
    b.onclick=()=>pick('word',word,null); w.appendChild(b);
    if(i<9) keymap[String(i+1)]=word;
  });
  const o2=document.getElementById('other'); o2.innerHTML=''; okeys={};
  (it.others||[]).forEach(([k,t,key])=>{
    const b=document.createElement('button');
    b.innerHTML=t+' <span class="k">'+key+'</span>';
    b.onclick=()=>act(k); o2.appendChild(b);
    okeys[key.toLowerCase()]=k;
  });
}
function nxt(){ if(it) go(it.nav_next); }
function prv(){ if(it) go(it.nav_prev); }
function act(k){
  if(k==='free'){const v=prompt('この黒は何ですか(語を入力)'); if(v) pick('free',null,v);}
  else if(k==='anomaly'){ flagIt(); }
  else pick(k,null,null);
}
// **異常の報告は語の答を上書きしない。次にも進まない**(見直しながら印を付けるため)
async function flagIt(){
  if(it.flagged){
    if(!confirm('この問の異常の印を取り消しますか')) return;
    await fetch('/api/flag',{method:'POST',body:JSON.stringify({order:cur,undo:1})});
    go(cur); return;
  }
  const v=prompt('何が混じっていますか(空でもよい。例: 別のコマの絵 / 貼り込んだ画像 / 3D)');
  if(v===null) return;
  await fetch('/api/flag',{method:'POST',body:JSON.stringify({order:cur,note:v})});
  go(cur);
}
function showImg(){
  document.getElementById('img').src='/'+(ctx?'ctx':'crop')+'?o='+cur+'&t='+Date.now();
  info();
}
function info(){
  if(!it) return;
  const sc=ctx?it.ctx_scale:it.scale;
  // **倍率が行列に無いとき、黙って「(原寸)」と出してはいけない。**
  // 2026-10-26 の点検で見つけた2件目: `sc` が undefined だと `sc<1` が false になり、
  // **縮小して出している絵を「原寸」と書く**ことになる(規律6 に直接触る)
  document.getElementById('info').textContent=
    (ctx?'[コマ全体] ':'[面の周り] ')+it.work+' '+it.pid+' コマ'+it.comp+' 面'+it.region+
    // **欠けていたら数を出さない。** 0 と出すのは作者に嘘をつくことになる
    (it.area==null?'':'  面積 '+it.area.toLocaleString()+'px')+
    (sc==null?'  (表示倍率 不明)':sc<1?'  (表示は '+(sc*100).toFixed(0)+'%)':'  (原寸)')+
    (it.saved?'  ← 記録済み: '+(it.saved.word||it.saved.free||it.saved.choice):'')+
    (it.flagged?'   ⚑ 異常として報告済み: '+it.flagged.note:'');
}
async function pick(choice,word,free){
  // **保存できなかったら進まない。** 2026-10-25: ここで戻り値を見ていなかったため、
  // 400 を返し続けたまま 72 問が無言で捨てられた
  let r;
  try{
    r=await fetch('/api/save',{method:'POST',body:JSON.stringify({
      order:cur,choice:choice,word:word,free:free,
      ms:Math.round(performance.now()-t0)})});
  }catch(e){ alert('保存できませんでした(接続): '+e+'\\nこの問は記録されていません'); return; }
  if(!r.ok){
    const t=await r.text();
    alert('保存できませんでした: '+t+'\\n**この問は記録されていません。**進みません');
    return;
  }
  nxt();
}
addEventListener('keydown',e=>{
  if(!it||e.metaKey||e.ctrlKey) return;
  const k=e.key.toLowerCase();
  if(keymap[k]) pick('word',keymap[k],null);
  else if(k==='c'){ctx=!ctx;showImg();}
  else if(okeys[k]) act(okeys[k]);
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
                p = S['dir'] / 'crops' / S['q'][o][key]
                return self._send(p.read_bytes(), 'image/png')
            self._send(b'not found', 'text/plain', 404)
        except Exception as e:                                      # noqa: BLE001
            self._send(str(e).encode(), 'text/plain', 500)

    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if urlparse(self.path).path == '/api/flag':
                flag(d, self.annotator)
            else:
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
    ap.add_argument('--from', dest='start', type=int, default=None,
                    help='この order から始める(先頭から見直すときは 0)')
    ap.add_argument('--queue', help='別の行列 (既定は results/l19_ui/queue.json)')
    ap.add_argument('--save', help='別の保存先 (labels/ からの相対でも絶対でも)')
    ap.add_argument('--extra', default=None,
                    help="語を足す。空文字で足さない (層の正解では '')")
    a = ap.parse_args()
    sv = (Path(a.save) if a.save and os.path.isabs(a.save) else
          (ROOT / a.save if a.save else
           LAB / ('fill_words.jsonl' if a.final else 'fill_words_trial.jsonl')))
    load_state(sv, a.reask, a.queue,
               None if a.extra is None else [w for w in a.extra.split(',') if w])
    m = S['meta']
    if S['only'] is None:
        print(f"行列 {m['n']} (要判断 {m.get('n_ask','?')} + "
              f"自動の確認 {m.get('n_check','?')})  "
              f"保存済み {len(S['done'])}  保存先 {S['save'].name}  "
              f"-> http://{a.host}:{a.port}", flush=True)
    else:
        print(f"出し直し {len(S['only'])} 問 (全 {m['n']} 問のうち)  "
              f"保存先 {S['save'].name} に追記  "
              f"-> http://{a.host}:{a.port}", flush=True)
    if a.start is not None:
        S['start'] = a.start
        print(f'  開始位置 {a.start}  (全問を B / N で見直せます)', flush=True)
    print(f'  異常の報告: X または「⚑ 異常を報告」ボタン -> {S["flagfile"]}'
          f'  既報 {len(S["flag"])} 件', flush=True)
    # **保存の読み込みで起きたことを印字する**(T144。黙って「済み」にしない。規律9)
    print(f'  保存の読み込み: **記録の中身で行列に結んだ**(`order` では引かない。T144)'
          f'  結んだ {len(S["done"])}', flush=True)
    for k_, v_ in sorted(S['load'].items()):
        print(f'    {v_:5d}  {k_}', flush=True)
    if not S['load']:
        print('    0  (order で読んだ行・この行列に無い行は無い)', flush=True)
    self_check()
    js_check()
    key_check()
    go_check()
    print(f'  起動の門: 行列 {len(S["q"])} 行すべてで記録を組み立てられた / '
          f'JS は構文検査を通った / JS の読む鍵が揃っている / '
          f'go() が解答欄を出した', flush=True)
    H.annotator = a.annotator
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == '__main__':
    main()

"""聞き取りの問いの形を読み、点検し、画面(interview_core.html)に流し込む。サーバー版と artifact 版で共通。

点検は指針 `doc/ui_guidelines_20261107.md` のうち、機械で言えるものだけ:
問いの1文がある / 単位がある / 「何が決まるか」がある / 選択肢に器具の用語らしい綴りが無い /
「どれでもない」「分からない」を問いの側で重ねていない / 問いごとに絵がある / ID が重ならない。
"""
import json
import re
from html import escape as html_escape
from pathlib import Path

CORE = Path(__file__).resolve().parent / "interview_core.html"
RESERVED = {"__none__", "__unknown__"}
TOGGLES_ONLY = "__toggles_only__"  # 札だけで保存した行の value(Track S 2026-11-27)
# **札のキーは当たってよい列の側で見る。** 数字=軸1 / `q w e r t y u i o`=軸2(E 2026-11-18)/
# `d 0 + - ; !` と Enter・矢印=画面の操作。禁じる列を数えるやり方は、軸が増えるたび取りこぼす(E 2026-11-27)
TOGGLE_KEYS = "zxcvbnm"
# 器具の用語らしい綴り(英字の snake_case / 列名)。画面には作者の言葉を出す(指針 §1-3)
JARGON = re.compile(r"[a-z]+_[a-z_]+|\b(stratum|v_layer_kind|layer_id|alpha|bbox)\b")


class SpecError(ValueError):
    pass


def load_questions(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    errs = []
    for k in ("interview_id", "question", "unit", "decides", "choices", "items"):
        if not spec.get(k):
            errs.append(f"`{k}` が無い")
    if errs:
        raise SpecError("; ".join(errs))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", spec["interview_id"]):
        errs.append("`interview_id` は英数字と _ . - だけ")
    if spec.get("initial_view", "fit") not in ("fit", "native"):
        errs.append("`initial_view` は fit(既定)か native")
    if not 1 <= len(spec["choices"]) <= 7:
        errs.append("選択肢は 1〜7 個(「どれでもない」「分からない」は自動で足す。数字キーは 9 まで)")
    # **2つ目の軸**(任意。`choices2` + `question2`)。1つの層が2つの軸の両方に当たる便で使う
    # (作者 2026-11-18)。**両方押すまで保存できない**のは画面の側
    if spec.get("choices2"):
        if not spec.get("question2"):
            errs.append("`choices2` が在るなら `question2`(2つ目の軸の問いの1文)が要る")
        if not 1 <= len(spec["choices2"]) <= 7:
            errs.append("2つ目の軸の選択肢も 1〜7 個")
        if spec.get("question2") and JARGON.search(spec["question2"]):
            errs.append(f"`question2` に器具の用語らしい綴り: {spec['question2']!r}")
    elif spec.get("question2"):
        errs.append("`question2` が在るなら `choices2` が要る")
    seen = set()
    for c in spec["choices"] + list(spec.get("choices2") or []):
        if c.get("value") in RESERVED or c.get("label") in ("どれでもない", "分からない", "わからない"):
            errs.append(f"選択肢 `{c.get('label')}` は自動で足すので書かない")
        if not c.get("value") or not c.get("label"):
            errs.append("選択肢に value と label が要る")
        if c.get("value") == TOGGLES_ONLY:
            errs.append(f"選択肢の value に `{TOGGLES_ONLY}` は使えない(札だけの保存の予約語)")
        # 軸ごとに別の列に入るので、2つの軸で同じ value が在ってもよい(軸の中で重なるのだけ見る)
        ax = "2" if spec.get("choices2") and c in spec["choices2"] else "1"
        if (ax, c.get("value")) in seen:
            errs.append(f"軸{ax} の選択肢の value が重なる: {c.get('value')}")
        seen.add((ax, c.get("value")))
        for field in ("label", "help"):
            if c.get(field) and JARGON.search(c[field]):
                errs.append(f"選択肢の {field} に器具の用語らしい綴り: {c[field]!r}(作者の言葉で書く)")
    # 札(任意): 主な答えと**組み合わせて**付ける(Track S 2026-11-27)。行の `toggles` 欄に value の列が入る
    choice_values = {c.get("value") for c in spec["choices"] + list(spec.get("choices2") or [])}
    tvals = set()
    for t in spec.get("toggles", []):
        if not t.get("value") or not t.get("label"):
            errs.append("札に value と label が要る")
        if t.get("value") in choice_values or t.get("value") in tvals or t.get("value") in RESERVED | {TOGGLES_ONLY}:
            errs.append(f"札の value が選択肢・他の札・予約語と重なる: {t.get('value')}")
        tvals.add(t.get("value"))
        k = t.get("key")
        if k is not None and (len(k) != 1 or k not in TOGGLE_KEYS):
            errs.append(f"札のキー `{k}` は `{' '.join(TOGGLE_KEYS)}` のどれか1文字"
                        f"(数字は1つ目の軸・q w e r t y u i o は2つ目の軸・d 0 + - ; ! は画面の操作)")
        for field in ("label", "help"):
            if t.get(field) and JARGON.search(t[field]):
                errs.append(f"札の {field} に器具の用語らしい綴り: {t[field]!r}")
        # **2つの軸が在る便では札だけでは保存できない**(画面の決め。E 2026-11-27)。
        # 画面が許さないことを札の説明で約束させない
        if spec.get("choices2") and "札だけ" in (t.get("help") or ""):
            errs.append("2つ目の軸が在る便では**札だけでは保存できない**ので、札の説明に「札だけ」と書けない"
                        f"(札 {t.get('value')})")
    for field in ("question", "decides"):
        if JARGON.search(spec[field]):
            errs.append(f"`{field}` に器具の用語らしい綴り: {spec[field]!r}")
    ids = set()
    for i, it in enumerate(spec["items"]):
        if not it.get("id"):
            errs.append(f"items[{i}] に id が無い")
        elif it["id"] in ids:
            errs.append(f"items の id が重なる: {it['id']}")
        ids.add(it.get("id"))
        if not it.get("images"):
            errs.append(f"items[{i}] に絵が無い")
        for k in ("prefill", "prefill2"):
            v = it.get(k)
            pool = spec["choices"] if k == "prefill" else (spec.get("choices2") or [])
            if v and v not in [c.get("value") for c in pool] + ["__none__", "__unknown__"]:
                errs.append(f"items[{i}] の {k} が選択肢に無い: {v}")
            if v and k == "prefill2":
                errs.append(f"items[{i}]: **見た目の軸に初期値は置かない**(作者が押していない答えを置かない)")
        for im in it.get("images", []):
            if not im.get("src"):
                errs.append(f"items[{i}] の絵に src が無い")
    if errs:
        raise SpecError("問いの形の点検に落ちた:\n  - " + "\n  - ".join(errs))
    spec["item_ids"] = ids
    spec.setdefault("batch_size", 0)  # 0 = 途中の区切り画面を出さない(既定。作者 2026-11-07「不要」)
    return spec


def render_page(spec, mode):
    data = {k: v for k, v in spec.items() if k != "item_ids"}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = CORE.read_text(encoding="utf-8")
    assert "/*__INTERVIEW_DATA__*/null" in html and '/*__INTERVIEW_MODE__*/"server"' in html
    # **使い捨ての artifact は名前で見分けられないと数えられない**(指針 §7-2)。
    # 題は `page_title`、無ければ `interview_id`
    title = html_escape(spec.get("page_title") or spec["interview_id"])
    html = html.replace("<title>聞き取り</title>", f"<title>{title}</title>", 1)
    return (html.replace("/*__INTERVIEW_DATA__*/null", blob)
                .replace('/*__INTERVIEW_MODE__*/"server"', json.dumps(mode)))

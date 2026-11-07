"""聞き取りの問いの形を読み、点検し、画面(interview_core.html)に流し込む。サーバー版と artifact 版で共通。

点検は指針 `doc/ui_guidelines_20261107.md` のうち、機械で言えるものだけ:
問いの1文がある / 単位がある / 「何が決まるか」がある / 選択肢に器具の用語らしい綴りが無い /
「どれでもない」「分からない」を問いの側で重ねていない / 問いごとに絵がある / ID が重ならない。
"""
import json
import re
from pathlib import Path

CORE = Path(__file__).resolve().parent / "interview_core.html"
RESERVED = {"__none__", "__unknown__"}
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
    if not 1 <= len(spec["choices"]) <= 7:
        errs.append("選択肢は 1〜7 個(「どれでもない」「分からない」は自動で足す。数字キーは 9 まで)")
    seen = set()
    for c in spec["choices"]:
        if c.get("value") in RESERVED or c.get("label") in ("どれでもない", "分からない", "わからない"):
            errs.append(f"選択肢 `{c.get('label')}` は自動で足すので書かない")
        if not c.get("value") or not c.get("label"):
            errs.append("選択肢に value と label が要る")
        if c.get("value") in seen:
            errs.append(f"選択肢の value が重なる: {c.get('value')}")
        seen.add(c.get("value"))
        for field in ("label", "help"):
            if c.get(field) and JARGON.search(c[field]):
                errs.append(f"選択肢の {field} に器具の用語らしい綴り: {c[field]!r}(作者の言葉で書く)")
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
        for im in it.get("images", []):
            if not im.get("src"):
                errs.append(f"items[{i}] の絵に src が無い")
    if errs:
        raise SpecError("問いの形の点検に落ちた:\n  - " + "\n  - ".join(errs))
    spec["item_ids"] = ids
    spec.setdefault("batch_size", 20)
    return spec


def render_page(spec, mode):
    data = {k: v for k, v in spec.items() if k != "item_ids"}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = CORE.read_text(encoding="utf-8")
    assert "/*__INTERVIEW_DATA__*/null" in html and '/*__INTERVIEW_MODE__*/"server"' in html
    return (html.replace("/*__INTERVIEW_DATA__*/null", blob)
                .replace('/*__INTERVIEW_MODE__*/"server"', json.dumps(mode)))

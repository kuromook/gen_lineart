# `tools/ui/` —— 聞き取りの UI の共通部品

指針: `doc/ui_guidelines_20261107.md`。**問いと絵を JSON に書けば、指針の形の画面になる。** 画面を一から作らない。

| ファイル | 役目 |
|---|---|
| `interview_core.html` | 画面の核(サーバー版と artifact 版で共通)。拡大縮小・キー・タッチ・保留・一言・追記の保存・続きから |
| `interview_spec.py` | 問いの JSON を読み、点検する(1文の問い・単位・決まること・器具の用語・自動の選択肢の重複) |
| `interview_server.py` | 手元のサーバー版。`--lan` で LAN の端末から(インターネットには出さない) |
| `build_interview_artifact.py` | artifact 版の1枚の HTML(絵を埋める、15MB まで) |
| `verify_answers.py` | 答えを問いと突き合わせる。artifact を消す前の行の数の一致 |
| `make_sample.py` | 煙試験の見本(合成の図形だけ。原稿は使わない) |

## 問いの形

```json
{
  "interview_id": "l40_layer_kind_20261107",
  "question": "印の付いた層は、線画ですか?",
  "unit": "層",
  "decides": "この層を面の境界に使うかが決まる",
  "choices": [{"value": "line", "label": "線画", "help": "仕上げの線"},
              {"value": "rough", "label": "下絵", "help": "下描き・あたり"}],
  "items": [{"id": "p1_l71", "version": "v4", "note": "(任意の補足)",
             "question": "(任意。この問いだけ言い換えるとき)",
             "images": [{"src": "crops/p1_comp.png", "caption": "合成した絵"},
                        {"src": "crops/p1_l71.png", "caption": "問う層だけ", "target": true}]}]
}
```

- **「どれでもない」「分からない」は自動で足す。書かない**(書くと点検で落ちる)
- 画面に出る文字は**作者の言葉**で。`v_layer_kind` のような綴りは点検で落ちる
- 絵の `src` は問いの JSON のフォルダー(か `--image-root`)からの相対パス。**縮小して渡さない**(拡大縮小は画面がやる)。サーバーが出す URL は **`/img/<src>`**(`crops/f01.png` なら `/img/crops/f01.png`。Track L 2026-11-07)
- `batch_size`(任意)を書くと、その数ごとに「一区切り」の画面を出す。**既定は出さない**(作者 2026-11-07「不要」)。進み具合は常に上に出ている
- 合成前の層を並べ、問う対象に `"target": true`。同じ大きさの絵は同じ拡大で動く
- `initial_view`(任意): `fit`(既定、画面に収める)/ `native`(**等倍で始める**。線の欠け・薄さを問うとき。指針 §3-3)
- `page_title`(任意): 画面と artifact の題。無ければ `interview_id`(使い捨ての artifact を一覧で数えるため)
- 行には答えのほかに **`pos`(出した位置)/ `prev_item_id`(直前に見せた問い)/ `shown_seq` / `shown_scale`(見せた倍率)** が入る(指針 §6-2。Track E の直し 2026-11-07)

## 回し方

```bash
# 手元(または LAN)のサーバー。答えは <interview_id>_answers.jsonl に追記
./venv/bin/python tools/ui/interview_server.py data/<...>/questions.json --lan

# artifact(使い捨て)。出力は追跡しない場所へ。publish に capabilities {"db": {}, "user": {}}
./venv/bin/python tools/ui/build_interview_artifact.py data/<...>/questions.json -o <scratchpad>/interview.html
# 終わったら: ArtifactData の list で answers を JSONL に移す → 突き合わせ → ユーザーの確認 → 削除
./venv/bin/python tools/ui/verify_answers.py data/<...>/questions.json data/<...>/answers.jsonl --expect-rows <db の行数>
```

**答えと問いの JSON は原稿の名前を含むので、track の `data/` 配下に置く(追跡しない)。**

## 試したこと(2026-11-07)

- サーバー: 頁・絵・答えの読み書き、`../` と `%2e%2e` の抜け道は 403、問いに無い item は 400、続きから再開
- ヘッドレスの Chrome で PC 幅とタブレット幅の描画。縮小時に線が途切れる欠陥を見つけて直した
- **試していないもの**: 実機のタッチ(ピンチ・スワイプ)とキー操作、artifact 版の db への保存(公開していない)

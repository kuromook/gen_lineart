# 構造 DB の断面(census が受け取った控え)

**この文書は公開して安全**(数・日時・sha256 だけ。原稿の固有名を書かない)。
**DB 自身は作業木の外**に置く —— `git` に巻き込まれる経路を無くすため
(2026-10-23 に 99MB を public repo に上げた事故がある。force-push では消えない)。

| | |
|---|---|
| 置き場 | **`/home/sh1/deepl/census_db/manuscripts.duckdb`**(**作業木の外。追跡しない**) |
| 大きさ | 157,298,688 バイト |
| 断面(供給元のファイル日時) | **2026-10-26 17:57**(JST) |
| 受領 | 2026-10-28 22:55、`scp` で `sh1@192.168.1.5` から |
| sha256 | `65434b73af6c91ca8e6a50b1436419e40f3fae6e67424ddf28b9c4ee4c77bbb9` |
| 照合 | **供給元で `certutil` が出した値と一致**(コピーの事故は無い) |
| 読み手 | `./venv/bin/python` + `duckdb 1.5.6`(**読み取り専用で開く**) |

## 役割(ユーザーの指示、2026-10-28)

> **各トラックで原稿データに関する情報が欲しい需要が生まれたら census に聞けばわかる
> (基盤にも抽出にも聞かなくても)という状態にしたい**

- **census は読むだけ。書き戻さない。** DB を作り直すのは抽出側
- **答えるときは供給元と断面を名乗る** —— **この数は census が測ったのではない**(T132)
- **差分の正本は便の CSV** —— この断面に入っていない便が在る(下)

## この断面に**入っていない**もの(2026-10-26 17:57 より後の便)

- 遮蔽の全頁走行の出力(10-28、層の表・頁の表 各3本)—— **基盤の領分**
- 制作時刻の調査(10-28、`ts_probe_*`)—— **否定の結果。器の中に時刻は無い**
- 画布の単位(便 n5)・マスクの被覆・r7/r8/n1/n2/n4 の各便は、**DB に入っているか未確認の列が在る**
  —— **数が食い違ったら、便の CSV が正本**

## 表(23。`pages` 9,653 / `layers` 437,561)

`cmc_pages` 9,269 / `deliveries` 9,693 / `file_hashes` 9,653 / `file_times` 9,653 /
`koma_frames` 7,658 / `koma_overlaps` 4,925 / `layer_colour` 72,839 /
`layer_colour_text` 72,839 / `layers` 437,561 / `page_colour` 7,634 / `page_ids` 9,653 /
`page_tables` 160,446 / `page_versions`(ビュー)872 / `page_versions_history` 1,744 /
`pages` 9,653 / `requests` 9,692 / `v_folder_descendants` 108,865 / `v_layer_kind` 437,561 /
`v_layer_name_marks` 437,561 / `v_page_order` 9,653 / `v_page_structure` 9,653 /
`v_request_gaps` 0 / `v_work_colour` 289

## 公開中の数との突き合わせ(**転記の点検**)

| 量 | DB | 公開 | |
|---|---:|---:|---|
| 頁(全) | 9,653 | 9,653 | 一致 |
| 層(フォルダー込み) | 437,561 | 437,561 | 一致 |
| フォルダー | 108,865 | 108,865 | 一致 |
| 異なり sha256 | 7,418 | 7,418 | 一致 |
| 形式ごとの頁 | `.clip` 7,667 / `.lip` 1,986 | 同じ | 一致 |

**これは転記の一致である** —— 公開した数も DB も同じ走行から出ているので、
**壊れた DB でも一致する**(T132)。**材料が正しいことは、この点検では分からない。**

## `config/`(337MB・230ファイル)は引かなかった —— **数で確かめた**

DB を作る材料8種は、**全部この DB の表になっている**:
`clip_structure.jsonl` → `pages`/`layers` / `clip_file_times.csv` → `file_times` /
`clip_file_sha256.csv` → `file_hashes` / `page_saturation_full.csv` → `page_colour` /
`m_colour_page_layers.csv` → `layer_colour` / `version_roles_*.csv` → `page_versions_history` /
`delivery_registry.csv` → `requests`/`deliveries` / `koma_pages.jsonl` → `koma_frames`/`koma_overlaps`。

**`config/` が要るのは DB を作り直すときだけ**で、**census は読むだけなので要らない。**
**引かないことで、二重管理(抽出側と Track M が更新し続ける台帳の古い写しを持つこと)も避けた。**

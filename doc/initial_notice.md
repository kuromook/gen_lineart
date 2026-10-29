# `ops-review` —— 引き継ぎ(**運用の棚卸し。track ではない**)

**開設**: 2026-10-30(ユーザーの指示)/ **元**: `foundation` @ `a3dfd5a`
**この文書は短く保つ。履歴は `doc/work_log.md` に。**

## 0. この枝は何で、何でないか

- **これは実験の track ではない。** 文字(A〜N)を振らない。`CLAUDE.md` §5 の track 一覧にも入れない
- **成果物は「差分の提案」だけ** —— `CLAUDE.md` / `doc/worktree_policy.md` /
  `.claude/settings*.json` / フック / skill の配置について、**こう変える、という提案**
- **自分で正本を書き換えない。** **運用方針の正本は `CLAUDE.md` と `doc/worktree_policy.md` で、
  正本を直すのは基盤**(規律3: **食い違ったら `CLAUDE.md` を直してから他を更新する**)
- **口座全体(`~/.claude/`)に効くものは、変更の前に影響範囲を書く** ——
  他の企画と、**いま idle の5セッション**に効く
- **1回で終わる形にする。** 棚卸しがそれ自体で終わらない企画になるのを防ぐのが、この枝を分けた理由

## 1. 始める前に分かっていること(基盤が 2026-10-30 に実測)

### **フックが1つも無い。これが一番大きい**

`~/.claude/settings.json` / `.claude/settings.json`(中身は `{}`)/ `.claude/settings.local.json`
の**どれにも `hooks` キーが無い**。**ハーネスが実行する点検はゼロで、全部「Claude が思い出して回す」形。**

**そして、そういう形になっている規則が、この企画で一番効いている規則である**:

| 規則 | 実体 | いまの担保 |
|---|---|---|
| 許可表に無い追跡画像・追跡重みが在れば止める | `tools/check_public_artifacts.py`(終了コード1) | **Claude が思い出して回す** |
| 規律4: 台帳を開いたら audit を回す | `tools/audit_track_ledger.py` | **同上** |
| **原稿由来の納品物は `data/` 配下。絶対にコミットしない** | 規則のみ。**器具なし** | **同上** |
| **`.gitignore` は門ではない**(`git add -f` も既追跡も止めない) | 規則のみ。**器具なし** | **同上** |

**全部「沈黙が正常動作」の点検である**(`feedback_names_are_not_checks`:
**点検の名前が最も危険**)。**フックはまさにこれのためのもので、Claude ではなくハーネスが実行する。**

**実例**: 2026-10-30 に基盤は `check_public_artifacts.py` を2回手で回した。
**回したのは覚えていたからで、忘れた場合に何も起きないことは変わらない。**

### 文書の大きさ(**`CLAUDE.md` 自身の規則に照らす**)

| | 行 | 規則 |
|---|---:|---|
| `CLAUDE.md` | **150** | **「目安 150 行以内」。ちょうど上限に張り付いている** |
| `doc/worktree_policy.md` | 373 | — |
| `doc/glossary.md` | 270 | 全読みしてよい、とされている |
| `doc/CURRENT.md` | **4,390** | 全読み禁止。節単位で読む、が規則 |

### 許可(`.claude/settings.local.json`)

- **78件**(`Bash` 70 / `Read` 3 / `WebFetch` 4 / `WebSearch` 1)
- **明らかな残骸がある**: `Bash(break)`、
  2026年夏の `run_warmstart.log` を読む `awk` 一発が丸ごと1件、
  `Bash(cd /home/sh1/deepl/lineart)` と `Bash(cd /home/sh1/deepl/lineart && *)` の重複など
- **`/fewer-permission-prompts` skill がこの用途のために在る**

### 口座全体に在るもの(**変更は他の企画に効く**)

- `~/.claude/settings.json` —— `model` / `theme` / 許可6件 / `enabledPlugins` / `extraKnownMarketplaces`
- `~/.claude/skills/experiment-montage-hygiene/` ——
  **この企画専用の作法が口座側に在り、全企画に効いている。** 置き場として正しいかは未検討
- `~/.claude/skills/synced/` —— UUID 名のものが1つ。**中身を見ていない**

### moat(**本命の1つ**)

- **効いている実体は install 済みの凍結コピー**
  `~/.claude/plugins/cache/moat/moat/0.1.0/skills/`(`moat` / `moat-audit` / `moat-controls` /
  `moat-register` / `moat-retract` の5本)
- **`/home/sh1/deepl/moat` の作業木を編集しても、branch を切り替えても、効く規則は動かない**
  (2026-10-26 に確認。`extraKnownMarketplaces` は directory 指定だが、**実体は cache の方**)
- **`CLAUDE.md` §3 が記録している穴**: **規律9 の全体 / 判定保留は正直な結論 /
  曝露や母集団も棒の一部** の3つは **install 済みの moat のどこにも無く、
  `CLAUDE.md` の箇条書きが唯一の実体**である(2026-10-26 に grep で確認)。
  規律9 は moat の `provisional-selection` branch の暫定規則11〜13 に在るが、**install 済み main には無い**

**この節の数は全部 2026-10-30 のもの。引用する前に引き直すこと**(`names_are_not_checks`)。
**特に moat の「作業木を直しても効かない」は、確かめ直してから書くこと。**

## 2. 進め方(提案。変えてよい)

1. **測る** —— 上の節を自分で引き直す。**合わなければ、合わない方を書く**
2. **フックの案を出す** —— **どの規則を、どの hook 事象で、何を実行して担保するか**を表にする。
   **「入れる」ではなく「入れるとこうなる」を書く。** 誤発火でユーザーの作業が止まる形は最悪なので、
   **止める hook と、警告だけの hook を分ける**
3. **許可を整理する案** —— 残骸の一覧と、残す理由。`/fewer-permission-prompts` を使ってよい
4. **skill の配置の案** —— 企画専用のものが口座側に在ってよいか
5. **`CLAUDE.md` の差分案** —— **150行の上限を守ったまま**。
   **縮めるときに命令形を落とさない**(2026-10-26 に一度落ちた実例がある)
6. **基盤に出す** —— `cp outbox/<file> ../lineart/inbox/`。**正本を直すのは基盤**

## 条例

<!-- 型は doc/ordinance_spec.md。この節だけが SessionStart で注入される。
     基本法の写しを置かない —— 凍っても害が無いのはそのためである。 -->

**上位法**: 規律は `../lineart/doc/CURRENT.md § Working Discipline`、運用は
`../lineart/CLAUDE.md`。**この木の `CLAUDE.md` は branch 時点の複製なので読まない**(T92)。

**判定の単位**: **この枝は絵も部品も判定しない。** 運用の棚卸しで、腕も棒も無い。

1. **`~/.claude/` を自分で書き換えない。** 提案を出す。**実行はユーザーの承認の後**
2. **`foundation` を直接触らない。** push はこの枝だけ
3. **口座全体に効く変更は、影響範囲を先に書く** —— 他の企画と、動いている他のセッションに効く
4. **Python は `/home/sh1/deepl/lineart/venv/bin/python`。この worktree に `venv/` は無い**
5. **1回で終わる形にする。** 棚卸しがそれ自体で終わらない企画になるのを防ぐのが、この枝を分けた理由
6. **名前で判定しない。中身を読む。** この枝は2回踏んだ ——
   `.claude/` の配布を T84/T92 と混ぜ、montage skill を読まずに「企画専用」と判定した

## 4. 最初の3手

1. `cat ~/.claude/settings.json .claude/settings.json .claude/settings.local.json` と
   `ls ~/.claude/skills ~/.claude/plugins/cache/moat/moat/0.1.0/skills` —— **§1 を引き直す**
2. **`CLAUDE.md` を全部読む**(150行)。**規則のうち、器具が在るものと、文字しか無いものを分ける表を作る**
3. **2 の表のうち「文字しか無い」側だけを対象に、フックの案を書く** ——
   **器具が在るものは、回し忘れを止めれば足りる**

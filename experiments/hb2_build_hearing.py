"""聞き取り HB2 の問いを作る(登録: doc/work_log.md 2026-11-27「登録 HB2」)。HB と同じ選び方で HB の20コマも除き、
「位置を移動した」を主な理由と組み合わせて付ける札にする。出力 data/hb2_why_changed_20261127/(追跡しない)。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import hb_build_hearing  # noqa: E402

OUT = ROOT / "data/hb2_why_changed_20261127"


def spec(items):
    return {"interview_id": "s_hb2_why_changed_20261127", "page_title": "線を変えた理由 HB2",
            "question": "このコマで、線画が下絵から変わった所(青=動かした・赤=足した)の主な理由はどれですか?", "unit": "コマ",
            "decides": "下絵から線画で線を動かした・足した理由を、作者の言葉で種類に分けるのに使います。位置の移動は理由と別に、下の札で付けてください",
            "choices": [{"value": "refine", "label": "清書で形を整えた", "help": "下絵の線を、線画で正確に確定した"},
                        {"value": "detail", "label": "下絵に無い細部を描き足した", "help": "髪・服のしわ・影の線・背景の細部など"},
                        {"value": "change", "label": "形・ポーズ・構図を変えた", "help": "下絵と違う形に描き直した(場所をずらしただけなら下の札)"},
                        {"value": "atari", "label": "下絵があたり程度で、形は線画で決めた", "help": "下絵は大まかな位置だけ"},
                        {"value": "instrument", "label": "色分けの誤り", "help": "ずれ・塗り・抽出の失敗などで色が正しくない(一言に書いてください)"}],
            "toggles": [{"value": "moved", "label": "位置を移動した", "key": "m",
                         "help": "下絵の形のまま場所をずらした所がある。上の理由と一緒に付けられます。移動だけなら札だけで保存できます"}],
            "items": items}


if __name__ == "__main__":
    hb_build_hearing.main(out=OUT, seed=20261127, also_skip=("hb_why_changed_20261128",), make_spec=spec, version="s4_t1_c050_hb2")

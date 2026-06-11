"""ako5の rough(下描き)/line(ペン入れ) のタイル単位のズレを局所平行移動で復元し、
残差ゲートで「正しく対応の取れたペアだけ」を抽出するパイプライン。

方式（検証で確定）:
- ページ単位の剛体/アフィン(ECC)は無効 → ズレは局所非一様 → タイルごとに局所平行移動を探索
- 探索: blur済みエッジマップのFFT相互相関で ±SEARCH px のベスト平行移動を求める
- 残差ゲート: アライン後の line画素→最寄りrough線 距離の中央値 <= RESID_TH のタイルだけ採用
  （描き直し/空白/ベタは残差大 or インク率帯外で自動棄却）

出力:
- 採用タイルの aligned-rough と line を dataset_480/train/{rough_aligned,line_aligned}/ に ako5a_ 接頭辞で保存
- リスト dataset_480/valid_train_ako5_aligned.txt
- 歩留まり統計 と QCモンタージュ results/ako5_aligned_qc.png
"""
import argparse
import io
import json
import os
import zipfile

import numpy as np
from numpy.fft import irfft2, rfft2
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage

ZIP_PATH = os.path.expanduser("~/dataset_ako5.zip")
ROUGH_OUT = "dataset_480/train/rough_aligned"
LINE_OUT = "dataset_480/train/line_aligned"
LIST_OUT = "dataset_480/valid_train_ako5_aligned.txt"
QC_OUT = "results/ako5_aligned_qc.png"

TS = 480
SEARCH = 128          # ±px local translation search
ROUGH_STD = 15        # rough autocontrast std floor (内容のあるタイルだけ)
INK_LO, INK_HI = 0.02, 0.25
RESID_TH = 5.0        # 採用残差(px): 対称チャンファ max(line→rough, rough→line)
PSR_MIN = 4.0         # 相関ピーク鋭さの下限（偽マッチ=平坦を棄却）
TONE_MAX = 450.0      # 網点トーン度の上限（網点タイルを棄却。クリーン線画~200-400, 網点>1000）


def load_page(zf, name):
    return np.asarray(Image.open(io.BytesIO(zf.open(f"dataset_ako5/{name}").read())).convert("L"))


def best_shift(line_mask, rough_big, M):
    """line_mask(TSxTS) を rough_big((TS+2M)^2) 内で最も重ねられる平行移動(dy,dx)とPSR。

    PSR(peak-to-sidelobe ratio): 相関ピークの鋭さ。本物の対応は鋭い単峰、
    偶然重なった別絵は相関面が平坦→低PSR。(peak-mean)/std で算出。
    """
    lm = ndimage.gaussian_filter(line_mask.astype(np.float32), 2)
    rb = ndimage.gaussian_filter(rough_big.astype(np.float32), 2)
    H, W = rb.shape
    corr = irfft2(rfft2(rb) * np.conj(rfft2(lm, s=(H, W))), s=(H, W))
    v = corr[:2 * M + 1, :2 * M + 1]
    iy, ix = np.unravel_index(np.argmax(v), v.shape)
    peak = v[iy, ix]
    # ピーク周辺11x11を除いた残りでPSRを計算
    mask = np.ones_like(v, bool)
    y0 = max(0, iy - 5); y1 = min(v.shape[0], iy + 6)
    x0 = max(0, ix - 5); x1 = min(v.shape[1], ix + 6)
    mask[y0:y1, x0:x1] = False
    side = v[mask]
    psr = float((peak - side.mean()) / (side.std() + 1e-6))
    return iy - M, ix - M, psr


def tone_score(gray):
    """網点スクリーントーン度: FFT中高域に鋭い周期ピークがあれば高い。

    網点=規則的なドット格子→周波数領域でDC外に明瞭なピーク。
    線画はブロードバンドでピークなし。annulus内 max/median で評価。
    """
    g = gray.astype(np.float32)
    g = g - g.mean()
    # ハニング窓で端のリークを抑制
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    F = np.fft.fftshift(np.abs(np.fft.fft2(g * win)))
    cy, cx = np.array(F.shape) // 2
    yy, xx = np.ogrid[:F.shape[0], :F.shape[1]]
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # 低域(構図のエッジ)を除外した中高域アニュラス
    ann = (rad >= F.shape[0] * 0.06) & (rad <= F.shape[0] * 0.45)
    vals = F[ann]
    if vals.size == 0:
        return 0.0
    return float(vals.max() / (np.median(vals) + 1e-6))


def process_page(R, L, M):
    """1ページから採用候補タイルを返す: list of dict."""
    h = min(R.shape[0], L.shape[0])
    w = min(R.shape[1], L.shape[1])
    R = R[:h, :w]
    L = L[:h, :w]
    out = []
    for r in range(h // TS):
        for c in range(w // TS):
            y0, x0 = r * TS, c * TS
            rt = R[y0:y0 + TS, x0:x0 + TS]
            lt = L[y0:y0 + TS, x0:x0 + TS]
            if rt.std() < ROUGH_STD:
                continue
            lb = lt < 128
            lo = lb.mean()
            if lo < INK_LO or lo > INK_HI:
                continue
            # rough big window (margin M, clamped)
            yb0, xb0 = max(0, y0 - M), max(0, x0 - M)
            yb1, xb1 = min(h, y0 + TS + M), min(w, x0 + TS + M)
            big = np.asarray(ImageOps.autocontrast(Image.fromarray(R[yb0:yb1, xb0:xb1]), 0))
            bdark = np.zeros((TS + 2 * M, TS + 2 * M), bool)
            bgray = np.full((TS + 2 * M, TS + 2 * M), 255, np.uint8)
            oy, ox = yb0 - (y0 - M), xb0 - (x0 - M)
            bdark[oy:oy + big.shape[0], ox:ox + big.shape[1]] = big < 100
            bgray[oy:oy + big.shape[0], ox:ox + big.shape[1]] = big
            dy, dx, psr = best_shift(lb, bdark, M)
            sy, sx = M + dy, M + dx
            rb_al = bdark[sy:sy + TS, sx:sx + TS]
            if not rb_al.any():
                continue
            # 対称チャンファ: 片方向(line->rough)だけだと密roughに騙されるので両方向のmax
            d_l2r = float(np.median(ndimage.distance_transform_edt(~rb_al)[lb]))
            d_r2l = float(np.median(ndimage.distance_transform_edt(~lb)[rb_al]))
            resid = max(d_l2r, d_r2l)
            al_gray = bgray[sy:sy + TS, sx:sx + TS]
            tone = max(tone_score(lt), tone_score(al_gray))  # line/rough どちらかが網点なら高い
            out.append({
                "r": r, "c": c, "resid": resid, "d_l2r": d_l2r, "d_r2l": d_r2l,
                "psr": psr, "tone": tone,
                "dy": dy, "dx": dx, "ink": lo,
                "rough_orig": rt.copy(),
                "rough_aligned": al_gray.copy(),
                "line": lt.astype(np.uint8).copy(),
            })
    return out


def make_qc(accepted, path, n=12):
    accepted = sorted(accepted, key=lambda t: t["resid"])
    # 採用域から等間隔サンプル(最良〜閾値ぎわまで)
    idx = np.linspace(0, len(accepted) - 1, min(n, len(accepted))).astype(int)
    pick = [accepted[i] for i in idx]
    TH = 220
    lab = 14
    canvas = Image.new("RGB", (TH * 3, (TH + lab) * len(pick)), (180, 180, 180))
    dr = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    for i, t in enumerate(pick):
        y = i * (TH + lab)
        for j, key in enumerate(["rough_orig", "rough_aligned", "line"]):
            canvas.paste(Image.fromarray(t[key]).convert("RGB").resize((TH, TH)), (j * TH, y))
        dr.text((3, y + TH + 1),
                f'{t["page"]} r{t["r"]}c{t["c"]} resid={t["resid"]:.1f} PSR={t["psr"]:.1f} tone={t["tone"]:.0f}  [rough|aligned|line]',
                fill=(0, 0, 0), font=font)
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=0, help="先頭Nページだけ処理(0=全部)")
    ap.add_argument("--search", type=int, default=SEARCH)
    ap.add_argument("--resid", type=float, default=RESID_TH)
    ap.add_argument("--psr-min", type=float, default=PSR_MIN, dest="psr_min")
    ap.add_argument("--tone-max", type=float, default=TONE_MAX, dest="tone_max")
    ap.add_argument("--save", action="store_true", help="採用タイルを実際に保存しリスト出力")
    args = ap.parse_args()
    M = args.search

    def passes(t):
        return (t["resid"] <= args.resid and t["psr"] >= args.psr_min
                and t["tone"] <= args.tone_max)

    cand = []
    accepted = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        man = json.load(zf.open("dataset_ako5/manifest.json"))
        pages = man if args.pages == 0 else man[:args.pages]
        for e in pages:
            R = np.asarray(ImageOps.autocontrast(Image.fromarray(load_page(zf, e["sketch"])), 0))
            L = load_page(zf, e["line"])
            pname = e["sketch"][6:13]  # page000X
            tiles = process_page(R, L, M)
            npass = 0
            for t in tiles:
                t["page"] = pname
                cand.append(t)
                if passes(t):
                    accepted.append(t)
                    npass += 1
            print(f"{pname}: candidates={len(tiles)} accepted={npass}")

    a = np.array([t["resid"] for t in cand], float)
    print(f"\n=== clean-ink帯 候補タイル: {len(a)} ===")
    print(f"ゲート: resid<={args.resid} かつ psr>={args.psr_min} かつ tone<={args.tone_max}")
    for th in (5, 8, 10, 12, 15):
        print(f"  resid<={th:2d}px(単独)        : {(a<=th).sum():4d} ({(a<=th).mean()*100:.1f}%)")
    print(f"\n採用(全ゲート通過): {len(accepted)} タイル")

    os.makedirs("results", exist_ok=True)
    make_qc(accepted, QC_OUT)
    print(f"QCモンタージュ: {QC_OUT}")

    if args.save and accepted:
        os.makedirs(ROUGH_OUT, exist_ok=True)
        os.makedirs(LINE_OUT, exist_ok=True)
        names = []
        for t in accepted:
            fn = f'ako5a_{t["page"][4:]}_{t["r"]:02d}_{t["c"]:02d}.jpg'
            Image.fromarray(t["rough_aligned"]).save(os.path.join(ROUGH_OUT, fn), quality=95)
            Image.fromarray(t["line"]).save(os.path.join(LINE_OUT, fn), quality=95)
            names.append(fn)
        with open(LIST_OUT, "w") as f:
            for n in sorted(names):
                f.write(n + "\n")
        print(f"保存: {len(names)} ペア → {ROUGH_OUT}/ {LINE_OUT}/  リスト {LIST_OUT}")


if __name__ == "__main__":
    main()

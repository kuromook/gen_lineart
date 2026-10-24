# -*- coding: utf-8 -*-
"""L16 準備: 隙間埋めの精度 p と、取れる面の数の関係 (計算のみ。棒なし)。

問い: 「1区間あたり精度 p で隙間を当てたとき、塗りつぶしで取れる面はどれだけか」。
新しい画像処理はしない。既存の CSV だけから出す。

  k_i  = 面 i が持つ開いた区間の数 (障壁=線だけ。これが「埋めねばならない隙間」の数)
  recall(p) = mean_i p^{k_i}        区間の成否が独立だと仮定したとき

母集団は **データセットの矩形の上の 1,325面** (`results/l14_arms/per_region.csv`)。
作者のコマの上の 1,563面 (`results/l15_zorder_occ/per_region.csv`) には区間数が入って
いないので、**区間数は矩形母集団の値である**と明記する。

天井の見積り: 隙間を全部閉じても IoU>=0.5 になるとは限らない (ベタは線の下にも伸びる)。
`diagnosis.csv` の「漏れていない面」(自由成分/面 が 0.5〜2) で IoU>=0.5 の割合を
P(照合する | 正しく囲めた) の代用とする。
"""
import csv, collections, os, statistics as st

OUT = '/home/sh1/deepl/lineart-fill-region/results/l16_tolerance'
REG = '/home/sh1/deepl/lineart-fill-region/results/l14_arms/per_region.csv'
RUN = '/home/sh1/deepl/lineart-fill-region/results/l15_open_runs/per_run.csv'
DIA = '/home/sh1/deepl/lineart-fill-region/results/l14_arms/diagnosis.csv'
WORKS = ('ako6', 'ako7')
PS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999]
TARGETS = (0.25, 0.50, 0.80)


def counts(variant, drop_O=False):
    """面ごとの区間数。区間を1本も持たない面は k=0 として分母に入れる。"""
    k = {}
    for r in csv.DictReader(open(REG)):
        k[(r['name'], r['region'])] = [r['work'], 0]
    n_missing = 0
    for r in csv.DictReader(open(RUN)):
        if r['arm'] != 'real' or r['variant'] != variant:
            continue
        if drop_O and r['kind'] == 'O':
            continue
        key = (r['name'], r['region'])
        if key not in k:
            n_missing += 1; continue
        k[key][1] += 1
    assert n_missing == 0, n_missing
    out = collections.defaultdict(list)
    for (nm, rg), (w, c) in k.items():
        out[w].append(c)
    return out


def q(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(f * len(v)))]


def recall(ks, p):
    return sum(p ** c for c in ks) / len(ks)


def need(ks, target):
    """recall(p) = target となる p を二分法で。届かなければ None。"""
    if recall(ks, 1.0) < target:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if recall(ks, mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def ceiling():
    """漏れていない面に限ったときの IoU>=0.5 の割合 (障壁 = 線+細い印)。"""
    out = {}
    for w in WORKS:
        s = [r for r in csv.DictReader(open(DIA)) if r['work'] == w]
        g = [r for r in s if 0.5 <= float(r['ratio']) <= 2.0]
        out[w] = (len(s), len(g),
                  sum(1 for r in g if float(r['iou0']) >= 0.5) / len(g) if g else 0.0,
                  sum(1 for r in g if float(r['iou1']) >= 0.5) / len(g) if g else 0.0)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows_dist, rows_curve, rows_need = [], [], []

    print('=' * 90)
    print('1. 区間数 k の分布 (面ごと。区間0本の面も分母に入れる)')
    print(f'{"障壁":10} {"作品":5} {"面":>5} {"k=0":>6} {"中央":>5} {"平均":>6} '
          f'{"p75":>5} {"p90":>5} {"p99":>5} {"最大":>5}')
    dists = {}
    for variant in ('line', 'line+gap'):
        for drop_O in (False, True):
            lab = variant + ('' if not drop_O else ' (O除く)')
            c = counts(variant, drop_O)
            for w in WORKS:
                ks = c[w]
                dists[(lab, w)] = ks
                row = dict(barrier=lab, work=w, n_region=len(ks),
                           k_zero=sum(1 for x in ks if x == 0),
                           k_median=st.median(ks), k_mean=round(sum(ks) / len(ks), 2),
                           k_p75=q(ks, .75), k_p90=q(ks, .90), k_p99=q(ks, .99),
                           k_max=max(ks))
                rows_dist.append(row)
                print(f'{lab:10} {w:5} {len(ks):5d} {row["k_zero"]:6d} '
                      f'{row["k_median"]:5.1f} {row["k_mean"]:6.2f} '
                      f'{row["k_p75"]:5d} {row["k_p90"]:5d} {row["k_p99"]:5d} {row["k_max"]:5d}')

    print('\n' + '=' * 90)
    print('2. 曲線 recall(p) = mean_i p^{k_i}   (区間の成否は独立と仮定)')
    print('   **障壁 = 線だけ。これが「埋めねばならない隙間」の数**')
    hdr = ' '.join(f'{p:>7}' for p in PS)
    print(f'{"作品":6} {hdr}')
    for w in WORKS:
        ks = dists[('line', w)]
        vals = [recall(ks, p) for p in PS]
        print(f'{w:6} ' + ' '.join(f'{v:7.4f}' for v in vals))
        for p, v in zip(PS, vals):
            rows_curve.append(dict(barrier='line', work=w, p=p, recall=round(v, 6)))
    print('\n   参考: O (口が無い区間) を除いたとき')
    for w in WORKS:
        ks = dists[('line (O除く)', w)]
        vals = [recall(ks, p) for p in PS]
        print(f'{w:6} ' + ' '.join(f'{v:7.4f}' for v in vals))
        for p, v in zip(PS, vals):
            rows_curve.append(dict(barrier='line (O除く)', work=w, p=p, recall=round(v, 6)))

    print('\n' + '=' * 90)
    print('3. 必要な p (面の 25% / 50% / 80% を取るのに要る1区間あたりの精度)')
    print(f'{"障壁":14} {"作品":6} ' + ' '.join(f'{int(t*100):>8}%' for t in TARGETS))
    for lab in ('line', 'line (O除く)'):
        for w in WORKS:
            ks = dists[(lab, w)]
            ns = [need(ks, t) for t in TARGETS]
            print(f'{lab:14} {w:6} ' +
                  ' '.join((f'{n:9.4f}' if n is not None else '    届かず') for n in ns))
            for t, n in zip(TARGETS, ns):
                rows_need.append(dict(barrier=lab, work=w, target=t,
                                      p_needed=(round(n, 5) if n is not None else '')))

    print('\n' + '=' * 90)
    print('4. 天井: 隙間を全部閉じても IoU>=0.5 になるとは限らない')
    print('   漏れていない面 (自由成分/面 が 0.5〜2、障壁=線+細い印) に限った IoU>=0.5 の割合')
    cl = ceiling()
    for w in WORKS:
        n, ng, c0, c1 = cl[w]
        print(f'  {w}: 面 {n}  うち漏れていない {ng} ({ng/n:.3f})  '
              f'IoU>=0.5 {c0:.4f} (1px膨張後 {c1:.4f})')
    print('\n   天井を掛けた現実的な上限 recall(p) x 天井')
    print(f'{"作品":6} {hdr}')
    for w in WORKS:
        ks = dists[('line', w)]
        c = cl[w][2]
        vals = [recall(ks, p) * c for p in PS]
        print(f'{w:6} ' + ' '.join(f'{v:7.4f}' for v in vals))
        for p, v in zip(PS, vals):
            rows_curve.append(dict(barrier='line x ceiling', work=w, p=p,
                                   recall=round(v, 6)))

    print('\n   天井つきで 25%/50% を取るのに要る p')
    for w in WORKS:
        ks = dists[('line', w)]
        c = cl[w][2]
        for t in TARGETS:
            tt = t / c if c > 0 else 9
            n = need(ks, tt) if tt <= 1.0 else None
            print(f'  {w} 面の {int(t*100)}%: ' +
                  (f'p = {n:.4f}' if n is not None else
                   f'**到達不能** (天井 {c:.4f} < {t})'))
            rows_need.append(dict(barrier='line x ceiling', work=w, target=t,
                                  p_needed=(round(n, 5) if n is not None else '')))

    for nm, rows in (('k_distribution', rows_dist), ('curve', rows_curve),
                     ('p_needed', rows_need)):
        with open(f'{OUT}/{nm}.csv', 'w', newline='', encoding='utf-8') as fh:
            w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w_.writeheader()
            for r in rows:
                w_.writerow(r)
        print(f'-> {OUT}/{nm}.csv  {len(rows)} 行')


if __name__ == '__main__':
    main()

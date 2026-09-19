#!/usr/bin/env python
"""Same load, same duration, before and after cleaning the cooler.

The machine runs at 1,499MHz (base 3.5GHz) at 91-95C under the project's own
jobs, and the kernel's idle_inject threads hold every core at ~47%. That is a
~2.5x tax on every CPU estimate in this project, so the fix is worth measuring
rather than eyeballing: run this with the machine otherwise idle, before and
after, and compare.
"""
import argparse, multiprocessing as mp, time
from pathlib import Path

import numpy as np


def burn(seconds):
    a = np.random.rand(512, 512)
    end = time.time() + seconds
    while time.time() < end:
        a = a @ a
        a /= (np.abs(a).max() + 1e-9)


def read_temps():
    t = []
    for p in sorted(Path("/sys/class/hwmon").glob("hwmon*/temp*_input")):
        try:
            t.append(int(p.read_text()) / 1000.0)
        except Exception:
            pass
    return t


def read_mhz():
    f = []
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("cpu MHz"):
            f.append(float(line.split(":")[1]))
    return f


def throttle_counts():
    d = Path("/sys/devices/system/cpu/cpu0/thermal_throttle")
    out = {}
    for k in ("package_throttle_count", "package_throttle_total_time_ms"):
        p = d / k
        if p.exists():
            out[k] = int(p.read_text())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=int, default=120)
    p.add_argument("--workers", type=int, default=mp.cpu_count())
    p.add_argument("--label", default="")
    a = p.parse_args()
    print(f"=== 全コア負荷テスト {a.seconds}秒 / {a.workers}並列 {a.label}")
    idle_t = read_temps(); idle_f = read_mhz(); t0 = throttle_counts()
    print(f"開始前: 温度 {max(idle_t):.0f}C  周波数 {max(idle_f):.0f}MHz")
    procs = [mp.Process(target=burn, args=(a.seconds,)) for _ in range(a.workers)]
    [q.start() for q in procs]
    samples = []
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        time.sleep(2)
        samples.append((max(read_temps()), float(np.mean(read_mhz()))))
    [q.join() for q in procs]
    t1 = throttle_counts()
    warm = samples[len(samples) // 3:]          # skip the ramp
    temps = [s[0] for s in warm]; freqs = [s[1] for s in warm]
    print(f"負荷時: 温度 最大 {max(temps):.0f}C / 平均 {np.mean(temps):.0f}C")
    print(f"        周波数 平均 {np.mean(freqs):.0f}MHz / 最低 {min(freqs):.0f}MHz  (定格3500 / 最大3900)")
    for k in t0:
        print(f"        {k}: +{t1.get(k, 0) - t0[k]}")
    time.sleep(20)
    print(f"冷却後20秒: 温度 {max(read_temps()):.0f}C")
    verdict = "良好" if np.mean(freqs) > 3000 and max(temps) < 85 else \
              "まだ熱制限" if np.mean(freqs) < 2500 else "改善したが余地あり"
    print(f"判定: {verdict}")


if __name__ == "__main__":
    main()

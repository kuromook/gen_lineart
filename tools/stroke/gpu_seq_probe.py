#!/usr/bin/env python
"""Longest stroke-token sequence a transformer can train on with this GPU.

One stroke = one token (d-dim embedding). Measures a full train step
(forward + backward + AdamW) at bf16 autocast with PyTorch SDPA, sweeping the
sequence length until OOM. Reports peak memory and step time.
"""
import argparse, time, torch, torch.nn as nn

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--ckpt", action="store_true")
    a = p.parse_args()
    layer = lambda: nn.TransformerEncoderLayer(a.d, a.heads, 4 * a.d, dropout=0.0, batch_first=True, norm_first=True)
    blocks = nn.ModuleList([layer() for _ in range(a.layers)]).cuda()
    head = nn.Linear(a.d, 1).cuda()
    params = sum(x.numel() for x in blocks.parameters())
    opt = torch.optim.AdamW(list(blocks.parameters()) + list(head.parameters()), 1e-4)
    print(f"d={a.d} L={a.layers} params={params/1e6:.1f}M batch={a.batch} ckpt={a.ckpt}")
    print("seq_len,peak_gb,step_s")
    n = 1000
    while n <= 256000:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(a.batch, n, a.d, device="cuda")
            for it in range(2):
                torch.cuda.synchronize(); t = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = x
                    for b in blocks:
                        h = torch.utils.checkpoint.checkpoint(b, h, use_reentrant=False) if a.ckpt else b(h)
                    loss = head(h).float().mean()
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); dt = time.time() - t
            print(f"{n},{torch.cuda.max_memory_allocated()/2**30:.2f},{dt:.2f}", flush=True)
            del x, h, loss
            if dt > 60: break
            n *= 2
        except torch.OutOfMemoryError:
            print(f"{n},OOM,"); break

if __name__ == "__main__":
    main()

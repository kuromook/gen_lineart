"""SWA-style blend of two checkpoints from the same training run (early
precise-but-soft epoch + late confident-but-drifting epoch), to test
whether combining them recovers both properties without new training.
See doc/work_log.md ("Next Weekend's Plan", option 1).

Two blend modes:
- weight: average the two state_dicts tensor-by-tensor, then a single
  forward pass with the averaged weights.
- logit: forward pass through each checkpoint separately, average the
  raw pre-sigmoid logits, then sigmoid once.

Writes results/{tag}/{base}_out.png in the existing convention so
evaluate_fixed_outputs.py / evaluate_stroke_stability.py /
make_multi_model_eval_compare.py work unchanged.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

from lineart.model_zoo import build_generator, load_generator_checkpoint


IMAGE_SIZE = 480


def read_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def load_tensor(path, autocontrast=False):
    image = Image.open(path).convert("L")
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=0)
    return TF.to_tensor(TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE)))


def weight_average(ckpt_a, ckpt_b, model_name, in_channels, device):
    model_a, _ = load_generator_checkpoint(ckpt_a, device)
    model_b, _ = load_generator_checkpoint(ckpt_b, device)
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    averaged = {k: (sd_a[k].float() + sd_b[k].float()) / 2.0 for k in sd_a}
    model = build_generator(model_name, in_channels=in_channels).to(device)
    model.load_state_dict(averaged)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", required=True, help="early precise-but-soft")
    parser.add_argument("--checkpoint-b", required=True, help="late confident-but-drifting")
    parser.add_argument("--model-name", default="unet_skip0")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--mode", choices=["weight", "logit"], required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--rough-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--autocontrast", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "weight":
        model = weight_average(args.checkpoint_a, args.checkpoint_b, args.model_name, args.in_channels, device)
        count = 0
        with torch.no_grad():
            for sample in read_list(args.file_list):
                base = Path(sample).stem
                x = load_tensor(Path(args.rough_dir) / f"{base}.jpg", autocontrast=args.autocontrast).unsqueeze(0).to(device)
                pred = torch.sigmoid(model(x))
                output = (1.0 - pred).clamp(0, 1)
                TF.to_pil_image(output[0].cpu()).save(output_dir / f"{base}_out.png")
                count += 1
        print(f"saved {count} files to {output_dir} mode=weight")
        return

    model_a, _ = load_generator_checkpoint(args.checkpoint_a, device)
    model_b, _ = load_generator_checkpoint(args.checkpoint_b, device)
    model_a.eval()
    model_b.eval()
    count = 0
    with torch.no_grad():
        for sample in read_list(args.file_list):
            base = Path(sample).stem
            x = load_tensor(Path(args.rough_dir) / f"{base}.jpg", autocontrast=args.autocontrast).unsqueeze(0).to(device)
            logit_a = model_a(x)
            logit_b = model_b(x)
            pred = torch.sigmoid((logit_a + logit_b) / 2.0)
            output = (1.0 - pred).clamp(0, 1)
            TF.to_pil_image(output[0].cpu()).save(output_dir / f"{base}_out.png")
            count += 1
    print(f"saved {count} files to {output_dir} mode=logit")


if __name__ == "__main__":
    main()

"""Batch routed line-art inference with per-expert model caching."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

from lineart.route_dataset_source import DEFAULT_FAMILY_GATE, DEFAULT_KURIP_AKO5_GATE, route_decision
from lineart.unetgenerator import UNetGenerator
from scripts.inference_routed import DEFAULT_REGISTRY, IMAGE_SIZE, load_registry, resolve_expert


def read_inputs(args):
    inputs = []
    if args.inputs:
        inputs.extend(args.inputs)
    if args.input_list:
        with open(args.input_list) as file:
            inputs.extend(line.strip() for line in file if line.strip())
    if not inputs:
        raise ValueError("No inputs provided")
    return inputs


def output_path_for(input_path, output_dir):
    stem = Path(input_path).stem
    return str(Path(output_dir) / f"{stem}_out.png")


def route_one(input_path, args, registry, experts):
    route_args = SimpleNamespace(
        input=input_path,
        inputs=args.route_inputs,
        family_gate=args.family_gate,
        kurip_ako5_gate=args.kurip_ako5_gate,
        family_threshold=args.family_threshold,
        source_threshold=args.source_threshold,
        crop_count=args.crop_count,
        prefer_filename=args.prefer_filename,
        device=args.device,
    )
    decision = route_decision(route_args)
    expert = resolve_expert(decision.get("route", "default"), registry, experts)
    return decision, expert


def load_model(checkpoint, device):
    model = UNetGenerator(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def load_tensor(input_path, autocontrast, device):
    image = Image.open(input_path).convert("L")
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=0)
    tensor = TF.to_tensor(TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE))).unsqueeze(0)
    return tensor.to(device)


def save_prediction(model, input_path, output_path, autocontrast, device):
    tensor = load_tensor(input_path, autocontrast, device)
    with torch.no_grad():
        out = torch.sigmoid(model(tensor))
    out = (1.0 - out).clamp(0, 1)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    TF.to_pil_image(out[0].cpu()).save(output_path)


def write_reports(rows, summary_json, summary_csv):
    os.makedirs(os.path.dirname(summary_json) or ".", exist_ok=True)
    with open(summary_json, "w") as file:
        json.dump(rows, file, indent=2)
        file.write("\n")
    if summary_csv:
        os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
        fields = [
            "input",
            "output",
            "expert",
            "checkpoint",
            "dataset_source",
            "confidence",
            "route_source",
        ]
        with open(summary_csv, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                decision = row["route_decision"]
                writer.writerow({
                    "input": row["input"],
                    "output": row["output"],
                    "expert": row["expert"],
                    "checkpoint": row["checkpoint"],
                    "dataset_source": decision.get("dataset_source", ""),
                    "confidence": decision.get("confidence", ""),
                    "route_source": decision.get("source", "gate"),
                })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=None)
    parser.add_argument("--input-list", default=None)
    parser.add_argument("--output-dir", default="results/routed_batch")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--family-gate", default=DEFAULT_FAMILY_GATE)
    parser.add_argument("--kurip-ako5-gate", default=DEFAULT_KURIP_AKO5_GATE)
    parser.add_argument("--family-threshold", type=float, default=0.80)
    parser.add_argument("--source-threshold", type=float, default=0.80)
    parser.add_argument("--crop-count", type=int, default=9)
    parser.add_argument("--route-inputs", nargs="+", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--autocontrast", action="store_true")
    parser.add_argument("--no-prefer-filename", dest="prefer_filename", action="store_false")
    parser.set_defaults(prefer_filename=True)
    parser.add_argument("--summary-json", default="results/routed_batch/routes.json")
    parser.add_argument("--summary-csv", default="results/routed_batch/routes.csv")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    registry, experts = load_registry(args.registry)
    inputs = read_inputs(args)
    rows = []
    by_checkpoint = {}
    for input_path in inputs:
        decision, expert = route_one(input_path, args, registry, experts)
        output_path = output_path_for(input_path, args.output_dir)
        row = {
            "input": input_path,
            "output": output_path,
            "route_decision": decision,
            "expert": expert["name"],
            "checkpoint": expert["checkpoint"],
            "autocontrast": bool(args.autocontrast),
        }
        rows.append(row)
        by_checkpoint.setdefault(expert["checkpoint"], []).append(row)

    for checkpoint, checkpoint_rows in by_checkpoint.items():
        model = load_model(checkpoint, device)
        for row in checkpoint_rows:
            save_prediction(model, row["input"], row["output"], args.autocontrast, device)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    write_reports(rows, args.summary_json, args.summary_csv)
    print(json.dumps({
        "count": len(rows),
        "experts": {
            checkpoint: len(checkpoint_rows)
            for checkpoint, checkpoint_rows in by_checkpoint.items()
        },
        "summary_json": args.summary_json,
        "summary_csv": args.summary_csv,
    }, indent=2))


if __name__ == "__main__":
    main()

"""Run line-art inference through the dataset/source expert router."""

import argparse
import json
import os
from types import SimpleNamespace

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

from lineart.route_dataset_source import (
    DEFAULT_FAMILY_GATE,
    DEFAULT_KURIP_AKO5_GATE,
    route_decision,
)
from lineart.unetgenerator import UNetGenerator


IMAGE_SIZE = 480
DEFAULT_REGISTRY = "config/expert_registry.json"


def load_registry(path):
    with open(path) as file:
        registry = json.load(file)
    experts = {expert["name"]: expert for expert in registry["experts"]}
    return registry, experts


def resolve_expert(route_name, registry, experts):
    if route_name in experts:
        return experts[route_name]
    default_name = registry.get("default_expert", "shape1")
    return experts[default_name]


def run_inference(checkpoint, input_path, output_path, autocontrast=False, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNetGenerator(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    image = Image.open(input_path).convert("L")
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=0)
    tensor = TF.to_tensor(TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE))).unsqueeze(0).to(device)

    with torch.no_grad():
        out = torch.sigmoid(model(tensor))
    out = (1.0 - out).clamp(0, 1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    TF.to_pil_image(out[0].cpu()).save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--route-inputs", nargs="+", default=None)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--family-gate", default=DEFAULT_FAMILY_GATE)
    parser.add_argument("--kurip-ako5-gate", default=DEFAULT_KURIP_AKO5_GATE)
    parser.add_argument("--family-threshold", type=float, default=0.80)
    parser.add_argument("--source-threshold", type=float, default=0.80)
    parser.add_argument("--crop-count", type=int, default=9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--autocontrast", action="store_true")
    parser.add_argument("--no-prefer-filename", dest="prefer_filename", action="store_false")
    parser.set_defaults(prefer_filename=True)
    parser.add_argument("--route-json", default=None)
    args = parser.parse_args()

    registry, experts = load_registry(args.registry)
    route_args = SimpleNamespace(
        input=args.input,
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
    checkpoint = expert["checkpoint"]
    run_inference(checkpoint, args.input, args.output, args.autocontrast, args.device)

    result = {
        "input": args.input,
        "output": args.output,
        "route_decision": decision,
        "expert": expert["name"],
        "checkpoint": checkpoint,
        "autocontrast": bool(args.autocontrast),
    }
    if args.route_json:
        os.makedirs(os.path.dirname(args.route_json) or ".", exist_ok=True)
        with open(args.route_json, "w") as file:
            json.dump(result, file, indent=2)
            file.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

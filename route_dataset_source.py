"""Route an input rough image to a dataset/source style label."""

import argparse
import json

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

from train_dataset_gate_cnn import GateCNN
from dataset_gate import label_from_name


DEFAULT_FAMILY_GATE = "checkpoints/dataset_gate/gate_family_cnn.pth"
DEFAULT_KURIP_AKO5_GATE = "checkpoints/dataset_gate/gate_kurip_ako5_cnn.pth"


def load_gate(path, device):
    checkpoint = torch.load(path, map_location=device)
    labels = checkpoint["labels"]
    image_size = int(checkpoint.get("image_size", 128))
    model = GateCNN(len(labels)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, labels, image_size, bool(checkpoint.get("autocontrast", False))


def crop_boxes(width, height, crop_count):
    side = min(width, height)
    if crop_count <= 1:
        return [(0, 0, width, height)]
    small = max(int(side * 0.72), 1)
    positions = [
        ((width - side) // 2, (height - side) // 2, side),
        (0, 0, small),
        (width - small, 0, small),
        (0, height - small, small),
        (width - small, height - small, small),
        ((width - small) // 2, 0, small),
        ((width - small) // 2, height - small, small),
        (0, (height - small) // 2, small),
        (width - small, (height - small) // 2, small),
    ]
    boxes = []
    for x, y, size in positions[:crop_count]:
        x = max(0, min(x, width - size))
        y = max(0, min(y, height - size))
        boxes.append((x, y, x + size, y + size))
    return boxes


def load_images(path, image_size, autocontrast=False, crop_count=1):
    source = Image.open(path).convert("L")
    images = []
    for box in crop_boxes(source.width, source.height, crop_count):
        image = source.crop(box)
        image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BICUBIC)
        if autocontrast:
            image = ImageOps.autocontrast(image, cutoff=0)
        images.append(TF.to_tensor(image))
    return torch.stack(images, dim=0)


def load_image(path, image_size, autocontrast=False):
    image = Image.open(path).convert("L")
    image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BICUBIC)
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=0)
    return TF.to_tensor(image).unsqueeze(0)


def predict_gate(path, gate_path, device, crop_count=1):
    model, labels, image_size, autocontrast = load_gate(gate_path, device)
    image = load_images(path, image_size, autocontrast, crop_count).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(image), dim=1).mean(dim=0).cpu()
    order = torch.argsort(probs, descending=True)
    return [
        {"label": labels[int(index)], "probability": float(probs[int(index)])}
        for index in order
    ]


def average_predictions(paths, gate_path, device, crop_count):
    predictions = []
    for path in paths:
        predictions.append(predict_gate(path, gate_path, device, crop_count))
    labels = [item["label"] for item in predictions[0]]
    totals = {label: 0.0 for label in labels}
    for prediction in predictions:
        for item in prediction:
            totals[item["label"]] += item["probability"]
    averaged = [
        {"label": label, "probability": totals[label] / len(predictions)}
        for label in labels
    ]
    return sorted(averaged, key=lambda item: item["probability"], reverse=True)


def route_decision(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = args.inputs or [args.input]
    filename_labels = [label_from_name(path.split("/")[-1]) for path in inputs]
    if args.prefer_filename and filename_labels[0] and all(label == filename_labels[0] for label in filename_labels):
        filename_label = filename_labels[0]
        route_name = {
            "kurip": "kurip_clean540",
            "ako5": "ako5",
        }.get(filename_label, "shape1")
        return {
            "inputs": inputs,
            "dataset_source": filename_label,
            "confidence": 1.0,
            "route": route_name,
            "source": "filename_prefix",
        }

    family = average_predictions(inputs, args.family_gate, device, args.crop_count)
    result = {
        "inputs": inputs,
        "family": family,
        "dataset_source": "unknown",
        "confidence": 0.0,
        "route": "default",
    }

    family_label = family[0]["label"]
    family_confidence = family[0]["probability"]
    if family_label == "general_lineart" and family_confidence >= args.family_threshold:
        result.update(
            dataset_source="general_lineart",
            confidence=family_confidence,
            route="shape1",
        )
    elif family_label == "aligned_scan" and family_confidence >= args.family_threshold:
        source = average_predictions(inputs, args.kurip_ako5_gate, device, args.crop_count)
        result["aligned_scan_source"] = source
        source_label = source[0]["label"]
        source_confidence = source[0]["probability"]
        if source_confidence >= args.source_threshold:
            result.update(
                dataset_source=source_label,
                confidence=source_confidence,
                route={"kurip": "kurip_clean540", "ako5": "ako5"}.get(source_label, "shape1"),
            )
        else:
            result.update(
                dataset_source="aligned_scan_unknown",
                confidence=source_confidence,
                route="shape1",
            )

    return result


def route(args):
    print(json.dumps(route_decision(args), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--inputs", nargs="+", default=None)
    parser.add_argument("--family-gate", default=DEFAULT_FAMILY_GATE)
    parser.add_argument("--kurip-ako5-gate", default=DEFAULT_KURIP_AKO5_GATE)
    parser.add_argument("--family-threshold", type=float, default=0.70)
    parser.add_argument("--source-threshold", type=float, default=0.70)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--crop-count", type=int, default=9)
    parser.add_argument("--prefer-filename", action="store_true", default=True)
    parser.add_argument("--no-prefer-filename", dest="prefer_filename", action="store_false")
    args = parser.parse_args()
    if not args.input and not args.inputs:
        parser.error("one of --input or --inputs is required")
    route(args)


if __name__ == "__main__":
    main()

"""Evaluate dataset gate accuracy with multi-tile voting."""

import argparse
import json
import random
from pathlib import Path

import torch

from dataset_gate import DEFAULT_LABELS
from route_dataset_source import load_gate, load_images
from train_dataset_gate_cnn import labels_for_mode, target_label_from_name


DEFAULT_ROUGH_DIRS = ["dataset/pairs_480/train/rough", "dataset/pairs_480/test/rough"]


def collect_rows(rough_dirs, target_mode, labels, max_per_label):
    grouped = {label: [] for label in labels}
    seen = set()
    for rough_dir in rough_dirs:
        for path in sorted(Path(rough_dir).glob("*.jpg")):
            if path.name in seen:
                continue
            seen.add(path.name)
            label = target_label_from_name(path.name, target_mode)
            if label in grouped:
                grouped[label].append(str(path))
    rng = random.Random(2468)
    rows = []
    for label, paths in grouped.items():
        rng.shuffle(paths)
        if max_per_label:
            paths = paths[:max_per_label]
        rows.extend((label, path) for path in paths)
    rng.shuffle(rows)
    return rows


def predict_path(model, path, image_size, autocontrast, crop_count, device):
    image = load_images(path, image_size, autocontrast, crop_count).to(device)
    with torch.no_grad():
        return torch.softmax(model(image), dim=1).mean(dim=0).cpu()


def confusion_matrix(labels, truth, pred):
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, pred_label in zip(truth, pred):
        matrix[label_to_index[true_label]][label_to_index[pred_label]] += 1
    return matrix


def evaluate(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint_labels, image_size, autocontrast = load_gate(args.gate, device)
    labels = labels_for_mode(args.target_mode, args.labels)
    if labels != checkpoint_labels:
        raise ValueError(f"labels mismatch: args={labels} checkpoint={checkpoint_labels}")

    rows = collect_rows(args.rough_dirs, args.target_mode, labels, args.max_per_label)
    label_to_index = {label: index for index, label in enumerate(labels)}
    predictions = []
    for label, path in rows:
        probs = predict_path(model, path, image_size, autocontrast, args.crop_count, device)
        predictions.append((label, path, probs))

    single_truth = [label for label, _, _ in predictions]
    single_pred = [labels[int(probs.argmax())] for _, _, probs in predictions]
    single_acc = sum(t == p for t, p in zip(single_truth, single_pred)) / max(len(single_truth), 1)

    by_label = {label: [] for label in labels}
    for item in predictions:
        by_label[item[0]].append(item)

    vote_truth = []
    vote_pred = []
    for label in labels:
        items = by_label[label]
        for start in range(0, len(items) - args.group_size + 1, args.group_size):
            group = items[start:start + args.group_size]
            mean_probs = torch.stack([item[2] for item in group]).mean(dim=0)
            vote_truth.append(label)
            vote_pred.append(labels[int(mean_probs.argmax())])

    vote_acc = sum(t == p for t, p in zip(vote_truth, vote_pred)) / max(len(vote_truth), 1)
    report = {
        "gate": args.gate,
        "target_mode": args.target_mode,
        "labels": labels,
        "sample_count": len(single_truth),
        "single_tile_accuracy": single_acc,
        "single_tile_confusion": confusion_matrix(labels, single_truth, single_pred),
        "group_size": args.group_size,
        "group_count": len(vote_truth),
        "voting_accuracy": vote_acc,
        "voting_confusion": confusion_matrix(labels, vote_truth, vote_pred),
        "crop_count": args.crop_count,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True)
    parser.add_argument(
        "--target-mode",
        choices=["source", "family", "kurip_binary", "kurip_ako5_binary"],
        required=True,
    )
    parser.add_argument("--rough-dirs", nargs="+", default=DEFAULT_ROUGH_DIRS)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--max-per-label", type=int, default=1000)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--crop-count", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", default="results/dataset_gate_voting_report.json")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

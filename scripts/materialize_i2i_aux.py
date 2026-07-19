import argparse
from pathlib import Path

from inference_i2i import run_inference


def read_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--rough-dir", required=True)
    parser.add_argument("--aux-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--autocontrast", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in read_list(args.file_list):
        base = Path(name).stem
        run_inference(
            args.checkpoint,
            str(Path(args.rough_dir) / f"{base}.jpg"),
            str(output_dir / f"{base}_out.png"),
            autocontrast=args.autocontrast,
            aux_input=str(Path(args.aux_dir) / f"{base}_out.png") if args.aux_dir else None,
        )


if __name__ == "__main__":
    main()

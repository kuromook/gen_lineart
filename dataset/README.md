# Dataset Layout

Top-level dataset folders are grouped under this directory.

```text
dataset/
  raw/        # Original rough/line source images
  pairs/      # Original 256px paired dataset
  pairs_256/  # 256px generated paired dataset
  pairs_480/  # 480px paired dataset used by current training/evaluation
```

The dataset contents are ignored by git. Keep small, durable metadata or notes in
tracked files such as this README.

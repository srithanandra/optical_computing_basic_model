# Optical computing basic model

Minimal PyTorch image classification training script.

## Setup

```bash
python -m venv .venv
```

Activate the venv:

- Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m src.main
```

### Use your own dataset (no MNIST)

Put images into `data/` using one of these layouts:

**Option A (recommended): train/test folders**

```text
data/
  train/
    class_a/
      img1.png
    class_b/
      img2.jpg
  test/            # or `val/`
    class_a/
    class_b/
```

Run:

```bash
python -m src.main --data-dir data --epochs 3 --batch-size 64 --img-size 28
```

**Option B: single folder (auto 80/20 split)**

```text
data/
  class_a/
  class_b/
```

Run the same command as above.


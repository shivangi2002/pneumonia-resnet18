# Pneumonia Detection using ResNet18

---

## 🎯 Task Objective

Classifies chest x-ray as:
* Normal
* Pneumonia

Constraints:

* Use transfer learning
* Freeze pretrained layers
* Replace final layer for 2-class output
* Write full PyTorch training loop manually (no high-level wrappers)

---


## Project Structure

```text
project/
├── src/
│   ├── dataset.py        # XRay dataset class
│   ├── model.py          # ResNet18 with frozen pretrained layers
│   ├── train.py          # training loop with early stopping
│   ├── eval.py           # validation + metrics (accuracy, precision, recall)
│   ├── visualize.py      # loss curve plotting
│   └── logger.py         # CSV summary + JSON history saving
│
├── data/                 # NOT in git — download separately
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│
├── checkpoints/          # saved model weights (one per run)
├── plots/                # loss curves (one per run)
├── results/
│   ├── hyperparameters_tuning_results.csv  # comparison across runs
│   └── history/                            # full per-epoch metrics per run
│
├── Notebooks/
│   └── exploration.ipynb # experiments/debugging
│
├── docs/
│   └── learning.md       # learning notes
│
├── main.py               # entry point — run_model(lr, batch, epochs)
├── requirements.txt
└── README.md
```

---

## Setup
Install dependencies:
```
pip install -r requirements.txt
```
Place your dataset under `data/train/` and `data/test/` (folders `NORMAL/`
and `PNEUMONIA/` inside each).

## Usage

```python
from main import run_model
run_model(lr=0.001, images_per_batch=32, num_epochs=10)
```
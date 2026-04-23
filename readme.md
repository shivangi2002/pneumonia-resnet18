# Pneumonia Detection using ResNet18

---

## 🎯 Task Objective

Instead of training a model from scratch, use a pretrained **ResNet18** model and adapt it to classify chest X-ray images into:

* Normal
* Pneumonia

Constraints:

* Use transfer learning
* Freeze pretrained layers
* Replace final layer for 2-class output
* Write full PyTorch training loop manually (no high-level wrappers)

---

## 🧠 Approach

### 🔹 Model Setup

* Loaded pretrained ResNet18 (`torchvision.models`)
* Froze all pretrained layers
* Replaced final fully connected layer (`fc`)

  * 1000 → 2 output classes

---

### 🔹 Data Pipeline

* Custom Dataset implemented
* DataLoader used for batching
* Transformations:

  * Resize to 224×224
  * Convert to tensor
  * Normalize (ImageNet mean & std)

---


```text
project/
├── src/
│   ├── dataset.py        # dataset class
│   ├── model.py          # model (ResNet18 etc.)
│   ├── train.py          # training loop
│   └── eval.py           # validation + metrics (accuracy, precision, recall)
│
├── data/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/              
│   └── test/
│
├── notebooks/
│   └── exploration.ipynb # experiments/debugging
│
├── docs/
│   └── learning.md       # your notes
│
├── main.py               # entry point (pipeline)
├── README.md
```

---

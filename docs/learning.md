# 🧠 Learning Notes

---

## 🔹 Pixel Range (Before vs After `ToTensor()`)

* Before:

  * Pixel values are integers → **0 to 255**

* After `ToTensor()`:

  * Values are scaled → **0 to 1 (floats)**

* Why?

  * Neural networks train better with smaller, normalized values
  * Large values can make training unstable

---

## 🔹 Why Resize to 224×224?

* All images must have the **same shape** for batching
* `DataLoader` stacks images → requires identical dimensions
* ResNet18 is designed for **224×224 input**

👉 So resizing ensures:

* batching works
* model input is valid

---

## 🔹 Why Normalize?

* After `ToTensor()`, values are in **[0,1]**

* Normalization shifts them to:

  ```
  (value - mean) / std
  ```

* Why?

  * Matches distribution of ImageNet (pretrained model)
  * Helps model converge faster and more stably

---

## 🔹 Why Load Data in Batches?

* Training on entire dataset at once:

  * ❌ too slow
  * ❌ memory issues

* Batching:

  * ✔️ faster computation (GPU-friendly)
  * ✔️ stable gradient updates

---

## 🔹 Why `shuffle=True`?

* Prevents model from learning **data order patterns**
* Ensures:

  * better generalization
  * less overfitting

---

## 🔹 What is Shape?

Shape describes tensor dimensions.

Examples:

* Image:

  ```
  [3, 224, 224]
  ```

  → channels, height, width

* Batch:

  ```
  [32, 3, 224, 224]
  ```

  → batch size, channels, height, width

---

## 🔹 Why Not Train Whole Model?

* Pretrained layers already learned:

  * edges
  * textures
  * patterns

* If we train everything:

  * ❌ risk of overfitting (small dataset)
  * ❌ may destroy useful features

* So we:

  * freeze early layers
  * train only final layer

---

## 🔹 Why Not Use BCE Loss?

* BCE is used when:

  ```
  output → [batch, 1]
  ```

* Our model outputs:

  ```
  [batch, 2]
  ```

* So we use:

  👉 `CrossEntropyLoss`

* Key idea:

  ```
  Loss depends on output format, not just number of classes
  ```

---


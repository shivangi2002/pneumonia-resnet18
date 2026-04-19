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
  Type of Loss depends on output format, not just number of classes
  ```

---


## 🔹 Autograd (Automatic Gradient Computation)

- We do NOT manually compute gradients

- When we call:
  loss.backward()

- PyTorch automatically computes gradients using autograd  
- It applies the chain rule across all layers

What backward() does:
- Computes d(Loss) / d(weights)
- Stores gradients in param.grad
- Does NOT update weights

---

## 🔹 Training Loop Order Matters

Correct order:

optimizer.zero_grad() → forward → loss → backward → optimizer.step()

Why:
- zero_grad() → clears old gradients  
- forward → computes predictions  
- loss → measures error  
- backward → computes gradients  
- step() → updates weights  
---

## 🔹 Why loss.backward() AND optimizer.step()?

- loss.backward():
  - computes gradients  
  - tells how weights should change  

- optimizer.step():
  - updates weights using gradients  
  - actually changes model parameters  


---

## 🔹 Why zero_grad()?

- Gradients accumulate by default in PyTorch

Without it:
new_grad = old_grad + current_grad

- This mixes gradients from different batches → incorrect updates

So we:
- reset gradients every batch

---

## 🔹 Why We Calculate Average Loss?

- Loss is computed per batch

- Using last batch loss is misleading

Why average?
- gives overall performance across dataset
- reduces noise from individual batches

Formula:
average_loss = total_loss / number_of_batches

Insight:
Average loss reflects true model performance over the dataset

---

## 🔹 Optimizer Insight (Adam)

Adam combines:

1. Momentum (past gradients)
   - uses previous gradients
   - smooths updates (reduces zig-zag)

2. Adaptive Learning Rate
   - each parameter has its own step size
   - large gradients → smaller updates  
   - small gradients → larger updates  

3. Internal tracking:
   - m → average of past gradients  
   - v → average of squared gradients  

Update intuition:
weight = weight - lr × (m / sqrt(v))

Behavior:
- stable gradients → faster learning  
- unstable gradients → slower updates  

Insight:
Adam adjusts both direction and step size per parameter  

---

## 🔹 Loss vs Gradient (Core Concept)

- Loss:
  how wrong the model is

- Gradient:
  how to change weights to reduce loss


---

## 🔹 Final Big Picture

```
Input → Model → Output → Loss → Backward → Gradients → Optimizer → Update Weights
```

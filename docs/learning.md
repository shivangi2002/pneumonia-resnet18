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
---

## 🔹 Evaluation Metrics (Understanding Model Performance)

---

### 🔸 Accuracy (Overall Correctness)

**What it tells us:**

```
Out of all patients, how many did the model classify correctly
(both pneumonia and normal cases)?
```

**understanding:**

* Looks at all predictions
* Counts both correct pneumonia and correct normal cases
* Gives a general idea of how well the model is doing

---

### 🔸 Precision (How reliable are pneumonia predictions?)

**What it tells us:**

```
When the model says “pneumonia”,
how often is it actually correct?
```

**understanding:**

* Focuses only on pneumonia predictions
* Tells us if we can trust the model when it raises an alert
* Low precision → many false alarms

---

### 🔸 Recall (Did we miss any pneumonia cases?)

**What it tells us:**

```
Out of all real pneumonia cases,
how many did the model successfully detect?
```

**understanding:**

* Focuses only on actual pneumonia patients
* Tells us if the model is missing sick patients
* Low recall → dangerous (missed cases)

## 🔹 Train / Val / Test Splits

Data is split into three parts, each with a clear purpose:

* **Training set**
  * Used to train the model
  * Model adjusts its weights based on this data
  * The whole point is for the model to learn the right weights from here

* **Validation set**
  * Has two purposes:
    1. Check if the model is overfitting (compare with training loss curve)
    2. Help with hyperparameter tuning (compare different combos)
  * Model does NOT update weights from this set
  * BUT — when we try many combos and pick the one with the best val score,
    we end up picking the combo that got LUCKIEST on this specific val set.
    The val score of the winner therefore over-estimates its true performance
    on unseen data.

* **Test set**
  * Used to estimate real-world performance
  * Since we tuned on val, val scores already favor whichever combo got luckiest
    there. Test gives a separate number on data we never used to choose anything,
    so we can trust it as an honest estimate.
  * **Used only ONCE, at the very end** — if we peek at test, retune, peek again,
    we'd be choosing the combo that happened to fit our specific test set best.
    That makes the test number an over-estimate too — same problem as val.

## 🔹 Parameters vs Hyperparameters

* **Parameter** — what the model learns (weights, biases). Updated by
  gradient descent INSIDE the training loop, every batch, based on data.

* **Hyperparameter** — how the model learns (learning rate, batch size,
  num epochs). Set OUTSIDE the training loop, before training starts, or
  adjusted at a higher level (e.g., a learning rate scheduler changes it
  between epochs based on rules, not based on training data directly).
---

## 🔹 Hyperparameter Tuning

Since hyperparameters aren't learned by gradient descent, we have to find good values by trying different combinations.

---

### 🔸 The infinite search space

The space of possible values is huge:
* Learning rate is a real number — infinite values between 0 and 1 alone
* Batch size is an integer — could be anything from 1 to thousands
* Combining multiple hyperparameters multiplies the space further

Testing every value is impossible — and computing many combinations is
expensive (each one requires a full training run).

---

### 🔸 Approaches to navigate the space

* **Grid search** — try every combination from a fixed list
* **Random search** — randomly sample combinations
* **Bayesian optimization** — use past results to decide what to try next
  (tools like Optuna, Hyperopt, Ray Tune)
* **Logarithmic spacing** — for values where magnitude matters (e.g., 0.1, 0.01, 0.001)
* **Coarse-to-fine** — start wide, zoom in around the winner

---

### 🔸 Tension

More combos compared on val → winner's val score becomes more over-optimistic.

---

### 🔸 Fair comparison

For runs to be comparable, fix randomness so only the hyperparameter being tuned changes between runs.

## 🔹 Loss Curves

After every epoch, we record both training loss and validation loss, and
plot them on a graph (epoch on X-axis, loss on Y-axis). The purpose is to
check whether the model is underfitting, fitting well, or overfitting —
and to stop training if needed.

* **Underfitting** — both curves drop together (small gap between them)
  but never reach a low loss value. Val_loss doesn't rise — it just
  plateaus at a high value alongside train_loss. The model hasn't
  captured enough signal yet.

* **Good fit** — train_loss and val_loss drop together and stay close to
  each other (with normal noise/zigzag), reaching a low value. The model
  is learning patterns that generalize.

* **Overfitting** — train_loss keeps dropping to low values but val_loss
  goes UP or forms a V-shape (drops, then rises). Model is memorizing
  instead of generalizing. The GAP between the two curves is the
  overfitting signal.

* **Why we need both curves (not just val_loss)** — val_loss tells us how
  well the model generalizes, but it can't tell us WHY a model is bad.
  Train_loss is the diagnostic tool:
    * High train + high val → underfitting
    * Low train + low val → good fit
    * Low train + high/rising val → overfitting
  Without train_loss, underfitting and overfitting can both look like
  "val_loss isn't great" — but they need opposite fixes.

* **What we do with it** — the lowest point of val_loss tells us the best
  epoch. Beyond that, the model overfits. Early stopping uses this signal
  to stop training, and the model checkpoint we save is from the best
  val_loss point — not the last epoch.

## 🔹 Randomness 

Many parts of training rely on randomness — train/val split, batch
shuffling, weight initialization. For fair comparison across runs, this
randomness must be reproducible.

* **Pseudo-random** — computers can't be truly random. They use math
  formulas that produce sequences that look random but are fully
  deterministic.
* **Seed** — the starting number fed to the formula. Same seed → same
  sequence. PyTorch uses algorithms like Fisher-Yates internally for
  shuffling.
* **Global random state** — PyTorch keeps a shared "current state" of
  randomness. All random operations (`nn.Linear` init, `random_split`,
  `DataLoader` shuffle) read from it by default. So setting
  `torch.manual_seed(42)` ONCE affects everything that follows.
* **Why fix it for tuning** — without a fixed seed, different runs get
  different splits, shuffles, and inits. Differences in val_loss could
  come from randomness rather than the hyperparameter we're tuning.

## 🔹 Early Stopping

Stops training automatically when val_loss stops improving — instead of
guessing num_epochs in advance. Prevents overfitting and saves compute.

* **patience** — how many epochs without improvement we tolerate before
  stopping (typical: 5–10).
* **min_delta** — minimum improvement that counts as real; smaller drops
  are treated as noise.

## 🔹 Why Track Metrics & Models Per Run

When tuning hyperparameters, we run the same training many times with
different settings. To make that useful, we need a record of each run —
both what happened and what came out of it.

* **Why track metrics** — we need a way to COMPARE runs and pick the
  winner. Without recorded metrics (val_loss, accuracy, etc.) per
  hyperparameter combo, all runs blur together and we can't decide
  which set of hyperparameters worked best.
* **Why keep the trained model** — knowing the best hyperparameters
  isn't enough. To actually USE the model later (evaluate on test set,
  deploy, share), we need the actual trained weights. Otherwise we'd
  have to retrain from scratch every time.
* **Together they answer two different questions** — metrics tell us
  WHICH run was best; the saved model gives us THE model itself.

## 🔹 How Image Transforms Work

The `Resize → ToTensor → Normalize` pipeline is pure math, no AI.

* **Resize (bilinear interpolation)** — for each new pixel, find the
  fractional position in the original (`i × original/new`), then take a
  weighted average of nearby original pixels (closer pixels get bigger
  weights). Weights sum to 1.
* **ToTensor** — converts PIL → PyTorch tensor AND divides every pixel
  by 255, so values go from `[0, 255]` integers to `[0, 1]` floats.
  Smaller values keep training stable (avoids exploding activations and
  saturated gradients).
* **Normalize** — applies `(value - mean) / std` per channel using
  ImageNet statistics. Makes our images match the distribution ResNet18
  was pretrained on, so the pretrained weights work as intended.
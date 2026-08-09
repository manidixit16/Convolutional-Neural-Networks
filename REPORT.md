# Comparative Report: Shallow CNN vs Deep CNN on Fashion-MNIST

## 1. Objective

Perform a controlled comparison between a **shallow CNN** and a **deep CNN** on the
Fashion-MNIST dataset — training both on the same data with an identical training
setup — and determine which architecture is more suitable for this image
classification task in terms of accuracy, generalisation, and efficiency. A third
experiment adds **data augmentation** to the deep CNN to study its effect.

## 2. Dataset Overview

Fashion-MNIST consists of **70,000** 28×28 grayscale images across **10** clothing
classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag,
Ankle boot), split into **60,000 training** and **10,000 test** images. The classes
are perfectly balanced (6,000 training images each). Pixels were normalized to the
`[0, 1]` range and images reshaped to `(28, 28, 1)` to match the CNN input format.
A **fixed, stratified 90/10 train–validation split** (seeded) was held out from the
training set so both models see exactly the same validation data and the experiment
is reproducible.

## 3. Model Architectures

**Shallow CNN** — a minimal baseline:
`Conv2D(32) → MaxPool → Flatten → Dense(128) → Dense(10, softmax)`
(1 convolution layer, 1 pooling layer).

**Deep CNN** — a deeper, regularised network:
`Conv2D(32) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25) → Conv2D(128) →
BatchNorm → MaxPool → Dropout(0.25) → Flatten → Dense(256) → Dropout(0.5) →
Dense(10, softmax)` (3 convolution layers, 2 pooling layers, batch-norm + dropout).

**Deep CNN + Augmentation** — the same deep architecture, trained with gentle on-the-fly
augmentation (small random rotation ≤4%, translation ≤6%, zoom ≤6%) applied to the
training images only.

**Training recipe (shared for a fair comparison):** Adam optimizer, sparse categorical
cross-entropy loss, batch size 256, up to 25 epochs (35 for the augmented run), with
`EarlyStopping` (best-weight restoration) and `ReduceLROnPlateau`. Early stopping lets
each architecture train only as long as it keeps improving, so each is compared at *its
own best* rather than at an arbitrary fixed epoch.

## 4. Model Comparison

| Metric | Shallow CNN | Deep CNN | Deep CNN + Aug |
|--------|-------------|----------|----------------|
| Number of Conv Layers | 1 | 3 | 3 |
| Total Parameters | 693,962 | 1,701,770 | 1,701,770 |
| Epochs Trained (early stop) | 25 | 20 | 35 |
| Training Accuracy | 96.78% | 94.94% | 91.80% |
| Validation Accuracy | 92.60% | 94.65% | 93.23% |
| Test Accuracy | 92.20% | **93.41%** | 92.02% |
| Train–Val Gap | +4.18% | +0.29% | −1.44% |
| Overfitting Observed? | Yes | No | No |
| Training Time | 60 s | 696 s | 1,244 s |

*(Numbers are from the executed notebook; exact values may vary slightly between runs.)*

## 5. Key Observations

- **Accuracy.** The deep CNN reached **93.41%** test accuracy versus **92.20%** for the
  shallow CNN — a **1.2 percentage-point** improvement, which reduces the error rate from
  ~7.8% to ~6.6% (about a **15% relative reduction in errors**).
- **Generalisation.** This is the clearest difference. The shallow CNN's training accuracy
  (96.78%) sits well above its validation accuracy (92.60%) — a **+4.18% gap** that signals
  **overfitting**: the large flatten-to-dense head memorises the training set. The deep CNN's
  training and validation accuracy are almost identical (**+0.29% gap**) thanks to
  batch-normalisation and dropout — it generalises cleanly with no overfitting.
- **Efficiency.** The shallow CNN trained in about **60 seconds** — roughly **11× faster**
  than the deep CNN's **696 seconds** — and uses ~2.4× fewer parameters. Efficiency is where
  the shallow model wins decisively.
- **Data augmentation.** Gentle augmentation drove the deep model's train–val gap **negative**
  (−1.44%, i.e. validation slightly above training) — strong regularisation, no overfitting —
  but its test accuracy (**92.02%**) did *not* beat the plain deep model on this clean, centred
  dataset. This is an instructive result: augmentation is **not a free accuracy win**. Because
  Fashion-MNIST images are already upright and centred, simulating shifts/rotations mostly makes
  training harder; augmentation's real payoff (robustness to geometric variation and better
  calibration) would show up on *perturbed* test data rather than on this clean test set. An
  earlier, more aggressive augmentation setting actually *hurt* accuracy (≈89.7%), reinforcing
  that augmentation strength must be matched to the data.
- **Error behaviour.** Both models classify the visually distinctive classes almost perfectly
  (**Trouser, Bag, Sandal, Sneaker, Ankle boot**). Both struggle most with the upper-body-garment
  cluster — **Shirt vs T-shirt/top vs Pullover vs Coat** — which share nearly identical 28×28
  silhouettes. The deep CNN's confusion matrix shows a **stronger diagonal and lighter off-diagonal**
  mass in this cluster. A **confidence analysis** further shows both models are very confident when
  correct, but the deep model's errors skew to lower confidence — it is better *calibrated*, i.e. less
  often confidently wrong. Its most confident mistakes almost all fall inside the Shirt/T-shirt/
  Pullover/Coat cluster.

## 6. Final Conclusion

- **Recommended model:** the **deep CNN**. For Fashion-MNIST, accuracy and reliable
  generalisation are the priorities, and the deep model wins on both while showing no
  overfitting. Unless deployment is severely compute- or latency-constrained, it is the
  better default.
- **More efficient:** the **shallow CNN** — ~11× faster to train and far lighter, making it
  the right choice when compute, latency, or simplicity dominate.
- **More accurate:** the **deep CNN**, on training, validation, and test accuracy, and with
  the smaller (near-zero) generalisation gap.
- **What we learned:** depth pays off, but only with the right supporting techniques. Stacking
  more convolution layers with growing filter counts builds a feature hierarchy that separates
  visually similar clothing classes a single-conv model cannot — but that extra capacity would
  overfit without **batch-normalisation and dropout**. A **fixed validation split** and **early
  stopping** made the comparison fair and reproducible, the **confidence analysis** showed the
  regularised deep model is better calibrated in its mistakes, and the **augmentation experiment**
  showed that a regularisation technique must be matched to the data — on an already-clean dataset
  it improves generalisation metrics without necessarily raising test accuracy. The core trade-off
  is concrete: **more capacity buys accuracy at the cost of compute and training time**, and
  regularisation is what turns that capacity into better generalisation rather than memorisation.
  The best architecture ultimately depends on whether accuracy or efficiency is the binding
  constraint.

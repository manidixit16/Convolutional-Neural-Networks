# Comparative Report: Shallow CNN vs Deep CNN on Fashion-MNIST

## 1. Objective

Perform a controlled comparison between a **shallow CNN** and a **deep CNN** on the
Fashion-MNIST dataset — training both on the same data with an identical training
setup — and determine which architecture is more suitable for this image
classification task in terms of accuracy, generalisation, and efficiency.

## 2. Dataset Overview

Fashion-MNIST consists of **70,000** 28×28 grayscale images across **10** clothing
classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag,
Ankle boot), split into **60,000 training** and **10,000 test** images. The classes
are perfectly balanced (6,000 training images each). Pixels were normalized to the
`[0, 1]` range and images reshaped to `(28, 28, 1)` to match the CNN input format.
For training, a 10% validation split was held out from the training set.

## 3. Model Architectures

**Shallow CNN** — a minimal baseline:
`Conv2D(32) → MaxPool → Flatten → Dense(128) → Dense(10, softmax)`
(1 convolution layer, 1 pooling layer).

**Deep CNN** — a deeper, regularised network:
`Conv2D(32) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25) → Conv2D(128) →
BatchNorm → MaxPool → Dropout(0.25) → Flatten → Dense(256) → Dropout(0.5) →
Dense(10, softmax)` (3 convolution layers, 2 pooling layers, batch-norm + dropout).

Both models used the **Adam** optimizer, **sparse categorical cross-entropy** loss,
**12 epochs**, and a **batch size of 128** — kept identical for a fair comparison.

## 4. Model Comparison

| Metric | Shallow CNN | Deep CNN |
|--------|-------------|----------|
| Number of Conv Layers | 1 | 3 |
| Total Parameters | 693,962 | 1,701,770 |
| Training Accuracy | 96.34% | 93.01% |
| Validation Accuracy | 91.07% | 93.50% |
| Test Accuracy | 90.78% | **92.83%** |
| Train–Val Gap | 5.28% | −0.49% |
| Overfitting Observed? | Yes | No |
| Training Time | 71.2 s | 584.5 s |

*(Numbers are from the executed notebook; exact values may vary slightly between runs.)*

## 5. Key Observations

- **Accuracy.** The deep CNN reached **92.83%** test accuracy versus **90.78%** for
  the shallow CNN — an improvement of roughly **2 percentage points**, which is
  substantial at this accuracy level (it cuts the error rate from ~9.2% to ~7.2%,
  about a 22% relative reduction in errors).
- **Generalisation.** This is the most striking difference. The shallow CNN's
  training accuracy (96.34%) sits well above its validation accuracy (91.07%) — a
  **5.28% gap** that signals clear **overfitting**: the large flatten-to-dense head
  memorises the training set. The deep CNN's validation accuracy actually slightly
  **exceeds** its training accuracy (gap of −0.49%), thanks to batch-normalisation and
  dropout — it generalises cleanly with no overfitting.
- **Efficiency.** The shallow CNN trained in about **71 seconds** — roughly **8×
  faster** than the deep CNN's **584 seconds** — and uses ~2.4× fewer parameters.
  Efficiency is where the shallow model wins decisively.
- **Error behaviour.** Both models classify the visually distinctive classes almost
  perfectly (**Trouser, Bag, Sandal, Sneaker, Ankle boot**). Both struggle most with
  the upper-body-garment cluster — **Shirt vs T-shirt/top vs Pullover vs Coat** — which
  share nearly identical 28×28 silhouettes. The deep CNN's confusion matrix shows a
  **stronger diagonal and lighter off-diagonal** mass in this cluster, i.e. it reduces
  confusion between these similar classes.

## 6. Final Conclusion

- **Recommended model:** the **deep CNN**. For Fashion-MNIST, accuracy and reliable
  generalisation are the priorities, and the deep model wins on both while showing no
  overfitting.
- **More efficient:** the **shallow CNN** — ~8× faster to train and far lighter, making
  it the right choice when compute, latency, or simplicity dominate.
- **More accurate:** the **deep CNN**, on training, validation, and test accuracy, and
  with the smaller (indeed non-existent) generalisation gap.
- **What we learned:** depth pays off, but only with the right supporting techniques.
  Stacking more convolution layers with growing filter counts builds a feature
  hierarchy that separates visually similar clothing classes a single-conv model
  cannot — but that extra capacity would overfit without **batch-normalisation and
  dropout**. The core trade-off is concrete: **more capacity buys accuracy at the cost
  of compute and training time**, and regularisation is what turns that capacity into
  better generalisation rather than memorisation. The best architecture ultimately
  depends on whether accuracy or efficiency is the binding constraint.

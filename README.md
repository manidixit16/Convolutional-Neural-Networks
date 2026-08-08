# Comparative Study of Shallow CNN vs Deep CNN on Fashion-MNIST

A hands-on assignment comparing a **shallow CNN** and a **deep CNN** on the
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset
(28×28 grayscale images, 10 clothing classes). Both models are trained on the
same data with the same training setup, then compared on accuracy,
generalisation, efficiency, and error behaviour.

## Repository contents

| File | Description |
|------|-------------|
| `Fashion_MNIST_Shallow_vs_Deep_CNN.ipynb` | Main deliverable — a fully executed notebook covering all six parts of the assignment (dataset loading & exploration, shallow CNN, deep CNN, comparison table, prediction/error analysis with confusion matrices, and the final conclusion). Outputs are embedded. |
| `REPORT.md` | Short comparative report summarising the objective, dataset, model comparison, key observations, and conclusion. |
| `build_notebook.py` | Script that regenerates the notebook from source (kept for reproducibility). |
| `results.json` | Headline metrics produced by the executed notebook. |
| `requirements.txt` | Python dependencies. |

## What the notebook does

- **Part 1 – Load & Explore (15):** loads Fashion-MNIST, prints data shapes and
  the number of classes, displays one sample image per class, checks the class
  balance, then normalizes pixels to `[0, 1]` and reshapes images to `(28, 28, 1)`
  for CNN input. Includes write-ups on *why normalization* and *why reshaping* are needed.
- **Part 2 – Shallow CNN (25):** 1 conv layer + 1 pooling layer + flatten + 1 dense
  hidden layer + output. Trained, evaluated on the test set, with training/validation
  accuracy and loss curves plotted.
- **Part 3 – Deep CNN (25):** 3 conv layers (32→64→128 filters) with batch-norm,
  2 pooling layers, dropout, a dense head, and output. Same training setup as the
  shallow model for a fair comparison.
- **Part 4 – Comparative Study (20):** a comparison table (conv layers, parameters,
  train/val/test accuracy, overfitting flag, training time) plus a written analysis.
- **Part 5 – Prediction & Error Analysis (15):** predictions from both models, 5
  correct + 5 incorrect samples per model with actual/predicted labels, and a
  confusion matrix for each model.
- **Part 6 – Conclusion (10):** which model to recommend, which was more efficient,
  which was more accurate, and lessons learned.

## How to run

```bash
pip install -r requirements.txt

# Option A: open and run the notebook interactively
jupyter notebook Fashion_MNIST_Shallow_vs_Deep_CNN.ipynb

# Option B: regenerate and execute from the command line
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace Fashion_MNIST_Shallow_vs_Deep_CNN.ipynb
```

The Fashion-MNIST dataset is downloaded automatically by
`tensorflow.keras.datasets.fashion_mnist` on first run.

## Environment

Trained on CPU with TensorFlow. See `requirements.txt` for exact packages.

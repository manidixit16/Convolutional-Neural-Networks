# Comparative Study of Shallow CNN vs Deep CNN on Fashion-MNIST

A hands-on assignment comparing a **shallow CNN** and a **deep CNN** on the
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset
(28×28 grayscale images, 10 clothing classes). Both models are trained on the
same data with the same training setup, then compared on accuracy,
generalisation, efficiency, and error behaviour. A third experiment adds data
augmentation to the deep CNN.

## Repository contents

| File | Description |
|------|-------------|
| `Fashion_MNIST_Shallow_vs_Deep_CNN.ipynb` | Main deliverable — a fully executed notebook covering all six parts of the assignment. Outputs (plots, tables, confusion matrices) are embedded. |
| `REPORT.md` | Short comparative report (objective, dataset, model comparison, key observations, conclusion). |
| `REPORT.pdf` | PDF version of the report, ready to attach to a submission. |
| `build_notebook.py` | Script that regenerates the notebook from source (reproducibility). |
| `make_report_pdf.py` | Renders `REPORT.md` to `REPORT.pdf`. |
| `results.json` | Headline metrics produced by the executed notebook. |
| `requirements.txt` | Python dependencies. |

## What the notebook does

- **Part 1 – Load & Explore (15):** loads Fashion-MNIST, prints data shapes and the
  number of classes, displays one sample image per class, checks the class balance,
  normalizes pixels to `[0, 1]`, reshapes images to `(28, 28, 1)`, and creates a
  **fixed, stratified train/validation split** for reproducibility. Includes write-ups
  on *why normalization* and *why reshaping* are needed.
- **Part 2 – Shallow CNN (25):** 1 conv layer + 1 pooling layer + flatten + 1 dense
  hidden layer + output. Trained with early stopping, evaluated on the test set, with
  training/validation accuracy and loss curves plotted.
- **Part 3 – Deep CNN (25):** 3 conv layers (32→64→128 filters) with batch-norm,
  2 pooling layers, dropout, a dense head, and output — same training setup as the
  shallow model. Includes a **data-augmentation enhancement** (a fourth model:
  the deep CNN retrained with gentle random rotation/translation/zoom).
- **Part 4 – Comparative Study (20):** a comparison table (conv layers, parameters,
  epochs, train/val/test accuracy, overfitting flag, training time) across the shallow,
  deep, and augmented-deep models, plus a written analysis.
- **Part 5 – Prediction & Error Analysis (15):** predictions from both models, 5
  correct + 5 incorrect samples per model with actual/predicted labels, a confusion
  matrix for each model, a **confidence analysis** (how sure each model is when right
  vs wrong), and the deep model's most-confident mistakes.
- **Part 6 – Conclusion (10):** which model to recommend, which was more efficient,
  which was more accurate, and lessons learned.

## Headline results

| Metric | Shallow CNN | Deep CNN | Deep CNN + Aug |
|--------|-------------|----------|----------------|
| Test accuracy | 92.20% | **93.41%** | 92.02% |
| Train–val gap | +4.18% (overfits) | +0.29% (no overfit) | −1.44% (regularised) |
| Training time | ~60 s | ~696 s | ~1,244 s |

The deep CNN is the most accurate and best-generalising model; the shallow CNN is far
more efficient. Data augmentation strongly regularises the deep model but does not raise
test accuracy on this already-clean dataset. *(Exact numbers may vary slightly per run.)*

## How to run

```bash
pip install -r requirements.txt

# Option A: open and run the notebook interactively
jupyter notebook Fashion_MNIST_Shallow_vs_Deep_CNN.ipynb

# Option B: regenerate and execute from the command line
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace Fashion_MNIST_Shallow_vs_Deep_CNN.ipynb

# Regenerate the PDF report
python make_report_pdf.py
```

The Fashion-MNIST dataset is downloaded automatically by
`tensorflow.keras.datasets.fashion_mnist` on first run.

## Environment

Trained on CPU with TensorFlow. See `requirements.txt` for exact packages.

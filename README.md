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
| `Fashion_MNIST_Shallow_vs_Deep_CNN.ipynb` | Main deliverable — a fully executed notebook covering all six parts, with a *"what this graph shows"* explanation after every figure. Outputs embedded. |
| `REPORT.md` | Comparative report **with the graphs embedded** and expanded explanations. |
| `REPORT.pdf` | PDF version of the report (graphs included), ready to attach to a submission. |
| `Assignment_Report.docx` | Comprehensive **Word report** — full write-up with graphs embedded, comparison table, and repository links. |
| `figures/` | All plots saved as PNGs (used by the report and Word doc). |
| `build_notebook.py` | Regenerates the notebook from source. |
| `make_report_pdf.py` | Renders `REPORT.md` → `REPORT.pdf` (embeds the figures). |
| `make_report_docx.js` | Builds `Assignment_Report.docx` from `figures/` + `results.json`. |
| `results.json` | Headline metrics produced by the executed notebook. |
| `requirements.txt` | Python dependencies. |

## What the notebook does

- **Part 1 – Load & Explore (15):** shapes, one sample image per class (with an
  explanation of which classes will be easy/hard), class-balance check, normalization,
  reshaping, and a **fixed, stratified train/validation split** for reproducibility.
- **Part 2 – Shallow CNN (25):** 1 conv + 1 pool + flatten + dense + output; trained
  with early stopping; accuracy/loss curves **with an interpretation of the overfitting**.
- **Part 3 – Deep CNN (25):** 3 conv (32→64→128) with batch-norm, 2 pool, dropout; same
  training setup; curves interpreted; plus a **data-augmentation enhancement** (a fourth model).
- **Part 4 – Comparative Study (20):** comparison table + a grouped accuracy bar chart, each
  with written analysis.
- **Part 5 – Prediction & Error Analysis (15):** correct/incorrect samples, **confusion
  matrices**, a **confidence analysis** (right vs wrong), and the deep model's most-confident
  mistakes — each explained.
- **Part 6 – Conclusion (10):** recommendation, efficiency, accuracy, and lessons learned.

## Headline results

| Metric | Shallow CNN | Deep CNN | Deep CNN + Aug |
|--------|-------------|----------|----------------|
| Test accuracy | 92.20% | **93.41%** | 91.64% |
| Train–val gap | +4.18% (overfits) | +0.29% (no overfit) | −2.53% (regularised) |
| Training time | ~127 s | ~1,132 s | ~1,527 s |

The deep CNN is the most accurate and best-generalising model; the shallow CNN is ~9× faster
to train. Data augmentation strongly regularises the deep model but does not raise test accuracy
on this already-clean dataset. *(Exact numbers may vary slightly per run.)*

## How to run

```bash
pip install -r requirements.txt

# Run the notebook (also writes figures/ and results.json)
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace Fashion_MNIST_Shallow_vs_Deep_CNN.ipynb

# Rebuild the reports
python make_report_pdf.py            # REPORT.md -> REPORT.pdf (needs markdown, playwright)
npm install docx && node make_report_docx.js   # -> Assignment_Report.docx
```

The Fashion-MNIST dataset is downloaded automatically by
`tensorflow.keras.datasets.fashion_mnist` on first run.

## Environment

Trained on CPU with TensorFlow. See `requirements.txt` for Python packages; the Word report
generator additionally needs the `docx` npm package (`npm install docx`).

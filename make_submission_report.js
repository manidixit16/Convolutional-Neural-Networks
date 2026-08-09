/*
 * Build the graded-assignment Word report in the house style used for earlier
 * submissions (kicker -> title -> subtitle -> metadata -> overview ->
 * results-at-a-glance -> per-part analysis with figures -> final conclusion).
 *
 * Output: Mani_Dixit_PGDSAI3_Fashion_MNIST_Report.docx
 * Run:  NODE_PATH=<docx node_modules> node make_submission_report.js
 * Needs figures/ and results.json produced by executing the notebook.
 */
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  ExternalHyperlink, BorderStyle, Table, TableRow, TableCell, WidthType,
  ShadingType, ImageRun,
} = require("docx");
const fs = require("fs");
const path = require("path");

const OUT = process.argv[2] || "Mani_Dixit_PGDSAI3_Fashion_MNIST_Report.docx";
const FIGDIR = "figures";
const R = JSON.parse(fs.readFileSync("results.json", "utf8"));

const NOTEBOOK_NAME = "Fashion_MNIST_Comparative_Study_Mani_PGDSAI3.ipynb";
const REPO = "https://github.com/manidixit16/Convolutional-Neural-Networks";
const NBVIEWER = "https://nbviewer.org/github/manidixit16/Convolutional-Neural-Networks/blob/main/" + NOTEBOOK_NAME;

const BLUE = "2B6CB0", DARK = "1A365D", GREY = "5A6472", INK = "1A202C";
const PAGE_W = 6.5 * 96;

const pct = (x) => (x * 100).toFixed(2) + "%";
const sgn = (x) => (x >= 0 ? "+" : "") + (x * 100).toFixed(2) + "%";
const pngSize = (b) => ({ w: b.readUInt32BE(16), h: b.readUInt32BE(20) });

function fig(file, caption, maxW = PAGE_W) {
  const buf = fs.readFileSync(path.join(FIGDIR, file));
  const { w, h } = pngSize(buf);
  const width = Math.min(maxW, w), height = Math.round(Math.min(maxW, w) * (h / w));
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width, height } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 170 },
      children: [new TextRun({ text: caption, italics: true, color: GREY, size: 17 })] }),
  ];
}
const H = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 90 },
  children: [new TextRun({ text, bold: true, color: BLUE, size: 25 })] });
const P = (runs, opts = {}) => new Paragraph({ spacing: { after: 140, ...(opts.spacing || {}) },
  children: (typeof runs === "string") ? [new TextRun(runs)] : runs, ...opts });
const B = (runs) => new Paragraph({ bullet: { level: 0 }, spacing: { after: 60 },
  children: (typeof runs === "string") ? [new TextRun(runs)] : runs });
const link = (url, text) => new ExternalHyperlink({ link: url,
  children: [new TextRun({ text: text || url, style: "Hyperlink", color: BLUE, underline: {} })] });

// ---- metadata + generic tables ----
function metaRow(label, valueChildren) {
  return new TableRow({ children: [
    new TableCell({ width: { size: 2600, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: "EBF4FF" },
      margins: { top: 60, bottom: 60, left: 120, right: 120 },
      children: [new Paragraph({ children: [new TextRun({ text: label, bold: true, color: DARK, size: 19 })] })] }),
    new TableCell({ width: { size: 6900, type: WidthType.DXA }, margins: { top: 60, bottom: 60, left: 120, right: 120 },
      children: [new Paragraph({ children: valueChildren })] }),
  ] });
}
function metaTable(rows) {
  return new Table({ columnWidths: [2600, 6900], width: { size: 9500, type: WidthType.DXA },
    borders: borderSet(), rows });
}
function borderSet() {
  const b = { style: BorderStyle.SINGLE, size: 4, color: "CBD5E0" };
  return { top: b, bottom: b, left: b, right: b, insideHorizontal: b, insideVertical: b };
}
function cell(t, { w = 2300, fill, bold, color, align } = {}) {
  return new TableCell({ width: { size: w, type: WidthType.DXA },
    shading: fill ? { type: ShadingType.CLEAR, fill } : undefined,
    margins: { top: 55, bottom: 55, left: 100, right: 100 },
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text: String(t), bold, color })] })] });
}
function dataTable(widths, header, rows) {
  const head = new TableRow({ children: header.map((t, i) =>
    cell(t, { w: widths[i], fill: "EBF4FF", bold: true, color: DARK })) });
  const body = rows.map((r, ri) => new TableRow({ children: r.map((t, i) =>
    cell(t, { w: widths[i], fill: ri % 2 ? "F7FAFC" : undefined, bold: i === 0 })) }));
  return new Table({ columnWidths: widths, width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    borders: borderSet(), rows: [head, ...body] });
}

const dTest = pct(R.deep.test_acc), sTest = pct(R.shallow.test_acc), aTest = pct(R.deep_aug.test_acc);
const speedup = Math.round(R.deep.train_time_s / R.shallow.train_time_s);

const children = [
  // ---- Title block ----
  new Paragraph({ spacing: { after: 30 },
    children: [new TextRun({ text: "DEEP LEARNING  —  GRADED ASSIGNMENT", bold: true, color: GREY, size: 17 })] }),
  new Paragraph({ spacing: { after: 20 },
    children: [new TextRun({ text: "Fashion-MNIST: Shallow CNN vs Deep CNN Comparative Study", bold: true, color: DARK, size: 34 })] }),
  new Paragraph({ border: { bottom: { color: BLUE, space: 4, style: BorderStyle.SINGLE, size: 18 } }, spacing: { after: 200 },
    children: [new TextRun({ text: "Image Classification  |  10 Clothing Classes  |  Shallow CNN vs Deep CNN", italics: true, color: BLUE, size: 20 })] }),

  // ---- Metadata ----
  metaTable([
    metaRow("Author", [new TextRun({ text: "Mani Dixit", size: 19 })]),
    metaRow("Batch", [new TextRun({ text: "PGDSAI3", size: 19 })]),
    metaRow("Notebook", [new TextRun({ text: NOTEBOOK_NAME, size: 19 })]),
    metaRow("Dataset", [new TextRun({ text: "Fashion-MNIST — 60,000 train + 10,000 test, 28×28 grayscale, 10 classes", size: 19 })]),
    metaRow("GitHub Repo", [link(REPO)]),
    metaRow("View Notebook Online", [link(NBVIEWER)]),
  ]),

  // ---- Overview ----
  H("Assignment Overview"),
  P("This report presents the complete solution to the assignment (Parts 1–6): a comparative deep-learning study classifying Fashion-MNIST clothing images with two convolutional neural networks. A shallow CNN and a deep CNN are trained on the same data with an identical, fair training setup — a fixed stratified 90/10 train/validation split, the Adam optimizer, early stopping with best-weight restoration, and a learning-rate scheduler — and compared on accuracy, generalisation, efficiency, and error behaviour. A third experiment adds gentle data augmentation to the deep CNN. The full executed notebook, all figures, and run instructions are in the GitHub repository linked above."),

  // ---- Results at a glance ----
  H("Results at a Glance"),
  dataTable([2600, 1900, 2200, 2800],
    ["Model", "Test Accuracy", "Train–Val Gap", "Overfitting?"],
    [
      ["Shallow CNN", sTest, sgn(R.shallow.gap), R.shallow.overfitting],
      ["Deep CNN", dTest, sgn(R.deep.gap), R.deep.overfitting],
      ["Deep CNN + Augmentation", aTest, sgn(R.deep_aug.gap), R.deep_aug.overfitting],
    ]),
  P([new TextRun({ text: "Best overall: ", bold: true }),
     new TextRun(`the Deep CNN (test accuracy ${dTest}) — the most accurate and the only model with essentially no overfitting.`)]),

  // ---- Part 1 ----
  H("Part 1 — Load & Explore the Dataset"),
  P("Fashion-MNIST provides 70,000 grayscale 28×28 images across 10 balanced clothing classes (6,000 training images each). Pixels are normalized to [0, 1] and images reshaped to (28, 28, 1) for CNN input; a seeded, stratified 90/10 split holds out a validation set shared by every model."),
  ...fig("01_samples_per_class.png", "Figure 1 — One sample image per class. Distinctive items (Trouser, Bag, Sandal, Sneaker, Ankle boot) are easy; the upper-body garments (T-shirt, Pullover, Coat, Shirt) share almost identical silhouettes and drive nearly all the errors."),
  P([new TextRun({ text: "Why normalize? ", bold: true }), new TextRun("raw 0–255 pixels give large, unevenly-scaled activations/gradients that make training slow and unstable; rescaling to [0, 1] lets the optimizer converge faster and more reliably. "),
     new TextRun({ text: "Why reshape? ", bold: true }), new TextRun("a Conv2D layer convolves over a (height, width, channels) tensor, so the (28, 28) images need an explicit channel axis → (28, 28, 1).")]),

  // ---- Part 2 ----
  H("Part 2 — Shallow CNN"),
  P("Architecture: Conv2D(32) → MaxPool → Flatten → Dense(128) → Dense(10, softmax) — one convolution and one pooling layer. Almost all parameters sit in the Flatten → Dense connection, which can memorise training-set detail."),
  ...fig("02_shallow_curves.png", "Figure 2 — Shallow CNN learning curves. Training accuracy climbs toward 0.97 while validation flattens near 0.92, and validation loss turns upward — textbook overfitting. Early stopping restores the best-validation weights (dashed line)."),
  B([new TextRun({ text: "Patterns learned: ", bold: true }), new TextRun("only low-level, local features (edges, strokes) — enough for the distinctive shapes, not the subtle differences between similar tops.")]),
  B([new TextRun({ text: "Fit: ", bold: true }), new TextRun(`clear overfitting — training accuracy ${pct(R.shallow.train_acc)} vs validation ${pct(R.shallow.val_acc)} (gap ${sgn(R.shallow.gap)}). Final test accuracy ${sTest}.`)]),

  // ---- Part 3 ----
  H("Part 3 — Deep CNN"),
  P("Architecture: Conv(32) → BatchNorm → Conv(64) → MaxPool → Dropout → Conv(128) → BatchNorm → MaxPool → Dropout → Flatten → Dense(256) → Dropout → Dense(10, softmax) — three convolution layers with growing filters, two pooling layers, batch-norm and dropout. Same optimizer, callbacks, batch size and validation set as the shallow model."),
  ...fig("03_deep_curves.png", "Figure 3 — Deep CNN learning curves. Training and validation rise together and stay close, and validation loss tracks training loss instead of rising — the signature of a model that generalises rather than memorises."),
  B([new TextRun({ text: "Extra representations: ", bold: true }), new TextRun("a feature hierarchy — edges → mid-level parts (sleeves, soles, collars) → class-discriminative shapes — that separates the look-alike upper-body garments.")]),
  B([new TextRun({ text: "Improvement: ", bold: true }), new TextRun(`meaningful — test accuracy ${dTest} (up from ${sTest}) with a near-zero gap (${sgn(R.deep.gap)}), i.e. genuine generalisation, not just more training-set fit.`)]),

  // ---- Enhancement ----
  H("Enhancement — Deep CNN with Data Augmentation"),
  P("The same deep architecture retrained with gentle on-the-fly augmentation (random rotation ≤4%, translation ≤6%, zoom ≤6%) on the training images only — kept small because the garments are already upright and centred."),
  ...fig("05_deep_aug_curves.png", "Figure 4 — Augmented deep CNN curves. Validation accuracy sits slightly above training accuracy (trained on harder augmented images, validated on clean ones) — heavy regularisation, mild underfitting, no overfitting."),
  B([new TextRun({ text: "Effect: ", bold: true }), new TextRun(`the train–val gap goes negative (${sgn(R.deep_aug.gap)}) but test accuracy (${aTest}) does not beat the plain deep model on this clean dataset — augmentation is not a free accuracy win; its payoff is robustness/calibration, and its strength must match the data.`)]),

  // ---- Part 4 ----
  H("Part 4 — Comparative Study"),
  dataTable([2900, 2100, 2100, 2100],
    ["Metric", "Shallow CNN", "Deep CNN", "Deep CNN + Aug"],
    [
      ["Conv layers", R.shallow.conv_layers, R.deep.conv_layers, R.deep_aug.conv_layers],
      ["Total parameters", R.shallow.params.toLocaleString(), R.deep.params.toLocaleString(), R.deep_aug.params.toLocaleString()],
      ["Training accuracy", pct(R.shallow.train_acc), pct(R.deep.train_acc), pct(R.deep_aug.train_acc)],
      ["Validation accuracy", pct(R.shallow.val_acc), pct(R.deep.val_acc), pct(R.deep_aug.val_acc)],
      ["Test accuracy", sTest, dTest, aTest],
      ["Train–Val gap", sgn(R.shallow.gap), sgn(R.deep.gap), sgn(R.deep_aug.gap)],
      ["Overfitting?", R.shallow.overfitting, R.deep.overfitting, R.deep_aug.overfitting],
      ["Training time (s)", R.shallow.train_time_s.toFixed(0), R.deep.train_time_s.toFixed(0), R.deep_aug.train_time_s.toFixed(0)],
    ]),
  ...fig("06_accuracy_comparison.png", "Figure 5 — Accuracy comparison. The shallow model's Training bar towers over its Validation/Test bars (overfitting); the deep model's bars are level with the tallest Test bar; the augmented model's Training bar is the shortest (mild underfitting)."),
  B([new TextRun({ text: "Better overall: ", bold: true }), new TextRun(`Deep CNN — highest accuracy and smallest gap.`)]),
  B([new TextRun({ text: "Efficiency: ", bold: true }), new TextRun(`Shallow CNN — ~${speedup}× faster to train and ~2.4× fewer parameters.`)]),
  B([new TextRun({ text: "Trade-off: ", bold: true }), new TextRun("compute/latency vs accuracy — shallow for speed and simplicity, deep for accuracy.")]),

  // ---- Part 5 ----
  H("Part 5 — Prediction & Error Analysis"),
  ...fig("09_confusion_matrices.png", "Figure 6 — Confusion matrices (rows = actual, columns = predicted). Both models are near-perfect on distinctive classes; errors concentrate in the Shirt / T-shirt / Pullover / Coat block. The deep CNN has a stronger diagonal and lighter off-diagonal cells there."),
  ...fig("10_confidence_hist.png", "Figure 7 — Prediction confidence for correct (green) vs incorrect (red) predictions. Both are confident when correct, but the deep model's errors lean to lower confidence — it is better calibrated (less often confidently wrong)."),
  B([new TextRun({ text: "Easiest classes: ", bold: true }), new TextRun("Trouser, Bag, Sandal, Sneaker, Ankle boot — distinctive silhouettes, near-perfect recall.")]),
  B([new TextRun({ text: "Most confused: ", bold: true }), new TextRun("Shirt with T-shirt/top, Pullover and Coat — nearly identical outlines at 28×28; Shirt is the hardest class for both models.")]),
  B([new TextRun({ text: "Deep CNN vs Shallow: ", bold: true }), new TextRun("the deep model reduces confusion within that look-alike cluster and is better calibrated in its mistakes.")]),

  // ---- Part 6 ----
  H("Part 6 — Final Conclusion"),
  P([new TextRun({ text: "1. Which model would you recommend? ", bold: true, color: INK }),
     new TextRun(`The Deep CNN — highest, most dependable accuracy (${dTest}) with no overfitting. Unless compute/latency is severely constrained, it is the better default.`)]),
  P([new TextRun({ text: "2. Which model was more efficient? ", bold: true, color: INK }),
     new TextRun(`The Shallow CNN — ~${speedup}× faster to train, ~2.4× fewer parameters, cheaper at inference.`)]),
  P([new TextRun({ text: "3. Which model was more accurate? ", bold: true, color: INK }),
     new TextRun(`The Deep CNN, on training, validation and test accuracy alike, and with the smaller generalisation gap.`)]),
  P([new TextRun({ text: "4. What did you learn from this comparative study? ", bold: true, color: INK }),
     new TextRun("Depth pays off, but only with the right supporting techniques. Extra convolution layers build a feature hierarchy that separates visually similar classes a single-conv model cannot — but that capacity would overfit without batch-norm and dropout. A fixed validation split and early stopping made the comparison fair and reproducible, the confidence analysis showed the regularised deep model is better calibrated, and the augmentation experiment showed a technique must be matched to the data. The core trade-off: more capacity buys accuracy at the cost of compute and training time, and regularisation turns that capacity into generalisation rather than memorisation.")]),
];

const doc = new Document({
  creator: "Mani Dixit",
  title: "Fashion-MNIST: Shallow CNN vs Deep CNN Comparative Study",
  styles: { default: { document: { run: { font: "Calibri", size: 21 } } } },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 } } }, children }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("Wrote", OUT, buf.length, "bytes"); });

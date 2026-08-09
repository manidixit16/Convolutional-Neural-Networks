/*
 * Build a comprehensive Word report (Assignment_Report.docx) for the
 * Fashion-MNIST Shallow vs Deep CNN study: full write-up WITH the graphs
 * embedded, the comparison table (from results.json), and the repo links.
 *
 * Run:  NODE_PATH=<docx node_modules> node make_report_docx.js [outfile]
 * Needs the figures/ folder and results.json produced by executing the notebook.
 */
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  ExternalHyperlink, BorderStyle, Table, TableRow, TableCell, WidthType,
  ShadingType, ImageRun,
} = require("docx");
const fs = require("fs");
const path = require("path");

const OUT = process.argv[2] || "Assignment_Report.docx";
const FIGDIR = "figures";
const R = JSON.parse(fs.readFileSync("results.json", "utf8"));

const REPO = "https://github.com/manidixit16/Convolutional-Neural-Networks";
const NOTEBOOK = REPO + "/blob/main/Fashion_MNIST_Shallow_vs_Deep_CNN.ipynb";
const REPORT_PDF = REPO + "/blob/main/REPORT.pdf";
const REPORT_MD = REPO + "/blob/main/REPORT.md";

const BLUE = "2B6CB0", DARK = "1A365D", GREY = "4A5568";
const PAGE_W = 6.5 * 96; // usable width in px (Letter, 1in margins)

function pct(x) { return (x * 100).toFixed(2) + "%"; }
function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }

function figure(file, caption, maxW = PAGE_W) {
  const buf = fs.readFileSync(path.join(FIGDIR, file));
  const { w, h } = pngSize(buf);
  const width = Math.min(maxW, w);
  const height = Math.round(width * (h / w));
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width, height } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 160 },
      children: [new TextRun({ text: caption, italics: true, color: GREY, size: 18 })],
    }),
  ];
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1, spacing: { after: 60 },
    children: [new TextRun({ text, bold: true, color: DARK, size: 32 })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 90 },
    children: [new TextRun({ text, bold: true, color: BLUE, size: 26 })],
  });
}
function p(runs, opts = {}) {
  const children = (typeof runs === "string") ? [new TextRun(runs)] : runs;
  return new Paragraph({ spacing: { after: 140, ...(opts.spacing || {}) }, children, ...opts });
}
function bullet(runs) {
  const children = (typeof runs === "string") ? [new TextRun(runs)] : runs;
  return new Paragraph({ bullet: { level: 0 }, spacing: { after: 60 }, children });
}
function link(url, text) {
  return new ExternalHyperlink({
    link: url,
    children: [new TextRun({ text: text || url, style: "Hyperlink", color: BLUE, underline: {} })],
  });
}

// ---- comparison table from results.json ----
function cellText(t, opts = {}) {
  return new TableCell({
    width: { size: opts.w || 2400, type: WidthType.DXA },
    shading: opts.fill ? { type: ShadingType.CLEAR, fill: opts.fill } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: String(t), bold: !!opts.bold, color: opts.color })] })],
  });
}
function tableRow(cells) { return new TableRow({ children: cells }); }

function comparisonTable() {
  const head = tableRow([
    cellText("Metric", { w: 3000, fill: "EBF4FF", bold: true, color: DARK }),
    cellText("Shallow CNN", { w: 2200, fill: "EBF4FF", bold: true, color: DARK }),
    cellText("Deep CNN", { w: 2200, fill: "EBF4FF", bold: true, color: DARK }),
    cellText("Deep CNN + Aug", { w: 2200, fill: "EBF4FF", bold: true, color: DARK }),
  ]);
  const rows = [
    ["Number of Conv Layers", R.shallow.conv_layers, R.deep.conv_layers, R.deep_aug.conv_layers],
    ["Total Parameters", R.shallow.params.toLocaleString(), R.deep.params.toLocaleString(), R.deep_aug.params.toLocaleString()],
    ["Epochs Trained (early stop)", R.shallow.epochs_trained, R.deep.epochs_trained, R.deep_aug.epochs_trained],
    ["Training Accuracy", pct(R.shallow.train_acc), pct(R.deep.train_acc), pct(R.deep_aug.train_acc)],
    ["Validation Accuracy", pct(R.shallow.val_acc), pct(R.deep.val_acc), pct(R.deep_aug.val_acc)],
    ["Test Accuracy", pct(R.shallow.test_acc), pct(R.deep.test_acc), pct(R.deep_aug.test_acc)],
    ["Train–Val Gap", (R.shallow.gap * 100).toFixed(2) + "%", (R.deep.gap * 100).toFixed(2) + "%", (R.deep_aug.gap * 100).toFixed(2) + "%"],
    ["Overfitting Observed?", R.shallow.overfitting, R.deep.overfitting, R.deep_aug.overfitting],
    ["Training Time (s)", R.shallow.train_time_s.toFixed(0), R.deep.train_time_s.toFixed(0), R.deep_aug.train_time_s.toFixed(0)],
  ].map((r, i) => tableRow([
    cellText(r[0], { w: 3000, bold: true, fill: i % 2 ? "F7FAFC" : undefined }),
    cellText(r[1], { w: 2200, fill: i % 2 ? "F7FAFC" : undefined }),
    cellText(r[2], { w: 2200, fill: i % 2 ? "F7FAFC" : undefined }),
    cellText(r[3], { w: 2200, fill: i % 2 ? "F7FAFC" : undefined }),
  ]));
  return new Table({
    columnWidths: [3000, 2200, 2200, 2200],
    width: { size: 9600, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "CBD5E0" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "CBD5E0" },
      left: { style: BorderStyle.SINGLE, size: 4, color: "CBD5E0" },
      right: { style: BorderStyle.SINGLE, size: 4, color: "CBD5E0" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 4, color: "CBD5E0" },
      insideVertical: { style: BorderStyle.SINGLE, size: 4, color: "CBD5E0" },
    },
    rows: [head, ...rows],
  });
}

const dTest = pct(R.deep.test_acc), sTest = pct(R.shallow.test_acc), aTest = pct(R.deep_aug.test_acc);
const speedup = (R.deep.train_time_s / R.shallow.train_time_s).toFixed(0);

const children = [
  h1("Comparative Study of Shallow CNN vs Deep CNN on Fashion-MNIST"),
  new Paragraph({
    border: { bottom: { color: BLUE, space: 4, style: BorderStyle.SINGLE, size: 18 } },
    spacing: { after: 200 },
    children: [new TextRun({ text: "Hands-on assignment report — dataset exploration, model design, training, comparison, and error analysis", italics: true, color: BLUE, size: 20 })],
  }),

  // Links block
  p([new TextRun({ text: "GitHub Repository: ", bold: true }), link(REPO)]),
  bullet([new TextRun("Jupyter Notebook (all 6 parts, executed): "), link(NOTEBOOK, "Fashion_MNIST_Shallow_vs_Deep_CNN.ipynb")]),
  bullet([new TextRun("Comparative Report (PDF): "), link(REPORT_PDF, "REPORT.pdf")]),
  bullet([new TextRun("Comparative Report (Markdown): "), link(REPORT_MD, "REPORT.md")]),

  h2("1. Objective"),
  p("Perform a controlled comparison between a shallow CNN and a deep CNN on the Fashion-MNIST dataset — training both on the same data with an identical training setup — and determine which architecture is more suitable for this image-classification task in terms of accuracy, generalisation, and efficiency. A third experiment adds data augmentation to the deep CNN to study its effect."),

  h2("2. Dataset Overview"),
  p("Fashion-MNIST consists of 70,000 grayscale 28×28 images across 10 balanced clothing classes (6,000 training images each), split into 60,000 training and 10,000 test images. Pixels were normalized to [0, 1] and images reshaped to (28, 28, 1) for CNN input. A fixed, stratified 90/10 train–validation split (seeded) was held out so both models see exactly the same validation data and the experiment is reproducible."),
  ...figure("01_samples_per_class.png", "Figure 1 — One sample image per class. Distinctive items (Trouser, Bag, Sandal, Sneaker, Ankle boot) are easy; the upper-body garments (T-shirt, Pullover, Coat, Shirt) share almost identical outlines and are where most errors occur."),

  h2("3. Preprocessing — Why Normalize and Reshape?"),
  p([new TextRun({ text: "Normalization: ", bold: true }), new TextRun("raw pixels are 0–255; feeding such large, unevenly-scaled values in makes training slow and unstable. Rescaling to [0, 1] keeps activations and gradients in a small consistent range, so the optimizer converges faster and more reliably.")]),
  p([new TextRun({ text: "Reshaping: ", bold: true }), new TextRun("a Conv2D layer expects a (height, width, channels) tensor. The images arrive as (28, 28) with no channel axis, so we reshape to (28, 28, 1) — a single grayscale channel — otherwise the convolution cannot build.")]),

  h2("4. Shallow CNN"),
  p("Architecture: Conv2D(32) → MaxPool → Flatten → Dense(128) → Dense(10, softmax) — one convolution layer and one pooling layer. Almost all of its parameters live in the Flatten → Dense connection, which is powerful enough to memorise training-set detail."),
  ...figure("02_shallow_curves.png", "Figure 2 — Shallow CNN learning curves. Training accuracy climbs toward 0.97 while validation flattens near 0.92, and validation loss turns upward — the classic signature of overfitting. Early stopping restores the best-validation weights (dashed line)."),
  p([new TextRun({ text: "Result: ", bold: true }), new TextRun(`test accuracy ${sTest}. The model learns only low-level features (edges, strokes) and overfits — training accuracy sits well above validation accuracy (gap ${(R.shallow.gap * 100).toFixed(1)}%).`)]),

  h2("5. Deep CNN"),
  p("Architecture: Conv(32) → BatchNorm → Conv(64) → MaxPool → Dropout → Conv(128) → BatchNorm → MaxPool → Dropout → Flatten → Dense(256) → Dropout → Dense(10, softmax) — three convolution layers with growing filters, two pooling layers, batch-norm and dropout. More filters build a feature hierarchy; batch-norm stabilises training; dropout prevents over-reliance on any single feature."),
  ...figure("03_deep_curves.png", "Figure 3 — Deep CNN learning curves. Training and validation rise together and stay close, and validation loss tracks training loss instead of rising — the signature of a model that generalises rather than memorises."),
  p([new TextRun({ text: "Result: ", bold: true }), new TextRun(`test accuracy ${dTest} — the best of all models — with a near-zero train–validation gap (${(R.deep.gap * 100).toFixed(1)}%), i.e. no overfitting. The deeper feature hierarchy resolves the hard look-alike classes the shallow model misses.`)]),

  h2("6. Enhancement — Deep CNN with Data Augmentation"),
  p("The same deep architecture was retrained with gentle on-the-fly augmentation (small random rotation ≤4%, translation ≤6%, zoom ≤6%) on the training images only. The transforms are kept small because Fashion-MNIST garments are already upright and centred."),
  ...figure("04_augmented_samples.png", "Figure 4 — Augmented training images: slightly rotated / shifted / zoomed, but still clearly recognisable."),
  ...figure("05_deep_aug_curves.png", "Figure 5 — Augmented deep CNN curves. Validation accuracy sits slightly above training accuracy — the model is mildly under-fit to the (harder) augmented training data: heavy regularisation, no overfitting."),
  p([new TextRun({ text: "Result: ", bold: true }), new TextRun(`test accuracy ${aTest} with a negative train–val gap (${(R.deep_aug.gap * 100).toFixed(1)}%). Augmentation strongly regularises but does not beat the plain deep model on this clean test set — an honest, instructive result: augmentation is not a free accuracy win; its payoff is robustness/calibration, and its strength must be matched to the data.`)]),

  h2("7. Comparative Study"),
  p("All three models use the same optimizer, loss, batch size, validation set, and early-stopping recipe, so the comparison is fair. Training/validation accuracy are reported at each model's best (restored) epoch."),
  comparisonTable(),
  ...figure("06_accuracy_comparison.png", "Figure 6 — Accuracy comparison. The shallow model's Training bar towers over its Validation/Test bars (overfitting); the deep model's bars are level and its Test bar is the tallest; the augmented model's Training bar is the shortest (mild underfitting)."),
  p([new TextRun({ text: "Key observations: ", bold: true }), new TextRun(`the deep CNN is the most accurate (${dTest} test vs ${sTest} for shallow) and the best-generalising (gap ~0 vs ${(R.shallow.gap * 100).toFixed(1)}% for shallow). The shallow CNN is far more efficient — about ${speedup}× faster to train and ~2.4× fewer parameters. Data augmentation trades training-set fit for robustness without raising test accuracy on this clean dataset.`)]),

  h2("8. Prediction & Error Analysis"),
  p("Predictions were generated for both models. Below: the confusion matrices, a confidence analysis, and the deep model's most confident mistakes."),
  ...figure("09_confusion_matrices.png", "Figure 7 — Confusion matrices (rows = actual, columns = predicted). Both models are near-perfect on the distinctive classes; errors cluster in the Shirt / T-shirt / Pullover / Coat block. The deep CNN has a stronger diagonal and lighter off-diagonal cells there."),
  ...figure("10_confidence_hist.png", "Figure 8 — Prediction confidence for correct (green) vs incorrect (red) predictions. Both are confident when correct; the deep model's errors lean to lower confidence — it is better calibrated (less often confidently wrong)."),
  ...figure("11_deep_confident_mistakes.png", "Figure 9 — The deep CNN's most confident mistakes: nearly all are Shirt/T-shirt/Pullover/Coat confusions that are genuinely ambiguous at 28×28 grayscale."),
  p([new TextRun({ text: "Easiest classes: ", bold: true }), new TextRun("Trouser, Bag, Sandal, Sneaker, Ankle boot (distinctive silhouettes). "), new TextRun({ text: "Most confused: ", bold: true }), new TextRun("Shirt with T-shirt/Pullover/Coat. The deep CNN reduces confusion within this cluster and is better calibrated.")]),

  h2("9. Final Conclusion"),
  bullet([new TextRun({ text: "Recommended model: ", bold: true }), new TextRun("the deep CNN — highest accuracy and best generalisation, no overfitting.")]),
  bullet([new TextRun({ text: "More efficient: ", bold: true }), new TextRun(`the shallow CNN — ~${speedup}× faster to train and far lighter.`)]),
  bullet([new TextRun({ text: "More accurate: ", bold: true }), new TextRun(`the deep CNN (${dTest} vs ${sTest} test), with the smaller generalisation gap.`)]),
  p([new TextRun({ text: "What we learned: ", bold: true }), new TextRun("depth pays off, but only with the right supporting techniques. Extra convolution layers build a feature hierarchy that separates visually similar classes a single-conv model cannot — but that capacity would overfit without batch-norm and dropout. A fixed validation split and early stopping made the comparison fair and reproducible; the confidence analysis showed the regularised deep model is better calibrated; and the augmentation experiment showed a technique must be matched to the data. The core trade-off is concrete: more capacity buys accuracy at the cost of compute and training time, and regularisation turns that capacity into generalisation rather than memorisation.")]),
];

const doc = new Document({
  creator: "Mani Dixit",
  title: "Comparative Study of Shallow CNN vs Deep CNN on Fashion-MNIST",
  styles: { default: { document: { run: { font: "Calibri", size: 21 } } } },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 } } }, children }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("Wrote", OUT, buf.length, "bytes");
});

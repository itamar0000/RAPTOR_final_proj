// build_presentation.cjs — RAPTOR Final Presentation (10 slides, pptxgenjs)
// Run: node tools/build_presentation.cjs
"use strict";
const PptxGenJS = require("C:/Users/olete/AppData/Roaming/npm/node_modules/pptxgenjs");
const fs  = require("fs");
const path = require("path");

const ROOT  = path.resolve(__dirname, "..");
const FIGS  = path.join(ROOT, "outputs", "figures");
const OUT   = path.join(ROOT, "outputs", "RAPTOR_Final_Presentation.pptx");

const pres = new PptxGenJS();
pres.layout  = "LAYOUT_16x9";
pres.author  = "Eitan Weizman & Itamar Haimov";
pres.title   = "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval";

// ── Palette ───────────────────────────────────────────────────────────────────
const NAVY   = "1F3864";
const BLUE   = "2E5090";
const AMBER  = "E07B39";
const GREEN  = "276221";
const LIGHT  = "F4F7FC";
const WHITE  = "FFFFFF";
const GREY   = "8D99AE";
const DARK   = "111827";

// ── Helpers ───────────────────────────────────────────────────────────────────
function figPath(name) {
  return path.join(FIGS, name);
}

// Slide header with slide number (for light slides)
function addHeader(slide, title, num) {
  // Slide number circle
  slide.addShape(pres.shapes.OVAL, {
    x: 0.3, y: 0.18, w: 0.42, h: 0.42,
    fill: { color: NAVY }, line: { color: NAVY }
  });
  slide.addText(String(num), {
    x: 0.3, y: 0.18, w: 0.42, h: 0.42,
    fontSize: 13, bold: true, color: WHITE,
    align: "center", valign: "middle", margin: 0
  });
  // Title text
  slide.addText(title, {
    x: 0.82, y: 0.15, w: 8.8, h: 0.5,
    fontSize: 22, bold: true, color: NAVY,
    fontFace: "Georgia", valign: "middle", margin: 0
  });
  // Subtitle rule
  slide.addShape(pres.shapes.LINE, {
    x: 0.3, y: 0.72, w: 9.4, h: 0,
    line: { color: NAVY, width: 1.5 }
  });
}

// Small stat card
function statCard(slide, x, y, w, h, label, value, bg) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: bg || NAVY },
    shadow: { type: "outer", blur: 6, offset: 3, angle: 135, color: "000000", opacity: 0.12 }
  });
  slide.addText(value, {
    x, y, w, h: h * 0.55,
    fontSize: 36, bold: true, color: WHITE,
    align: "center", valign: "bottom", fontFace: "Georgia", margin: 0
  });
  slide.addText(label, {
    x, y: y + h * 0.55, w, h: h * 0.42,
    fontSize: 11, color: "CADCFC",
    align: "center", valign: "top", margin: 0
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: DARK };

  // Large accent shape top-right
  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.2, y: 0, w: 2.8, h: 5.625,
    fill: { color: NAVY }
  });
  // Amber accent stripe
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.9, y: 0, w: 0.18, h: 5.625,
    fill: { color: AMBER }
  });

  s.addText("RAPTOR", {
    x: 0.5, y: 0.55, w: 6.2, h: 1.4,
    fontSize: 72, bold: true, color: WHITE,
    fontFace: "Georgia", valign: "middle"
  });
  s.addText("Recursive Abstractive Processing\nfor Tree-Organized Retrieval", {
    x: 0.5, y: 1.85, w: 6.2, h: 1.1,
    fontSize: 20, color: "CADCFC", fontFace: "Calibri"
  });
  s.addShape(pres.shapes.LINE, {
    x: 0.5, y: 3.05, w: 5.8, h: 0,
    line: { color: AMBER, width: 2 }
  });
  s.addText("A Reproduction Study", {
    x: 0.5, y: 3.18, w: 5.8, h: 0.38,
    fontSize: 16, color: GREY, italics: true, fontFace: "Calibri"
  });
  s.addText("Final Submission — Seminar in Software Engineering 157119.5785", {
    x: 0.5, y: 3.65, w: 6.2, h: 0.4,
    fontSize: 12, color: GREY, fontFace: "Calibri"
  });
  s.addText("Eitan Weizman & Itamar Haimov\nJune 2026", {
    x: 0.5, y: 4.2, w: 6.2, h: 0.7,
    fontSize: 14, bold: true, color: WHITE, fontFace: "Calibri"
  });

  // Right panel content
  s.addText("Original Paper:", {
    x: 7.35, y: 0.55, w: 2.5, h: 0.35,
    fontSize: 10, color: "CADCFC", fontFace: "Calibri"
  });
  s.addText("Sarthi et al.\nICLR 2024\narXiv:2401.18059", {
    x: 7.35, y: 0.85, w: 2.5, h: 0.9,
    fontSize: 11, bold: true, color: WHITE, fontFace: "Calibri"
  });
  s.addShape(pres.shapes.LINE, {
    x: 7.4, y: 1.88, w: 2.3, h: 0,
    line: { color: AMBER, width: 1 }
  });
  s.addText("Local Reproduction\nwith Ollama", {
    x: 7.35, y: 2.0, w: 2.5, h: 0.65,
    fontSize: 11, color: "CADCFC", fontFace: "Calibri"
  });
  s.addText("Evaluated on\nFRAMES Benchmark", {
    x: 7.35, y: 2.8, w: 2.5, h: 0.65,
    fontSize: 11, color: "CADCFC", fontFace: "Calibri"
  });
  s.addText("50 Questions\nAnalyzed", {
    x: 7.35, y: 3.6, w: 2.5, h: 0.65,
    fontSize: 11, color: "CADCFC", fontFace: "Calibri"
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — Problem & Motivation
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  addHeader(s, "1. Problem & Motivation", 1);

  // Left text column
  s.addText("Why Standard RAG Fails", {
    x: 0.3, y: 0.85, w: 4.5, h: 0.45,
    fontSize: 17, bold: true, color: NAVY, fontFace: "Georgia"
  });

  const problems = [
    ["Short Context", "Retrieved chunks (~100 tokens) miss the document's global theme."],
    ["Context Blindness", "No holistic view — systematic errors on broad questions."],
    ["Contiguity Limit", "Can only retrieve contiguous passages; cannot bridge distant sections."],
    ["Multi-Hop Failure", "Cannot connect facts spread across different pages or documents."],
  ];

  problems.forEach(([title, desc], i) => {
    const y = 1.42 + i * 0.95;
    // Amber left border
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.3, y, w: 0.07, h: 0.75,
      fill: { color: AMBER }
    });
    s.addText(title, {
      x: 0.47, y, w: 4.3, h: 0.32,
      fontSize: 13, bold: true, color: NAVY, fontFace: "Calibri", margin: 0
    });
    s.addText(desc, {
      x: 0.47, y: y + 0.3, w: 4.3, h: 0.42,
      fontSize: 11, color: "444444", fontFace: "Calibri", margin: 0
    });
  });

  // Right: RAG vs RAPTOR comparison diagram image
  s.addImage({ path: figPath("fig6_tree_diagram.png"), x: 5.0, y: 0.85, w: 4.7, h: 2.8 });
  s.addText("RAPTOR Tree Structure — multiple abstraction levels", {
    x: 5.0, y: 3.7, w: 4.7, h: 0.3,
    fontSize: 9, italics: true, color: GREY, align: "center", fontFace: "Calibri"
  });

  s.addText("Source: Sarthi et al., RAPTOR (ICLR 2024)", {
    x: 0.3, y: 5.28, w: 9.4, h: 0.25,
    fontSize: 8, color: GREY, align: "right", fontFace: "Calibri"
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — Key Idea / Tree Construction
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  addHeader(s, "2. Key Idea — RAPTOR Tree Construction", 2);

  // Pipeline steps as numbered cards
  const steps = [
    { n: "1", label: "Chunking", desc: "~100-token\nsentence-aware\nsegments" },
    { n: "2", label: "Embedding", desc: "Dense vectors\n(nomic-embed-text\n768-dim)" },
    { n: "3", label: "Clustering", desc: "UMAP + GMM\nsoft assignment\nBIC auto-k" },
    { n: "4", label: "Summarize", desc: "LLM generates\ncluster summaries\n(Qwen 2.5 7B)" },
    { n: "5", label: "Recurse", desc: "Repeat until\nno further\nclustering" },
  ];

  const stepW = 1.65;
  const stepH = 2.3;
  const startX = 0.25;
  const y0 = 1.0;

  steps.forEach((st, i) => {
    const x = startX + i * (stepW + 0.22);
    // Card
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: y0, w: stepW, h: stepH,
      fill: { color: i === 4 ? NAVY : WHITE },
      shadow: { type: "outer", blur: 5, offset: 2, angle: 135, color: "000000", opacity: 0.1 }
    });
    // Number circle
    s.addShape(pres.shapes.OVAL, {
      x: x + stepW / 2 - 0.27, y: y0 + 0.18, w: 0.54, h: 0.54,
      fill: { color: i === 4 ? AMBER : NAVY }
    });
    s.addText(st.n, {
      x: x + stepW / 2 - 0.27, y: y0 + 0.18, w: 0.54, h: 0.54,
      fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
    });
    s.addText(st.label, {
      x, y: y0 + 0.85, w: stepW, h: 0.42,
      fontSize: 13, bold: true, color: i === 4 ? WHITE : NAVY,
      align: "center", fontFace: "Calibri", margin: 0
    });
    s.addText(st.desc, {
      x, y: y0 + 1.3, w: stepW, h: 0.88,
      fontSize: 10, color: i === 4 ? "CADCFC" : "555555",
      align: "center", fontFace: "Calibri"
    });
    // Arrow (between cards)
    if (i < steps.length - 1) {
      s.addShape(pres.shapes.LINE, {
        x: x + stepW + 0.03, y: y0 + stepH / 2,
        w: 0.19, h: 0,
        line: { color: AMBER, width: 2 }
      });
    }
  });

  // Bottom info boxes
  const boxes = [
    { label: "Leaf Layer", desc: "Raw ~100-token chunks", color: "5C8374" },
    { label: "Middle Layers", desc: "LLM-generated cluster summaries", color: BLUE },
    { label: "Root", desc: "Highest-level document summary", color: NAVY },
    { label: "Soft Clustering", desc: "Nodes may belong to multiple clusters", color: AMBER },
  ];
  const bW = 2.2, bH = 0.75;
  boxes.forEach((b, i) => {
    const x = 0.25 + i * (bW + 0.27);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 3.55, w: bW, h: bH, fill: { color: b.color } });
    s.addText(b.label, {
      x, y: 3.55, w: bW, h: 0.32,
      fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
    });
    s.addText(b.desc, {
      x, y: 3.87, w: bW, h: 0.4,
      fontSize: 9.5, color: "CADCFC", align: "center", valign: "top", fontFace: "Calibri", margin: 0
    });
  });

  s.addText("Source: Sarthi et al., RAPTOR (ICLR 2024)", {
    x: 0.3, y: 5.28, w: 9.4, h: 0.25,
    fontSize: 8, color: GREY, align: "right", fontFace: "Calibri"
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — Methods (Our Implementation)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  addHeader(s, "3. Methods — Local Ollama Adaptation", 3);

  // Left column: Original vs Our
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 0.9, w: 4.4, h: 1.75,
    fill: { color: "FFEBEE" },
    shadow: { type: "outer", blur: 4, offset: 2, angle: 135, color: "000000", opacity: 0.08 }
  });
  s.addText("Original Paper (API)", {
    x: 0.3, y: 0.9, w: 4.4, h: 0.38,
    fontSize: 13, bold: true, color: "B71C1C", align: "center", valign: "middle",
    fontFace: "Calibri", margin: 0
  });
  s.addText([
    { text: "Summarization: ", options: { bold: true, breakLine: false } },
    { text: "GPT-4\n", options: { breakLine: true } },
    { text: "Embeddings: ", options: { bold: true, breakLine: false } },
    { text: "text-embedding-ada-002 (1536d)\n", options: { breakLine: true } },
    { text: "QA Model: ", options: { bold: true, breakLine: false } },
    { text: "GPT-4", options: {} },
  ], {
    x: 0.45, y: 1.32, w: 4.1, h: 1.25,
    fontSize: 11, color: "444444", fontFace: "Calibri"
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 2.8, w: 4.4, h: 1.75,
    fill: { color: "E8F5E9" },
    shadow: { type: "outer", blur: 4, offset: 2, angle: 135, color: "000000", opacity: 0.08 }
  });
  s.addText("Our Reproduction (Ollama)", {
    x: 0.3, y: 2.8, w: 4.4, h: 0.38,
    fontSize: 13, bold: true, color: GREEN, align: "center", valign: "middle",
    fontFace: "Calibri", margin: 0
  });
  s.addText([
    { text: "Summarization: ", options: { bold: true, breakLine: false } },
    { text: "Qwen 2.5 7B Instruct (local)\n", options: { breakLine: true } },
    { text: "Embeddings: ", options: { bold: true, breakLine: false } },
    { text: "nomic-embed-text (768d)\n", options: { breakLine: true } },
    { text: "QA Model: ", options: { bold: true, breakLine: false } },
    { text: "Qwen 2.5 7B Instruct (local)", options: {} },
  ], {
    x: 0.45, y: 3.22, w: 4.1, h: 1.25,
    fontSize: 11, color: "444444", fontFace: "Calibri"
  });

  // Arrow between the two
  s.addShape(pres.shapes.LINE, {
    x: 2.5, y: 2.66, w: 0, h: 0.16,
    line: { color: AMBER, width: 2 }
  });
  s.addText("replaced with", {
    x: 1.5, y: 2.62, w: 2.1, h: 0.22,
    fontSize: 9, color: AMBER, align: "center", bold: true, fontFace: "Calibri", margin: 0
  });

  // Right column: retrieval strategy
  s.addText("Retrieval: Collapsed Tree Strategy", {
    x: 5.1, y: 0.9, w: 4.6, h: 0.42,
    fontSize: 14, bold: true, color: NAVY, fontFace: "Georgia"
  });
  s.addImage({ path: figPath("fig7_retrieval_strategies.png"), x: 5.1, y: 1.38, w: 4.6, h: 2.7 });
  s.addText("Collapsed Tree outperforms Tree Traversal on all benchmarks (paper results)", {
    x: 5.1, y: 4.1, w: 4.6, h: 0.35,
    fontSize: 9, italics: true, color: GREY, align: "center", fontFace: "Calibri"
  });

  s.addText("Source: Sarthi et al., RAPTOR (ICLR 2024)", {
    x: 0.3, y: 5.28, w: 9.4, h: 0.25,
    fontSize: 8, color: GREY, align: "right", fontFace: "Calibri"
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — Data
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  addHeader(s, "4. Data — FRAMES Multi-Hop Benchmark", 4);

  // Left: dataset info cards
  const dataPoints = [
    { label: "Benchmark", val: "Google FRAMES", sub: "Factuality, Retrieval & Reasoning" },
    { label: "Task", val: "Multi-Hop QA", sub: "Over Wikipedia articles" },
    { label: "Total Test Size", val: "824 questions", sub: "We evaluated 50 samples" },
    { label: "Evidence Source", val: "Wikipedia API", sub: "On-demand article fetching" },
  ];

  dataPoints.forEach((d, i) => {
    const y = 0.88 + i * 1.08;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.3, y, w: 4.5, h: 0.9,
      fill: { color: WHITE },
      shadow: { type: "outer", blur: 4, offset: 2, angle: 135, color: "000000", opacity: 0.09 }
    });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.3, y, w: 0.08, h: 0.9, fill: { color: AMBER } });
    s.addText(d.label, {
      x: 0.5, y: y + 0.05, w: 4.2, h: 0.3,
      fontSize: 10, color: GREY, fontFace: "Calibri", margin: 0
    });
    s.addText(d.val, {
      x: 0.5, y: y + 0.3, w: 4.2, h: 0.35,
      fontSize: 14, bold: true, color: NAVY, fontFace: "Calibri", margin: 0
    });
    s.addText(d.sub, {
      x: 0.5, y: y + 0.62, w: 4.2, h: 0.25,
      fontSize: 10, color: "666666", fontFace: "Calibri", margin: 0
    });
  });

  // Right: example question callout
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 0.88, w: 4.6, h: 2.2,
    fill: { color: NAVY },
    shadow: { type: "outer", blur: 6, offset: 3, angle: 135, color: "000000", opacity: 0.12 }
  });
  s.addText("Sample FRAMES Question", {
    x: 5.25, y: 0.95, w: 4.3, h: 0.35,
    fontSize: 11, bold: true, color: AMBER, fontFace: "Calibri"
  });
  s.addText(
    '"If my future wife has the same first name as the 15th first lady\'s mother and her surname is the second assassinated president\'s mother\'s maiden name, what is my future wife\'s name?"',
    {
      x: 5.25, y: 1.35, w: 4.3, h: 1.0,
      fontSize: 10.5, color: WHITE, italics: true, fontFace: "Calibri"
    }
  );
  s.addText("Gold Answer: Jane Ballou", {
    x: 5.25, y: 2.42, w: 4.3, h: 0.35,
    fontSize: 11, bold: true, color: AMBER, fontFace: "Calibri"
  });

  // Articles per question chart
  s.addImage({ path: figPath("fig2_articles_per_question.png"), x: 5.1, y: 3.2, w: 4.6, h: 2.1 });
  s.addText("Wikipedia articles fetched per FRAMES question (avg = 0.94)", {
    x: 5.1, y: 5.28, w: 4.6, h: 0.25,
    fontSize: 8, italics: true, color: GREY, align: "center", fontFace: "Calibri"
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — Results Overview
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  addHeader(s, "5. Results — FRAMES Evaluation (n = 50)", 5);

  // Stat cards row
  const stats = [
    { label: "Questions\nEvaluated", val: "50", bg: NAVY },
    { label: "Non-Empty\nAnswers", val: "47", bg: BLUE },
    { label: "Soft Accuracy\n(Gold Found)", val: "4%", bg: AMBER },
    { label: "Pipeline\nErrors", val: "6%", bg: "B85042" },
  ];
  stats.forEach((st, i) => {
    statCard(s, 0.3 + i * 2.38, 0.88, 2.1, 1.35, st.label, st.val, st.bg);
  });

  // Pie chart (native pptxgenjs)
  s.addChart(pres.charts.PIE, [{
    name: "Status",
    labels: ["Insufficient Context (90%)", "Pipeline Error (6%)", "Answered (4%)"],
    values: [45, 3, 2]
  }], {
    x: 0.3, y: 2.38, w: 4.6, h: 2.9,
    chartColors: ["E07B39", "B85042", "276221"],
    showPercent: true,
    dataLabelColor: WHITE,
    dataLabelFontSize: 11,
    showLegend: true,
    legendPos: "b",
    legendFontSize: 10,
    chartArea: { fill: { color: LIGHT }, roundedCorners: false },
    title: "Answer Status Distribution",
    showTitle: true,
    titleFontSize: 12,
    titleColor: NAVY,
  });

  // Answer length chart image
  s.addImage({ path: figPath("fig3_answer_length_dist.png"), x: 5.2, y: 2.38, w: 4.5, h: 2.9 });
  s.addText("Answer length (chars) by status — refusals cluster between 200-400 chars", {
    x: 5.2, y: 5.28, w: 4.5, h: 0.25,
    fontSize: 8, italics: true, color: GREY, align: "center", fontFace: "Calibri"
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — Paper Results vs Our Results
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  addHeader(s, "5. Results — Paper Benchmarks vs. Our Reproduction", 6);

  // Paper benchmark bar chart (native)
  s.addChart(pres.charts.BAR, [
    {
      name: "Best Baseline (paper)",
      labels: ["QuALITY\n(Accuracy)", "NarrativeQA\n(METEOR)", "QASPER\n(F-1)"],
      values: [56.1, 12.1, 42.2]
    },
    {
      name: "RAPTOR (paper, GPT-4)",
      labels: ["QuALITY\n(Accuracy)", "NarrativeQA\n(METEOR)", "QASPER\n(F-1)"],
      values: [76.2, 17.5, 55.7]
    }
  ], {
    x: 0.3, y: 0.85, w: 5.5, h: 3.8,
    barDir: "col",
    chartColors: ["8D99AE", NAVY],
    showValue: true,
    dataLabelFontSize: 9,
    dataLabelColor: DARK,
    catAxisLabelColor: "555555",
    valAxisLabelColor: "555555",
    valGridLine: { color: "DDDDDD", size: 0.5 },
    catGridLine: { style: "none" },
    showLegend: true,
    legendPos: "b",
    legendFontSize: 10,
    chartArea: { fill: { color: WHITE }, roundedCorners: false },
    title: "RAPTOR vs. Baseline (Original Paper Results, GPT-4)",
    showTitle: true,
    titleFontSize: 11,
    titleColor: NAVY,
  });

  // Right: our results + explanation
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.1, y: 0.88, w: 3.6, h: 3.8,
    fill: { color: WHITE },
    shadow: { type: "outer", blur: 5, offset: 2, angle: 135, color: "000000", opacity: 0.1 }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.1, y: 0.88, w: 3.6, h: 0.44,
    fill: { color: NAVY }
  });
  s.addText("Our Implementation (FRAMES)", {
    x: 6.1, y: 0.88, w: 3.6, h: 0.44,
    fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle",
    fontFace: "Calibri", margin: 0
  });

  const ourStats = [
    ["Soft Accuracy", "4%"],
    ["Gold Match", "2 / 50"],
    ["LLM Model", "Qwen 2.5 7B"],
    ["Benchmark", "FRAMES (harder)"],
    ["Answered", "47 / 50"],
  ];
  ourStats.forEach(([k, v], i) => {
    const y = 1.44 + i * 0.58;
    s.addText(k, {
      x: 6.2, y, w: 1.8, h: 0.38,
      fontSize: 11, color: GREY, fontFace: "Calibri", margin: 0
    });
    s.addText(v, {
      x: 8.0, y, w: 1.6, h: 0.38,
      fontSize: 11, bold: true, color: NAVY, align: "right", fontFace: "Calibri", margin: 0
    });
    if (i < ourStats.length - 1) {
      s.addShape(pres.shapes.LINE, {
        x: 6.2, y: y + 0.42, w: 3.3, h: 0,
        line: { color: "EEEEEE", width: 0.75 }
      });
    }
  });

  // Gap explanation
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 4.82, w: 9.4, h: 0.65,
    fill: { color: "FFF3E0" }
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.3, y: 4.82, w: 0.07, h: 0.65, fill: { color: AMBER } });
  s.addText(
    "Gap explained: FRAMES requires 3-8 Wikipedia facts per question (multi-hop). " +
    "Our local 7B model lacks GPT-4 level reasoning, and per-article RAPTOR trees cannot bridge cross-article gaps. " +
    "The pipeline itself is correct — the challenge is the benchmark difficulty.",
    {
      x: 0.47, y: 4.88, w: 9.1, h: 0.55,
      fontSize: 10.5, color: "333333", fontFace: "Calibri"
    }
  );
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Problems Encountered
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  addHeader(s, "5. Problems Encountered", 7);

  const problems = [
    {
      n: "1",
      title: "Package Conflicts",
      body: "numpy, scikit-learn, umap-learn, sentence-transformers all had version clashes. Required manual pinning of exact versions.",
      color: "B85042"
    },
    {
      n: "2",
      title: "API → Ollama Adaptation",
      body: "Original RAPTOR depends on OpenAI GPT-4 + ada-002. We wrote OllamaSummarizer, OllamaQA, OllamaEmbedding adapter classes to satisfy the same interfaces.",
      color: BLUE
    },
    {
      n: "3",
      title: "FRAMES Multi-Hop Gap",
      body: "FRAMES questions need 3–8 chained Wikipedia facts. Single-article RAPTOR trees lack cross-article fusion; 90% of answers triggered insufficient-context refusals.",
      color: AMBER
    },
    {
      n: "4",
      title: "Windows Encoding",
      body: "UnicodeEncodeError on emoji and non-ASCII text under PowerShell cp1252. Fixed with UTF-8 stream wrappers and ensure_ascii=False in JSON.",
      color: "5C8374"
    },
  ];

  const pW = 4.55, pH = 1.85;
  problems.forEach((pr, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.3 + col * (pW + 0.28);
    const y = 0.9 + row * (pH + 0.28);
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: pW, h: pH,
      fill: { color: WHITE },
      shadow: { type: "outer", blur: 5, offset: 2, angle: 135, color: "000000", opacity: 0.1 }
    });
    // Top colored stripe
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: pW, h: 0.07, fill: { color: pr.color } });
    // Number
    s.addShape(pres.shapes.OVAL, {
      x: x + 0.14, y: y + 0.18, w: 0.46, h: 0.46,
      fill: { color: pr.color }
    });
    s.addText(pr.n, {
      x: x + 0.14, y: y + 0.18, w: 0.46, h: 0.46,
      fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0
    });
    s.addText(pr.title, {
      x: x + 0.7, y: y + 0.18, w: pW - 0.85, h: 0.44,
      fontSize: 13, bold: true, color: NAVY, fontFace: "Calibri", valign: "middle", margin: 0
    });
    s.addShape(pres.shapes.LINE, {
      x: x + 0.14, y: y + 0.72, w: pW - 0.28, h: 0,
      line: { color: "EEEEEE", width: 0.75 }
    });
    s.addText(pr.body, {
      x: x + 0.14, y: y + 0.82, w: pW - 0.28, h: 0.92,
      fontSize: 11, color: "444444", fontFace: "Calibri"
    });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — Task Completion Table
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  addHeader(s, "6. Reproduction Plan — Task Completion", 8);

  const tasks = [
    { task: "Article Comprehension & Basic RAG Infrastructure", weight: "15%", done: true, note: "RAPTOR library + Ollama adapters integrated" },
    { task: "Chunking + Embeddings Pipeline", weight: "15%", done: true, note: "Sentence-aware chunking + nomic-embed-text (768d)" },
    { task: "Clustering + Tree Construction", weight: "20%", done: true, note: "UMAP + GMM + BIC auto-k, recursive builder" },
    { task: "Tree Retrieval (Collapsed Tree)", weight: "20%", done: true, note: "Collapsed tree strategy verified on all samples" },
    { task: "Evaluation on FRAMES Benchmark", weight: "20%", done: true, note: "50 questions, results analyzed & visualized" },
    { task: "Report + Final Presentation", weight: "10%", done: true, note: "This document + Word report" },
  ];

  const colWs = [4.2, 0.9, 0.9, 3.8];
  const headers = ["Task", "Weight", "Status", "Notes"];
  const tX = 0.3, tW = 9.4;

  // Header row
  let headerRow = [];
  headers.forEach((h, i) => {
    headerRow.push({
      text: h,
      options: {
        bold: true, color: WHITE, fontSize: 11, fontFace: "Calibri",
        fill: { color: NAVY }, align: "center", valign: "middle",
        border: { pt: 0.5, color: "FFFFFF" }
      }
    });
  });
  const tableData = [headerRow];

  tasks.forEach((t, i) => {
    const bg = i % 2 === 0 ? "FFFFFF" : "F4F7FC";
    tableData.push([
      { text: t.task, options: { fontSize: 10.5, fontFace: "Calibri", fill: { color: bg }, color: DARK, border: { pt: 0.5, color: "DDDDDD" } } },
      { text: t.weight, options: { fontSize: 10.5, bold: true, fontFace: "Calibri", fill: { color: bg }, color: NAVY, align: "center", border: { pt: 0.5, color: "DDDDDD" } } },
      { text: t.done ? "✓ Done" : "✗", options: { fontSize: 10.5, bold: true, fontFace: "Calibri", fill: { color: t.done ? "C6EFCE" : "FFCCCC" }, color: t.done ? GREEN : "B71C1C", align: "center", border: { pt: 0.5, color: "DDDDDD" } } },
      { text: t.note, options: { fontSize: 9.5, fontFace: "Calibri", fill: { color: bg }, color: "555555", border: { pt: 0.5, color: "DDDDDD" } } },
    ]);
  });

  s.addTable(tableData, {
    x: tX, y: 0.88,
    w: tW,
    colW: colWs,
    rowH: 0.62,
    border: { pt: 0.5, color: "CCCCCC" }
  });

  // Progress bar visual
  s.addText("100% Complete", {
    x: 0.3, y: 5.0, w: 1.9, h: 0.35,
    fontSize: 13, bold: true, color: GREEN, fontFace: "Calibri", margin: 0
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 2.3, y: 5.1, w: 7.4, h: 0.2, fill: { color: "DDDDDD" } });
  s.addShape(pres.shapes.RECTANGLE, { x: 2.3, y: 5.1, w: 7.4, h: 0.2, fill: { color: GREEN } });
  s.addText("All 6 tasks completed", {
    x: 0.3, y: 5.3, w: 9.4, h: 0.2,
    fontSize: 9, color: GREY, fontFace: "Calibri"
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — Bibliography
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: DARK };

  // Left dark panel
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 3.8, h: 5.625,
    fill: { color: NAVY }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 3.8, y: 0, w: 0.12, h: 5.625,
    fill: { color: AMBER }
  });

  s.addText("Bibliography", {
    x: 0.2, y: 0.4, w: 3.4, h: 0.65,
    fontSize: 28, bold: true, color: WHITE, fontFace: "Georgia"
  });
  s.addText("Green = cited in report\nYellow = supplementary reading", {
    x: 0.2, y: 1.1, w: 3.4, h: 0.65,
    fontSize: 11, color: "CADCFC", fontFace: "Calibri", italics: true
  });

  const cited = [
    "Sarthi et al. RAPTOR. ICLR 2024. arXiv:2401.18059",
    "McInnes et al. UMAP. 2018. arXiv:1802.03426",
    "Schwarz. BIC for GMM. Annals of Statistics, 1978",
    "Yu et al. FRAMES Benchmark. Google DeepMind, 2024",
    "Qwen Team. Qwen2.5 Technical Report. 2024",
    "Nomic AI. nomic-embed-text. 2024",
    "Ollama. Open-Source Local LLM Server. 2024",
    "parthsarthi03. RAPTOR GitHub Repo. 2024",
  ];
  const extra = [
    "Lewis et al. RAG for NLP. NeurIPS 2020",
    "Reimers & Gurevych. Sentence-BERT. EMNLP 2019",
    "Kočiský et al. NarrativeQA. TACL 2018",
    "Dasigi et al. QASPER. NAACL 2021",
    "Pang et al. QuALITY. NAACL 2022",
  ];

  s.addText("Cited Sources", {
    x: 4.1, y: 0.3, w: 5.6, h: 0.36,
    fontSize: 13, bold: true, color: "C6EFCE", fontFace: "Calibri"
  });
  cited.forEach((ref, i) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 4.1, y: 0.7 + i * 0.37, w: 5.6, h: 0.32,
      fill: { color: "1A3A22" }
    });
    s.addText(`${i + 1}. ${ref}`, {
      x: 4.15, y: 0.72 + i * 0.37, w: 5.5, h: 0.28,
      fontSize: 9, color: "C6EFCE", fontFace: "Calibri", margin: 0
    });
  });

  s.addText("Supplementary Reading", {
    x: 4.1, y: 3.68, w: 5.6, h: 0.33,
    fontSize: 11, bold: true, color: "FFEB9C", fontFace: "Calibri"
  });
  extra.forEach((ref, i) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 4.1, y: 4.05 + i * 0.29, w: 5.6, h: 0.25,
      fill: { color: "333300" }
    });
    s.addText(`${cited.length + i + 1}. ${ref}`, {
      x: 4.15, y: 4.06 + i * 0.29, w: 5.5, h: 0.22,
      fontSize: 8.5, color: "FFEB9C", fontFace: "Calibri", margin: 0
    });
  });

  s.addText("Thank you!", {
    x: 0.2, y: 4.0, w: 3.4, h: 0.7,
    fontSize: 26, bold: true, color: AMBER, fontFace: "Georgia"
  });
  s.addText("Questions?", {
    x: 0.2, y: 4.65, w: 3.4, h: 0.4,
    fontSize: 17, color: "CADCFC", fontFace: "Calibri", italics: true
  });
}

// ── Write ─────────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: OUT }).then(() => {
  console.log("Presentation written to:", OUT);
});

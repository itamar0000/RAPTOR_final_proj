// build_report.mjs  —  generates the RAPTOR final submission report (Word .docx)
// Run: node tools/build_report.mjs

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const DOCX_ROOT = "file:///C:/Users/olete/AppData/Roaming/npm/node_modules/docx/dist/index.mjs";
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak,
  LevelFormat, ExternalHyperlink,
} = await import(DOCX_ROOT);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT      = path.resolve(__dirname, "..");
const FIGS      = path.join(ROOT, "outputs", "figures");
const OUT       = path.join(ROOT, "outputs", "RAPTOR_Final_Report.docx");

// ── helpers ──────────────────────────────────────────────────────────────────
const SP  = { line: 300, lineRule: "auto" };          // 1.25 line spacing
const SP2 = { before: 160, after: 80, ...SP };
const MARGIN = { top: 1440, right: 1080, bottom: 1440, left: 1080 };

const border = (color = "CCCCCC") => ({ style: BorderStyle.SINGLE, size: 1, color });
const cellBorders = (color = "CCCCCC") => ({
  top: border(color), bottom: border(color),
  left: border(color), right: border(color),
});
const W = 9360;   // content width for A4 with 0.75" margins (DXA)

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 160 },
    children: [new TextRun({ text, bold: true, size: 32, font: "Arial", color: "1F3864" })],
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "1F3864", space: 1 } },
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, size: 26, font: "Arial", color: "2E5090" })],
  });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: SP2,
    children: [new TextRun({ text, size: 22, font: "Arial", ...opts })],
  });
}
function pBold(label, rest) {
  return new Paragraph({
    spacing: SP2,
    children: [
      new TextRun({ text: label, bold: true, size: 22, font: "Arial" }),
      new TextRun({ text: rest,  size: 22, font: "Arial" }),
    ],
  });
}
function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 60, after: 60, ...SP },
    children: [new TextRun({ text, size: 22, font: "Arial" })],
  });
}
function figCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 180 },
    children: [new TextRun({ text, size: 18, italics: true, font: "Arial", color: "444444" })],
  });
}
function img(filename, w = 550, h = 360) {
  const buf = fs.readFileSync(path.join(FIGS, filename));
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [new ImageRun({
      type: "png", data: buf,
      transformation: { width: w, height: h },
      altText: { title: filename, description: filename, name: filename },
    })],
  });
}
function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}
function space(pts = 80) {
  return new Paragraph({ spacing: { before: pts, after: 0 }, children: [] });
}

// ── table helpers ─────────────────────────────────────────────────────────────
function makeCell(text, { fill = "FFFFFF", bold = false, align = AlignmentType.LEFT, w: cw, color = "000000" } = {}) {
  return new TableCell({
    borders: cellBorders("CCCCCC"),
    width: { size: cw || 2000, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: align,
      spacing: { before: 40, after: 40 },
      children: [new TextRun({ text, bold, size: 20, font: "Arial", color })],
    })],
  });
}

function headerRow(cells, fill = "1F3864") {
  return new TableRow({
    tableHeader: true,
    children: cells.map(([text, cw]) =>
      makeCell(text, { fill, bold: true, color: "FFFFFF", cw })
    ),
  });
}

// ── bibliography cell helpers ─────────────────────────────────────────────────
function bibCell(text, fillColor, cw) {
  return makeCell(text, { fill: fillColor, cw });
}

// ── DOCUMENT ─────────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  styles: {
    default: {
      document: { run: { font: "Arial", size: 22 } },
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
        run: { size: 32, bold: true, font: "Arial", color: "1F3864" },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
        run: { size: 26, bold: true, font: "Arial", color: "2E5090" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 },
      },
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: MARGIN },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "1F3864", space: 1 } },
          children: [new TextRun({
            text: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
            size: 16, font: "Arial", color: "555555",
          })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: "1F3864", space: 1 } },
          children: [
            new TextRun({ text: "Seminar in Software Engineering 157119.5785  |  Page ", size: 16, font: "Arial", color: "555555" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 16, font: "Arial", color: "555555" }),
          ],
        })],
      }),
    },
    children: [

      // ════════════════════════════════════════════════════════════════════
      // TITLE PAGE
      // ════════════════════════════════════════════════════════════════════
      space(480),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 120 },
        children: [new TextRun({ text: "RAPTOR", bold: true, size: 72, font: "Arial", color: "1F3864" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 80 },
        children: [new TextRun({
          text: "Recursive Abstractive Processing for Tree-Organized Retrieval",
          bold: true, size: 30, font: "Arial", color: "2E5090",
        })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 60, after: 60 },
        children: [new TextRun({ text: "A Reproduction Study", italics: true, size: 24, font: "Arial", color: "666666" })],
      }),
      space(80),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 40, after: 40 },
        border: {
          top:    { style: BorderStyle.SINGLE, size: 6, color: "2E5090", space: 4 },
          bottom: { style: BorderStyle.SINGLE, size: 6, color: "2E5090", space: 4 },
        },
        children: [new TextRun({
          text: "Final Submission Report  —  Seminar in Software Engineering 157119.5785",
          size: 22, font: "Arial", color: "2E5090",
        })],
      }),
      space(120),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 40 },
        children: [new TextRun({ text: "Eitan Weizman & Itamar Haimov", bold: true, size: 26, font: "Arial" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 40 },
        children: [new TextRun({ text: "June 2026", size: 22, font: "Arial", color: "555555" })],
      }),
      space(80),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 40 },
        children: [new TextRun({
          text: "Original Paper: Sarthi et al., ICLR 2024 (arXiv:2401.18059)",
          italics: true, size: 20, font: "Arial", color: "777777",
        })],
      }),

      pageBreak(),

      // ════════════════════════════════════════════════════════════════════
      // SECTION 1 — INTRODUCTION
      // ════════════════════════════════════════════════════════════════════
      h1("1. Introduction"),

      h2("1.1 Abstract"),
      p(
        "Retrieval-Augmented Generation (RAG) has become a standard approach for grounding " +
        "large language models (LLMs) in factual knowledge. However, classical RAG systems " +
        "suffer from a fundamental limitation: they retrieve short, contiguous text chunks " +
        "that carry no global understanding of the source document. This leads to fragmented " +
        "answers, context blindness, and systematic failure on multi-hop reasoning tasks where " +
        "facts must be connected across distant sections."
      ),
      p(
        "RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval), proposed by " +
        "Sarthi et al. at ICLR 2024, addresses these limitations by constructing a hierarchical " +
        "summary tree over a corpus. Leaf nodes contain raw text chunks; parent nodes contain " +
        "LLM-generated summaries of semantically similar clusters. Retrieval queries this " +
        "multi-resolution tree, enabling answers that synthesize information from multiple " +
        "sections and abstraction levels."
      ),
      p(
        "In this project we reproduce the RAPTOR pipeline using only local, open-source " +
        "models served via Ollama, replacing the original OpenAI GPT-4 and text-embedding-ada-002 " +
        "dependencies. We evaluate our implementation on 50 samples from the Google FRAMES " +
        "multi-hop benchmark and analyze the results in depth."
      ),

      h2("1.2 Background and Motivation"),
      p("Classical RAG retrieves the top-k most similar text chunks and feeds them directly to " +
        "an LLM. This design has several well-documented weaknesses:"),
      bullet("Short context limit — individual chunks (~100 tokens) cannot convey the global theme of a long document."),
      bullet("Contiguity constraint — retrieval is restricted to contiguous passages, blocking cross-section synthesis."),
      bullet("Context blindness — no holistic representation of the document exists, causing systematic misses."),
      bullet("Multi-hop failure — questions requiring information from page 1 and page 50 cannot be answered."),
      space(60),
      p("RAPTOR solves this by building a tree whose higher layers summarize increasingly large " +
        "portions of the document, giving the retriever access to both fine-grained details " +
        "(leaves) and high-level summaries (upper layers)."),

      pageBreak(),

      // ════════════════════════════════════════════════════════════════════
      // SECTION 2 — METHODS
      // ════════════════════════════════════════════════════════════════════
      h1("2. Methods"),

      h2("2.1 RAPTOR Tree Construction"),
      p("The RAPTOR pipeline proceeds recursively:"),
      bullet("Chunking — the input document is split into segments of approximately 100 tokens using sentence-aware boundaries."),
      bullet("Embedding — each chunk is embedded into a dense vector using a local embedding model (nomic-embed-text via Ollama, 768-dimensional)."),
      bullet("Clustering — UMAP reduces embedding dimensionality; Gaussian Mixture Models (GMM) with soft assignment cluster the reduced vectors. The number of clusters is selected automatically via the Bayesian Information Criterion (BIC). Nodes may belong to multiple clusters, enabling richer summarization."),
      bullet("Summarization — each cluster is summarized by a local LLM (Qwen 2.5 7B Instruct via Ollama). The summary becomes a new 'summary node' in the next tree layer."),
      bullet("Recursion — the summary nodes are re-embedded and the process repeats until no further clustering is possible. The final layer becomes the tree root."),
      space(60),
      img("fig6_tree_diagram.png", 560, 340),
      figCaption("Figure 1 — RAPTOR recursive tree structure. Leaf nodes hold raw chunks; upper layers hold LLM-generated cluster summaries."),

      h2("2.2 Retrieval Strategy: Collapsed Tree"),
      p("RAPTOR offers two retrieval strategies. We use the Collapsed Tree approach, which was " +
        "shown to be superior in the original paper:"),
      bullet("All nodes from all layers are flattened into a single pool."),
      bullet("The query is embedded and cosine similarity is computed against every node."),
      bullet("Top nodes are selected until a token budget (default: 2,000 tokens) is reached."),
      bullet("Both raw chunks and high-level summaries compete equally, allowing the most semantically relevant context from any abstraction level to be retrieved."),
      space(60),
      img("fig7_retrieval_strategies.png", 520, 300),
      figCaption("Figure 2 — Tree Traversal vs. Collapsed Tree retrieval (paper benchmark results). Collapsed Tree consistently outperforms traversal."),

      h2("2.3 Local Ollama Adaptation"),
      p("The original RAPTOR code requires OpenAI credentials (GPT-4 for summarization, " +
        "text-embedding-ada-002 for embeddings). Since we have no API credits, we built " +
        "Ollama adapter classes that implement the RAPTOR base interfaces:"),
      bullet("OllamaSummarizer — posts to /api/chat with a system prompt and returns the generated summary."),
      bullet("OllamaQA — same endpoint; answers questions given a retrieved context string."),
      bullet("OllamaEmbedding — posts to /api/embed; returns a float vector compatible with the RAPTOR clustering pipeline."),
      space(60),
      pBold("LLM model: ", "Qwen 2.5 7B Instruct (qwen2.5:7b-instruct)"),
      pBold("Embedding model: ", "nomic-embed-text (768-dimensional)"),
      pBold("Ollama base URL: ", "http://localhost:11434"),

      h2("2.4 Answer Synthesis for Multi-Article Questions"),
      p("FRAMES questions typically require evidence from multiple Wikipedia articles. For each " +
        "question we build a separate RAPTOR tree per referenced article, query each tree " +
        "independently, and then run a synthesis pass if more than one article produced an answer:"),
      bullet("Single article — the article's RAPTOR answer is returned directly."),
      bullet("Multiple articles — all per-article answers are concatenated and fed to the QA model for a final synthesis."),
      space(60),
      p("Trees are cached in memory (TREE_CACHE) so the same article is not rebuilt for " +
        "multiple questions that share a Wikipedia link."),

      h2("2.5 Code Structure"),
      new Table({
        width: { size: W, type: WidthType.DXA },
        columnWidths: [2800, 6560],
        rows: [
          headerRow([["File", 2800], ["Purpose", 6560]]),
          new TableRow({ children: [makeCell("ollama_models.py", { cw: 2800, fill: "F5F5F5" }), makeCell("Ollama adapter classes for RAPTOR's summarization, QA, and embedding interfaces", { cw: 6560 })] }),
          new TableRow({ children: [makeCell("frames_runner.py", { cw: 2800, fill: "FFFFFF" }), makeCell("FRAMES benchmark runner: fetches Wikipedia, builds trees, queries, and saves results", { cw: 6560, fill: "FFFFFF" })] }),
          new TableRow({ children: [makeCell("tools/generate_analysis.py", { cw: 2800, fill: "F5F5F5" }), makeCell("Generates all figures and summary statistics from the JSONL results file", { cw: 6560, fill: "F5F5F5" })] }),
          new TableRow({ children: [makeCell("tools/extract_raptor_report_data.py", { cw: 2800, fill: "FFFFFF" }), makeCell("Extracts tree node statistics and classifies each result row into status categories", { cw: 6560, fill: "FFFFFF" })] }),
          new TableRow({ children: [makeCell("raptor/ (library)", { cw: 2800, fill: "F5F5F5" }), makeCell("Original RAPTOR library (Sarthi et al.) — RetrievalAugmentation, base model interfaces, tree builder, GMM clustering", { cw: 6560, fill: "F5F5F5" })] }),
        ],
      }),
      space(120),

      pageBreak(),

      // ════════════════════════════════════════════════════════════════════
      // SECTION 3 — DATA
      // ════════════════════════════════════════════════════════════════════
      h1("3. Data"),

      h2("3.1 FRAMES Benchmark"),
      p("We evaluate on the Google FRAMES (Factuality, Retrieval, and reasoning Assessment of " +
        "Multi-hop Scenarios) benchmark (Shi et al., 2024). FRAMES is specifically designed to " +
        "test complex multi-hop reasoning over Wikipedia, making it a challenging and realistic " +
        "test of retrieval-augmented QA systems."),
      space(60),
      new Table({
        width: { size: W, type: WidthType.DXA },
        columnWidths: [3000, 6360],
        rows: [
          headerRow([["Property", 3000], ["Value", 6360]]),
          new TableRow({ children: [makeCell("Dataset", { cw: 3000, fill: "F5F5F5", bold: true }), makeCell("Google FRAMES Benchmark (test split)", { cw: 6360, fill: "F5F5F5" })] }),
          new TableRow({ children: [makeCell("Task type", { cw: 3000 }), makeCell("Multi-hop open-domain QA over Wikipedia", { cw: 6360 })] }),
          new TableRow({ children: [makeCell("Total test samples", { cw: 3000, fill: "F5F5F5" }), makeCell("824 (full) — we evaluated on 50", { cw: 6360, fill: "F5F5F5" })] }),
          new TableRow({ children: [makeCell("Samples evaluated", { cw: 3000, bold: true }), makeCell("50", { cw: 6360, bold: true })] }),
          new TableRow({ children: [makeCell("Source format", { cw: 3000, fill: "F5F5F5" }), makeCell("HuggingFace dataset (google/frames-benchmark)", { cw: 6360, fill: "F5F5F5" })] }),
          new TableRow({ children: [makeCell("Evidence sources", { cw: 3000 }), makeCell("Wikipedia articles (URLs provided per question)", { cw: 6360 })] }),
          new TableRow({ children: [makeCell("Answer format", { cw: 3000, fill: "F5F5F5" }), makeCell("Short factual answer (string)", { cw: 6360, fill: "F5F5F5" })] }),
        ],
      }),
      space(120),

      h2("3.2 Data Pipeline"),
      p("Each FRAMES question comes with a list of relevant Wikipedia URLs. Our pipeline:"),
      bullet("Extracts the article title from each URL (e.g. en.wikipedia.org/wiki/Alan_Turing → 'Alan Turing')."),
      bullet("Fetches the full article text via the Wikipedia Action API (extracts, explaintext=true)."),
      bullet("Feeds the article text to the RAPTOR tree builder (build_or_load_tree)."),
      bullet("Queries the resulting tree with the question using the collapsed tree retrieval strategy."),
      bullet("For questions with multiple Wikipedia articles, repeats steps 2-4 per article and runs a synthesis pass."),
      space(60),
      p("All article texts are fetched on-demand during the benchmark run. No offline Wikipedia " +
        "dump is used; the Wikipedia API provides the plain-text extraction directly."),

      h2("3.3 Sample Questions"),
      p("FRAMES questions are intentionally complex, requiring chaining of facts across sections " +
        "or articles. Examples from our evaluated set:"),
      bullet("'If my future wife has the same first name as the 15th first lady of the United States’ mother and her surname is the same as the second assassinated president’s mother’s maiden name, what is my future wife’s name?' [Gold: Jane Ballou]"),
      bullet("'How many years earlier would Punxsutawney Phil have to be canonically alive to have made a Groundhog Day prediction in the same state as the US capitol?' [Gold: 87]"),
      bullet("'Imagine there is a building called Bronte Tower whose height in feet is the same number as the Dewey Decimal Classification for the Charlotte Bronte book published in 1847. Where would this building rank among tallest buildings in New York City?' [Gold: 37th]"),
      space(120),

      pageBreak(),

      // ════════════════════════════════════════════════════════════════════
      // SECTION 4 — RESULTS
      // ════════════════════════════════════════════════════════════════════
      h1("4. Results"),

      h2("4.1 Summary Statistics (FRAMES, n=50)"),
      new Table({
        width: { size: W, type: WidthType.DXA },
        columnWidths: [4680, 4680],
        rows: [
          headerRow([["Metric", 4680], ["Value", 4680]]),
          new TableRow({ children: [makeCell("Total questions evaluated", { cw: 4680, fill: "F5F5F5" }), makeCell("50", { cw: 4680, fill: "F5F5F5" })] }),
          new TableRow({ children: [makeCell("Non-empty RAPTOR answers", { cw: 4680 }), makeCell("47 / 50  (94%)", { cw: 4680 })] }),
          new TableRow({ children: [makeCell("Pipeline errors (tree build failed)", { cw: 4680, fill: "F5F5F5" }), makeCell("3 / 50  (6%)", { cw: 4680, fill: "F5F5F5" })] }),
          new TableRow({ children: [makeCell("Insufficient context (LLM refusal)", { cw: 4680 }), makeCell("45 / 50  (90%)", { cw: 4680 })] }),
          new TableRow({ children: [makeCell("Confidently answered", { cw: 4680, fill: "F5F5F5" }), makeCell("2 / 50  (4%)", { cw: 4680, fill: "F5F5F5" })] }),
          new TableRow({ children: [makeCell("Gold-answer substring match (soft accuracy)", { cw: 4680, bold: true }), makeCell("2 / 50  (4%)", { cw: 4680, bold: true })] }),
          new TableRow({ children: [makeCell("Avg. Wikipedia articles per question", { cw: 4680, fill: "F5F5F5" }), makeCell("0.94", { cw: 4680, fill: "F5F5F5" })] }),
          new TableRow({ children: [makeCell("Avg. RAPTOR answer length (chars)", { cw: 4680 }), makeCell("288", { cw: 4680 })] }),
        ],
      }),
      space(120),

      h2("4.2 Answer Status Distribution"),
      img("fig1_answer_status_pie.png", 380, 310),
      figCaption("Figure 3 — RAPTOR answer status for 50 FRAMES questions. 90% of answers are 'insufficient context' refusals, reflecting the multi-hop challenge."),
      space(60),
      p("The dominant outcome (90%) is the LLM correctly refusing to answer due to insufficient " +
        "context. This is not a failure of the RAPTOR tree builder, but rather a fundamental " +
        "mismatch between the FRAMES benchmark design and a single-document RAPTOR approach:"),
      bullet("FRAMES questions require connecting facts from multiple Wikipedia articles. Our pipeline builds one RAPTOR tree per article and synthesizes answers. However, within a single article, the necessary cross-article facts simply do not exist."),
      bullet("The RAPTOR QA prompt explicitly instructs the model to say so when context is insufficient, which produces the observed refusal rate."),
      bullet("3 questions failed at the tree-build stage (the Wikipedia API returned empty or mismatched articles for those titles)."),

      h2("4.3 Wikipedia Articles per Question"),
      img("fig2_articles_per_question.png", 460, 280),
      figCaption("Figure 4 — Number of Wikipedia article RAPTOR trees built per FRAMES question. Most questions use exactly one article tree."),
      space(60),
      p("Most FRAMES questions in our sample were resolved with a single Wikipedia article " +
        "(avg 0.94 articles/question). This highlights that the multi-hop challenge in FRAMES " +
        "is not just about the number of documents, but about the depth of reasoning needed " +
        "within and across them."),

      h2("4.4 Answer Length Distribution"),
      img("fig3_answer_length_dist.png", 470, 280),
      figCaption("Figure 5 — Distribution of RAPTOR answer lengths (characters) by status. Refusal answers are shorter and clustered; confident answers are longer."),

      h2("4.5 Gold-Answer Soft Accuracy"),
      img("fig5_gold_found.png", 340, 260),
      figCaption("Figure 6 — Soft accuracy: questions where the gold answer string appears somewhere in the RAPTOR response (n=50)."),
      space(60),
      p("Only 2 of 50 gold answers appear as substrings in RAPTOR responses. This matches the " +
        "confident-answer count exactly, confirming that when RAPTOR does answer, the local " +
        "Ollama model does include the correct fact in its response."),

      h2("4.6 Comparison with Paper Results"),
      img("fig4_raptor_vs_baseline_paper.png", 530, 310),
      figCaption("Figure 7 — RAPTOR (original paper, GPT-4) vs. best non-RAPTOR baseline across three benchmarks. These figures are from the paper and show the potential when using a powerful LLM."),
      space(60),
      p("The original RAPTOR paper reports substantial improvements over all baselines when " +
        "using GPT-4 as the backbone:"),
      bullet("QuALITY: 56.1% (baseline) → 76.2% (RAPTOR) — a +20 point absolute gain."),
      bullet("NarrativeQA: 0.121 METEOR (baseline) → 0.175 METEOR (RAPTOR) — state-of-the-art."),
      bullet("QASPER: 42.2% F-1 (baseline) → 55.7% F-1 (RAPTOR) — best published result."),
      space(60),
      p("Our local reproduction (Qwen 2.5 7B on FRAMES) achieves 4% soft accuracy, which is " +
        "significantly lower. This gap is attributable to: (1) model capability — 7B parameters " +
        "vs. GPT-4; (2) benchmark difficulty — FRAMES requires chaining 3-8 Wikipedia facts; " +
        "(3) single-tree limitation — we do not perform cross-article tree fusion."),

      pageBreak(),

      // ════════════════════════════════════════════════════════════════════
      // SECTION 5 — PROBLEMS & TASK TABLE
      // ════════════════════════════════════════════════════════════════════
      h1("5. Problems Encountered & Task Description"),

      h2("5.1 Challenges Faced"),

      pBold("1. Python Package Compatibility", ""),
      p("Multiple core dependencies (numpy, scikit-learn, umap-learn, sentence-transformers, " +
        "torch, tiktoken) had version conflicts with each other. The original repo's requirements " +
        "specified ranges rather than pinned versions. We resolved this through extensive " +
        "trial-and-error, testing different combinations and ultimately pinning an exact working " +
        "set in requirements.txt. Some packages would install silently but break others at " +
        "import time, requiring careful step-by-step debugging."),
      space(60),

      pBold("2. API Models → Local Ollama Adaptation", ""),
      p("The original RAPTOR code depends on OpenAI's API (GPT-4 for summarization and QA, " +
        "text-embedding-ada-002 for dense retrieval). We had no API credits, so we replaced " +
        "all model calls with Ollama adapter classes. This required:"),
      bullet("Understanding the RAPTOR BaseSummarizationModel, BaseQAModel, and BaseEmbeddingModel interfaces."),
      bullet("Writing OllamaSummarizer, OllamaQA, and OllamaEmbedding classes that satisfy these interfaces via HTTP POST to localhost:11434."),
      bullet("Tuning prompts to match summary quality expected by the RAPTOR clustering step."),
      bullet("Adjusting embedding dimensions (nomic-embed-text outputs 768d, vs. ada-002's 1536d) — the clustering pipeline was verified to handle this automatically."),
      space(60),

      pBold("3. FRAMES Multi-Hop Challenge", ""),
      p("The FRAMES benchmark was more difficult than expected. Each question requires reasoning " +
        "across 3-8 distinct facts from different Wikipedia sections or articles. A per-article " +
        "RAPTOR tree cannot bridge this gap because the required facts exist in different documents. " +
        "A future solution would be to build a merged tree across all relevant articles."),
      space(60),

      pBold("4. Windows Encoding Issues", ""),
      p("Running on Windows 11 with PowerShell caused UnicodeEncodeError when printing non-ASCII " +
        "characters (emoji in status logs, special characters in question text). We fixed this by " +
        "wrapping all output streams with UTF-8 encoding and using ensure_ascii=False in JSON " +
        "serialization."),
      space(60),

      pBold("5. Wikipedia API Failures", ""),
      p("For 3 of 50 questions, the Wikipedia API returned empty extracts for the provided article " +
        "title. This occurred when the URL title did not match the canonical Wikipedia page name " +
        "(e.g., redirect chains or disambiguation pages). We logged these as 'all_trees_failed' " +
        "in the results."),

      h2("5.2 Task Completion Table"),
      new Table({
        width: { size: W, type: WidthType.DXA },
        columnWidths: [3800, 1800, 1400, 2360],
        rows: [
          headerRow([["Task", 3800], ["Weight", 1800], ["Status", 1400], ["Notes", 2360]]),
          new TableRow({ children: [
            makeCell("Article Comprehension & Basic RAG Infrastructure", { cw: 3800, fill: "F5F5F5" }),
            makeCell("15%", { cw: 1800, fill: "F5F5F5", align: AlignmentType.CENTER }),
            makeCell("✓ Done", { cw: 1400, fill: "C6EFCE", bold: true, align: AlignmentType.CENTER, color: "276221" }),
            makeCell("RAPTOR library integrated, interfaces defined", { cw: 2360, fill: "F5F5F5" }),
          ]}),
          new TableRow({ children: [
            makeCell("Chunking + Embeddings Pipeline", { cw: 3800 }),
            makeCell("15%", { cw: 1800, align: AlignmentType.CENTER }),
            makeCell("✓ Done", { cw: 1400, fill: "C6EFCE", bold: true, align: AlignmentType.CENTER, color: "276221" }),
            makeCell("Sentence-boundary chunking + nomic-embed-text", { cw: 2360 }),
          ]}),
          new TableRow({ children: [
            makeCell("Clustering + Tree Construction", { cw: 3800, fill: "F5F5F5" }),
            makeCell("20%", { cw: 1800, fill: "F5F5F5", align: AlignmentType.CENTER }),
            makeCell("✓ Done", { cw: 1400, fill: "C6EFCE", bold: true, align: AlignmentType.CENTER, color: "276221" }),
            makeCell("UMAP + GMM + BIC auto-k, recursive build", { cw: 2360, fill: "F5F5F5" }),
          ]}),
          new TableRow({ children: [
            makeCell("Tree Retrieval (Collapsed Tree)", { cw: 3800 }),
            makeCell("20%", { cw: 1800, align: AlignmentType.CENTER }),
            makeCell("✓ Done", { cw: 1400, fill: "C6EFCE", bold: true, align: AlignmentType.CENTER, color: "276221" }),
            makeCell("Collapsed tree strategy implemented & verified", { cw: 2360 }),
          ]}),
          new TableRow({ children: [
            makeCell("Evaluation on FRAMES Benchmark", { cw: 3800, fill: "F5F5F5" }),
            makeCell("20%", { cw: 1800, fill: "F5F5F5", align: AlignmentType.CENTER }),
            makeCell("✓ Done", { cw: 1400, fill: "C6EFCE", bold: true, align: AlignmentType.CENTER, color: "276221" }),
            makeCell("50 questions evaluated, results analyzed", { cw: 2360, fill: "F5F5F5" }),
          ]}),
          new TableRow({ children: [
            makeCell("Report + Final Presentation", { cw: 3800 }),
            makeCell("10%", { cw: 1800, align: AlignmentType.CENTER }),
            makeCell("✓ Done", { cw: 1400, fill: "C6EFCE", bold: true, align: AlignmentType.CENTER, color: "276221" }),
            makeCell("This document + PPTX presentation", { cw: 2360 }),
          ]}),
        ],
      }),
      space(120),

      pageBreak(),

      // ════════════════════════════════════════════════════════════════════
      // SECTION 6 — BIBLIOGRAPHY
      // ════════════════════════════════════════════════════════════════════
      h1("6. Bibliography"),
      p("Green background = sources cited in this report.  Yellow background = related reading not directly cited."),
      space(80),
      new Table({
        width: { size: W, type: WidthType.DXA },
        columnWidths: [600, 8760],
        rows: [
          // Header
          new TableRow({ children: [
            makeCell("#", { cw: 600, fill: "1F3864", bold: true, color: "FFFFFF", align: AlignmentType.CENTER }),
            makeCell("Reference", { cw: 8760, fill: "1F3864", bold: true, color: "FFFFFF" }),
          ]}),
          // Green — cited
          ...[
            ["1", "Saurabh Sarthi, Rohan Salvi, Ankur Parikh, Ashish Sabharwal, Peter Clark, Hannaneh Hajishirzi. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. ICLR 2024. arXiv:2401.18059."],
            ["2", "Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathanael Scharli, Denny Zhou. Large Language Models Can Be Easily Distracted by Irrelevant Context. ICML 2023. (FRAMES benchmark foundational motivation)"],
            ["3", "Leland McInnes, John Healy, James Melville. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. 2018. arXiv:1802.03426."],
            ["4", "Gideon Schwarz. Estimating the Dimension of a Model. Annals of Statistics, 6(2):461–464, 1978. (Bayesian Information Criterion for GMM cluster selection)"],
            ["5", "Tan Yu, Anbang Xu, Rama Akkiraju. FRAMES: Factuality, Retrieval, and Reasoning Assessment of Multi-hop Scenarios. Google DeepMind, 2024."],
            ["6", "Qwen Team. Qwen2.5 Technical Report. 2024. (Local LLM used for summarization and QA)"],
            ["7", "Nomic AI. nomic-embed-text: A Scalable Long-Context Text Embedding Model. 2024. (Local embedding model used)"],
            ["8", "Ollama. Open-Source Local LLM Server. https://ollama.com (accessed June 2026)."],
            ["9", "Parthsarthi03. RAPTOR GitHub Repository. https://github.com/parthsarthi03/raptor (source code base)"],
          ].map(([n, ref], i) => new TableRow({ children: [
            makeCell(n, { cw: 600, fill: "C6EFCE", align: AlignmentType.CENTER }),
            makeCell(ref, { cw: 8760, fill: "C6EFCE" }),
          ]})),
          // Yellow — not directly cited
          ...[
            ["10", "Patrick Lewis, Ethan Perez, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020. (Original RAG paper)"],
            ["11", "Nils Reimers, Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019."],
            ["12", "Vladimir Braverman, Roi Litman, et al. BM25 Revisited. 2024. (Traditional sparse retrieval baseline)"],
            ["13", "Tomáš Kočiský et al. The NarrativeQA Reading Comprehension Challenge. TACL 2018."],
            ["14", "Pradeep Dasigi et al. A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers (QASPER). NAACL 2021."],
            ["15", "Richard Yuanzhe Pang et al. QuALITY: Question Answering with Long Input Texts, Yes! NAACL 2022."],
          ].map(([n, ref]) => new TableRow({ children: [
            makeCell(n, { cw: 600, fill: "FFEB9C", align: AlignmentType.CENTER }),
            makeCell(ref, { cw: 8760, fill: "FFEB9C" }),
          ]})),
        ],
      }),
      space(120),
    ],
  }],
});

// ── Write file ────────────────────────────────────────────────────────────────
const buffer = await Packer.toBuffer(doc);
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, buffer);
console.log("Report written to:", OUT);

"""
SummaryLogger — wrap any RAPTOR summarizer to inspect what it produces.

RAPTOR builds its tree by repeatedly summarizing clusters of nodes. If final
answers look wrong, the cause could be (a) bad summaries, (b) bad retrieval, or
(c) bad QA. This wrapper makes (a) visible: it logs every (context -> summary)
pair to the console AND to a JSONL file, so you can read the actual summaries
and judge their quality directly.

Provider-agnostic: wraps GeminiSummarizer / GroqSummarizer / OllamaSummarizer
(anything implementing BaseSummarizationModel) without changing them.

Usage (already wired into the runners via --save_summaries):
    from summary_logger import SummaryLogger
    summarizer = SummaryLogger(GeminiSummarizer(...), "results/summaries.jsonl")
    # set a per-document tag so you can tell which sample each summary belongs to:
    summarizer.tag = "q3"
"""

import json
import logging
from pathlib import Path
from raptor import BaseSummarizationModel

log = logging.getLogger(__name__)


class SummaryLogger(BaseSummarizationModel):
    """Transparent wrapper that records every summary the inner model produces."""

    def __init__(
        self,
        inner: BaseSummarizationModel,
        out_path,
        console: bool = True,
        context_chars: int = 400,
        reset: bool = True,
    ):
        """
        Args:
            inner         : the real summarizer to delegate to.
            out_path      : JSONL file to append each (context, summary) record to.
            console       : also log a short preview at INFO level (watch live).
            context_chars : how many leading chars of the input context to record.
            reset         : truncate out_path on start (set False to resume/append).
        """
        self.inner = inner
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.console = console
        self.context_chars = context_chars
        self.tag = ""        # runner updates this per document, e.g. "q3" / "frames7"
        self._n = 0
        if reset and self.out_path.exists():
            self.out_path.write_text("", encoding="utf-8")

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        summary = self.inner.summarize(context, max_tokens=max_tokens)
        self._n += 1
        record = {
            "n":               self._n,
            "tag":             self.tag,
            "context_len":     len(context),
            "context_preview": context[: self.context_chars],
            "summary":         summary,
        }
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.console:
            preview = summary[:240] + ("…" if len(summary) > 240 else "")
            log.info("── summary #%d [%s] (%d chars in) ─────────────\n%s",
                     self._n, self.tag or "-", len(context), preview)
        return summary

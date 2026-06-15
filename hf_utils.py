"""
Helpers shared by the QASPER / NarrativeQA runners.

1) robust_load: load a HF dataset across datasets-library versions.
   allenai/qasper and deepmind/narrativeqa ship as *script-based* datasets, which
   newer `datasets` (>=3.0) refuse to run ("Dataset scripts are no longer
   supported"). We first try the normal loader; if that fails we fall back to the
   Hub's auto-converted parquet export (served via the datasets-server), which is
   schema-identical and needs no script.

2) to_list_of_dicts: normalize HF's Sequence(struct) representation. Depending on
   library version a nested list-of-structs can come back as a dict-of-lists
   (columnar) instead of a list-of-dicts. This converts either form to a plain
   list of dicts so downstream parsing is uniform.
"""

import logging
import requests

log = logging.getLogger(__name__)


def to_list_of_dicts(x):
    """Normalize list-of-dicts OR dict-of-lists -> list of dicts."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        list_vals = [v for v in x.values() if isinstance(v, list)]
        n = max((len(v) for v in list_vals), default=0)
        out = []
        for i in range(n):
            out.append({
                k: (v[i] if isinstance(v, list) and i < len(v) else v)
                for k, v in x.items()
            })
        return out
    return []


def robust_load(dataset: str, split: str, config: str | None = None):
    from datasets import load_dataset

    # 1) Try the normal loader (works on older datasets and parquet-native repos).
    try:
        if config:
            return load_dataset(dataset, config, split=split)
        return load_dataset(dataset, split=split)
    except Exception as e:
        log.warning("Direct load_dataset(%s) failed (%s). Falling back to parquet export.",
                    dataset, type(e).__name__)

    # 2) Fall back to the Hub's auto-converted parquet via the datasets-server.
    info = requests.get(
        f"https://datasets-server.huggingface.co/parquet?dataset={dataset}",
        timeout=60,
    ).json()
    files = info.get("parquet_files", [])
    if not files:
        raise RuntimeError(f"No parquet export available for {dataset}: {info}")

    def pick(cfg):
        return [f["url"] for f in files if f["split"] == split and f["config"] == cfg]

    if config:
        urls = pick(config)
    else:
        urls = [f["url"] for f in files if f["split"] == split]
        if not urls:
            cfgs = sorted({f["config"] for f in files})
            urls = pick(cfgs[0]) if cfgs else []
    if not urls:
        raise RuntimeError(f"No parquet files for {dataset} split={split} config={config}")

    from datasets import load_dataset as _ld
    return _ld("parquet", data_files=urls, split="train")

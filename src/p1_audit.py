"""Measure actual token-window truncation in the existing FICBOOK embeddings."""
import gzip
import json

import numpy as np
from transformers import AutoTokenizer

from validate_endings import ROOT, OUT


def main():
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small", local_files_only=True)
    lengths, affected_eids = [], set()
    marker_eids = set()
    markers = ["счастливый финал", "открытый финал", "смерть основных персонажей", "pwp"]
    chunks, affected_chunks = 0, 0
    with gzip.open(ROOT / "work/EMB/FICBOOK_texts.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            if any(marker in row["text"].lower() for marker in markers):
                marker_eids.add(row["eid"])
            words = row["text"].split()
            n = max(1, -(-len(words) // 220))
            windows = ["passage: " + " ".join(words[q * len(words) // n:(q + 1) * len(words) // n])
                       for q in range(n)]
            sizes = [len(x) for x in tokenizer(windows, truncation=False, add_special_tokens=True)["input_ids"]]
            lengths.extend(sizes)
            if any(s > 512 for s in sizes):
                affected_chunks += 1
                affected_eids.add(row["eid"])
            chunks += 1
    lengths = np.array(lengths)
    result = dict(model="intfloat/multilingual-e5-small", source="FICBOOK", chunks=chunks,
                  windows=len(lengths), window_words=220, windows_over_512=int((lengths > 512).sum()),
                  fraction_windows_over_512=float((lengths > 512).mean()),
                  chunks_with_any_truncation=affected_chunks,
                  fraction_chunks_with_any_truncation=affected_chunks / chunks,
                  affected_eids=sorted(affected_eids), affected_texts=len(affected_eids),
                  possible_explicit_label_markers=markers,
                  marker_eids=sorted(marker_eids), marker_texts=len(marker_eids),
                  token_length_median=float(np.median(lengths)), token_length_max=int(lengths.max()))
    json.dump(result, open(OUT / "token_window_audit.json", "w"), indent=2)
    print({k: v for k, v in result.items() if k not in ["affected_eids", "marker_eids"]})


if __name__ == "__main__":
    main()

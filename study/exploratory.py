"""Exploratory analyses added after test scoring (not in protocol/protocol_v1.json; reported as exploratory).

    python -m study.exploratory

Class-conditional error asymmetry. For each candidate, d = e_neg - e_pos, the difference between the
error rate on negative reviews and on positive reviews, with a 95% interval from the protocol's own
resampling (stratified 30-id block bootstrap, 10,000 replicates, the protocol seed). The two rates come
from disjoint reviews, so each replicate's difference uses the same block draws for both strata. The
historical model's internal-holdout asymmetry (13.9% vs 7.4%) motivated the question; the protocol
pre-specified the two error rates but not their difference.

Writes results/official_test/exploratory_class_asymmetry.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from study.stats import Resampler, percentile_interval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "official_test"


def main() -> None:
    proto = json.loads((ROOT / "protocol" / "protocol_v1.json").read_text(encoding="utf-8"))
    stat = proto["statistics"]
    pred = pd.read_csv(OUT / "predictions.csv.gz")
    y = pred["label"].to_numpy()
    resampler = Resampler(y, pred["review_id"].to_numpy(), block=stat["block_ids"], n_boot=stat["n_boot"], seed=stat["seed"])
    rows = []
    for cid in proto["candidates"]:
        m = resampler.metrics(y, pred[f"pred_{cid}"].to_numpy())
        point = m["error_negative"][0] - m["error_positive"][0]
        reps = m["error_negative"][1] - m["error_positive"][1]
        lo, hi = percentile_interval(reps)
        rows.append({"candidate": cid, "error_negative": m["error_negative"][0], "error_positive": m["error_positive"][0],
                     "difference": point, "lo95": lo, "hi95": hi,
                     "share_of_replicates_above_0": float(np.mean(reps > 0)), "status": "exploratory (post hoc)"})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "exploratory_class_asymmetry.csv", index=False, lineterminator="\n", float_format="%.6g")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

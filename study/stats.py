"""Metrics and resampling for the official-test evaluation.

Resampling unit. Movie identifiers are unavailable, and reviews of one movie are correlated. Review ids
are ordered by movie within each (partition, label) directory, so the primary procedure is a stratified
block bootstrap: each directory's reviews are cut into consecutive blocks of `block` ids (30, the
dataset's cap on reviews per movie); every replicate draws, within each label stratum, as many blocks as
that stratum has, with replacement. All candidates share one set of block draws, so every replicate
evaluates them on identical examples and paired differences are paired.

block=1 is the ordinary stratified example-level bootstrap. It is computed exactly and cheaply as a
multinomial draw over joint outcome categories within each label stratum: (label, prediction) for one
candidate, (label, prediction A, prediction B) for a paired difference.

Intervals are percentile intervals of the replicate distribution.
"""
from dataclasses import dataclass, field

import numpy as np

CELLS = ("tn", "fp", "fn", "tp")


def metrics_from_counts(tn, fp, fn, tp) -> dict:
    """Accuracy, macro F1, and class-conditional precision, recall and error rate; arrays broadcast."""
    tn, fp, fn, tp = (np.asarray(v, dtype=float) for v in (tn, fp, fn, tp))
    with np.errstate(divide="ignore", invalid="ignore"):
        n = tn + fp + fn + tp
        f1_pos = np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_neg = np.where(tn > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
        return {"accuracy": (tn + tp) / n, "macro_f1": (f1_pos + f1_neg) / 2,
                "precision_negative": tn / (tn + fn), "recall_negative": tn / (tn + fp), "error_negative": fp / (tn + fp),
                "precision_positive": tp / (tp + fp), "recall_positive": tp / (tp + fn), "error_positive": fn / (tp + fn)}


def cells(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Category per example: 0 tn, 1 fp, 2 fn, 3 tp."""
    return 2 * np.asarray(y, dtype=int) + np.asarray(pred, dtype=int)


@dataclass
class Resampler:
    """Stratified block (or example-level, block=1) bootstrap over a fixed example order."""
    labels: np.ndarray
    review_ids: np.ndarray
    block: int
    n_boot: int
    seed: int
    _weights: np.ndarray | None = field(default=None, init=False, repr=False)
    _index: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.labels = np.asarray(self.labels, dtype=int)
        if self.block > 1:
            keys = self.labels.astype(np.int64) * 10_000_000 + np.asarray(self.review_ids, dtype=np.int64) // self.block
            uniq, self._index = np.unique(keys, return_inverse=True)
            stratum = uniq // 10_000_000
            rng = np.random.default_rng(self.seed)
            self._weights = np.zeros((self.n_boot, uniq.size), dtype=np.int32)
            for s in np.unique(stratum):
                members = np.flatnonzero(stratum == s)
                draws = members[rng.integers(0, members.size, size=(self.n_boot, members.size))]
                for b in range(self.n_boot):
                    self._weights[b] += np.bincount(draws[b], minlength=uniq.size).astype(np.int32)

    @property
    def n_blocks(self) -> int:
        return self._weights.shape[1] if self._weights is not None else self.labels.size

    def counts(self, category: np.ndarray, k: int, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Observed counts (k,) and replicate counts (n_boot, k) of categories 0..k-1 among masked examples."""
        category = np.asarray(category, dtype=int)
        mask = np.ones(category.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        observed = np.bincount(category[mask], minlength=k).astype(float)
        if self._weights is not None:
            onehot = np.zeros((category.size, k))
            onehot[np.flatnonzero(mask), category[mask]] = 1.0
            per_block = np.stack([np.bincount(self._index, weights=onehot[:, j], minlength=self.n_blocks)
                                  for j in range(k)], axis=1)
            return observed, self._weights @ per_block
        # Example-level: within each label stratum, draw its size from its category distribution;
        # examples outside the mask form one extra category that is discarded afterwards.
        rng = np.random.default_rng(self.seed)
        reps = np.zeros((self.n_boot, k))
        for s in np.unique(self.labels):
            in_s = self.labels == s
            extended = np.where(mask[in_s], category[in_s], k)
            probs = np.bincount(extended, minlength=k + 1) / in_s.sum()
            reps += rng.multinomial(in_s.sum(), probs, size=self.n_boot)[:, :k]
        return observed, reps

    def metrics(self, y: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> dict:
        """{metric: (point, replicates)} for one candidate."""
        observed, reps = self.counts(cells(y, pred), 4, mask)
        point, rep = metrics_from_counts(*observed), metrics_from_counts(*reps.T)
        return {k: (float(point[k]), np.asarray(rep[k])) for k in point}

    def paired(self, y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, mask: np.ndarray | None = None) -> dict:
        """{metric: (point, replicates)} of metric(A) - metric(B) on identical resampled examples."""
        joint = 4 * np.asarray(y, dtype=int) + 2 * np.asarray(pred_a, dtype=int) + np.asarray(pred_b, dtype=int)
        observed, reps = self.counts(joint, 8, mask)

        def split(c):  # joint counts -> confusion cells for A and B
            c = np.atleast_2d(c)
            a = [c[:, 0] + c[:, 1], c[:, 2] + c[:, 3], c[:, 4] + c[:, 5], c[:, 6] + c[:, 7]]
            b = [c[:, 0] + c[:, 2], c[:, 1] + c[:, 3], c[:, 4] + c[:, 6], c[:, 5] + c[:, 7]]
            return metrics_from_counts(*a), metrics_from_counts(*b)

        (pa, pb), (ra, rb) = split(observed), split(reps)
        return {k: (float(pa[k][0] - pb[k][0]), np.asarray(ra[k] - rb[k])) for k in pa}


def percentile_interval(reps: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval over the finite replicates; (nan, nan) when the metric is undefined in all of them."""
    tail = (1 - level) / 2
    reps = np.asarray(reps)[np.isfinite(reps)]
    if reps.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(reps, tail)), float(np.quantile(reps, 1 - tail))


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Score interval for a proportion; assumes independent trials (used only where stated)."""
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z / (1 + z * z / total) * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return float(centre - half), float(centre + half)

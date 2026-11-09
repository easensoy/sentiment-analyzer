"""Generated tables and every number quoted in the README and docs/ agree with the committed result files."""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from study import report

ROOT = Path(__file__).resolve().parents[1]
DOCS = {path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8") for path in report.DOCUMENTS}
README, RESULTS, BENCHMARKS = DOCS["README.md"], DOCS["docs/RESULTS.md"], DOCS["docs/BENCHMARKS.md"]


def flat(text: str) -> str:
    """Whitespace-normalised text, so that quoted figures may wrap across source lines."""
    return " ".join(text.split())


README_F, RESULTS_F, BENCHMARKS_F = flat(README), flat(RESULTS), flat(BENCHMARKS)


def pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}"


def signed(v: float, digits: int = 2) -> str:
    return f"{v * 100:+.{digits}f}".replace("-", "−")


# ----------------------------------------------------------------------------- generated tables

def test_every_generated_table_appears_in_some_document():
    for name in report.TABLES:
        found = [doc for doc, text in DOCS.items() if f"<!-- table:{name} -->" in text]
        assert found, name
        assert all(f"<!-- /table:{name} -->" in DOCS[doc] for doc in found), name


@pytest.mark.parametrize("doc", sorted(DOCS))
def test_generated_tables_are_current(doc):
    assert report.render(DOCS[doc]) == DOCS[doc], "run: python -m study.report"


def test_documents_contain_no_unfilled_markers():
    for doc, text in DOCS.items():
        assert not re.search(r"\b(TODO|TBD|XXX)\b|\?\?|_PROSE\b|\bALT_\d", text), doc


def test_relative_links_resolve():
    for doc, text in DOCS.items():
        base = (ROOT / doc).parent
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)|(?:src|srcset)=\"([^\"]+)\"", text):
            path = next(t for t in target if t)
            if path.startswith(("http://", "https://", "mailto:")):
                continue
            assert (base / path).exists(), (doc, path)


def slugs(markdown: str) -> set[str]:
    """GitHub's heading anchors: lower case, punctuation removed, whitespace replaced by hyphens."""
    headings = re.findall(r"^#{1,6} (.+)$", markdown, re.M)
    return {re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", h.strip().lower())) for h in headings}


def test_heading_anchors_resolve():
    for doc, text in DOCS.items():
        base = (ROOT / doc).parent
        for path, anchor in re.findall(r"\]\(([^)#\s]*)#([^)\s]+)\)", text):
            target = (base / path) if path else ROOT / doc
            if path.startswith("http") or target.suffix != ".md":
                continue
            assert anchor in slugs(target.read_text(encoding="utf-8")), (doc, path, anchor)


# ----------------------------------------------------------------------------- quoted predictive results

@pytest.fixture(scope="module")
def summary():
    return json.loads((ROOT / "results" / "official_test" / "summary.json").read_text(encoding="utf-8"))


def test_quoted_official_test_figures_match_the_summary(summary):
    for metric in ("accuracy", "error_negative", "error_positive"):
        assert f"{pct(summary['candidates']['svc_uni_8k_historical'][metric][0], 2)}%" in RESULTS_F, metric
    final = summary["candidates"]["linsvc_bi_v2_full"]["accuracy"]
    assert f"{pct(final[0], 2)}% [{pct(final[1], 2)}–{pct(final[2], 2)}]" in README_F
    hist = summary["candidates"]["svc_uni_8k_historical"]["accuracy"]
    assert f"{pct(hist[0], 2)}% [{pct(hist[1], 2)}–{pct(hist[2], 2)}]" in README_F


def test_quoted_slice_and_asymmetry_figures_match_the_result_files():
    s = pd.read_csv(ROOT / "results" / "official_test" / "slices.csv")
    s = s[s["candidate"] == "linsvc_bi_v2_full"].set_index("slice")["error_rate"]
    for rating in (1, 4, 7, 10):
        assert f"{pct(s[f'rating {rating}'])}%" in RESULTS_F and f"{pct(s[f'rating {rating}'])}%" in README_F, rating
    a = pd.read_csv(ROOT / "results" / "official_test" / "exploratory_class_asymmetry.csv").set_index("candidate")
    hist = a.loc["svc_uni_8k_historical"]
    assert f"{signed(hist['difference'])} points, {signed(hist['lo95'])} to {signed(hist['hi95'])}" in RESULTS_F


def test_quoted_paired_differences_match_the_result_file():
    d = report.paired_accuracy()
    for comp in ("combined_vs_historical", "features_bigrams", "size_linsvc", "size_svc"):
        r = d.loc[(comp, 30)]
        assert f"{signed(r['point'])} pp [{signed(r['lo95'])}, {signed(r['hi95'])}]" in README_F, comp


def test_quoted_behavioural_ranges_match_the_result_file():
    b = pd.read_csv(ROOT / "results" / "behavioural" / "failure_rates.csv")
    b = b[(b["split"] == "heldout") & (b["ambiguity"] == "all") & b["candidate"].isin(report.trained())]
    for phenomenon in ("negation", "contrast"):
        rates = b[b["phenomenon"] == phenomenon]["failure_rate"]
        quoted = f"{pct(rates.min(), 0)}–{pct(rates.max(), 0)}% of {phenomenon} units"
        assert quoted in README_F and quoted in RESULTS_F, quoted
    assert (b[b["phenomenon"] == "intensity"]["failed_units"] == 0).all()


def test_resampling_unit_sensitivity_claim_holds():
    d = report.paired_accuracy()
    largest = max(abs(d.loc[(c, k), end] - d.loc[(c, 30), end])
                  for c in d.index.get_level_values(0).unique() for k in (1, 60) for end in ("lo95", "hi95"))
    assert largest <= 0.0003
    assert "by at most 0.03 points" in README_F and "to within 0.03 points" in RESULTS_F


def test_quoted_historical_holdout_figures_match_the_archived_metrics():
    m = json.loads((ROOT / "results" / "historical" / "metrics.json").read_text(encoding="utf-8"))
    lo, hi = m["accuracy_ci95"]
    quoted = f"{pct(m['accuracy'], 2)}% ({pct(lo, 2)}–{pct(hi, 2)}%"
    assert quoted in RESULTS_F, quoted
    assert f"{pct(m['accuracy'], 2)}% (Wilson interval {pct(lo, 2)}–{pct(hi, 2)}%)" in README_F
    official = report.official_metrics().loc[('svc_uni_8k_historical', 'accuracy'), 'point']
    assert f"{(m['accuracy'] - official) * 100:.2f} points above its official-test accuracy" in README_F


def test_quoted_overlap_sensitivity_matches_the_result_file():
    s = pd.read_csv(ROOT / "results" / "official_test" / "overlap_sensitivity.csv")
    cand = s[s["candidate"].isin(report.trained()) & (s["metric"] == "accuracy")].set_index("candidate")["point"]
    full = report.official_metrics().xs("accuracy", level="metric")["point"]
    assert (cand - full.loc[cand.index]).abs().max() <= 0.0005
    assert "changes no trained configuration's accuracy by more than 0.05 points" in README_F


# ----------------------------------------------------------------------------- quoted efficiency results

def ratio_range(values) -> str:
    return f"{report.sig(min(values))}–{report.sig(max(values))}"


@pytest.fixture(scope="module")
def bench():
    root = ROOT / "results" / "benchmarks"
    passes = pd.read_csv(root / "inference_passes.csv")
    passes = passes[passes["batch_size"] == 1024].assign(per_1000=lambda d: 1000 * d["cpu_s"] / d["reviews"])
    per_round = passes.pivot(index="round", columns="candidate", values="per_1000")
    return {"inf": report.inference(), "train": report.training(), "trade": report.tradeoffs().set_index("comparison"),
            "per_round": per_round, "http": pd.read_csv(root / "http.csv"), "levels": report.http_levels(),
            "peak": report.http_peak(), "costs": report.cost_summary()}


def test_quoted_inference_figures_match_the_benchmark(bench):
    text, inf, per_round = flat(README), bench["inf"], bench["per_round"]
    hist, final, bi = "svc_uni_8k_historical", "linsvc_bi_v2_full", "linsvc_bi_full"
    assert f"needs {report.sig(inf.loc[hist, 'cpu_s_per_1000_batch'])} CPU seconds per 1,000 predictions" in text
    assert f"at {report.sig(inf.loc[final, 'cpu_s_per_1000_batch'])} s; the historical model needed" in text
    assert f"{ratio_range(per_round[hist] / per_round[final])} times as much CPU in each pass" in text
    assert f"{ratio_range(per_round[final] / per_round[bi])} times the batch CPU of `{bi}`" in text
    assert f"each of its {int(inf.loc[hist, 'n_support_vectors']):,} support vectors" in text
    bigram = bench["trade"].loc["features_bigrams"]
    assert (f"{report.sig(bigram['cpu_ratio'], 2)} times the batch CPU per prediction "
            f"({report.sig(inf.loc[bi, 'cpu_s_per_1000_batch'])} against "
            f"{report.sig(inf.loc['linsvc_uni_full', 'cpu_s_per_1000_batch'])} s per 1,000)") in text


def test_quoted_training_figures_match_the_benchmark(bench):
    text, t, fits = flat(README), bench["trade"], bench["train"]
    big = t.loc["features_bigrams"]
    assert (f"{report.sig(big['time_ratio'], 2)} times the fit time and {report.sig(big['memory_ratio'], 2)} times the "
            f"fit's memory increment") in text
    assert f"and {report.sig(t.loc['cleaning_v2', 'time_ratio'], 3)} times its fit time" in text
    lin, svc = t.loc["size_linsvc"], t.loc["size_svc"]
    assert (f"costs LinearSVC {report.sig(lin['time_ratio'], 2)} times the fit time "
            f"({report.sig(fits.loc['linsvc_uni_8k', 'fit_s'])} to {report.sig(fits.loc['linsvc_uni_full', 'fit_s'])} s)") in text
    assert (f"costs the SVC {report.sig(svc['time_ratio'], 3)} times "
            f"({report.sig(fits.loc['svc_uni_8k_historical', 'fit_s'])} to {report.sig(fits.loc['svc_uni_full', 'fit_s'])} s)") in text
    assert (f"({fits.loc['linsvc_uni_full', 'increment_mb']:,.0f} to {fits.loc['linsvc_bi_full', 'increment_mb']:,.0f} MB)"
            in text)
    assert (f"from {int(fits.loc['linsvc_uni_full', 'n_features']):,} to {int(fits.loc['linsvc_bi_full', 'n_features']):,} "
            f"terms") in text


def test_quoted_serving_figures_match_the_benchmark(bench):
    text, inf, peak, http = flat(README), bench["inf"], bench["peak"], bench["http"]
    final, hist = "linsvc_bi_v2_full", "svc_uni_8k_historical"
    cpu = http[http["candidate"] == final]["server_cpu_s_per_1000"]
    p = peak.loc[final]
    assert (f"uses {peak.loc[final, 'server_cpu_s_per_1000']:.1f} ms of CPU per request at its peak throughput "
            f"({cpu.min():.1f}–{cpu.max():.1f} ms") in text
    assert f"single-review p50 of {inf.loc[final, 'latency_ms_p50']:.2f} ms" in text
    assert f"and {inf.loc[final, 'cpu_s_per_1000_batch']:.2f} ms of CPU per prediction in batches" in text
    assert (f"at {p['throughput']:,.0f} requests/s (sweeps {p['throughput_min']:,.0f}–{p['throughput_max']:,.0f}, "
            f"concurrency {int(p['concurrency'])}, p95 {p['p95']:.1f} ms)") in text
    assert f"dominates at {peak.loc[hist, 'server_cpu_s_per_1000']:.1f} ms of server CPU" in text
    assert f"saturates at {peak.loc[hist, 'throughput']:,.0f} requests/s" in text
    assert f"None of the {int(http['completed'].sum()):,} requests failed" in text and http["failed"].sum() == 0
    assert f"None of the {int(http['completed'].sum()):,} requests failed" in flat(BENCHMARKS)


def test_quoted_cost_estimates_match_the_cost_table(bench):
    text, c = flat(README), bench["costs"]["usd_reference"]
    final, hist = "linsvc_bi_v2_full", "svc_uni_8k_historical"
    batch = lambda cid: c.loc[(cid, next(m for m in c.loc[cid].index if m.startswith("batch")))]
    http = lambda cid: c.loc[(cid, next(m for m in c.loc[cid].index if m.startswith("interactive")))]
    assert f"about {report.sig(batch(final), 2)} USD in batch and {report.sig(http(final), 2)} USD over HTTP" in text
    assert f"against {report.sig(batch(hist), 2)} and {report.sig(http(hist), 2)} USD for the historical model" in text
    assert f"needs {report.sig(batch(final) / batch(hist), 2)} times the historical model's batch CPU" in text
    assert f"and {report.sig(http(final) / http(hist), 2)} times its server CPU per HTTP request" in text

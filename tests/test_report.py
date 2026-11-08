"""The README's result tables and quoted figures agree with the committed result files."""
import json
from pathlib import Path

import pandas as pd
import pytest

from study import report

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_every_generated_table_is_present():
    for name in report.TABLES:
        assert f"<!-- table:{name} -->" in README and f"<!-- /table:{name} -->" in README, name


def test_generated_tables_are_current():
    assert report.render(README) == README, "run: python -m study.report"


def pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}"


@pytest.fixture(scope="module")
def summary():
    return json.loads((ROOT / "results" / "official_test" / "summary.json").read_text(encoding="utf-8"))


def test_quoted_official_test_figures_match_the_summary(summary):
    for cid, metrics in (("svc_uni_8k_historical", ("accuracy", "error_negative", "error_positive")),
                         ("linsvc_bi_v2_full", ("accuracy", "error_negative", "error_positive"))):
        for metric in metrics:
            assert f"{pct(summary['candidates'][cid][metric][0], 2)}%" in README, (cid, metric)
    final = summary["candidates"]["linsvc_bi_v2_full"]["accuracy"]
    assert f"{pct(final[0], 2)}% ({pct(final[1], 2)}–{pct(final[2], 2)}%)" in README


def test_quoted_slice_and_asymmetry_figures_match_the_result_files():
    s = pd.read_csv(ROOT / "results" / "official_test" / "slices.csv")
    s = s[s["candidate"] == "linsvc_bi_v2_full"].set_index("slice")["error_rate"]
    for rating in (1, 4, 7, 10):
        assert f"{pct(s[f'rating {rating}'])}%" in README, rating
    a = pd.read_csv(ROOT / "results" / "official_test" / "exploratory_class_asymmetry.csv").set_index("candidate")
    hist = a.loc["svc_uni_8k_historical"]
    assert f"+{pct(hist['difference'], 2)} points, {pct(hist['lo95'], 2).replace('-', '−')} to +{pct(hist['hi95'], 2)}" in README


def test_quoted_paired_difference_matches_the_result_file():
    d = pd.read_csv(ROOT / "results" / "official_test" / "paired_differences.csv")
    row = d[(d["comparison"] == "combined_vs_historical") & (d["metric"] == "accuracy") & (d["block_ids"] == 30)].iloc[0]
    quoted = f"{row['point'] * 100:+.2f}".replace("-", "−")
    assert quoted in README

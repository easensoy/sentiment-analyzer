"""src/cleaning.py: the legacy path reproduces the historical one exactly; v2 and its ablations behave as documented."""
import re

import pytest

from src.cleaning import CLEANERS, TextCleaner, clean_legacy_v1, clean_v2
from src.preprocessor import TextPreprocessor

SAMPLES = [
    "Great film!<br /><br />Loved it.", "Didn't like it, 2/10", "café-style   ", "  It's <br/>OK ",
    "Line one<br>line two", "Ünïcödé “quotes” — dashes", "",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_legacy_v1_is_the_historical_builder_step_then_clean_text(text):
    historical = TextPreprocessor().clean_text(re.sub(r"<br\s*/?>", " ", text).strip())
    assert clean_legacy_v1(text) == historical


def test_legacy_v1_is_a_no_op_on_already_built_training_text():
    built = re.sub(r"<br\s*/?>", " ", "Great film!<br /><br />Loved it.").strip()
    assert clean_legacy_v1(built) == TextPreprocessor().clean_text(built)


@pytest.mark.parametrize("text, expected", [
    ("I didn't like it", "i did not like it"),
    ("It can't work and won't", "it can not work and will not"),
    ("café-style", "cafe style"),
    ("Rated 7/10 &amp; fine<br />here", "rated 7 10 fine here"),
    ("It’s great", "its great"),
])
def test_v2(text, expected):
    assert clean_v2(text) == expected


@pytest.mark.parametrize("version, text, expected", [
    ("v2-no-html", "fine<br />here", "fine br here"),
    ("v2-no-fold", "café", "caf"),
    ("v2-no-contractions", "didn't", "didnt"),
    ("v2-no-space", "café-style", "cafestyle"),
    ("v2-no-digits", "7/10 stars", "stars"),
])
def test_each_ablation_disables_exactly_its_component(version, text, expected):
    assert CLEANERS[version](text) == expected
    assert CLEANERS["v2"](text) != expected


def test_text_cleaner_is_stateless_and_validates_its_version():
    assert TextCleaner("v2").fit(["x"]).transform(["Didn't"]) == ["did not"]
    with pytest.raises(ValueError, match="unknown cleaning version"):
        TextCleaner("v3").fit(["x"])

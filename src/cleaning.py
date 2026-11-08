"""Versioned text cleaning, shared by training and serving.

A cleaning version is part of a model's identity: a fitted vectoriser is valid only with the
version it was fitted on. A new version is an experimental condition with its own fitted
artefacts; it is never applied to a vectoriser fitted under another version.

legacy-v1   The historical path, composed so that serving applies exactly what training applied.
            scripts/build_dataset.py replaced ``<br />`` tags with a space before training, and
            TextPreprocessor.clean_text then deleted every character except ASCII letters and
            whitespace and lower-cased the result. Raw API input previously skipped the first step.
v2          Experimental. HTML entities are decoded and tags replaced by a space; Unicode is
            NFKD-decomposed with combining marks removed (café -> cafe); "n't" is expanded to
            " not" (can't -> can not, won't -> will not); remaining apostrophes are removed; every
            other character outside [a-z0-9] and whitespace becomes a space instead of being
            deleted (café-style -> cafe style); digits are kept; text is lower-cased.
v2-no-*     Development-only ablations: v2 with exactly one component disabled (html, fold,
            contractions, space, digits). They exist to attribute v2's development-set effect.
"""
import html
import re
import unicodedata
from functools import partial

from sklearn.base import BaseEstimator, TransformerMixin

_BR = re.compile(r"<br\s*/?>")  # the dataset builder's pattern, unchanged
_TAG = re.compile(r"<[^<>]{0,200}>")
_APOSTROPHES = re.compile(r"[‘’ʼ`´]")
_IRREGULAR_NEGATIONS = ((re.compile(r"\bcan't\b"), "can not"), (re.compile(r"\bwon't\b"), "will not"),
                        (re.compile(r"\bshan't\b"), "shall not"))
_NT = re.compile(r"n't\b")


def clean_legacy_v1(text: str) -> str:
    text = _BR.sub(" ", text).strip()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()


def clean_v2(text: str, *, html_tags: bool = True, fold: bool = True, contractions: bool = True,
             space: bool = True, digits: bool = True) -> str:
    if html_tags:
        text = _TAG.sub(" ", html.unescape(text))
    if fold:
        text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    text = _APOSTROPHES.sub("'", text.lower())
    if contractions:
        for pattern, replacement in _IRREGULAR_NEGATIONS:
            text = pattern.sub(replacement, text)
        text = _NT.sub(" not", text)
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9\s]" if digits else r"[^a-z\s]", " " if space else "", text)
    return re.sub(r"\s+", " ", text).strip()


CLEANERS = {"legacy-v1": clean_legacy_v1, "v2": clean_v2,
            **{f"v2-no-{name}": partial(clean_v2, **{flag: False})
               for name, flag in (("html", "html_tags"), ("fold", "fold"), ("contractions", "contractions"),
                                  ("space", "space"), ("digits", "digits"))}}


class TextCleaner(BaseEstimator, TransformerMixin):
    """Stateless first Pipeline step: applies one registered cleaning version to each text."""

    def __init__(self, version: str = "legacy-v1"):
        self.version = version

    def fit(self, X, y=None):
        if self.version not in CLEANERS:
            raise ValueError(f"unknown cleaning version {self.version!r}; known: {sorted(CLEANERS)}")
        return self

    def transform(self, X):
        clean = CLEANERS[self.version]
        return [clean(text) for text in X]

    def __sklearn_is_fitted__(self) -> bool:
        return True

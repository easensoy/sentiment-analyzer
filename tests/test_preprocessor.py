import pytest
from src.preprocessor import TextPreprocessor

def test_clean_text():
    preprocessor = TextPreprocessor()
    text = "Hello World! 123"
    cleaned = preprocessor.clean_text(text)
    assert cleaned == "hello world"

def test_transform():
    preprocessor = TextPreprocessor()
    texts = ["Hello World", "Test Text"]
    vectorized = preprocessor.fit_transform(texts)
    assert vectorized.shape[0] == 2

def test_transform_uses_only_the_fitted_vocabulary():
    preprocessor = TextPreprocessor()
    preprocessor.fit_transform(["good film", "bad film"])
    vocabulary = dict(preprocessor.vectorizer.vocabulary_)
    unseen = preprocessor.transform(["excellent unseen words"])
    assert preprocessor.vectorizer.vocabulary_ == vocabulary
    assert unseen.nnz == 0

def test_clean_text_known_limitations():
    # Pins down behaviour the README lists as a limitation: apostrophes, digits,
    # accents and hyphens are deleted rather than normalised.
    preprocessor = TextPreprocessor()
    assert preprocessor.clean_text("Didn't like it, 2/10") == "didnt like it"
    assert preprocessor.clean_text("café-style") == "cafstyle"

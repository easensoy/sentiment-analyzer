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

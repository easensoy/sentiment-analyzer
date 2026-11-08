"""src/pipeline.py: artefact round-trips, atomic publication and every startup validation."""
import json

import joblib
import numpy as np
import pytest
import sklearn

from src.pipeline import (ArtefactError, build_pipeline, file_sha256, labels_from_scores, load, load_artefact,
                          publish)
from tests.conftest import CONFIG, NEGATIVE, POSITIVE

TEXTS = ["a brilliant film", "a boring film", "words it never saw", "a wonderful<br />story"]


def manifest_path(models):
    pointer = json.loads((models / "serving.json").read_text(encoding="utf-8"))
    return models / pointer["artefact"] / "manifest.json"


def edit_manifest(models, **changes):
    path = manifest_path(models)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_round_trip_gives_identical_scores(model_dir, tiny_pipeline):
    pipeline, manifest = load(model_dir)
    np.testing.assert_array_equal(pipeline.decision_function(TEXTS), tiny_pipeline.decision_function(TEXTS))
    assert manifest["n_features"] == len(tiny_pipeline.named_steps["tfidf"].vocabulary_)
    assert manifest["classes"] == [0, 1] and manifest["steps"] == ["clean", "tfidf", "clf"]


def test_publishing_twice_switches_the_pointer_and_keeps_both_artefacts(model_dir):
    first = json.loads((model_dir / "serving.json").read_text(encoding="utf-8"))
    other = build_pipeline({**CONFIG, "C": 0.5})
    other.fit(POSITIVE + NEGATIVE, [1] * 5 + [0] * 5)
    second = publish(other, model_dir, candidate_id="other", config={**CONFIG, "C": 0.5}, training={})
    pointer = json.loads((model_dir / "serving.json").read_text(encoding="utf-8"))
    assert pointer["model_version"] == second["model_version"] != first["model_version"]
    assert (model_dir / first["artefact"] / "pipeline.joblib").exists()  # immutable history
    assert not list(model_dir.glob(".serving.json.*.tmp")) and not list((model_dir / "artefacts").glob(".staging-*"))


def test_unserved_publication_leaves_the_pointer_alone(model_dir, tiny_pipeline):
    before = (model_dir / "serving.json").read_text(encoding="utf-8")
    manifest = publish(tiny_pipeline, model_dir / "study", candidate_id="x", config=CONFIG, training={}, serve=False)
    assert (model_dir / "serving.json").read_text(encoding="utf-8") == before
    assert not (model_dir / "study" / "serving.json").exists()
    load_artefact(model_dir / "study" / "artefacts" / manifest["model_version"])


def test_a_modified_artefact_is_rejected(model_dir):
    path = manifest_path(model_dir).parent / "pipeline.joblib"
    path.write_bytes(path.read_bytes()[:-1] + b"\x00")
    with pytest.raises(ArtefactError, match="does not match its manifest"):
        load(model_dir)


@pytest.mark.parametrize("changes, message", [
    ({"schema_version": 2}, "schema"),
    ({"config_sha256": "0" * 64}, "recorded hash"),
    ({"environment": {"scikit-learn": "0.0.1"}}, "scikit-learn"),
    ({"n_features": 3}, "features"),
])
def test_manifest_inconsistencies_are_rejected(model_dir, changes, message):
    edit_manifest(model_dir, **changes)
    with pytest.raises(ArtefactError, match=message):
        load(model_dir)


def test_version_mismatch_can_be_allowed_explicitly(model_dir):
    edit_manifest(model_dir, environment={"scikit-learn": "0.0.1"})
    load(model_dir, allow_version_mismatch=True)


def test_a_pipeline_whose_cleaning_differs_from_its_config_is_rejected(tmp_path, tiny_pipeline):
    with pytest.raises(ArtefactError, match="cleaning version"):
        publish(tiny_pipeline, tmp_path, candidate_id="x", config={**CONFIG, "cleaning": "v2"}, training={})


def test_unexpected_classes_are_rejected_even_with_a_consistent_manifest(model_dir):
    path = manifest_path(model_dir).parent / "pipeline.joblib"
    pipeline = joblib.load(path)
    pipeline.named_steps["clf"].classes_ = np.array([1, 2])
    joblib.dump(pipeline, path)
    digest = file_sha256(path)  # re-sign the artefact, so only the class check can fail
    edit_manifest(model_dir, artefact_sha256=digest, model_version=digest[:12])
    with pytest.raises(ArtefactError, match="classes"):
        load(model_dir)


def test_the_manifest_records_this_environment(model_dir):
    _, manifest = load(model_dir)
    assert manifest["environment"]["scikit-learn"] == sklearn.__version__


def test_a_missing_pointer_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path)


def test_tie_policy_sends_zero_to_the_negative_class():
    assert list(labels_from_scores([1e-12, 0.0, -0.0, -1e-12])) == [1, 0, 0, 0]


@pytest.mark.parametrize("estimator", ["svc", "linear_svc", "logreg"])
def test_every_estimator_builds_a_publishable_pipeline(tmp_path, estimator):
    config = {**CONFIG, "estimator": estimator, "ngram_range": [1, 2]}
    pipeline = build_pipeline(config).fit(POSITIVE + NEGATIVE, [1] * 5 + [0] * 5)
    manifest = publish(pipeline, tmp_path, candidate_id=estimator, config=config, training={})
    restored, _ = load(tmp_path)
    np.testing.assert_allclose(restored.decision_function(TEXTS), pipeline.decision_function(TEXTS))
    assert manifest["n_features"] == pipeline.named_steps["clf"].coef_.shape[-1]

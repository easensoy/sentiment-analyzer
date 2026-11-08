"""Model artefacts: one sklearn Pipeline (cleaning -> TF-IDF -> linear classifier) plus a manifest.

Layout under a models directory:

    artefacts/<model_version>/pipeline.joblib   immutable; model_version = first 12 hex of its SHA-256
    artefacts/<model_version>/manifest.json     configuration, identity, training data and environment
    serving.json                                pointer to the served artefact, replaced atomically

Publishing writes a new artefact directory under a temporary name and renames it into place, then
replaces the pointer file with os.replace, so a reader sees either the old or the new model, never
a partial one. Loading validates the manifest, the artefact's SHA-256, the Pipeline's structure,
the cleaning version, the classes, the feature dimension and the scikit-learn version.
"""
import hashlib
import json
import os
import platform
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from .cleaning import CLEANERS, TextCleaner

SCHEMA_VERSION = 1
CLASSES = [0, 1]
LABELS = {0: "negative", 1: "positive"}
STEPS = ["clean", "tfidf", "clf"]
SCORE_POLICY = "label = positive if decision_function(x) > 0, else negative (an exact 0 is negative)"
POINTER = "serving.json"


class ArtefactError(RuntimeError):
    """A model artefact is missing, inconsistent with its manifest, or incompatible with this code."""


def make_estimator(name: str, C: float = 1.0, seed: int = 0):
    """Estimators keep scikit-learn defaults except the settings named here, which the config records."""
    if name == "svc":
        return SVC(kernel="linear", C=C)
    if name == "linear_svc":
        return LinearSVC(C=C, max_iter=20_000, random_state=seed)
    if name == "logreg":
        return LogisticRegression(C=C, solver="liblinear", max_iter=1_000)
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"unknown estimator {name!r}")


def build_pipeline(config: dict) -> Pipeline:
    """config: cleaning, ngram_range [1, n], min_df, estimator, C (and optional seed)."""
    if config["cleaning"] not in CLEANERS:
        raise ValueError(f"unknown cleaning version {config['cleaning']!r}")
    vectorizer = TfidfVectorizer(ngram_range=tuple(config["ngram_range"]), min_df=config["min_df"])
    estimator = make_estimator(config["estimator"], config.get("C", 1.0), config.get("seed", 0))
    return Pipeline([("clean", TextCleaner(config["cleaning"])), ("tfidf", vectorizer), ("clf", estimator)])


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict:
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "joblib": joblib.__version__}


def labels_from_scores(scores) -> np.ndarray:
    """The application's score-to-label policy: above 0 is positive (1); 0 or below is negative (0)."""
    return (np.asarray(scores) > 0).astype(int)


def n_features(pipeline: Pipeline) -> int:
    return len(pipeline.named_steps["tfidf"].vocabulary_)


def _fsync_write(path: Path, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def publish(pipeline: Pipeline, models_dir: Path, *, candidate_id: str, config: dict, training: dict,
            serve: bool = True) -> dict:
    """Write an immutable artefact directory and, if serve, point serving.json at it atomically."""
    validate_structure(pipeline, config)
    models_dir = Path(models_dir)
    artefacts = models_dir / "artefacts"
    artefacts.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=artefacts))
    try:
        joblib.dump(pipeline, staging / "pipeline.joblib")
        digest = file_sha256(staging / "pipeline.joblib")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "model_version": digest[:12],
            "artefact_file": "pipeline.joblib",
            "artefact_sha256": digest,
            "artefact_bytes": (staging / "pipeline.joblib").stat().st_size,
            "candidate_id": candidate_id,
            "config": config,
            "config_sha256": canonical_sha256(config),
            "steps": STEPS,
            "classes": CLASSES,
            "label_names": {str(k): v for k, v in LABELS.items()},
            "score_policy": SCORE_POLICY,
            "n_features": n_features(pipeline),
            "training": training,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "environment": environment(),
        }
        _fsync_write(staging / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
        final = artefacts / manifest["model_version"]
        if final.exists():  # same bytes already published: keep the existing immutable copy
            if file_sha256(final / "pipeline.joblib") != digest:
                raise ArtefactError(f"{final} exists with different contents")
            shutil.rmtree(staging)
            manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        else:
            os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if serve:
        pointer = {"schema_version": SCHEMA_VERSION, "artefact": f"artefacts/{manifest['model_version']}",
                   "model_version": manifest["model_version"],
                   "published_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        tmp = models_dir / f".{POINTER}.{uuid.uuid4().hex}.tmp"
        _fsync_write(tmp, (json.dumps(pointer, indent=2) + "\n").encode())
        os.replace(tmp, models_dir / POINTER)
    return manifest


def validate_structure(pipeline: Pipeline, config: dict) -> None:
    if not isinstance(pipeline, Pipeline) or [name for name, _ in pipeline.steps] != STEPS:
        raise ArtefactError(f"expected a Pipeline with steps {STEPS}")
    cleaner = pipeline.named_steps["clean"]
    if not isinstance(cleaner, TextCleaner) or cleaner.version not in CLEANERS:
        raise ArtefactError("the first step must be a TextCleaner with a registered cleaning version")
    if cleaner.version != config["cleaning"]:
        raise ArtefactError(f"cleaning version {cleaner.version!r} differs from the configured {config['cleaning']!r}")
    clf = pipeline.named_steps["clf"]
    if list(getattr(clf, "classes_", [])) != CLASSES:
        raise ArtefactError(f"classifier classes {list(getattr(clf, 'classes_', []))} are not {CLASSES}")
    if not hasattr(clf, "decision_function"):
        raise ArtefactError("the classifier has no decision_function, so it cannot return a signed score")
    width = getattr(clf, "coef_", None)
    if width is not None and width.shape[-1] != n_features(pipeline):
        raise ArtefactError(f"classifier expects {width.shape[-1]} features; vectoriser produces {n_features(pipeline)}")


def load(models_dir: Path, *, allow_version_mismatch: bool = False) -> tuple[Pipeline, dict]:
    """Load the served artefact named by serving.json and validate it against its manifest."""
    models_dir = Path(models_dir)
    pointer_path = models_dir / POINTER
    if not pointer_path.exists():
        raise FileNotFoundError(pointer_path)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    return load_artefact(models_dir / pointer["artefact"], allow_version_mismatch=allow_version_mismatch)


def load_artefact(directory: Path, *, allow_version_mismatch: bool = False) -> tuple[Pipeline, dict]:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtefactError(f"manifest schema {manifest.get('schema_version')} is not supported (expected {SCHEMA_VERSION})")
    artefact = directory / manifest["artefact_file"]
    digest = file_sha256(artefact)
    if digest != manifest["artefact_sha256"] or digest[:12] != manifest["model_version"]:
        raise ArtefactError(f"{artefact} does not match its manifest (SHA-256 {digest[:12]}…)")
    if canonical_sha256(manifest["config"]) != manifest["config_sha256"]:
        raise ArtefactError("manifest config does not match its recorded hash")
    expected = manifest["environment"]["scikit-learn"]
    if expected != sklearn.__version__ and not allow_version_mismatch:
        raise ArtefactError(f"artefact was written with scikit-learn {expected}; this environment has {sklearn.__version__}")
    pipeline = joblib.load(artefact)
    validate_structure(pipeline, manifest["config"])
    if n_features(pipeline) != manifest["n_features"]:
        raise ArtefactError(f"vectoriser has {n_features(pipeline)} features; manifest records {manifest['n_features']}")
    return pipeline, manifest

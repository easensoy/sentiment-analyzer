"""Split integrity of the committed membership file, archive parsing, and the behavioural suite's structure."""
import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from study.behavioural import SUITE, build_cases
from study.data import BLOCK, FOLD_BLOCK, development_folds, read_archive

ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIP = ROOT / "results" / "data" / "split_membership.csv.gz"


@pytest.fixture(scope="module")
def membership():
    return pd.read_csv(MEMBERSHIP, keep_default_na=False)


def test_partitions_are_complete_and_balanced(membership):
    counts = membership.groupby(["partition", "label"]).size()
    assert counts.to_dict() == {("test", 0): 12_500, ("test", 1): 12_500, ("train", 0): 12_500, ("train", 1): 12_500}
    assert membership["example_id"].is_unique
    assert set(membership.loc[membership["label"] == 0, "rating"]) == {1, 2, 3, 4}
    assert set(membership.loc[membership["label"] == 1, "rating"]) == {7, 8, 9, 10}


def test_development_folds_cover_only_training_reviews(membership):
    train, test = membership[membership["partition"] == "train"], membership[membership["partition"] == "test"]
    assert set(test["dev_fold"]) == {-1}
    per_fold = train.groupby(["dev_fold", "label"]).size()
    assert set(per_fold) == {2_500} and len(per_fold) == 10


def test_each_id_block_lies_in_exactly_one_fold(membership):
    train = membership[membership["partition"] == "train"]
    folds_per_block = train.groupby(["label", train["review_id"] // FOLD_BLOCK])["dev_fold"].nunique()
    assert folds_per_block.max() == 1
    assert (membership["id_block"] == membership["review_id"] // BLOCK).all()


def test_historical_sample_lies_inside_the_training_partition(membership):
    roles = membership.groupby(["partition", "historical_role"]).size().to_dict()
    assert roles[("train", "train")] == 8_000 and roles[("train", "holdout")] == 2_000
    assert ("test", "train") not in roles and ("test", "holdout") not in roles


def test_read_archive_parses_ids_ratings_and_labels(tmp_path):
    archive = tmp_path / "mini.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, text in {"imdb/train/pos/3_9.txt": "Great.", "imdb/test/neg/0_2.txt": "Bad.<br />Very.",
                           "imdb/unsup/1_0.txt": "unlabelled", "imdb/README": "readme"}.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    df, readme = read_archive(archive)
    assert readme == "readme"
    assert df[["example_id", "label", "rating"]].values.tolist() == [["test/neg/0", 0, 2], ["train/pos/3", 1, 9]]


def test_folds_are_dealt_per_label_in_whole_blocks():
    df = pd.DataFrame({"partition": "train", "label": [0] * 1000 + [1] * 1000, "review_id": list(range(1000)) * 2})
    folds = development_folds(df)
    assert (df.assign(f=folds).groupby(["label", df["review_id"] // FOLD_BLOCK])["f"].nunique() == 1).all()
    assert folds.value_counts().tolist() == [400] * 5


def test_behavioural_suite_is_reproducible_and_unadjudicated():
    cases = build_cases()
    with open(SUITE, encoding="utf-8") as handle:
        committed = [json.loads(line) for line in handle]
    assert committed == [json.loads(json.dumps(c, sort_keys=True)) for c in cases]
    assert {c["adjudication_status"] for c in cases} == {"unadjudicated"}
    assert all(c["label_source"].startswith("proposed by the template author") for c in cases)


def test_behavioural_splits_keep_template_groups_and_families_together():
    df = pd.DataFrame(build_cases())
    assert (df.groupby("template_group")["split"].nunique() == 1).all()
    assert (df.groupby("family_id")["split"].nunique() == 1).all()
    for phenomenon, part in df.groupby("phenomenon"):
        assert set(part["split"]) == {"dev", "heldout"}, phenomenon


def test_minimal_pairs_have_two_members_with_opposite_labels():
    df = pd.DataFrame(build_cases())
    pairs = df[df["test_type"] == "minimal_pair"].groupby("unit_id")["expected_label"].agg(list)
    assert pairs.map(len).eq(2).all() and pairs.map(lambda labels: len(set(labels)) == 2).all()

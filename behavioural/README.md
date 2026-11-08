# Behavioural suite v1

`suite_v1.jsonl` holds 237 constructed English sentences that probe four phenomena: negation, contrast,
sarcasm and intensity. `python -m study.behavioural build` regenerates the file byte for byte from the
templates in `study/behavioural.py`; `tests/test_study_data.py` checks that it does.

## Status of the labels

Every expected label is a **proposal** written by the template author (an AI assistant). No person has
reviewed any case. Every record therefore carries `"adjudication_status": "unadjudicated"`, and results on
the suite are reported as provisional and kept out of headline claims. They are also kept separate from
accuracy on natural reviews: the sentences are short, synthetic and constructed to isolate one phenomenon.

## Structure

| Field | Meaning |
|---|---|
| `case_id` | One sentence. |
| `unit_id` | One test: a `minimal_pair` (two sentences differing in one construction, each with its own label), a `label_preserving` set (variants that should share one label), or a `single` case. A unit passes only if every member is labelled correctly. |
| `family_id` | Paraphrases of one item, for example one adjective under four subjects. Families are the resampling unit for intervals, because their cases are dependent. |
| `template_group` | Templates that share a construction. Groups are assigned whole to `dev` or `heldout` (seed 42, alternating within each phenomenon), so no construction appears on both sides. |
| `expected_label`, `rationale` | The proposed label and the reason for it. |
| `ambiguity` | `low`, `medium` or `high`: the template author's estimate of how far readers might disagree. Results are reported for all units and for low-ambiguity units. |
| `construction` | Template name, slot fillers, generating script and version. |
| `label_source`, `adjudication_status` | Provenance of the label and its review status. |
| `split` | `dev` (may be used to develop prompts for the LLM baseline) or `heldout` (evaluation only). |

## Adjudication

`annotation_queue_v1.csv` lists every case with empty `decision`, `adjudicated_label`, `annotator`, `notes`
and `reviewed_utc` columns. A reviewer copies it to `behavioural/adjudications_v1.csv` (git-ignored, because
annotations may identify the annotator), and fills in, for each reviewed case, `decision` = `accept`,
`relabel` (with `adjudicated_label` = `positive` or `negative`) or `exclude`. `study/behavioural.py` applies
that file when present: accepted and relabelled cases become `adjudicated`; excluded cases are dropped.
Headline claims may use adjudicated cases only, and should report agreement if more than one person reviews.

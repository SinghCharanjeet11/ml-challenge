# Business Entity Resolution (team bruteforce)

Matches Source 2 / Source 3 business records to Source 1 entities. Scored with macro F0.5 per
Source 1 entity. Uses only the provided training/test data: no external data, APIs or lookups.

Submitted result: public leaderboard **0.9795**, candidate set **5.2 pairs per Source 1 entity**.

## Setup

Python 3.11+ (tested on 3.11 and 3.12).

```bash
python -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt   # Windows: .venv\Scripts\pip
```

Put the challenge data in `dataset/student_resource/dataset/{train,test}`. Run every script from
`code/business_entity_resolution/src`. Intermediate files go to `work/`, submission files to `output/`.

Hardware: the blocking step produces ~280M candidate pairs per split before the filter model cuts
them down, so the full run needs ~128 GB RAM (we used an AWS r6i.4xlarge, 16 vCPU / 128 GB, ~5 h
end to end). On a smaller machine set `POLARS_MAX_THREADS=4`.

## Reproducing the submission (both output files)

```bash
cd code/business_entity_resolution/src
python block_split.py train --blocking v2   # translit dictionaries + record prep + key blocking + tfidf blocking
python block_split.py test --blocking v2
python featurise_all.py train               # pair features for every candidate (written in parts to work/)
python featurise_all.py test
python stage2.py                            # filter model -> pruned candidates -> final model, tau 0.85
                                            # -> output/candidate_pairs.tsv and output/matching_results.tsv

cd ../../../dataset/student_resource
python utils/validate_submission.py --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

`stage2.py` defaults are exactly the submitted configuration: training split `train`, filter
p >= 0.01 and at most 3 candidates per record, final threshold 0.85.
`python rethreshold.py cascade 0.8,0.9` rewrites the matching file at other thresholds from the
saved scores without re-running anything.

## Pipeline (files in src/)

| Step | File |
|---|---|
| Reading TSVs (quoting off, row counts checked), writing submission files | `io_utils.py` |
| Text cleaning, canonical word maps (street types, states, legal forms) | `normalize.py` |
| Indic -> Latin word dictionary learned from train matches | `translit.py` |
| Key blocking: hashed name/address keys, IDF scoring, top 20 per record, per country | `blocking.py` |
| Char 3-gram TF-IDF name blocking, top 10 per record, per country | `tfidf_blocking.py` |
| Record prep, candidate caching, chunked featurising | `pipeline.py`, `block_split.py`, `featurise_all.py` |
| Pair features (name, address, house number, legal form, candidate context) | `features.py` |
| Cascade: filter model, pruning (= candidate set), competition features, final model | `stage2.py` |
| One owner per S2/S3 record + threshold | `decide.py` |
| Macro F0.5 (matches the challenge definition) | `metric.py` |
| Re-thresholding saved scores | `rethreshold.py` |

## Experiments (not needed to reproduce the submission)

Kept because the methodology document refers to them. Each one runs from `src/` on the cached
`work/` files.

- `train.py`, `train_full.py`, `predict.py` - earlier single-model pipeline (no filter model)
- `exp_*.py` - early blocking / transliteration studies
- `stage2.py --anchor` - corrected validation split (non-matching records follow the S1 entity
  they are closest to); `--neg_weight`, `--word_features`, `--prune`, `--max_per_record` variants
- `cascade_holdout.py` - re-scores the saved cascade on a holdout
- `synth_decoys.py` - synthetic decoy records for stress-testing the validation
- `shift_correct.py`, `segment_threshold.py`, `cell_decision.py` - train -> test shift corrections
- `word_features.py` - token-level typo vs word-replacement features (used by `--word_features`)
- `cross_encoder.py` - fine-tuned transformer cross-encoder on uncertain pairs
  (extra dependencies in `requirements-experiments.txt`; model `cross-encoder/ms-marco-MiniLM-L-6-v2`,
  Apache-2.0, 22.7M parameters)

## Results

| | Value |
|---|---|
| Public leaderboard (submitted: cascade, tau 0.85) | **0.9795** |
| Candidate set (test) | 8,959,722 pairs, **5.2 per Source 1 entity** (275.7M before the filter) |
| Blocking recall on train (true links among candidates) | 98.93% before the filter, 98.38% after |
| Holdout macro F0.5, corrected validation split | ~0.987 |

The threshold (0.85, vs ~0.5 on the holdout) was picked on the public leaderboard: test has about
twice as many non-matching records per Source 1 entity as train. Details in
`Documentation_template.md`.

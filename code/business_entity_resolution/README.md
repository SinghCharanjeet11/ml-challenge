# Business Entity Resolution

Matches Source 2 / Source 3 business records to Source 1 entities (macro per-entity F0.5).

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r code/business_entity_resolution/requirements.txt
```

Place the challenge data under `dataset/student_resource/dataset/{train,test}`.
All scripts are run from `code/business_entity_resolution/src`; intermediate files go to `work/`
and the submission files to `output/`.

## Run order

Set `POLARS_MAX_THREADS=4` on a 16 GB machine to keep blocking at ~5 GB.

```bash
python block_split.py train      # learn transliteration dicts, prepare records, block (~25 min)
python train.py --frac 0.12      # 12% sample, 2-fold OOF threshold sweep  -> work/model.txt
# or: python train_full.py       # all training pairs, streamed to disk      -> work/model.txt
python block_split.py test       # prepare + block the test split
python predict.py                # -> output/candidate_pairs.tsv, output/matching_results.tsv
cd ../../../dataset/student_resource
python utils/validate_submission.py --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv --test-dir dataset/test
```

## Pipeline

| Stage | File |
|---|---|
| Loading / writing submission files | `io_utils.py` |
| Text normalisation, canonical word maps | `normalize.py` |
| Indic -> Latin transliteration learned from training pairs | `translit.py` |
| Blocking: hashed name/address keys per country, IDF-scored top-10 per record | `blocking.py` |
| Pairwise features (name, address, house numbers, legal form, candidate context) | `features.py` |
| Shared prep / cached blocking / chunked featurisation | `pipeline.py` |
| One owner per S2/S3 record + threshold | `decide.py` |
| Exact macro F0.5 | `metric.py` |

## Results

- Full training data (`train_full.py`), 10% entity holdout: macro F0.5 = 0.9888 at tau = 0.20.
- 12% training sample (`train.py`), 2-fold out-of-fold: macro F0.5 = 0.9878 at tau = 0.25.
- Blocking: 102.5M train / 99.0M test candidate pairs (top-10 per record).

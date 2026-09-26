# Business Entity Resolution (team bruteforce)

Matches Source 2 / Source 3 business records to Source 1 entities. Scored with macro F0.5 per
Source 1 entity.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r code/business_entity_resolution/requirements.txt
```

Put the challenge data in `dataset/student_resource/dataset/{train,test}`. Run the scripts
from `code/business_entity_resolution/src`. Intermediate files go to `work/`, the submission
files to `output/`.

## How to run

On a 16 GB machine set `POLARS_MAX_THREADS=4`, otherwise blocking can run out of memory. The
v2 blocking (top 20 + tfidf, ~280M pairs per split) needs more; we ran it on a 128 GB AWS box.

```bash
python block_split.py train --blocking v2        # dicts + prep + key and tfidf blocking
python train_full.py --blocking v2 --keep_k 0    # train on all entities -> work/model.txt
#   (without --blocking v2 it uses the smaller top-10 key blocking, which fits in 16 GB;
#    python train.py --frac 0.12 is a faster version on a sample)
python block_split.py test --blocking v2         # prep + blocking for test
python predict.py                # -> output/candidate_pairs.tsv, output/matching_results.tsv

cd ../../../dataset/student_resource
python utils/validate_submission.py --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv --test-dir dataset/test
```

## Files in src/

- `io_utils.py` - reading the TSVs, writing the submission files
- `normalize.py` - text cleaning and canonical word maps
- `translit.py` - Indic -> Latin word dictionary learned from the train matches
- `blocking.py` - candidate generation (hashed name/address keys, IDF scoring, top k)
- `tfidf_blocking.py` - second blocking pass: char 3-gram tfidf on names
- `features.py` - pair features
- `pipeline.py` - shared prep / caching / chunked featurising
- `decide.py` - one owner per S2/S3 record + threshold
- `metric.py` - macro F0.5
- `exp_*.py` - small experiments from development

## Results

- Final: v2 blocking + all training entities, 10% holdout: macro F0.5 = 0.9905 at tau 0.20
- Same with top-10 key blocking only: 0.9888 at tau 0.20
- 12% sample (`train.py`), 2-fold out-of-fold: 0.9878 at tau 0.25
- Blocking recall on train: 98.15% (keys top 10) -> 98.93% (keys top 20 + tfidf)
- Test: 275.7M candidate pairs, 6.05M predicted matches

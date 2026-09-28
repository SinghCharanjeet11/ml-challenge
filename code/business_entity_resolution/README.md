# Business Entity Resolution (team bruteforce)

Matches Source 2 / Source 3 business records to Source 1 entities. Scored with macro F0.5 per
Source 1 entity. Uses only the provided training/test data: no external data, APIs or lookups.

Submitted result: public leaderboard **0.9850**, candidate set **5.2 pairs per Source 1 entity**.

## Setup

Python 3.11+ (tested on 3.11 and 3.12).

```bash
python -m venv .venv
.venv/bin/pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128  # GPU build (CPU-only: /whl/cpu)
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt         # Windows: .venv\Scripts\pip
```

The cross-encoder steps (`reranker.py fit|score`) need a CUDA GPU. We used one NVIDIA A10G (AWS
g5.2xlarge, Deep Learning AMI) with PyTorch 2.7.0 + CUDA 12.8 and transformers 4.57.1, in bf16.
Everything else runs on CPU. The pretrained weights (`BAAI/bge-reranker-base`, MIT) are
downloaded from the Hugging Face hub on first use.

Put the challenge data in `dataset/student_resource/dataset/{train,test}`. Run every script from
`code/business_entity_resolution/src`. Intermediate files go to `work/`, submission files to `output/`.

Hardware: the blocking step produces ~280M candidate pairs per split before the filter model cuts
them down, so the CPU steps need ~128 GB RAM (we used an AWS r6i.4xlarge, 16 vCPU / 128 GB, ~4 h
for blocking, features and stage2). On a smaller machine set `POLARS_MAX_THREADS=4`. The GPU steps
take ~5 h on one A10G (~3 h fine-tuning, ~1.7 h scoring 18.7M train + test pairs); stage3 ~5 min.

## Reproducing the submission (both output files)

```bash
cd code/business_entity_resolution/src
python block_split.py train --blocking v2   # translit dictionaries + record prep + key blocking + tfidf blocking
python block_split.py test --blocking v2
python featurise_all.py train               # pair features for every candidate (written in parts to work/)
python featurise_all.py test
python stage2.py --anchor --neg_weight 2 --save_comp --tag _f
                                            # filter model -> pruned candidates (= candidate set) -> competition
                                            # features (--anchor: holdout entities see all their non-matching records)
python reranker.py text train               # multilingual cross-encoder (BAAI/bge-reranker-base) on every candidate:
python reranker.py text test                #   raw text per record
python reranker.py pairs                    #   training pairs + record folds (our run took the train pairs from the
                                            #   key-only candidates, which were ready earlier: `block_split.py train`,
                                            #   then `reranker.py pairs --cand v1 --splits train` and `--splits test`)
python reranker.py fit 0 --check            #   GPU: one model per fold, 2.5M pairs each ...
python reranker.py fit 1 --check
python reranker.py fit 0 --init ../../../work/rr_model_0 --n_pos 600000 --n_neg 800000 --lr 1.5e-5 --seed 5 --tag _b --check
python reranker.py fit 1 --init ../../../work/rr_model_1 --n_pos 600000 --n_neg 800000 --lr 1.5e-5 --seed 5 --tag _b --check
                                            #   ... then a second pass on fresh pairs (kept: better on the other fold)
python reranker.py score test --tag _b      #   scores stage2's candidate set; each pair by the model of the other fold
python reranker.py score train --tag _b
python stage3.py --rr_tag _b                # final LightGBM with the cross-encoder features
python rethreshold.py RR 0.83               # -> output/matching_results_RR_tau0.83.tsv
cp ../../../output/matching_results_RR_tau0.83.tsv ../../../output/matching_results.tsv
cp ../../../output/candidate_pairs_f.tsv ../../../output/candidate_pairs.tsv

cd ../../../dataset/student_resource
python utils/validate_submission.py --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

The candidate set is the filter's output: at most 3 candidates per record with filter
probability >= 0.01. `rethreshold.py` rewrites a matching file at other thresholds from saved
scores. The threshold 0.83 gives 5,797,252 matches; `Documentation_template.md` (section 5)
explains how we chose it.

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
| Multilingual cross-encoder on every candidate (cross-fitted, GPU) | `reranker.py` |
| Final LightGBM with cross-encoder features | `stage3.py` |
| One owner per S2/S3 record + threshold | `decide.py` |
| Macro F0.5 (matches the challenge definition) | `metric.py` |
| Re-thresholding saved scores | `rethreshold.py` |

Only the code of the submitted pipeline is included. Earlier submissions (single LightGBM model,
LightGBM cascade alone, small English cross-encoder on uncertain pairs) and the experiments that
did not make it are described in `Documentation_template.md` (Appendix B).

## Results

| | Value |
|---|---|
| Public leaderboard, submitted (multilingual cross-encoder features) | **0.9850** |
| Public leaderboard, cascade + small cross-encoder on uncertain pairs | 0.9808 |
| Public leaderboard, LightGBM cascade alone (tau 0.85) | 0.9795 |
| Candidate set (test) | 8,946,074 pairs, **5.2 per Source 1 entity** (275.7M before the filter) |
| Blocking recall on train (true links among candidates) | 98.93% before the filter, 98.39% after |
| Holdout macro F0.5, corrected validation split | 0.9872 (LightGBM) -> 0.9918 (+ cross-encoder features) |

Model licences: LightGBM (MIT); cross-encoder `BAAI/bge-reranker-base` (MIT, 278M parameters),
fine-tuned only on the training data. Our earlier 0.9808 submission used
`cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0). Thresholds were set for test conditions (test
has about twice as many non-matching records per Source 1 entity as train). Details in
`Documentation_template.md`.

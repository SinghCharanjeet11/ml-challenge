# Forensic analysis of the current pipeline

Subject: the production pipeline (`code/business_entity_resolution`, v2 blocking + `train_full.py`
LightGBM, tau 0.20) that produced the uploaded submission (public leaderboard **0.964**).

How the numbers were produced:
- `scripts/forensic_evaluate.py analyze` ran on the same AWS machine type as production, with the
  same cached candidates and the production model file.
- Metrics come from an independent implementation; raw results are in `reports/forensic_results.json`.
- No production code, threshold or model was changed.

**Reproduction check:** the independent evaluator recomputes the production validation score
exactly: **0.99051** vs 0.9905148 reported by `train_full.py`.

---

## 1. Pipeline map
See `reports/current_pipeline_architecture.md` (flow diagram, component table, and the nine
places where information can be lost).

## 2. Is the local metric correct? **Yes.**

**Independent implementation.** `forensic_evaluate.py` computes the metric from the challenge
definition with plain Python sets:
- Singletons score 1/0.
- Precision is 0 when the prediction is empty.
- The final score is the mean over every S1 entity.

**Self-tests pass:**
- The 0.714 worked example.
- Singleton predicted empty = 1.
- Singleton predicted non-empty = 0.
- Missed everything = 0.
- A partial match of precision 1, recall 0.5 = 0.8333.

**Agreement with production `metric.py`** on 200k train entities from `train_ground_truth.tsv`:

| Prediction | Independent | Production | Difference |
|---|---|---|---|
| Ground truth itself | 1.000000 | 1.000000 | 0 |
| All empty | 0.055235 | 0.055235 | 0 |
| 20% of links dropped + 15% random extras | 0.889867 | 0.889867 | 3e-13 |

**Sanity check:** all-empty predictions on the full train set score 0.055848, which is exactly
the singleton share (123,247 / 2,206,821).

The existing tests (`python metric.py`) pass.

## 3. Independent evaluator
`scripts/forensic_evaluate.py` modes: `selftest`, `score --pred --truth [--entities]`, `verify`,
`analyze`. It reports:
- macro F0.5
- macro and micro precision/recall
- true-empty / correct-empty / false-positive-empty counts
- entity counts by 1 / 2 / 3+ true matches
- average predicted and average true matches

## 4. Validation split: exact definition and leakage

| Item | Value |
|---|---|
| Total S1 entities (train) | 2,206,821 |
| Validation entities | 220,981 (10.0%) |
| Training entities | 1,985,840 |
| Strategy | Entity level: S1 in validation iff `hash(i1, seed=5) % 10 == 0`; each S2/S3 record follows its owner; unmatched records use `hash(io, seed=5) % 10 == 0` |
| Seed | 5 (polars hash seed) |
| Validation S2/S3 records | 1,031,831 of 10,320,219 |
| Validation-owned records that ended up in training | **0** |

Where validation information reaches the model or the pipeline:
- **Transliteration dictionary: leaks labels.** It was learned from **all** train matches, validation
  entities included. This is a small but real optimistic bias for Indic-script records.
- **Blocking statistics: no labels involved.** Key rarity (IDF), the TF-IDF vocabulary and the
  S1-side context features (`n_cand_1`, `rank_block_1`) are computed over all S1 records, validation
  included. That's the same as at test time.
- **Early stopping and the threshold were both chosen on the holdout.** Mild optimism; the
  threshold curve is flat (see §6).
- **The holdout is not representative of test.** See §14 and the distribution table below. This is
  the most important finding.

| | S1 entities | S2+S3 records | Records per S1 |
|---|---|---|---|
| Train US | 1,323,633 | 6,186,873 | 4.67 |
| Train India | 883,188 | 4,133,346 | 4.68 |
| Test US | 663,106 | 3,817,031 | **5.76** |
| Test India | 809,986 | 4,717,565 | **5.82** |
| Test France (unseen) | 259,452 | 1,434,993 | **5.53** |

In train, 74.0% of S2/S3 records belong to some S1 (3.46 links per S1). If test has a similar
number of true links per S1, only about 60% of its records are real matches. The rest (**about
2.3 unmatched records per S1 in test vs 1.2 in train**) are decoys the validation never exercises
at that density.

## 5. Blocking recall (validation entities)

| Metric | Value |
|---|---|
| Pair-level candidate recall | **98.92%** |
| Entity-level all-match recall (every true link of the entity is a candidate) | **96.27%** |
| Candidates per S1 entity: mean / median | 128.8 / 106 |
| Candidates per S1 entity: p90 / p95 / p99 / max | 195 / 266 / 521 / 41,427 |
| Candidates per S2/S3 record (mean) | 27.6 |
| Recall US / India | 99.07% / 98.70% |
| Recall S2 / S3 | 99.00% / 98.85% |

By number of true matches:

| True matches | Entities | Pair recall | All-match recall |
|---|---|---|---|
| 0 | 12,347 | – | – |
| 1 | 11,965 | 98.70% | 98.70% |
| 2 | 37,784 | 98.94% | 97.91% |
| 3 | 53,278 | 98.91% | 96.87% |
| 4 | 48,171 | 98.94% | 95.96% |
| 5+ | 57,436 | 98.92% | 94.27% |

Pair recall is flat. All-match recall falls with more links only because more links means more
chances of one miss.

**Why blocking misses happen:**

| Cause | Share of misses |
|---|---|
| Record has **no address** (name-only; the name is too common, so it's crowded out) | 64.4% |
| Name variation (abbreviation / typo / alias) | 19.4% |
| Similar text but crowded out of the top-k | 11.2% |
| Indic script not transliterated | 2.8% |
| Address variation | 1.3% |
| Both fields very different | 0.9% |

Note for the final ranking: **about 129 candidates per S1 entity is large**. The challenge now ranks
smaller candidate sets higher.

## 6. Matcher inside the candidate pool (28.4M validation pairs, 755,718 true)

- **Average precision** (pair level, 5M-pair sample): **0.9989**.
- **Threshold grid.** "Raw" = every pair with p ≥ t. "Assigned" = after the one-owner rule, which
  is what production uses. Assigned-pair precision also counts validation records assigned to
  non-validation S1s as errors.

| t | Raw P | Raw R | Assigned P | Assigned R | **Macro F0.5** | Predictions per S1 |
|---|---|---|---|---|---|---|
| 0.01 | 0.819 | 0.9994 | 0.922 | 0.9789 | 0.98588 | 3.413 |
| 0.02 | 0.862 | 0.9987 | 0.940 | 0.9788 | 0.98771 | 3.406 |
| 0.03 | 0.889 | 0.9980 | 0.948 | 0.9786 | 0.98844 | 3.402 |
| 0.05 | 0.919 | 0.9967 | 0.958 | 0.9781 | 0.98927 | 3.397 |
| 0.07 | 0.932 | 0.9959 | 0.964 | 0.9778 | 0.98974 | 3.394 |
| 0.10 | 0.945 | 0.9948 | 0.970 | 0.9773 | 0.99015 | 3.389 |
| 0.15 | 0.957 | 0.9932 | 0.976 | 0.9765 | 0.99045 | 3.384 |
| **0.20** (production) | 0.965 | 0.9916 | 0.980 | 0.9756 | **0.99051** | 3.380 |
| 0.25 | 0.970 | 0.9900 | 0.982 | 0.9748 | 0.99051 | 3.376 |
| 0.30 | 0.975 | 0.9885 | 0.984 | 0.9739 | 0.99037 | 3.372 |
| 0.40 | 0.982 | 0.9850 | 0.987 | 0.9717 | 0.99000 | 3.363 |
| 0.50 | 0.988 | 0.9810 | 0.990 | 0.9688 | 0.98933 | 3.352 |
| 0.60 | 0.991 | 0.9769 | 0.993 | 0.9655 | 0.98835 | 3.340 |
| 0.70 | 0.995 | 0.9713 | 0.995 | 0.9606 | 0.98681 | 3.322 |
| 0.80 | 0.997 | 0.9657 | 0.997 | 0.9552 | 0.98509 | 3.303 |
| 0.90 | 0.998 | 0.9565 | 0.998 | 0.9461 | 0.98202 | 3.271 |

On validation the curve is flat between 0.15 and 0.30 (within 0.00015), so the threshold is not
what limits validation. Micro precision is **0.9979** and micro recall **0.9756**: **validation is
recall-limited**.

## 7. Ranking quality

**S1-centric** (for each validation entity with true matches, its candidates ranked by model score):

| Metric | Value |
|---|---|
| Top-1 candidate is a true match | **99.91%** |
| Top-3 / 5 / 10 / 20 | 99.92% (flat) |
| MRR | 0.9992 |
| Multi-match entities with **all** true matches in top-3 / 5 / 10 / 20 | 45.0% / 84.1% / 96.1% / 96.1% |

The all-in-top-3/5 values are low only because many entities have more than 3–5 links. Top-10
and top-20 are capped by blocking (96.3% all-match recall).

**Record-centric** (this is what production selection uses, because each record picks one S1):
- The true S1 is the record's top-1 candidate for **97.92%** of true pairs.
- It's in the record's top-3 for 98.56%.
- Blocking recall is 98.92%, so **1.0% of true pairs are in the pool but lose the argmax** to another
  S1.

Conclusion: **ranking quality is not the problem**. The misses are at blocking and at the
one-owner argmax.

## 8. False positives (all 15,576 on validation)

- **Source:** 61.9% come from records that belong to no S1 (decoys); 38.1% are owned records sent to
  the wrong S1.
- **Median profile:** name core similarity 100, address token-set 90, first house number *different*,
  legal form equal, 7 shared blocking keys, TF-IDF similarity 0.80, model score 0.47.
- **Country always agrees:** blocking is per country. There are no postal codes in the data.

| Pattern (first matching rule) | Share |
|---|---|
| Missing address (name-only record) | **32.1%** |
| Same name + street, **different house number** | **25.8%** |
| Extra/missing word in the name (branch or franchise variant) | 14.2% |
| Same address / different business | 10.4% |
| Common business name | 6.7% |
| Legal-form confusion | 4.4% |
| Near-identical record (duplicate/decoy) | 1.3% |
| Name variation (abbreviation/typo) | 1.2% |
| Similar name / different address | 0.4% |
| Website / dba alias | 0.3% |
| Transliteration | 0.1% |
| Other | 3.2% |

- **Score bands:** 22.7% at 0.2–0.3, 31.0% at 0.3–0.5, 32.2% at 0.5–0.8, 14.1% at 0.8–1.0. Almost half
  are confidently wrong, so a higher threshold alone can't remove them.
- **By source:** S2 48.7%, S3 51.3%.

## 9. False negatives (18,608 = 2.44% of true pairs)

| Where the link was lost | Share |
|---|---|
| Never generated as a candidate (blocking) | **44.3%** |
| In the pool, but another S1 scored higher (one-owner rule) | **41.2%** |
| Ranked first, score just below tau | 10.3% |
| Ranked first, low model score (< 0.05) | 4.2% |

Patterns among the in-pool misses:

| Pattern | Share |
|---|---|
| **Missing address** | **70.5%** |
| Different house number | 9.7% |
| Same address / different name | 9.2% |
| Common name | 3.7% |
| Extra word | 3.2% |
| Other | < 4% |

Blocking misses are **64% missing-address records** (see §5). No false negatives were lost to top-k
trimming or entity aggregation: production v2 uses `keep_k = 0`.

- **By source:** S3 53%, S2 47%. **By country:** US 58%, India 42%.

**Missing-address records are the single biggest error type on both sides:** 32% of false positives
and about 65–70% of false negatives. With no address, the model can't tell same-named businesses apart.

## 10. Multi-match analysis

| True matches | Entities | Macro F0.5 | Precision | Recall | Avg predicted | Avg true | All-match recall | Share of total loss |
|---|---|---|---|---|---|---|---|---|
| 0 | 12,347 | 0.9896 | – | – | 0.011 | 0 | – | 6.2% |
| 1 | 11,965 | **0.9704** | 0.994 | 0.973 | 0.979 | 1 | 97.3% | 16.9% |
| 2 | 37,784 | 0.9891 | 0.996 | 0.976 | 1.96 | 2 | 95.3% | 19.6% |
| 3 | 53,278 | 0.9916 | 0.998 | 0.976 | 2.93 | 3 | 93.1% | 21.2% |
| 4 | 48,171 | 0.9925 | 0.998 | 0.976 | 3.91 | 4 | 91.0% | 17.1% |
| 5+ | 57,436 | 0.9930 | 0.999 | 0.975 | 5.51 | 5.64 | 87.6% | 19.1% |

- **Single-match entities are the weakest group (0.970).** One miss or one extra costs a lot there.
- **Multi-match entities score well (0.992).** F0.5 forgives one missed link out of many.
- Recall per link is flat at about 97.5% in every group, so there's **no multi-match-specific recall
  problem**.
- Singleton accuracy is 98.96%: 12,218 of 12,347 correctly empty, 129 false positives.

## 11. Source-specific analysis (pair level)

| | Candidate recall | Precision | Recall | F0.5 | FP rate | FN rate | Macro F0.5 if this source were perfect |
|---|---|---|---|---|---|---|---|
| S2 | 99.00% | 0.9794 | 0.9764 | 0.9788 | 2.06% | 2.36% | 0.99505 (+0.0045) |
| S3 | 98.85% | 0.9796 | 0.9749 | 0.9787 | 2.04% | 2.51% | 0.99565 (+0.0051) |

The two sources are almost identical; S3 contributes slightly more loss (+0.0051 vs +0.0045).
Pair precision here also counts validation records assigned to non-validation S1s.

## 12. Country analysis (validation)

| | Entities | Candidate recall | Macro F0.5 | Micro P | Micro R | Empty accuracy | Avg candidates per S1 | Avg predictions | F0.5 if perfect |
|---|---|---|---|---|---|---|---|---|---|
| US | 132,384 | 99.07% | 0.99078 | 0.9980 | 0.9762 | 99.07% | 128.1 | 3.38 | 0.99604 |
| India | 88,597 | 98.70% | 0.99012 | 0.9978 | 0.9748 | 98.78% | 129.8 | 3.38 | 0.99448 |

India is slightly worse: blocking recall is lower, because of transliteration and address formats.

**Unseen countries.** Structurally they're handled:
- Blocking, TF-IDF and every aggregation loop over whatever country labels exist.
- Country is not a model feature.
- France is processed like any other label.
- Nothing in the pipeline filters or one-hot encodes {US, India}.

What France does *not* get:
- A transliteration dictionary (not needed: France is Latin script).
- Any France-specific validation. Its accuracy is unknown locally.

## 13. Feature importance and ablation

**Model:** LightGBM binary classifier, 2000 trees, 50 features.

Top features by gain share:

| Feature | Gain share | Splits |
|---|---|---|
| margin_2nd (lead over the record's runner-up candidate) | 29.4% | 7,571 |
| rank_combo (rank by name+address score among the record's candidates) | 27.4% | 5,837 |
| gap_combo | 17.1% | 3,507 |
| rank_block | 6.1% | 6,574 |
| gap_block | 5.0% | 3,278 |
| hn_min_logdiff (house-number distance) | 4.7% | 4,277 |
| hn0_logdiff | 1.7% | 8,009 |
| ad_tset | 0.8% | 5,852 |
| nm_ratio | 0.7% | 16,409 |
| extra_all_o | 0.7% | 5,314 |

**Permutation ablation.** Each group is shuffled jointly on 54,994 validation entities (baseline
0.99198):

| Feature group shuffled | Macro F0.5 | Drop |
|---|---|---|
| Cross-field / candidate context (rank, gap, margin) | 0.5041 | **0.4879** |
| Name | 0.9144 | 0.0776 |
| Blocking | 0.9599 | 0.0321 |
| Numeric / house number | 0.9809 | 0.0111 |
| Record flags | 0.9904 | 0.0015 |
| Address text | 0.9908 | 0.0012 |
| Country/state | – | Not a feature; no state or postal features exist |

**The model mostly asks "is this S1 clearly the best of this record's candidates?"** It relies on
comparisons **within the record's candidate list**, and there are **no comparisons across records
competing for the same S1**. That's exactly the view decoys defeat: a decoy of a real business is
often the best candidate *for itself*.

## 14. Bottleneck

**Validation-side accounting.** Oracle fixes are applied to the production predictions (baseline
0.99051):

| Oracle | Macro F0.5 | Gain |
|---|---|---|
| Perfect decisions on the current candidates (blocking unchanged) | 0.99661 | +0.0061 |
| Add every missed true pair that is in the pool (argmax + threshold losses) | 0.99472 | +0.0042 |
| Add every true pair that blocking missed | 0.99400 | +0.0035 |
| Remove every false positive | 0.99242 | +0.0019 |
| Remove false positives from unmatched records only | 0.99172 | +0.0012 |

**Test-side evidence:**
- Public leaderboard **0.964** vs validation 0.9905: a gap of **0.027**. That's about 3× the *entire*
  validation loss (0.0095).
- Test has about 1.9× more unmatched records per S1 than train (2.3 vs 1.2). If test decoys behaved
  like train distractors, scaling the unmatched-record false positives by 1.9× would cost only
  about 0.0012 (row 5 above). So **the gap cannot be explained by "more of the same distractors"**.
  Either test decoys are a harder kind, or one population (France, 15% of test entities, never seen
  in training) scores far lower. France at about 0.82 with the others at about 0.99 would give
  exactly 0.964.

**Classification: H (combination), led by G.**
- **Primary: G, data-distribution-limited.**
  - The holdout doesn't reproduce test conditions: 23% more records per S1, an unseen country,
    decoys at about 2× density.
  - The local score (0.9905) overstates test (0.964) by 0.027.
- **Secondary, inside validation:**
  - **D, selection-limited (one-owner argmax).** 41% of false negatives are true pairs in the pool
    that lost the argmax; 38% of false positives are owned records sent to the wrong S1.
  - **A, blocking-limited.** 1.08% of true pairs are never generated; 44% of false negatives.
    Candidate sets are also large: 129 per S1.
  - Both are concentrated in **missing-address records** (about 65–70% of misses, 32% of false
    positives).
- **Not the bottleneck:**
  - **B (ranking):** top-1 true rate 99.91%, MRR 0.999.
  - **C (threshold):** flat curve, ±0.00015 between 0.15 and 0.30 on validation.
  - **E (singletons):** 98.96% correctly empty; only 6% of the loss.
  - **F (features):** average precision 0.9989.

## Summary

```
CURRENT_LOCAL_F05        = 0.99051   (reproduces production 0.990515 exactly)
VALIDATION_ENTITIES      = 220,981
CANDIDATE_PAIR_RECALL    = 0.98921
ALL_MATCH_RECALL         = 0.96275
TOP1_TRUE_RATE           = 0.99914   (S1-centric; record-centric true-S1-is-top-1 = 0.97918)
TOP5_TRUE_RATE           = 0.99917
TOP10_TRUE_RATE          = 0.99917
PRECISION                = 0.99790   (micro; macro 0.99770)
RECALL                   = 0.97564   (micro; macro 0.97563)
F05                      = 0.99051   (macro)
SOURCE2_F05              = 0.97882   (pair level)
SOURCE3_F05              = 0.97868   (pair level)
US_F05                   = 0.99078
INDIA_F05                = 0.99012
SINGLE_MATCH_F05         = 0.97044
MULTI_MATCH_F05          = 0.99179
TRUE_EMPTY_ACCURACY      = 0.98955
PUBLIC_LEADERBOARD       = 0.964

PRIMARY_BOTTLENECK       = G  data distribution: validation does not reproduce test (records per S1
                              5.8 vs 4.7, ~2x decoys, unseen France); gap 0.027 = 3x all validation loss
SECONDARY_BOTTLENECK     = D + A  one-owner selection (41% of FN, 38% of FP) and blocking (44% of FN),
                              both concentrated in missing-address records

MOST_PROMISING_IMPROVEMENT = First locate the 0.027 test gap with 1-2 diagnostic leaderboard
  submissions (e.g. the same file with all France rows emptied: the score change gives France's own
  F0.5 directly). If France is the gap, work on French normalisation/blocking. If not, it's the
  decoys: add S1-side competition features (how each record compares with the other records
  competing for the same S1, including same-source near-copies with a different house number) and
  make validation decoy-dense. Both also cut the candidate set per S1, which now counts in the
  ranking. Missing-address records are the biggest single error class on validation.
```

## Existing tests
- `python code/business_entity_resolution/src/metric.py`: **metric tests pass**.
- `python scripts/forensic_evaluate.py selftest`: **all metric checks pass**.
- `python scripts/forensic_evaluate.py verify`: **production metric agrees** (difference ≤ 3e-13).

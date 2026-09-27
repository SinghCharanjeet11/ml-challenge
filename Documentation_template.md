# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** bruteforce  
**Team Members:** Charanjeet Singh, Vanshika Srivastava  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
We use a cascade: blocking, then a light filter model, then a final matcher. Every Source 2/3
record looks up likely Source 1 entities in its own country with two cheap passes (hashed
name/address keys and character 3-gram TF-IDF on names). A small LightGBM filter keeps at most
three plausible candidates per record. That cuts the candidate set from ~159 to
**5.2 pairs per Source 1 entity** while keeping 98.4% of the true links. A second LightGBM model
makes the final call from 50 pair features plus "competition" features that compare each record
with the other records claiming the same Source 1 entity. Each record goes to at most one entity.

What helped most:
- A small Indic-to-Latin word dictionary learned from the training matches.
- Features aimed at near-duplicate decoy records (house-number distance, extra words).
- The competition features.
- Choosing the decision threshold for test conditions rather than the training holdout.

Results: **public leaderboard 0.9795**; macro F0.5 ~0.987 on a correctly built 10% holdout of
the training entities (section 5 explains why this differs from the leaderboard and from our
earlier holdout numbers).

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from exploring the data that shaped the pipeline:

- **One owner per record.** Every S2/S3 record is matched to at most one S1 entity (all 7.6M train
  links are unique). So we search from the S2/S3 side and allow one owner per record.
- **No cross-country links.** Blocking per country label loses nothing.
- **Precision matters.** 5.6% of S1 entities have no match, and about a quarter of S2/S3 records
  match nothing. A wrong match on a no-match entity turns a 1.0 into a 0.0, and a similar name
  alone isn't evidence.
- **Indic scripts.** About 40% of Indian S2/S3 names are written in Indic scripts (9 different
  scripts), usually a word-by-word phonetic spelling of the English name.
- **No postal codes.** No Indian PINs and under 0.01% US ZIPs, so we can't block on them; we use
  street, city and house number instead.
- **France is test-only.** France is about 15% of test S1 and never appears in training, so the
  features are language-neutral and country is not a feature.
- **Parsing traps.** Some names contain literal quote characters, and some are literally "NA" or
  "null". Everything is read with quoting and NA-conversion off, and row counts are asserted.
- **Test is denser than train.** It has 5.8 vs 4.7 S2+S3 records per S1. The distribution of
  true links per entity looks the same, so test has roughly twice as many records that match
  nothing (~2.3 vs ~1.2 per S1). This turned out to be the main difficulty (section 5).

Typical noise in S2/S3 records:
- abbreviations (Pvt Ltd / Private Limited, St / Street)
- digits used as letters (Sh4rma)
- dba/aka aliases and website-style names
- reordered words and typos
- empty addresses (~3%) and house numbers a few digits off

The non-matching records ("decoys") often copy a real business with one change: a
house number a few apart, one business word swapped, a different legal form, or a different name
at the same address.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage LightGBM cascade (filter, then matcher) + one-owner
assignment  
**Core Innovation:**
- a learned transliteration dictionary;
- decoy-focused pair features;
- a filter model that doubles as a ~30x candidate reduction;
- competition features across all records claiming the same S1 entity;
- test-aware threshold selection.

Flow:
1. Load and check the TSVs.
2. Normalise and transliterate.
3. Block (key pass + TF-IDF pass).
4. Pair features.
5. Filter model; keep the top candidates (= `candidate_pairs.tsv`).
6. Competition features.
7. Final model.
8. Best candidate per record if it passes the threshold (= `matching_results.tsv`).

---

## 3. Candidate Generation (Blocking)

Blocking has three steps. All run per country label and treat country as an open set.

**Step 1: key blocking.** Hashed keys for every record:
- single "core" name tokens (legal forms and filler words removed), up to 4
- order-free pairs of core name tokens
- the whole core name joined without spaces ("sharmatraders" vs "sharma traders")
- address tokens with 3+ characters plus every number (house numbers), up to 10
- neighbouring address token pairs
- name token x address token combinations

Keys shared by more than 1,000 S1 records of a country are dropped. Each query uses its rarest keys
first (at most 24 keys, 10,000 S1 postings). Candidates are scored by the summed IDF of shared keys,
and we keep the top 20.

**Step 2: character 3-gram TF-IDF on the core name.** This catches typos and merged words that
share no whole word. Trigrams in more than 2% of S1 names are ignored. The sparse top-10 search
uses `sparse_dot_topn`. The two lists are merged; a pair found by one pass only gets 0 for the
other pass's score.

**Step 3: learned filter.** Steps 1–2 give ~27 candidates per record (284M pairs on train, 276M on
test).
- A light LightGBM (20 cheap features, 300 trees) scores every pair. The features are blocking
  and TF-IDF scores, ranks and gaps, core-name ratio, token sort, Jaro-Winkler, address token set,
  house-number overlap and distance, source, and missing address.
- It's trained out-of-fold on train (2 folds split by entity).
- We keep, per record, at most its top 3 candidates with filter probability ≥ 0.01.
- **These pairs are the candidate set:** exactly what the final model scores and what
  `candidate_pairs.tsv` contains.

| Candidate set (train) | Pairs per S1 entity | True links kept |
|---|---|---|
| Key blocking, top 10 | 46 | 98.15% |
| Key blocking, top 20 | 92 | 98.50% |
| Keys top 20 + TF-IDF | 128.9 | 98.93% |
| **+ filter (p ≥ 0.01, top 3 per record), submitted** | **4.4** | **98.38%** |
| Filter alternatives: p ≥ 0.001 top 5 / p ≥ 0.05 top 2 | 5.4 / 4.0 | 98.64% / 97.95% |

- **Blocking keys used:** hashed name tokens, name token pairs, joined name, address tokens,
  address bigrams, name x address combos (IDF scoring), char 3-gram TF-IDF on names, plus a
  learned filter.
- **Candidate pairs generated:** 8,959,722 test pairs, i.e. **5.2 per S1 entity**, 20,582
  entities with none (275.7M before the filter). Train: 9,786,681 pairs.
- **How we ensured true matches were not lost:**
  - Country blocking is lossless.
  - Text is normalised and transliterated before keys are built.
  - Two independent retrieval passes find a record when either its name or its address is messy.
  - The filter costs 0.55 points of recall (98.93% → 98.38%) for a ~29x smaller candidate set.

Memory was the main practical constraint:
- On our 16 GB laptop, blocking runs in chunks (250k queries, 5M join rows, 4 Polars threads).
- Taking the top k per group with `group_by().head()` kept whole chunk tables alive, and memory
  grew past 26 GB. Filtering on a rank column fixed it.
- The full cascade ran on an AWS r6i.4xlarge (16 vCPU, 128 GB) using the challenge credits.

---

## 4. Matching Model

Normalisation is the same on both sides:
- accents removed, lower case, punctuation to spaces
- canonical forms for street types, directions, ordinals, US/Indian states, some French street
  words and legal forms (private -> pvt, incorporated -> inc, ...)
- digit-as-letter fixes and website suffixes removed
- Indic words replaced through the learned dictionary

**Features used** (50 pair features + 17 competition features):
- **Name features:**
  - RapidFuzz ratio / token-set on the full name
  - token-sort, token-set, partial ratio and Jaro-Winkler on the core name
  - ratio / partial ratio on the name without spaces
  - first token equal, core name equal, name lengths
  - extra words on each side, legal form equal / conflicting
  - TF-IDF name similarity
- **Address features:**
  - token-set, token-sort and partial ratio on the normalised address; address-missing flag
  - house numbers: counts, overlap, Jaccard, first number equal, log distance between first
    numbers, smallest log distance between any numbers, "close but different" flag (1–20 apart),
    prefix flag (67 vs 6705)
- **Candidate context:**
  - blocking score and shared keys
  - per record: candidate count, rank and gap by blocking score and by name+address score, gap to
    the best name / address, margin over the runner-up, same-core count
  - per S1: how many records retrieved it
  - record flags: source, Indic script, website-style name, alias marker
- **Competition features**, from the filter probability over the pruned set:
  - per record: rank, gap to its best candidate, second-best probability, number of candidates
  - per S1, over records whose best candidate it is: count, count with p > 0.5, sum and max, this
    pair's gap to the best, the same within the same source, and this pair's rank among all pairs
    pointing at the S1
- Country is deliberately **not** a feature.

**Model type:**
- Filter: LightGBM, 20 features, 300 trees, learning rate 0.1.
- Final matcher: LightGBM, 67 features, 127 leaves, learning rate 0.05, 2000 rounds, trained on
  the pruned pairs of 90% of training entities (8.8M rows).
- Both are MIT-licensed LightGBM models with no pre-trained components.

**Threshold selection method:** each S2/S3 record keeps only its highest-scoring candidate, then a
global threshold applies.
- On the training holdout the best threshold is ~0.5 (with the corrected split, section 5).
- Test has about twice as many non-matching records per entity, so we set the threshold for
  test on the public leaderboard: 0.80 → 0.9793, **0.85 → 0.9795**, 0.90 → 0.9790, 0.95 → 0.9769.
- The metric code reproduces the 0.714 worked example and was cross-checked against an
  independent implementation (difference ≤ 3e-13).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - **Public leaderboard 0.9795** (submitted file).
  - Corrected holdout (220,981 training entities): 0.9866 at its best threshold, 0.9841 at the
    submitted 0.85.

Leaderboard progression:

| Submission | Leaderboard F0.5 |
|---|---|
| Single model, v2 blocking, tau 0.2 (threshold from the holdout) | 0.964 |
| Same model, tau 0.4 / 0.5 / 0.6 / 0.7 / 0.8 | 0.972 / 0.9748 / 0.9765 / 0.9776 / 0.9778 |
| **Cascade, tau 0.80 / 0.85 / 0.90 / 0.95** | 0.9793 / **0.9795** / 0.9790 / 0.9769 |

**Validation lesson (important).** For most of the project our holdout reported ~0.991. The split
put each S1 entity and its true records in the holdout, but assigned records that match nothing
by a random hash of the record id. So each holdout entity faced only ~10% of its decoys, and the
holdout understated the decoy problem about tenfold. The correct split assigns a non-matching
record to the fold of the S1 entity it is closest to (its top blocking candidate). With it, the
same model scores 0.987 at best, prefers a much stricter threshold, and behaves much more like
the leaderboard. We found this late; the submitted model's threshold was already tuned on the
leaderboard, but earlier design choices were made against the optimistic number.

**Where errors come from** (corrected holdout, threshold 0.6, score 0.9865; gain if each error
type were fixed):

| Error type | Gain if fixed |
|---|---|
| True links rejected although in the candidate pool (56% are records without an address) | +0.0057 |
| True links lost before the model (blocking 1.07% + filter 0.55% of links) | +0.0053 |
| False matches from records that match nothing (roughly doubled on test) | +0.0027 |
| False matches to the wrong S1 entity | +0.0002 |

- **Common false positives (wrong merges):**
  - decoys copying a real business with one change: house number a few apart (6705 vs 6708), one
    extra or swapped business word ("Dds Forestry" vs "Dds Services"), a different legal form;
  - different businesses at the same address;
  - records with no address that share a name with several S1 entities.
  - On test, these decoys are more numerous and harder than in train, which is why a stricter
    threshold (0.85) wins there.
- **Common false negatives (missed matches):**
  - records without an address, where several same-named S1 entities compete;
  - names with typos or merged words that no blocking key retrieves;
  - Indic words missing from the learned dictionary;
  - heavily abbreviated names.

---

## 6. Conclusion
Two retrieval passes, a learned filter that doubles as a 5-per-entity candidate set, and a LightGBM
matcher with competition features give 0.9795 on the public leaderboard.

What we learned:
- **Validation design matters as much as the model.** Our holdout hid most decoys for most of the
  project.
- **Train and test differ.** Test has about twice the non-matching records, of a harder kind, plus
  an unseen country. So the threshold and anything tuned on train needs to be checked against test
  conditions.
- **Several ideas that looked better on the (corrected) holdout did not transfer** (section B).
  The one that looked most promising, a fine-tuned cross-encoder, we could no longer submit
  before the upload limit.

With more time, we would validate on the corrected split from the start, add a transformer
matcher trained on GPU, and work on records without an address.

---

## Appendix

### A. Code Artefacts
Everything is in `code/business_entity_resolution/`. `README.md` has exact commands; versions are
pinned in `requirements.txt`. Run from `src/`:

1. `python block_split.py train --blocking v2` and `python block_split.py test --blocking v2`:
   transliteration dictionaries, prepared records, key + TF-IDF candidates.
2. `python featurise_all.py train` and `python featurise_all.py test`: pair features for every
   candidate.
3. `python stage2.py`: filter model, pruning, competition features, final model →
   `output/candidate_pairs.tsv` and `output/matching_results.tsv` (tau 0.85).

Modules:
- `io_utils.py`, `normalize.py`, `translit.py`
- `blocking.py`, `tfidf_blocking.py`
- `features.py`, `pipeline.py`
- `decide.py`, `metric.py`, `rethreshold.py`

Experiments are listed in the README. No external data, APIs or lookups are used. Word maps are
hand-written; the dictionaries are learned from the provided training data only.

### B. Additional Results
Experiments after the submitted model. Each one was checked on the leaderboard at the same number
of matches as the submission, unless noted.

| Experiment | Local evidence | Leaderboard |
|---|---|---|
| Separate lenient threshold for records without an address | holdout +0.0020 | 0.9776 (worse) |
| Stricter threshold for France (0.95) | – | 0.9792 (worse) |
| Word-level features (typo vs replaced word, legal form, digit edits) + "twin" features | corrected holdout +0.0019 | 0.9749 (worse): on test it accepted same-name records with a different house number |
| Per-cell thresholds from a train/test density-ratio estimate (by house-number agreement) | estimated gain | 0.9784 (worse): most likely because France, absent from the holdout, looked like "excess" in the estimate |
| Non-matching records weighted 2x in training | corrected holdout +0.0007 | not submitted |
| Fine-tuned cross-encoder (MiniLM-L6, Apache-2.0) on the 1.4M uncertain test pairs, stacked with the cascade | corrected holdout +0.0022 at every threshold (AUC 0.939 → 0.958 on uncertain pairs) | not submitted (upload limit reached) |
| Synthetic decoys to simulate test density | could not reproduce the leaderboard's behaviour | – |

Run times on AWS r6i.4xlarge:
- key blocking ~16 min per split; TF-IDF ~86 min (train)
- features for all 284M train pairs ~22 min
- filter model ~20 min; final model ~6 min
- test features + scoring ~1.5–2 h

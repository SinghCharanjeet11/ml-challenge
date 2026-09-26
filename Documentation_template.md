# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** bruteforce  
**Team Members:** Charanjeet Singh, Vanshika Srivastava  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary
We use a classic blocking + classifier setup. Every Source 2/3 record looks up its 10 most
likely Source 1 entities using hashed name/address keys (only within its own country), a
LightGBM model scores those pairs, and each record is then given to at most one Source 1
entity if the score is high enough. The two things that helped most were a small
Indic-to-Latin word dictionary learned from the training matches, and features aimed at
near-duplicate "distractor" records (house number distance, extra words in the name). Our
best validation score is 0.9905 macro F0.5 on a 10% holdout of the training entities.

---

## 2. Methodology

### 2.1 Problem Analysis
Things we found while exploring the data that ended up shaping the pipeline:

- Every S2/S3 record is matched to at most one S1 entity (all 7.6M train links are unique),
  so it made sense to search from the S2/S3 side and allow only one owner per record.
- No link crosses countries, so blocking per country loses nothing.
- 5.6% of S1 entities have no match, and roughly a quarter of S2/S3 records match nothing.
  A wrong match on a no-match entity turns a 1.0 into a 0.0, so precision matters a lot, and
  a similar name alone isn't enough evidence.
- About 40% of Indian S2/S3 names are written in Indic scripts (9 different scripts), usually
  as a word-by-word phonetic spelling of the English name.
- Postal codes are basically missing (no Indian PINs, under 0.01% US ZIPs), so we couldn't
  block on them and used street, city and house number instead.
- France only appears in test (about 15% of S1). We never had French labels, so we kept the
  features language-neutral and added some French address words to the normaliser.
- A few parsing traps: names with literal quote characters, and names that are literally
  "NA" or "null". We read everything with quoting and NA-conversion turned off and check row
  counts.
- The test pool is a bit denser than train (5.75 vs 4.68 S2+S3 records per S1), so we expect
  test precision to be slightly harder.

Typical noise in S2/S3: abbreviations (Pvt Ltd / Private Limited, St / Street), digits used as
letters (Sh4rma), dba/aka aliases, website-style names (sharmatraders.com), reordered words,
typos, empty addresses (~3%) and house numbers that are a few digits off.

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier (LightGBM) + a simple assignment step  
**Core Innovation:** learned transliteration dictionary, distractor-focused features, and
candidate-context features computed over the full candidate pool.

The flow is: load and check the TSVs, normalise and transliterate, block (top 10 per S2/S3
record), build pair features, score with LightGBM, keep the best candidate per record if it
passes the threshold, write the two output files.

The context features (how a candidate ranks against the other candidates of the same record,
the margin over the runner-up, how many records point at the same S1) turned out to be the
most important ones. We compute them over the whole candidate table, for train as well as
test, so they mean the same thing in both.

---

## 3. Candidate Generation (Blocking)

Each S2/S3 record is compared only against S1 records with the same country label. We build
hashed keys for every record:

- single "core" name tokens (legal forms like pvt/ltd/inc and filler words removed), up to 4
- pairs of core name tokens (order doesn't matter)
- the whole core name joined without spaces (catches "sharmatraders" vs "sharma traders")
- address tokens with 3+ characters plus every number (house numbers), up to 10
- neighbouring address token pairs
- name token x address token combinations

Keys that appear in more than 1,000 S1 records of a country are dropped as too common. Each
query uses its rarest keys first (at most 24 keys, and at most 10,000 S1 postings in total),
candidates are scored by the summed IDF of the keys they share, and we keep the top 20.

The key-based search needs at least one whole word in common, so typos and merged words slip
through. We added a second pass for that: character 3-gram TF-IDF on the core name, also per
country, keeping the 10 most similar S1 names per record (trigrams found in more than 2% of S1
names are ignored, and the sparse top-n search uses `sparse_dot_topn`). The two candidate lists
are merged; a pair found by only one pass gets 0 for the other pass's score.

- **Blocking keys used:** hashed name tokens, name token pairs, joined name, address tokens,
  address bigrams, name x address combos (IDF scoring, no PIN/ZIP since they're missing), plus
  char 3-gram TF-IDF similarity on names
- **Candidate pairs generated:** 284.4M on train and 275,746,000 on test, about 27 per S2/S3
  record (key pass 196.6M + TF-IDF pass 98.5M on test, minus the overlap).
- **How you ensured true matches were not lost:** country blocking loses nothing, text is
  normalised and transliterated before keys are built, and because there are several
  independent kinds of keys a record can still be found when either its name or its address
  is messy. On train, recall went from 98.15% (keys, top 10) to 98.50% (keys, top 20) and
  98.93% with the TF-IDF pass added, so about 42% fewer true links are lost at this stage.

Memory was the main practical problem on our 16 GB laptop. Blocking runs in chunks (250k
queries at a time, 5M join rows per step, 4 Polars threads) and takes about 25 minutes and
~5 GB per split. The final, bigger candidate set was built on an AWS r6i.4xlarge (16 vCPU,
128 GB) using the challenge credits. One thing we learned the hard way: taking the top k per group with
`group_by().head()` kept a reference to each whole chunk table and memory kept growing past
26 GB; filtering on a rank column instead fixed it.

`output/candidate_pairs.tsv` is exactly the list of pairs our model scores.

---

## 4. Matching Model

Before computing features, both sides go through the same normalisation: accents removed,
lower case, punctuation to spaces, canonical forms for street types, directions, ordinals,
US/Indian states, some French street words and legal forms (private -> pvt, incorporated ->
inc, ...), digit-as-letter fixes, website suffixes removed, and Indic words replaced using the
learned dictionary.

**Features used** (50 in total):
- Name features: RapidFuzz ratio and token-set ratio on the full name; token-sort, token-set,
  partial ratio and Jaro-Winkler on the core name; ratio and partial ratio on the name without
  spaces; first token equal; core name equal; name lengths; number of extra words on each side;
  legal form equal / conflicting.
- Address features: token-set, token-sort and partial ratio on the normalised address; address
  missing flag; house numbers (counts, overlap, Jaccard, first number equal, log distance
  between first numbers, smallest log distance between any numbers, "close but different"
  flag for numbers 1-20 apart, prefix flag like 67 vs 6705).
- Other: blocking score and number of shared keys; per record: number of candidates, rank and
  gap by blocking score, rank and gap by name+address score, gap to the best name and best
  address, margin over the second best, how many candidates have the same core name; per S1
  record: how many records retrieved it and its rank among them; record flags (source 2/3,
  Indic script, website-style name, alias marker); the TF-IDF name similarity. Country is not a
feature on purpose, so the
  model has a chance on France.

**Model type:** LightGBM binary classifier (num_leaves 127, learning rate 0.05,
min_data_in_leaf 100, feature and bagging fraction 0.8, 2000 rounds).

Training data (`train_full.py`): we hold out 10% of the S1 entities (with all their records)
plus 10% of the unmatched records for validation. From the other 90% we keep all positives,
the two hardest negatives of every record (top 2 by name+address score or by blocking score),
and 1 in 10 of the remaining easy negatives with weight 10 so the model still sees the right
balance. That gives 49.3M training rows (6.80M positives) and 28.4M validation rows.

**Threshold selection method:** each S2/S3 record keeps only its highest-scoring candidate,
then we sweep the threshold tau and pick the value with the best macro F0.5 on the held-out
entities (our metric code reproduces the 0.714 worked example from the problem statement).
Best tau was 0.20 (the curve is flat: 0.9903-0.9905 for tau 0.15-0.30).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9905 on 220,981 held-out training entities (final model, tau =
  0.20). Earlier versions on the way: 0.9812 before the distractor features, 0.9878 on a 12%
  sample, 0.9888 with all the data but only the top-10 key blocking.

Threshold sweep on the holdout:

| tau | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 | 0.50 |
|---|---|---|---|---|---|---|---|
| macro F0.5 | 0.99012 | 0.99042 | 0.99051 | 0.99045 | 0.99034 | 0.99003 | 0.98933 |

Where the errors come from (12% sample model with the old top-10 blocking, 914,635 true
links): 96.8% matched correctly; 1.8% never made it into the top 10 candidates (top 20 plus
the TF-IDF pass later cut these misses to about 1.1%); 0.8% had another S1 ranked above the right
one; 0.6% had the right S1 on top but below the threshold. There were 18,977 false matches,
13,381 of them from records that don't belong to any S1.

- **Common false positives (wrong merges):** mostly distractors that look almost identical to
  a real entity, e.g. the same name with one extra word ("Sharma Traders" vs "Sharma Traders
  & Sons") or the same street with a slightly different house number (6705 vs 6708). Branches
  of chains with the same name on nearby streets are the other big group.
- **Common false negatives (missed matches):** mostly blocking misses, i.e. names with typos
  or merged words that share no full token with the S1 name, and records whose address is
  missing or very different. After that: Indic words that aren't in our dictionary and very
  heavily abbreviated names.

On test we predict 6,045,857 matches, and 5.2% of S1 entities get no match (5.6% on train).

---

## 6. Conclusion
IDF-scored multi-key blocking, a LightGBM model on string, house number and context features,
and a one-owner-per-record rule got us to 0.9905 macro F0.5 on held-out training entities.
Most of the improvement over plain string matching came from handling distractors explicitly
and from the transliteration dictionary. On a 16 GB machine we also had to be careful with
memory: chunked blocking, writing features to disk and scoring in chunks is what made
training on all the data possible. Adding the character 3-gram TF-IDF pass to the blocking was the last big gain; with more
time we'd look at the ~1.1% of links that still never get retrieved.

---

## Appendix

### A. Code Artefacts
Everything is in `code/business_entity_resolution/` (see `README.md` there, versions pinned
in `requirements.txt`). Scripts are run from `src/`, with `POLARS_MAX_THREADS=4` on a 16 GB
machine:

1. `python block_split.py train --blocking v2` - learns the transliteration dictionaries,
   prepares the train records and builds the train candidates (key pass + TF-IDF pass)
2. `python train_full.py --blocking v2 --keep_k 0` - trains the model, writes
   `work/model.txt` and `work/config.json`
3. `python block_split.py test --blocking v2` - prepares the test records and builds the
   test candidates
4. `python predict.py` - writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`

Modules: `io_utils.py` (reading/writing), `normalize.py`, `translit.py`, `blocking.py`,
`tfidf_blocking.py`, `features.py`, `pipeline.py` (shared prep, caching, chunked features),
`decide.py`,
`metric.py`. `train.py` is a quicker version that trains on a sample with 2-fold validation;
the `exp_*.py` files are small experiments we used while developing the blocking and
transliteration.

We don't use any external data, APIs or pre-trained models. The word maps are written by
hand and the dictionaries are learned from the provided training data only.

### B. Additional Results
Run times for the final model on AWS r6i.4xlarge (16 vCPU, 128 GB): key blocking ~16 min and
TF-IDF ~86 min on train, test blocking ~72 min, featurising 284M train pairs ~24 min,
LightGBM (2000 rounds, 49.3M rows) ~70 min (partly sharing the CPU with test blocking), test
scoring + writing ~97 min. The earlier top-10 version ran on our 16 GB laptop (blocking
~25 min per split, training ~38 min).

Most important features by gain: margin over the second-best candidate, rank by name+address
score, gap to the best candidate, log distance of the first house number, blocking score gap,
the "near house number" flag, extra word count and house-number Jaccard.

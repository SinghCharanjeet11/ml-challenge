# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary
Every Source 2 / Source 3 record retrieves its top-10 Source 1 candidates through IDF-scored, hashed
name and address keys, blocked within each country label. A LightGBM classifier then scores each
pair on 49 string, house-number and candidate-context features, and a decision layer gives every
S2/S3 record at most one owner and applies an F0.5-tuned threshold. The main additions are a learned
Indic-to-Latin word dictionary for the ~40% of Indian records written in Indic scripts, and
house-number-distance and extra-word features that separate true matches from near-duplicate
distractors. On a held-out 10% of training entities, macro F0.5 = **0.9888**.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from EDA over all train and test files (no ground truth used on test):

| Finding | Evidence | Consequence |
|---|---|---|
| Each S2/S3 record belongs to at most one S1 entity | 7,638,365 links, each ID matched exactly once | Search from the S2/S3 side; enforce one owner per record |
| Links never cross country labels | 0 cross-country links | Block within each country (lossless) |
| Many S1 entities have no match | 123,247 of 2.2M (5.6%) in train | A wrong match on a singleton turns 1.0 into 0.0, so precision matters |
| About a quarter of S2/S3 records match nothing | S2 26.6%, S3 25.4% unmatched | Distractors everywhere: a similar name is not proof |
| Indian S2/S3 names are often in Indic scripts | ~40% of Indian S2/S3 names, 9 scripts | Transliteration needed before comparing |
| Postal codes are practically absent | India PIN: 0; US ZIP < 0.01% | Block on street, city and house number instead |
| France appears only in test (15% of S1) | 259,452 French S1 entities | Rules must be language-agnostic; French address words added |
| Parsing traps | Literal quote characters in names, "NA"/"null" names | Read with quoting and NA conversion disabled; row counts asserted |
| Test pool is relatively larger | S2+S3 per S1: train 4.68, test 5.75 | Expect a slightly harder precision problem on test |

Noise seen in S2/S3: abbreviations (`Pvt Ltd` / `Private Limited`, `St` / `Street`), leetspeak
(`Sh4rma`), aliases (`dba`, `aka`), domain-style names (`sharmatraders.com`), reordered words,
typos, missing addresses (about 3%) and house numbers a few digits off.

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier + assignment decision layer  
**Core Innovation:**
1. A word-level Indic-to-Latin dictionary learned from aligned matched training pairs. It covers
   93% of Indic words in the test set, with no external data.
2. Distractor-aware features: house-number distance, prefix matches and extra-word counts. In the
   first model, 56% of false matches were near-duplicates that these features target.
3. Candidate-context features (rank, gap to the best candidate, margin over the runner-up) computed
   over the full candidate pool, so they have the same distribution at train and test time.

Pipeline: `raw TSVs -> load + integrity checks -> normalise + transliterate -> blocking (top-10
per S2/S3 record) -> pair features -> LightGBM -> one owner per record + threshold -> output files`.

---

## 3. Candidate Generation (Blocking)

Every S2/S3 record queries the Source 1 records **with the same country label** (`blocking.py`).

- **Blocking keys used** (all hashed to 64-bit integers):
  - `n|`: single core-name tokens (legal forms and filler words removed), up to 4 per record
  - `nn|`: order-free pairs of core-name tokens
  - `c|`: the concatenated core name (catches `sharmatraders` vs `sharma traders`)
  - `a|`: address tokens of 3+ characters, and all numeric tokens (house numbers), up to 10
  - `aa|`: adjacent address-token bigrams
  - `na|`: name token x address token combinations
- **Scoring:** keys shared by more than 1,000 S1 records in the country are dropped. Each query
  uses its rarest keys first, up to 24 keys or a total of 10,000 S1 postings. Candidates are
  scored by the summed IDF of shared keys, and the top 10 per query are kept.
- **Candidate pairs generated:** 102.5M on train (10.3M queries); **98.95M on test**
  (France 14.3M, India 46.6M, US 38.0M). That is about 10 per S2/S3 record, versus 5.7 x 10^12
  possible cross-source pairs in test.
- **How true matches were kept:**
  - Blocking is lossless across countries.
  - Text is normalised and transliterated before keys are built.
  - Several independent key families are used, so a record matches even when its name or its
    address is noisy.
  - On training, 98.2% of true links are in the top-10 candidates.
- **Engineering:** blocking runs in chunks of 250k queries and 5M join rows with Polars on 4
  threads, at about 5 GB of RAM for a full split (about 25 minutes). The top-k per query is taken
  with a rank filter rather than `group_by().head()`, which retains the whole chunk table in
  memory.

`output/candidate_pairs.tsv` is exactly the set of pairs the model scores.

---

## 4. Matching Model

Text normalisation (`normalize.py`) applies to both sides before any feature is computed:
- Unicode NFKD, accent stripping and lower-casing.
- Canonical word maps: street types, directions, ordinals, US and Indian state names, French
  street words, and legal forms (`private` -> `pvt`, `incorporated` -> `inc`, ...).
- Leetspeak repair and domain stripping.
- Indic words replaced through the learned dictionary (`translit.py`).

**Features used** (49, `features.py`):
- **Name features:**
  - RapidFuzz ratio and token-set ratio on the full name.
  - Token-sort, token-set, partial ratio and Jaro-Winkler on the core name.
  - Ratio and partial ratio on the space-free core name.
  - First-token equality, exact core-name equality, core-name lengths.
  - Extra-word counts on each side (core-name and full-name level).
  - Legal-form equality and conflict flag.
- **Address features:**
  - Token-set, token-sort and partial ratio on the canonicalised address; missing-address flag.
  - House numbers: counts, overlap, Jaccard, first-number equality, log-distance between first
    numbers and minimum log-distance across numbers, "near but different" flag (1-20 apart),
    prefix flag (`67` vs `6705`).
- **Other:**
  - Blocking score and shared-key count.
  - Candidate context per query: number of candidates, rank and gap by blocking score, rank and
    gap by name+address score, gap to the best name and best address, margin over the runner-up,
    number of candidates with the same core name.
  - Crowding of the S1 record: how many queries retrieved it, and its rank among them.
  - Record flags: source (2/3), Indic script, web-style name, alias marker.
  - Country is deliberately **not** a feature, so the model transfers to France.

**Model type:** LightGBM binary classifier (`num_leaves=127`, `learning_rate=0.05`,
`min_data_in_leaf=100`, feature/bagging fraction 0.8, 2000 rounds).

**Training data** (`train_full.py`, all training entities):
- 10% of S1 entities, with all their records, plus 10% of unmatched records are held out for
  validation.
- From the remaining 90%, every positive and every hard negative is kept: the top-2 candidates of
  the query by name+address score or by blocking score.
- One in ten easy negatives is kept, with weight 10, so the loss stays unbiased.
- In total: 31.9M training rows (6.75M positives) and 10.2M validation rows.

**Threshold selection method:**
- Each S2/S3 record keeps only its highest-probability candidate.
- A threshold tau is swept on **exact macro F0.5** (re-implemented and unit-tested on the worked
  example, 0.714) over the held-out entities.
- Best: tau = 0.20.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9888** on 220,981 held-out training entities (full-data model,
  tau = 0.20). The earlier 12% sample model scored 0.9878 (2-fold out-of-fold); before the
  distractor features, 0.9812.

| tau | 0.10 | 0.15 | **0.20** | 0.25 | 0.30 | 0.40 | 0.50 |
|---|---|---|---|---|---|---|---|
| macro F0.5 | 0.98848 | 0.98877 | **0.98883** | 0.98880 | 0.98867 | 0.98832 | 0.98761 |

Error breakdown (12% sample model, 914,635 true links among sampled records):

| Outcome | Count | Share |
|---|---|---|
| Matched correctly | 885,087 | 96.8% |
| True S1 not in the top-10 candidates (blocking miss) | 16,893 | 1.8% |
| Another S1 ranked first | 7,136 | 0.8% |
| True S1 ranked first but below the threshold | 5,519 | 0.6% |
| False matches | 18,977 | 13,381 from unmatched distractor records, 5,596 to the wrong S1 |

- **Common false positives (wrong merges):** distractor records that match no S1 but look
  almost identical to one: the same name with an extra word (`Sharma Traders` vs `Sharma Traders &
  Sons`), or the same street with a house number a few digits off (`6705` vs `6708`). Branches of
  chains with the same name on nearby streets are the other common case.
- **Common false negatives (missed matches):** mainly blocking misses. They are names with typos
  or merged words sharing no whole token with the S1 name, plus records with missing or very
  different addresses. The rest are Indic names whose words are outside the learned dictionary,
  and heavily abbreviated names.

---

## 6. Conclusion
A multi-key IDF blocking stage, a LightGBM matcher on string, house-number and candidate-context
features, and a one-owner-per-record decision layer reach 0.9888 macro F0.5 on held-out training
entities. Most of the gain over simple string matching came from modelling distractors explicitly
and from a data-driven transliteration dictionary. The main lesson on a 16 GB machine was memory
discipline: chunked blocking, streaming features to disk and chunked scoring made full-data
training possible. The next gains are in blocking recall (the 1.8% of links never retrieved),
for example with an extra character 3-gram TF-IDF nearest-neighbour pass.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` (see its `README.md`; dependencies pinned in `requirements.txt`).
Scripts run from `src/`; set `POLARS_MAX_THREADS=4` on a 16 GB machine.

| Step | Command | Output |
|---|---|---|
| 1 | `python block_split.py train` | transliteration dictionaries, prepared train records, train candidates |
| 2 | `python train_full.py` | `work/model.txt`, `work/config.json` |
| 3 | `python block_split.py test` | prepared test records, test candidates |
| 4 | `python predict.py` | `output/candidate_pairs.tsv`, `output/matching_results.tsv` |

Source modules: `io_utils.py` (loading, integrity checks, writers), `normalize.py`,
`translit.py`, `blocking.py`, `features.py`, `pipeline.py` (shared prep, caching, chunked
featurisation), `decide.py`, `metric.py`. `train.py` is the faster 12% sample variant with 2-fold
out-of-fold validation. `exp_*.py` and `train_eval.py` are exploratory scripts.

No external data, APIs or pre-trained models are used. Every table and dictionary is either
hand-written (canonical word maps) or learned from the provided training data.

### B. Additional Results
- Training time on a 16 GB, CPU-only laptop:
  - Blocking: about 25 minutes per split.
  - Full-data featurisation: about 10 minutes.
  - LightGBM, 2000 rounds on 31.9M rows: about 38 minutes.
  - Test scoring and writing: about 15-25 minutes.
- Most important features by gain: margin over the runner-up, rank by name+address score, gap to
  the best candidate, first-house-number log-distance, blocking-score gap, near-house-number flag,
  extra-word count, house-number Jaccard.

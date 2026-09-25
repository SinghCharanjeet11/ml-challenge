# PRD — Business Entity Resolution (Amazon ML Challenge 2026)

| | |
|---|---|
| **Status** | Draft v1 |
| **Date** | 2026-09-25 |
| **Source documents** | `problem statement/ps_for_amazon.pdf`, `dataset/student_resource/README.md`, `Documentation_template.md`, `utils/validate_submission.py` |
| **Data facts** | Measured directly on the provided files (full-file scans, not samples, unless marked "sampled") |

---

## 1. Problem in one paragraph

We get business records from **three independent sources** that share no common key. **Source 1 (S1)** is the deduplicated reference list. For **every S1 entity in the test set**, we must output the list of **S2 and S3 records that describe the same real-world business** (zero, one, or many). The output is scored with a **per-entity macro-averaged F0.5**, which weights precision twice as heavily as recall: a wrong merge hurts more than a missed link. The data is large (≈1.7M S1 × ≈10M S2+S3 in test), noisy (typos, abbreviations, reorderings, Indic scripts, missing addresses), and the test set has a **country that never appears in training (France)**.

---

## 2. Goals and non-goals

### Goals
1. **Maximise private-leaderboard macro F0.5** on the test set. Final ranking uses the private split only.
2. Produce a **valid** submission every time. A rejected file scores nothing.
3. Generalise to **France** (zero training labels) without special-casing the country list.
4. Ship a **reproducible, auditable** package (code + both output files + methodology doc) that passes the organisers' fair-play and license review.

### Non-goals
- Using any external data source, API, geocoder, or registry (forbidden, see §5.4).
- Deduplicating *within* S2 or S3 as a deliverable. Only S1-anchored matches are scored, though intra-S2/S3 similarity may be used as a feature.
- Real-time or online serving. This is a batch pipeline.

---

## 3. Inputs

### 3.1 Files

All files are **UTF-8, tab-separated, with a header row**.

| File | Rows | Columns | Notes |
|---|---:|---|---|
| `train/train_source1.tsv` | 2,206,821 | entity_id, business_name, business_address, country | US 60.0%, India 40.0% |
| `train/train_source2.tsv` | 5,034,616 | same | US 3.02M, India 2.02M |
| `train/train_source3.tsv` | 5,285,603 | same | US 3.17M, India 2.12M |
| `train/train_ground_truth.tsv` | 2,206,821 | source1_entity_id, matched_entity_ids | one row per train S1 |
| `test/test_source1.tsv` | **1,732,544** | same as sources | **US 38.3%, India 46.8%, France 15.0%** |
| `test/test_source2.tsv` | 4,887,273 | same | US 1.87M, India 2.31M, France 0.70M |
| `test/test_source3.tsv` | 5,082,316 | same | US 1.95M, India 2.41M, France 0.73M |

Total on disk: ≈2.4 GB.

### 3.2 Column semantics
- `entity_id`: prefix `S1-`/`S2-`/`S3-` + a **random integer**. IDs carry no signal: there is zero ID overlap between train and test, and no duplicate IDs in any file.
- `business_name`: noisy name (abbreviations, legal suffixes, typos, transliterations, DBA/aliases).
- `business_address`: noisy address (abbreviations, reordering, missing parts, landmarks).
- `country`: free-text label. **It must be treated as an open set.**
- Ground truth `matched_entity_ids`: comma-separated S2/S3 IDs, empty for singletons.

### 3.3 What the data actually looks like (EDA findings)

**Ground-truth structure (train)**

| Fact | Value | Why it matters |
|---|---|---|
| Singletons (S1 with no match) | 123,247 = **5.6%** | Only a small share, but each is worth a full 1.0 if predicted empty and 0.0 if we add even one wrong match. |
| Matches per S1 | 1: 5.4% · 2: 17.0% · 3: 24.1% · 4: 21.9% · 5: 14.6% · 6: 7.5% · 7+: 4.0% (max 11). **Mean 3.46** | This is a one-to-many problem. Stopping at the single best match caps recall. |
| S2 matches per S1 | 0–5 | Several records from the *same* source can match one S1, so S2 and S3 are not internally deduplicated. |
| S3 matches per S1 | 0–6 | Same as above. |
| Times any S2/S3 ID is matched | **Exactly 1, always** (7,638,365 links, 7,638,365 distinct IDs) | **Each S2/S3 record belongs to at most one S1.** This is a hard constraint we can enforce as an assignment step. |
| Unmatched S2/S3 records (distractors) | S2 26.6%, S3 25.4% | About a quarter of the pool is noise that matches nothing, which creates false-merge risk. |
| Cross-country links | **0** | Matching only within the same country label is lossless. |
| S2+S3 records per S1 | train **4.68**, test **5.75** | The test pool is relatively bigger, meaning more matches per S1, more distractors, or both. Expect precision to be harder on test than on train validation. |

**Field quality**

| Aspect | S1 | S2 / S3 |
|---|---|---|
| Missing name | 0 | 0 |
| Missing address | 0 | ≈3.3% train, ≈2.6% test |
| Non-Latin script | ≈0 | ≈40% of India records (sampled): Devanagari most common, plus Kannada, Telugu, Tamil, Gujarati, Bengali, Malayalam, Gurmukhi, Oriya |
| Postal codes | practically absent (India 6-digit PIN: 0; US ZIP: <0.01%) | practically absent |
| Style | Title Case, clean | S2 is mostly UPPERCASE with abbreviated streets and state codes. S3 is Title Case with full state names. |

**Noise patterns observed in matched pairs** (examples come from the training data)

- *Name:* typos and letter swaps (`Occupational → Occuptinfal`); leetspeak (`Sh0shana`, `5hoshana`, `lnfra`, `Carg0`); random diacritics (`Émpire`, `Ínc`, `Réalty`); dropped or added legal suffixes (`LLC`, `Pvt Ltd`, `Private Limited`, `(Inc.)`, `((Llc))`); added filler words (`Center`, `Services`, `Group`, `Holdings`); word reordering (`INC Ferros PAN`, `PVT MODERN FINANCE LTD`); duplicated words (`ÁNNALY ÁNNALY`); junk prefixes (`--`, `<<`, `#`); **domain or handle forms** (`pediatricmedicine.com`, `#globalsafe`, `rcannaly.com`); **aliases** (`Zephsol fka …`, `Pyraaria dba …`, `Vantagetavo One a/k/a …`); **completely different trade names at the same address** (`TAVOFLUX` ↔ `East Advanced Cohen LLC`); **full transliteration into Indic scripts** (`Shree Best Business Private Limited` ↔ `श्री बेस्ट बिजनेस प्राइवेट लिमिटेड`).
- *Address:* component reordering (`AZ, Tempe, Unit 1324, 1001 Playa Del Norte Drive`); street-type abbreviations (`Rd/Road`, `Pkwy`, `Cv`); state code vs full name (`TN/Tennessee`, `UP/Uttar Pradesh`, `KL/Kerala/Keralam`); state written in native script (`उत्तर प्रदेश`); house-number noise (`2046 ↔ 2048`, `120 ↔ 120-122`, `98 ↔ 103`); injected prefixes (`#`, `##`, `Door No 851`, `HN 361`, `PO Box 7886`); neighbouring or renamed cities (`New Castle ↔ Ossining`, `Aurangabad ↔ CH.SAMBHAJI NAGAR`, `Gurgaon ↔ Gurugram`); city typos (`Nasshville`); street-only or city-only addresses; empty addresses.
- *France (test only):* legal forms `SARL/SAS/SASU/EURL/SCI/SNC/SA`; street abbreviations `R./RUE`, `AV.`, `BD.`, `ALL`, `IMPASSE`, `N°/NO`; region vs département labels (`Hauts-de-France` ↔ `Nord`, `Nouvelle-Aquitaine` ↔ `Gironde`); accents added or removed (`Lyçee`, `MÉRIGNAC`); records concentrated in a few cities (Lille, Roubaix, Bordeaux, Nantes, Saint-Nazaire, …); **very generic names** (`Association des Coeur`, `Clinique Jeanne`, `Espace Comite SA`), which make false merges likely.

**File-parsing hazards**
- About 800 test rows (≈130 in S1) contain CSV-style quoted fields such as `"""ehpad Club SAS"` (train has only 6). Parsing them with quote handling on can merge or split rows.
- About 100 test rows have literal `NA`/`null`/`N/A` values. Pandas turns these into NaN by default.
- The Windows console is cp1252, so printing Indic text crashes unless `PYTHONIOENCODING=utf-8` is set.

---

## 4. Outputs (what we must produce)

### 4.1 `output/matching_results.tsv` (scored)
```
source1_entity_id<TAB>matched_entity_ids
S1-00001<TAB>S2-00047,S2-00193,S3-00812
S1-00002<TAB>S3-00004
S1-00003<TAB>
```
- **Exactly one row per test S1 entity**: 1,732,544 rows plus the header.
- The ID list is comma-joined with no spaces and no quotes. The field is **empty** for singletons.
- Only IDs with an `S2-` or `S3-` prefix that exist in the test files are allowed. No duplicates within a list and no duplicate S1 rows.

### 4.2 `output/candidate_pairs.tsv` (not scored, but audited)
- Same shape with header `source1_entity_id<TAB>candidate_entity_ids`.
- It must be **the exact candidate set the matching model runs inference on**, meaning the last blocking or filtering stage, not an earlier, looser one.
- Every matched ID must also be a candidate (matches ⊆ candidates).
- The organisers use this file to measure blocking recall ceiling and reduction ratio.

### 4.3 Final submission zip
```
<team_name>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md   (filled in; .md or .pdf)
```
The code must regenerate both output files from the raw data using only that folder.

---

## 5. Constraints

### 5.1 Format (a violation means rejection)
1. Tab separator, exact header names, UTF-8.
2. Every test S1 appears exactly once. Missing or duplicate rows are rejected.
3. Only S2/S3 IDs may appear. S1 IDs (self-matches) are rejected.
4. No duplicate IDs inside one list.
5. Use **LF line endings**. Pandas `to_csv` on Windows writes CRLF by default, and the validator's `rstrip("\n")` leaves a `\r` glued to the last ID of each row. The validator does not flag this unless run with `--check-ids`, but the scorer would see an ID that doesn't exist.

### 5.2 Model
- The final model must be under an **MIT or Apache-2.0 license** with **≤ 8B parameters**.
  - Allowed examples (verify the model card at time of use): LightGBM (MIT), XGBoost/CatBoost (Apache-2.0), `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0), `LaBSE` (Apache-2.0), `intfloat/multilingual-e5-*` (MIT), `BAAI/bge-m3` (MIT), `Qwen2.5-7B-Instruct` (Apache-2.0), `Mistral-7B` (Apache-2.0).
  - **Not allowed:** Llama (custom license), Gemma (custom terms), any model over 8B, and any model under a non-commercial license.
- To be safe, keep every library in the pipeline permissively licensed as well. For example, avoid `Unidecode` (GPL) and use `anyascii` (ISC) or a mapping learned from the training data instead.

### 5.3 Evaluation (the metric decides the design)
- Per S1 entity: `F0.5 = 1.25·P·R / (0.25·P + R)`, then a **plain average over all S1 entities**.
- Truth empty and prediction empty scores **1.0**. Truth empty and any prediction scores **0.0**. Truth non-empty and prediction empty scores **0.0**.
- Consequences:
  - Every S1 counts the same, so an entity with 1 match matters as much as one with 11.
  - For an S1 with true matches, predicting **one** correct match already earns F0.5 = 0.833 for k = 2, 0.714 for k = 3, and 0.556 for k = 5 (from P = 1, R = 1/k). Adding a wrong match costs more than a missing one: with k = 3, predicting 2 correct plus 0 wrong gives 0.909, while predicting 3 correct plus 1 wrong gives 0.789.
  - The public leaderboard is only a subset. **Do not overfit thresholds to it.** The final rank uses the private split.

### 5.4 Fair play (a violation means disqualification)
- **Forbidden:** external ER services, business-registry lookups, **geocoding APIs**, and any internet data augmentation.
- The organisers review code, so everything must be reproducible from the provided data.
- **Grey areas, to confirm with the organisers or document explicitly:**
  - Small hand-written normalisation tables: street-type abbreviations, US state and India state codes, legal-suffix lists. These are domain rules rather than data lookup, but must be declared.
  - Offline transliteration libraries.
  - Pretrained multilingual embedding models. These are allowed by the license rule, but their use should be stated.
  - Unsupervised use of **test** features, for example fitting IDF on the test corpus. This is not prohibited, but should be stated.

### 5.5 Compute (this machine)
- Intel i5-13500H (16 threads), **15.6 GB RAM** (often under 1 GB free with other apps open), **no CUDA GPU** (Intel Iris Xe only), **≈12 GB free disk**.
- Python 3.11, pandas 3.0, numpy 2.4, reportlab. **Missing:** rapidfuzz, lightgbm, polars, faiss. **Broken:** torch 2.1 CPU does not work with numpy 2.x. A separate virtual environment is needed.
- Implications:
  - Process one country at a time, in chunks.
  - Store IDs as int64 and strings as Arrow or categorical types.
  - Never build the full cross product (India alone is ~0.8M × 4.7M ≈ 3.8·10¹² pairs).
  - CPU transformer embeddings for about 22M records take many hours, so restrict them to the records that need them, or use a free cloud GPU for that step only. That is compute, not data, so it is allowed.

---

## 6. Solution design

### 6.1 Pipeline overview
```
raw TSVs
  └─► [0] Robust load + integrity checks
       └─► [1] Normalise (name variants, address parse, transliteration)
            └─► [2] Blocking — multi-pass, per country label, S2/S3 → S1 top-K
                 └─► candidate_pairs.tsv   (exact model inference set)
                      └─► [3] Pairwise features
                           └─► [4] Matcher (GBDT, optional cross-encoder re-rank)
                                └─► [5] Decision layer (assignment + F0.5-optimal selection)
                                     └─► matching_results.tsv ─► validator ─► submit
```

### 6.2 Stage requirements

**[0] Load**
- Read with `pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, na_filter=False, encoding="utf-8")`, or with pyarrow/polars using equivalent options.
- **Assert** that each row count equals `wc -l − 1` and that every row has 4 fields.
- Parse IDs into (source, int64).

**[1] Normalisation.** Keep the raw text alongside the normalised text; features use both.
- Text: NFKC, casefold, keep an accent-stripped copy, strip quotes and junk punctuation and prefixes, collapse whitespace.
- Leetspeak repair inside alphabetic tokens (`0→o, 1→l/i, 5→s, 3→e`).
- **Name variants:**
  - Split aliases on `dba | d/b/a | fka | f/k/a | aka | a/k/a | formerly` into several name variants.
  - Parse domains and handles (`x.com`, `#x`) into a concatenated-name key.
  - Extract the legal form to a canonical value (`pvt|private→private`, `ltd|limited→limited`, `llc`, `inc|incorporated`, `corp|corporation`, `co|company`, `sarl`, `sas`, `sasu`, `eurl`, `sci`, `snc`, `sa`, …) and keep a "core name" without legal form or filler words.
- **Transliteration of Indic scripts to Latin.** Prefer a **mapping learned from the training pairs**: S1 English names are aligned with S2/S3 Indic names of the same entity, which yields token dictionaries such as `प्राइवेट→private` and `लिमिटेड→limited`. Use `anyascii` or `indic-transliteration` as a fallback, then apply phonetic folding (`shree/shri/sri`).
- **Address parse:**
  - Extract house number(s) and ranges, unit/suite/flat/door number, PO box, street name plus canonical street type, city, and state/region.
  - Canonicalise US state codes, India state codes (including native-script names and `Keralam`), and French region/département pairs. Where possible, *learn* the French pairs from co-occurrence in the test data rather than hard-coding them.
  - Handle known city renames through a small alias table learned from training co-occurrence.

**[2] Blocking.** The recall ceiling is set here.
- **Partition by the exact `country` string** found in the data (open set, no hard-coded list). This loses no true links, since there are 0 cross-country links.
- **Query direction: each S2/S3 record retrieves its top-K S1 candidates.** This follows from the constraint that each S2/S3 record has at most one owner, and it keeps the candidate count bounded by K × 10M.
- Union of passes:
  1. Rare-token keys on core name: 1–2 highest-IDF tokens, with a cap on block size.
  2. Char 3-gram TF-IDF cosine top-K on core name (sparse chunked matmul or ANN).
  3. Address key: (house number, first street token) and (state, city, house number).
  4. Concatenated-name key, so domains and handles match spaced names.
  5. Transliterated and multilingual-embedding kNN for non-Latin records.
  6. An address-only pass for records whose name is an unrelated trade name.
- **Targets on validation:** pair recall ≥ 98% of true links. Report the mean candidates per S2/S3 record and the reduction ratio, and keep the `candidate_pairs.tsv` size reasonable (under ~1 GB).

**[3] Features.** Country-agnostic: **country is not a model feature**, so France is not out-of-vocabulary.
- Name:
  - Jaro-Winkler, normalised Levenshtein, token-set, token-sort and partial ratios on raw, core, and transliterated forms.
  - Char n-gram TF-IDF cosine, IDF-weighted token Jaccard, first-token equality, acronym match.
  - Legal-form agree/conflict/missing, alias-variant best match, concatenated or domain match, embedding cosine.
  - Script flags (Latin vs Indic) and name length.
- Address:
  - House number exact / near (±small) / range-contains / conflict / missing.
  - Street-name similarity, street-type agreement, city match or fuzzy match, state match, unit match.
  - Address missing on either side, and address token Jaccard.
- Context:
  - Source (S2 vs S3, whose noise styles differ).
  - Rank of this S1 among the S2/S3 record's candidates, score gap to the runner-up, candidate-block size.
  - Rarity (IDF) of the shared name tokens: generic names are riskier.

**[4] Matcher**
- Baseline: **LightGBM** binary classifier on candidate pairs. Labels come from the ground truth, and the model is trained with GroupKFold by S1 ID.
- Optional: a fine-tuned multilingual cross-encoder (≤ 8B, MIT/Apache) re-ranks only the uncertain band (e.g. 0.2 < p < 0.8). This should run on a cloud GPU.

**[5] Decision layer** (turns probabilities into the submission)
1. **Assignment:** each S2/S3 record goes to its best-scoring S1 only. It is dropped if p < τ, or if the gap to the second-best S1 is below a margin.
2. **Per-S1 selection:** sort the assigned candidates by p and keep the prefix that maximises *expected* F0.5 under the calibrated probabilities. This handles singletons naturally, since an empty list is chosen when nothing is confident.
3. Tune τ, the margin, and calibration on **out-of-fold** predictions against the exact macro F0.5 metric. Prefer settings that are stable across US and India over ones that are best for a single country.

### 6.3 Validation protocol
- **Metric:** a local re-implementation of per-entity F0.5 that follows §5.3 exactly, including the singleton and empty-prediction rules. It is unit-tested on the PDF example (expected 0.714).
- **Main estimate:** out-of-fold predictions over all training S1 entities (5-fold GroupKFold by S1). Blocking runs over the full S2/S3 pool so that distractors are realistic.
- **France proxy:** **leave-one-country-out**. Train on US and score India, and the reverse. The gap indicates how much to discount for France and guides which features to trust.
- **Blocking report:** recall ceiling, candidates per record, and the F0.5 upper bound given the candidates.
- Record every experiment: config, validation score, and leaderboard score if submitted.

---

## 7. Milestones

| # | Milestone | Exit criteria |
|---|---|---|
| M0 | Environment + loader + metric + writer | Clean virtual environment, pinned requirements, loaders pass the row-count asserts, metric unit tests pass, and a trivial all-empty submission passes the validator. |
| M1 | Rule baseline on the leaderboard | Normalisation plus exact and rare-token blocking plus a hand threshold on name/address similarity. First SCORED submission; baseline logged. |
| M2 | Blocking v2 | Multi-pass blocking reaches ≥ 98% pair recall on training validation within the memory budget. `candidate_pairs.tsv` is generated. |
| M3 | ML matcher | LightGBM with the full feature set. Out-of-fold macro F0.5 clearly beats M1. |
| M4 | Multilingual + France robustness | Learned Indic transliteration, French normalisation rules, leave-one-country-out gap measured and reduced. |
| M5 | Decision layer | Assignment plus expected-F0.5 selection. Thresholds tuned out-of-fold and checked on the leaderboard. |
| M6 | Package | Zip built, fresh-clone reproduction run end-to-end, `Documentation_template.md` filled in, validator PASS (also with `--check-ids`). |

---

## 8. Do / Don't checklist

**Do**
- Always read with `sep="\t"`, `dtype=str`, `quoting=QUOTE_NONE`, and `keep_default_na=False`, and assert row counts.
- Write with `sep="\t"`, `index=False`, `lineterminator="\n"`, and `encoding="utf-8"`.
- Treat `country` as an arbitrary string. Group by it and never enumerate it.
- Include **every** test S1 in both output files, including France and singletons.
- Enforce "each S2/S3 record belongs to at most one S1".
- Optimise thresholds for **macro per-entity F0.5**, not pairwise F1 or AUC.
- Make `candidate_pairs.tsv` exactly the model's inference set, and make sure matches ⊆ candidates.
- Run `python utils/validate_submission.py --matching … --candidate … --test-dir dataset/test` before every upload, and occasionally add `--check-ids`.
- Pin versions, fix random seeds, and avoid absolute paths in the shipped code.
- Record which model and library licenses are used, and every hand-written rule table, in the methodology document.

**Don't**
- Don't call any external API or geocoder, or download any business, address, or registry data.
- Don't one-hot, filter, or hard-code `{US, India}`, and don't drop rows whose country is unseen.
- Don't use Llama, Gemma, or other non-MIT/Apache models, or any model over 8B parameters, in the final pipeline.
- Don't predict a match just because it is the "best available". On a singleton, one false match turns 1.0 into 0.0.
- Don't tune against the public leaderboard, which is only a subset.
- Don't assume ZIP or PIN codes exist. They are essentially absent from all sources.
- Don't rely on train validation scores for France. Expect a drop and leave precision headroom.
- Don't write CRLF, quoted, or comma-separated output files, and don't put spaces after commas in ID lists.

---

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| France distribution shift (unseen country, generic names, 15% of test) | Precision drop on 15% of entities | Country-agnostic features, IDF-aware genericity features, leave-one-country-out tuning, more conservative decision margin. |
| Test pool relatively larger (5.75 vs 4.68 S2/S3 per S1) | More distractors, so more false merges | Assignment constraint, runner-up margin, precision-leaning τ. |
| Indic-script records (~40% of Indian S2/S3) | Recall loss if not transliterated | Transliteration learned from training pairs plus an address-driven blocking pass. |
| RAM (16 GB, little free) | Crashes or thrashing | Per-country chunking, int IDs, Arrow strings, on-disk intermediate files (parquet). |
| No GPU | Slow embeddings or cross-encoder | GBDT on string features first. Embeddings only for the hard subset, or on a cloud GPU. |
| Parsing bugs (quotes, NA, CRLF) | A silently lost row means rejection | Loader asserts, validator run on every output, LF writer. |
| Rule-table or library "external data" interpretation | Disqualification | Keep tables small and derived from the data where possible, and disclose everything. |

---

## 10. Open questions (for the team or organisers)
1. Competition deadline and **daily leaderboard submission limit** (not stated in the materials).
2. Team name and members, needed for the zip and the documentation.
3. Are hand-written normalisation tables and offline transliteration libraries acceptable? (Our reading is yes, since they are not data lookups, but we will disclose them.)
4. Is unsupervised use of test-set text (IDF fitting, co-occurrence mining) acceptable? (Not prohibited. We will disclose it.)
5. Does the MIT/Apache rule cover only the "final model" or every dependency? (Assume every dependency to be safe.)
6. The problem statement links a video ("Click Here") that is not included in the files. Someone should watch it for any extra rules.

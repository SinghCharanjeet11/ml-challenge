# Current pipeline architecture (production submission, leaderboard 0.964)

Scope: the only pipeline in the repository, `code/business_entity_resolution/src` (git `main` =
`full-training` after PR #1). It produced the uploaded `output/matching_results.tsv`
(v2 blocking + `train_full.py` model, holdout macro F0.5 0.9905, public leaderboard 0.964).
Nothing in this document changes production code.

## Execution flow

```
train/test TSVs (S1, S2, S3, train ground truth)
   │  io_utils.read_tsv / load_split
   │    all columns as strings, quoting OFF, NA-conversion OFF, row counts asserted;
   │    S2 and S3 stacked into one "others" table with src = 2 / 3
   ▼
learned dictionaries (train only)                       translit.py
   │  Indic word -> Latin word, from matched train pairs with equal word counts
   │    (min_count 2, min_share 0.5); Indic address component -> last S1 address component
   │    (min_count 5, min_share 0.6). Saved as work/name_dict, work/addr_dict.
   ▼
preprocessing                                           pipeline.prep_records, normalize.py, features.prepare
   │  S2/S3 text transliterated (unknown Indic words kept as-is) -> clean_expr:
   │    NFKD, accents stripped, lower case, "&" spaced, .com/.in/.net/.org/.co/.fr/www removed,
   │    punctuation -> space; multi-word US/Indian state names -> codes (address only)
   │  per-record columns: nm, ad, core (name minus legal/filler words, leetspeak fixed,
   │    NAME_CANON applied), legal (legal-form tokens), adc (ADDR_CANON + state maps),
   │    nums (unique digit runs in address), num0 (first number), flags indic / webname / alias
   │  cached: work/<split>_s1p.parquet, work/<split>_op.parquet
   ▼
candidate generation ("v2")                             block_split.py -> pipeline.block_cached
   │  per country label (open set, loops over whatever labels exist):
   │  (a) key blocking (blocking.py): hashed keys = core name tokens (<=4), core-token pairs,
   │      joined core name, address tokens (>=3 chars or numeric, <=10), address bigrams,
   │      name x address token combos. Keys with S1 document frequency > 1000 dropped.
   │      Each S2/S3 query uses its rarest keys first (<=24 keys, <=10,000 postings);
   │      score = sum of IDF of shared keys; keep top 20 per query.
   │  (b) char 3-gram TF-IDF on core name (tfidf_blocking.py): trigrams in > 2% of S1 names
   │      dropped; sparse top-10 cosine per query (sim >= 0.3) with sparse_dot_topn.
   │  (a) ∪ (b) merged; missing side's score = 0.   cached: work/<split>_cand_v2.parquet
   ▼
features                                                pipeline.featurise_chunks, features.build_features
   │  S1-side context over the FULL candidate table: n_cand_1 (queries retrieving this S1),
   │    rank_block_1; keep_k = 0 (no trimming in v2)
   │  per pair (50): RapidFuzz name/core/concat ratios, JW, address token set/sort/partial,
   │    house-number counts/overlap/Jaccard/first-equal/log-distance/near/prefix, legal-form
   │    equal/conflict, extra-word counts, lengths, flags, tfidf_sim, block_score, n_shared,
   │    query-side context (rank/gap by block score and by name+address "combo", margin over
   │    runner-up, same-core count). Country is NOT a feature.
   │  chunks of 400k queries
   ▼
model                                                   train_full.py (training), predict.py (inference)
   │  LightGBM binary, 127 leaves, lr 0.05, 2000 rounds, early stopping on the holdout
   │  training rows: all positives + top-2 hard negatives per query (by combo or block rank)
   │    + 1 in 10 easy negatives with weight 10
   ▼
selection                                               decide.assign
   │  each S2/S3 record -> its single highest-probability S1 candidate (one-owner rule),
   │  kept only if p >= tau (0.20, chosen on the holdout). No S1-side cap, no joint decisions.
   │  Multi-match = several records independently choosing the same S1.
   │  Singleton / no-match = an S1 that no record chose above tau (no explicit no-match model).
   ▼
output                                                  io_utils.write_id_lists_chunked
   output/candidate_pairs.tsv  = every pair the model scored (all v2 candidates)
   output/matching_results.tsv = assigned pairs; one row per S1 (empty when none);
                                 written in 200k-S1 slices, LF endings, sorted unique ids
```

## Component map

| Concern | Where |
|---|---|
| Data loading | `io_utils.read_tsv`, `io_utils.load_split` |
| Normalisation | `normalize.clean_expr`, `add_clean_columns`, word maps; `features.prepare` |
| Transliteration | `translit.py` (dictionaries learned from train matches) |
| Candidate generation | `blocking.py` (keys), `tfidf_blocking.py` (char 3-grams), merged in `pipeline.block_cached` |
| Caching / indexing | `pipeline.prep_cached`, `pipeline.block_cached` (parquet in `work/`) |
| Feature engineering | `features.build_features`, `pipeline.top_k_with_s1_context`, `pipeline.featurise_chunks` |
| Train/validation split | `train_full.py`: entity-level, `hash(i1, seed=5) % 10 == 0` -> holdout |
| Model training | `train_full.py` (production), `train.py` (sample variant) |
| Thresholding + multi-match + singletons | `decide.assign` (argmax per record, then `p >= tau`) |
| Inference | `predict.py` (chunked scoring) |
| Configuration | `work/config.json` (tau, keep_k, blocking version, feature list), `pipeline.BLOCK_PARAMS`, `V2_*` |
| Evaluation | `metric.macro_f05`; threshold sweep inside `train_full.py` |
| Submission generation | `predict.py` -> `io_utils.write_id_lists_chunked`; validator in `dataset/student_resource/utils` |
| Tests | `metric.py` self-tests (`python metric.py`) |

## Where information can be lost

1. **Normalisation.** Punctuation and domain suffixes are removed, and `&` / legal words are
   canonicalised, so `A&B Co` and `AB Co` can collapse or diverge. `normalize.ALIAS_SPLIT` is defined
   but never used: `dba` / `aka` names are not split into their two parts, only flagged.
2. **Transliteration.** Indic words missing from the learned dictionary stay in their script and can't
   match Latin text. Only names with equal word counts contribute to the dictionary.
3. **Column trimming.** `prep_records` keeps only derived columns; the raw name/address are not
   available to features.
4. **Key blocking.**
   - Keys shared by more than 1,000 S1 records are dropped (very common names and streets).
   - Each query is capped at 24 keys and 10,000 postings.
   - Only the top 20 per query are kept.
5. **TF-IDF blocking.**
   - It uses the name only.
   - Trigrams found in more than 2% of names are dropped.
   - Similarity must be ≥ 0.3, and only the top 10 per query are kept.
6. **Country blocking.** Lossless on train (0 cross-country links).
7. **Selection.** One owner per record: if the wrong S1 is ranked first, the true link is lost even
   when it scored high. Threshold `tau` is global (all countries, sources, and candidate crowding).
8. **No S1-side reasoning.** Each record decides alone; there is no comparison between records
   competing for the same S1, no per-S1 cap and no explicit "this S1 is a singleton" signal.
9. **Validation vs test mismatch.** Test has more S2/S3 records per S1 (about 5.8 vs 4.7), so it has
   a larger share of records that match nothing. The holdout can't show the effect of that
   (see the forensic report).

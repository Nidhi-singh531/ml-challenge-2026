# Business Entity Resolution — ML Challenge 2026

Pipeline for matching business records across three noisy sources.
**Two-channel blocking (IDF-weighted inverted indexes) → pairwise GBDT →
per-country threshold + one-to-one assignment.** Pure Python / NumPy / SciPy /
scikit-learn (LightGBM and rapidfuzz optional). No external data, no network
calls, no pretrained models.

> This file is the full handoff context. If you are an agent picking this up:
> read §2 (measured facts), §7 (rules) and §8 (streaming refactor complete;
> classifier optimization complete, AWS test output pending) before changing anything. Several non-obvious decisions here were
> driven by measurement, not intuition, and reverting them will silently cost
> score.

---

## 1. The task and the metric

Source 1 is the deduplicated reference source. For **every** Source-1 record,
output the set of Source-2 / Source-3 records referring to the same real
business. A Source-1 entity may match zero, one, or many records.

Scored by **macro-averaged F₀.₅**, computed per Source-1 entity and then
averaged over all of them, singletons included:

```
F_0.5 = (1.25 × P × R) / (0.25 × P + R)
```

- Precision is weighted 2× recall. A false merge costs more than a missed link.
- A singleton scores **1.0** for an empty prediction and **0.0** for any
  prediction.
- Averaging is per entity, so the marginal candidate on a 2-member cluster
  costs exactly as much as one on an 8-member cluster.
- **Recall is cheap.** With a fraction *m* of an entity's links missed and no
  false positives, F₀.₅ ≈ 1 − 0.2·m. A +1 pt blocking-recall gain is worth only
  ~+0.002 macro F₀.₅; one false merge on a singleton costs a full 1.0.

Hard constraints: every Source-1 entity gets exactly one row; only S2-/S3- IDs;
no duplicates within a list; TSV. **External data lookup of any kind is
disqualification.** Final model must be MIT/Apache-2.0 and ≤ 8B params (a GBDT
is trivially fine).

---

## 2. Measured facts about the data

| Fact | Value | Implication |
| --- | --- | --- |
| Rows train S1 / S2 / S3 | 2.21M / 5.03M / 5.29M | blocking decides everything |
| Rows test S1 / S2 / S3 | 1.73M / 4.89M / 5.08M | test adds **France** (259k S1, 1.43M targets) |
| Ground-truth links | 7.64M | mean 3.46 matches per entity |
| Cluster sizes | 0: 5.6%, 1: 5.4%, 2: 17%, 3: 24%, 4: 22%, 5+: 26% | few singletons |
| Each S2/S3 ID appears in exactly one S1 row | all 7,638,365 | a partition → one-to-one constraint |
| `country` agreement on true pairs | 100.00% | hard partition by country |
| True pairs sharing ≥1 token (name+addr) | 100.00% | token blocking has a perfect ceiling |
| Non-Latin name on the S2/S3 side | 6.7% | Devanagari, Bengali, Gujarati, Odia, Tamil, Telugu, Kannada |
| IDs | all `S<k>-<int < 1e9>`, no leading zeros | packed losslessly into int64 (`blocking.encode_id`) |
| **Blocking recall, mini subset vs full pool** | **0.988 vs 0.946** (single channel, K=75) | the mini subset is badly optimistic, see §8 |

**Noise operators observed** (synthetic, a fixed set worth targeting):

- *Names*: leet (`8rands`, `Visi0n`), typos, token reordering, legal-suffix
  add/drop, filler words, web-ified names (`securecloudservices.com`,
  `APPLIANCEPLATINUM.COM`, `#ridgefellowship`), `DBA:` / `aka` prefixes with a
  fabricated brand, script transliteration, accent insertion, and names
  replaced outright by a fabricated word (`Solgildriza`, `KELOYUMA`) with the
  address intact.
- *Addresses*: `Rd`↔`Road`, `TX`↔`Texas`↔`महाराष्ट्र`, city variants,
  house-number mutation (`17337`→`7337`, `669`→`0669`), component drop and
  reorder, `<NULL>` placeholders, **truncation to 2–3 components**
  (`'501, Ahmedabad, GJ'`).

Nothing may be hard-coded to `{US, India}`. Every rule is language-agnostic and
`country` is an open set of labels.

---

## 3. Layout

```
ml-challenge-2026/
├── .vscode/launch.json             <- one debug config per stage (mini)
├── dataset/{train,test,mini}/
├── utils/validate_submission.py    <- organiser-provided validator
├── output/      matching_results.tsv, candidate_pairs.tsv   (the submission)
├── work/        intermediates (never zipped); work/runlog.txt = experiment log
└── code/business_entity_resolution/
    ├── src/
    │   ├── ertext.py       normalisation, tokens, aliases, romanisation, skeletons, name keys
    │   ├── aliases.py      mines the alias lexicon (+ bootstrap merge mode)
    │   ├── blocking.py     two-channel index, top-K, token cache, parallel queries
    │   ├── features.py     32 text features + 15 context features
    │   ├── pipeline.py     build / train / predict / evaluate
    │   ├── analyze.py      recall curves per channel + error decomposition
    │   └── make_subset.py  carve a small dev subset
    ├── README.md
    └── requirements.txt
```

## 4. Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r code\business_entity_resolution\requirements.txt
pip install rapidfuzz lightgbm     # installed in .venv already
```

Commands below run from the workspace root. Set `$env:PYTHONIOENCODING="utf-8"`
before printing Indic examples in PowerShell.

---

## 5. Pipeline stages

Let `$s = "code/business_entity_resolution/src"`.

### 5.0 Dev subset (once)

```powershell
python $s/make_subset.py --data-dir dataset/train --out dataset/mini --clusters 8000 --distractors 120000
```

Fine for checking that code runs. **Do not use it to choose K, caps or
features:** it keeps ~5% of distractors, so blocking recall is 4 pts too high.

### 5.1 Alias lexicon

```powershell
# mini (fast)
python $s/aliases.py --data-dir dataset/mini --prefix train `
  --ground-truth dataset/mini/train_ground_truth.tsv --out work/aliases.tsv
# full (≈ 5 min, 1,708 entries) - use this for anything on full data
python $s/aliases.py --data-dir dataset/train --prefix train `
  --ground-truth dataset/train/train_ground_truth.tsv `
  --out work/aliases_full.tsv --max-clusters 400000
```

### 5.2 Blocking

```powershell
python $s/blocking.py --data-dir dataset/train --prefix train `
  --out work/dev/candidate_pairs.tsv --aliases work/aliases_full.tsv `
  --topk 50 --name-topk 25 --gen-df-cap 20000 --name-gen-df-cap 20000 `
  --cache work/tokcache_train.npz --workers 5 --no-tsv `
  [--store-sample 200000 | --s1-sample 3000]
```

| flag | meaning |
| --- | --- |
| `--topk` / `--name-topk` | K of the `all` channel / `name` channel; the output is their union (~69 per entity at 50/25) |
| `--gen-df-cap` / `--name-gen-df-cap` | tokens above this df only *add score* to candidates, they do not *generate* them (§6) |
| `--df-cap` | tokens above this df are ignored entirely (200k) |
| `--cache` | token-id cache. Written if missing, reused if present. **Delete it after changing aliases, `ertext` tokenisation or `name_keys`.** Tokenising the full train set takes ~12 min; loading the cache takes 2 s |
| `--workers N` | parallel query processes sharing the index through memory maps (verified byte-identical to serial) |
| `--store-sample N` | query **every** S1 record (faithful contention statistics) but store pairs for only N random ones: **the realistic dev run** |
| `--s1-sample N` | query only N random S1 records (fast blocking-recall probe; contention features are **not** faithful in this mode) |
| `--no-tsv` | skip the submission-format candidate TSV |
| `--part-pairs` | pairs per sidecar part (2M) |

Writes the sidecar directory `<out>_scores/`:
- `part_NNNN.npz`: pairs, streamed to disk as they arrive, with each entity's
  list contiguous. Columns: `s1, cand` (int64 codes), `score, shared, rank`
  (`all` channel), `nscore, nshared, nrank` (`name` channel; `nrank = 999`
  means not in the name top-K).
- `meta.npz`: `queried`, `stored`, and per-target contention statistics
  (`t_codes` sorted, `t_best`, `t_second`, `t_claims`). These are accumulated
  streaming over *every* query by `TargetStats` and match a full recomputation
  exactly.

Every consumer (`build`, `analyze`) takes the directory, or the `.tsv` /
`_scores.npz` path, which resolves to it.

### 5.3 Features

```powershell
python $s/pipeline.py build --data-dir dataset/mini --prefix train `
  --cand-scores work/mini/candidate_pairs_scores `
  --ground-truth dataset/mini/train_ground_truth.tsv `
  --aliases work/aliases.tsv --out work/mini/pairs.npz --workers 4 `
  [--max-s1 40000] [--prefilter work/model.pkl]
```

Streams over the sidecar parts one at a time; nothing is ever held for the
whole run. For each part it:
1. selects the stored entities (optionally subsampled with `--max-s1`);
2. computes context features, with contention read from `meta.npz`;
3. applies the prefilter;
4. computes text features in `--workers` processes.

Writes a manifest `pairs.npz` (entities, their countries, feature names) plus
`pairs.partNNN.npz` shards. Prints the **recall ceiling**, which no classifier
change can exceed.

### 5.4 Train, predict, evaluate

```powershell
python $s/pipeline.py train --pairs work/mini/pairs.npz `
  --ground-truth dataset/mini/train_ground_truth.tsv --out work/mini/model.pkl `
  [--refit] [--drop-features f1 f2 ...] [--unseen max|global]

python $s/pipeline.py predict --pairs work/mini/pairs.npz --model work/mini/model.pkl `
  --source1 dataset/mini/train_source1.tsv --out work/mini/matching_results.tsv `
  [--candidates-out ...] [--confident-out ... --confident-threshold 0.98]

python $s/pipeline.py evaluate --results work/mini/matching_results.tsv `
  --ground-truth dataset/mini/train_ground_truth.tsv
```

`train` holds out 25% of entities (a stable split over sorted entity codes) and
then:

- sweeps a global threshold and **one threshold per country**; a country
  absent from training (France) gets the **most conservative learned
  threshold** (`--unseen max`);
- fits a **stage-1 prefilter**, a small GBDT on context features only, whose
  threshold keeps 99.9% of the links the full model accepts. On mini it drops
  91.5% of pairs for −0.0004 F₀.₅. `build --prefilter model.pkl` applies it
  before the expensive text features;
- prints the oracle F₀.₅ over candidates;
- `--drop-features` neutralises columns for ablations on the same split;
- `--refit` retrains on all pairs after validation (use for the final model).

### 5.5 Diagnostics

```powershell
python $s/analyze.py blocking --cand-scores <sidecar>.npz `
  --ground-truth dataset/train/train_ground_truth.tsv --data-dir dataset/train
python $s/analyze.py errors --pairs <pairs>.npz --model <model>.pkl `
  --ground-truth dataset/train/train_ground_truth.tsv --data-dir dataset/train
```

`blocking` prints recall of the name channel alone and of the union, plus
examples of links never retrieved. (Its `recall@K` rows rank by the `all` score
inside the union and are no longer meaningful with two channels.) `errors`
splits the loss into blocking / FP / FN, prints feature importances, and dumps
concrete examples. **The examples tell you what to fix next.**

### 5.6 Test set → submission

The first command (test blocking) can instead run on AWS; see §8. Its output
is the same `output/cp_all_scores/` directory.

```powershell
python $s/blocking.py --data-dir dataset/test --prefix test --out output/cp_all.tsv --no-tsv `
  --aliases work/aliases_full.tsv --topk 50 --name-topk 25 --gen-df-cap 20000 `
  --name-gen-df-cap 20000 --cache work/tokcache_test.npz --workers 6
python $s/pipeline.py build --data-dir dataset/test --prefix test `
  --cand-scores output/cp_all_scores --aliases work/aliases_full.tsv `
  --prefilter work/dev/model40k.pkl --out work/pairs_test.npz
python $s/pipeline.py predict --pairs work/pairs_test.npz --model work/dev/model_optimized.pkl `
  --source1 dataset/test/test_source1.tsv --out output/matching_results.tsv `
  --candidates-out output/candidate_pairs.tsv --confident-out work/test_confident.tsv

# optional: alias bootstrap on test (queue item 5), then rerun blocking->predict
python $s/aliases.py --data-dir dataset/test --prefix test `
  --ground-truth work/test_confident.tsv --merge-with work/aliases_full.tsv `
  --out work/aliases_boot.tsv

python utils/validate_submission.py -m output/matching_results.tsv `
  -c output/candidate_pairs.tsv -t dataset/test
```

`output/candidate_pairs.tsv` is the set the classifier actually scored (after
the prefilter), written by `predict --candidates-out`. The validator must
print `PASS` before every upload.

---

## 6. Design decisions worth preserving

**Partition by country.** 100% agreement on true pairs, so it is free. France
appears as a third partition at test time.

**Two blocking channels, unioned.** On the full 10M-record pool, most links the
original single index missed had an *intact name* but an empty or truncated
address. A long Source-1 address lets hundreds of neighbours outscore them
(`'Pune Human Pvt. Ltd.' | ''`). The `name` channel has its own top-K over
name-only keys (`ertext.name_keys`):
- the canonical name tokens;
- `#` + concatenations (every prefix run, every order of ≤3 core tokens), which
  reach web-ified names;
- `~` + consonant skeletons, which reach other-script names.

Every pair in the union is scored in both channels, whichever channel found it.

**Two tiers of tokens.** One token with 100k+ postings dominates query time. So
only tokens with df ≤ `gen-df-cap` *generate* candidates, through a batched
sparse product `Q @ P`. Commoner tokens (city, state, 'private') only *add
score* to generated candidates, read from the per-record token lists (the
cache doubles as the forward index). A bound (rare score + maximum possible
common score < current K-th best) skips candidates that cannot enter the top-K
before that lookup. Score and shared-token count come out of one product via
the weight `2^20 + idf²`.

**Mechanical romanisation, no tables.** `ertext.romanize` uses only the stdlib
Unicode character names (`DEVANAGARI LETTER SHA`, `... VOWEL SIGN I`,
`... SIGN VIRAMA`) plus the abugida inherent-vowel rule. That covers all
Brahmic scripts in U+0900–U+0DFF. `skeleton` removes vowels and merges
consonant classes blurred by transliteration (c/k/q/g/j, d/t, b/p/f/v/w, z/s):
`शिव प्रोडक्ट्स` and `Shiva Products` both become `sv prtkts`. It feeds blocking
keys and the `name_skel_*` features.

**Index all tokens, rank by IDF; don't filter to rare tokens.** Rare-token
blocking caps recall at 92.6%.

**One-to-one assignment.** Every S2/S3 record belongs to exactly one S1 entity,
so only the highest-probability claim on a target is kept. Free precision under
F₀.₅.

**Indic scripts need a custom tokenizer.** `ertext._TOKEN_SPLIT` keeps
`U+0900–U+0DFF` plus ZWJ/ZWNJ inside tokens, and `strip_accents` only strips
marks on Latin bases. **Don't "simplify" either.** The ASCII fast path in
`normalize` produces identical output, just 3× faster.

**`<NULL>` / `null` are filtered** (`ertext.NULL_TOKENS`).

**Alias mining needs consistency and a word filter, not just PMI.** PMI alone
maps `rd → buchert`, so the canonical must accompany the variant in ≥30% of its
occurrences. Latin variants need 10× the evidence. Latin variants that are
ordinary Source-1 words are rejected (`max_word_ratio`): otherwise
`new → ny` (from New York) would also turn New Delhi and New Jersey into `ny`.
Also rejected this way: `west → wv`, `llc → pc`, `of → dc`, `tn → tamil` (TN is
also Tennessee). Known casualty: `ltd → limited`, harmless since both are legal
suffixes.

**Conflict features matter as much as similarity** (`num_conflict`,
`extra_core`, `missing_core`, `core_disjoint`): the dominant false positive is
the same building with a different unit.

**Contention features need every S1 record queried.** `target_n_claims`,
`target_best_ratio`, `rev_rank` and `target_margin` describe how the *other*
S1 entities score the same target. In `--s1-sample` mode only 0.1–2% of S1
queries the index, so these features are distorted. Faithful dev runs use
`--store-sample`: every S1 record is queried and feeds the streaming
`TargetStats` (best / second-best score and claim count per target), but pairs
are stored for only a sample. `rev_rank` is therefore 0 / 1 / 2 (best claimant
/ runner-up / further down), which is exactly what a streaming top-2 can
support; on mini it cost nothing measurable.

**Scale engineering.**
- IDs are int64 end to end.
- Record texts live in one bytes blob (`pipeline.TextStore`, ~1 GB instead of
  ~3 GB of Python strings).
- Pair files are sharded.
- Blocking queries run in parallel on memory-mapped shared indexes.
- The machine has 16 GB but only **~5 GB actually available**, so design for
  that.

---

## 7. Rules for changing this code

1. **Measure before you change.** Run `analyze.py`, read the printed examples,
   and fix what you actually see.
2. **Measure on the full target pool, not mini** (§2, last row). Use a
   `--s1-sample 3000` probe for blocking questions: 20 s once the cache
   exists.
3. **Rerun matrix.** Edited `ertext.py` / `aliases.py`, or blocking keys →
   delete the token cache → 5.1 → 5.2 → 5.3 → 5.4. Edited `features.py` →
   5.3 → 5.4. Model params only → 5.4.
4. **A change worth under 0.003 on 10k validation entities is a tie.** Keep the
   simpler version.
5. `TEXT_FEATURE_NAMES` must match `text_features()` order, and
   `CONTEXT_FEATURE_NAMES` must match `ContextBuilder.features()` columns.
   `build` asserts the total width.
6. Language-agnostic only: no `if country == ...`, no hard-coded state lists.
7. Never fetch anything.
8. Log every run in `work/runlog.txt` as `(date | data | change | K | val F0.5 | oracle | notes)`.

---

## 8. Current status (2026-09-26)

### What changed in this session

| Area | Change | Evidence |
| --- | --- | --- |
| blocking speed | batched sparse `Q @ P` instead of per-query merge; tokenise once; token cache | mini 77 s → 22 s, identical recall (only tie-order differences) |
| blocking speed | two token tiers (`--gen-df-cap`), flag lookup instead of `np.isin`, exact bound pruning | full pool 36 ms → ~11 ms per query, recall unchanged |
| blocking speed | `--workers`, memory-mapped shared index | byte-identical to serial on mini |
| blocking recall | `name` channel (tokens, concatenation keys, skeletons), unioned with `all` | full pool, see table below |
| text | `normalize` ASCII fast path; `romanize`, `skeleton`, `name_keys` | examples in §6 |
| aliases | word filter; `--merge-with` bootstrap mode | `new→ny` etc. gone; 1,708 entries from 400k clusters |
| features | +`name_skel_jacc/cont/ratio`, `name_compact_partial`, `addr_skel_cont` (text); +`rev_rank`, `target_margin`, `name_block_*` ×4 (context). Now 32 text + 15 context = 47 | mini ablation +0.0011 (a tie on mini) |
| pipeline | sharded pair files, parallel text features, `TextStore`, per-country thresholds, unseen-country fallback, stage-1 prefilter, `--refit`, `--drop-features`, `--candidates-out`, `--confident-out` | mini runs end to end |
| analyze | ported to the npz/int64 formats; per-channel recall | n/a |
| launch.json | stages updated to the current flags | n/a |

### Numbers

**Mini subset** (8,000 clusters, 2,000 validation entities; optimistic):

| run | recall ceiling | oracle F₀.₅ | val macro F₀.₅ |
| --- | --- | --- | --- |
| baseline as handed over (old split) | .9868 | .9964 | .9869 |
| + name channel, new features, per-country thresholds (split v3) | **.9953** | **.9989** | **.9874** |

The two rows use different validation splits (the split is now stable), so the
F₀.₅ difference is within noise. The recall and oracle gains are real.

**Full 10.3M-target pool, blocking recall only** (3,000 S1 queries,
`aliases_full`):

| configuration | cands/entity | union recall | ms/query (1 proc) |
| --- | --- | --- | --- |
| single channel, K=75, no gen cap | 75 | .9464 | ~33 |
| single channel, K=75, gen cap 20k | 75 | .9318 | 6.5 |
| two channels 50/25, no gen cap | 69 | .9686 | 48 |
| **two channels 50/25, gen caps 20k/20k (chosen)** | 69 | **.9619** | ~11 after pruning |
| two channels 100/50, gen caps 20k/20k | 139 | .9710 | ~11 |
| two channels 100/50, gen caps 50k/20k | 139 | .9758 | 32 |

Chosen: 50/25 with 20k/20k caps. The larger settings buy ≤ +1.4 pts recall
(≤ ~+0.003 F₀.₅ by §1) for 2× the pairs and 3× the time.

The first full-pool classifier numbers are below, under "Full-data dev run";
everything above them is mini or blocking-only.

### Streaming refactor: done

- Blocking writes sidecar parts to disk and accumulates target stats in
  `TargetStats`.
- `ContextBuilder` works per part from `meta.npz`, and `build` streams over the
  parts.
- Mini regenerated end to end: recall ceiling .9953, oracle .9989, val .9866
  (.9874 before; noise).
- Streamed stats were checked against a full recomputation: exact.

### Full-data dev run (26 Sep, 02:40–06:30): first real validation number

**Blocking.** All 2,206,821 training S1 records were queried against the full
10.3M-record pool, and pairs were stored for 200,000 random entities.

```powershell
python $s/blocking.py --data-dir dataset/train --prefix train --out work/dev/candidate_pairs.tsv `
  --topk 50 --name-topk 25 --gen-df-cap 20000 --name-gen-df-cap 20000 `
  --cache work/tokcache_train.npz --workers 5 --store-sample 200000 --no-tsv
```

- **Time:** 3 h 14 min. India ran at 6.3 ms/query wall, US at 4.7 ms/query,
  with 5 workers. Parallel scaling is only ~2× (8 logical / ~4 physical
  cores; the sparse products are memory-bound).
- **Output:** 13,740,903 pairs (68.7 per entity) in 7 parts, in
  `work/dev/candidate_pairs_scores/`.
- **Blocking recall** over the 200k stored entities (692,993 true links):
  **0.9708**, matching the 3k-query probe (.9711).

**Features + model** (`work/dev/after_blocking.ps1`): `build --max-s1 40000`
gives 2,747,427 pairs and 134,822 positives, in 307 s with 5 workers. `train`
fits on 30k entities (2.06M pairs) and validates on **10,000 entities**.

| | macro F₀.₅ |
| --- | --- |
| model alone, global threshold (0.675) | 0.9561 |
| per-country thresholds (India .725, US .675) | 0.9561 (no gain: a tie) |
| **model + stage-1 prefilter (what the test pipeline does)** | **0.9629** |
| oracle over candidates (perfect classifier) | 0.9899 |

By country, with the model alone: **India 0.9416**, **US 0.9659**. France,
unseen in training, gets the stricter threshold, 0.725.

**Error decomposition** (10k validation entities, model alone):

| | entities |
| --- | --- |
| fully correct | 7,449 (74.5%) |
| missed matches only | **1,887** |
| false positives only | 518 (47 true singletons wrongly merged) |
| both | 146 |

1,552 true links sat inside the candidates but scored below the threshold;
704 wrong links scored above it.

**What this says**

- **Blocking is no longer the bottleneck.** The oracle is 0.99; the ~0.03
  headroom is inside the candidate set, in the classifier. So no blocking
  change is needed, and the frozen-blocking rule below holds.
- **Missed matches dominate.** Many look easy but are ambiguous by
  construction:
  - `Gulf Alliance Inc.` | *empty address* scored p=0.11 and is a true match;
  - `Royal Properties Private [Limited]` | *Chennai* scored p=1.00 against a
    *Pune* S1 record, and is a *different* business.
  The data contains many distinct businesses with the same name, so name-only
  records are genuinely hard to call.
- **Other false positives:**
  - same building, different business (`Zalais Biologics` vs
    `Develompment Economic (Trust)` at the same address);
  - neighbouring house numbers (`595` vs `600A White Cliffs Dr`).
- **The prefilter adds +0.0068** because it vetoes some of the model's false
  merges. That suggests a second-stage model would add more.
- **Top features:** `name_len_ratio`, `addr_token_set`, `name_tri_jacc`,
  `target_n_claims`, `block_score`, `rank`. The new `name_compact_partial`,
  `target_margin` and `name_block_*` features all rank in the top ~20.

Logs: `work/dev/analyze_blocking.log`, `build40k.log`, `train40k.log`,
`errors40k.log`. Model: `work/dev/model40k.pkl`.

### Test blocking on AWS, in parallel (approved 26 Sep)

Test blocking doesn't need the model, so it runs on an AWS EC2 machine
(challenge Free plan credits) while the laptop does the training side. It
saves ~3 h.
- Machine: **m7i-flex.large** (2 vCPU, 8 GB), Amazon Linux 2023, ap-south-1,
  30 GB disk.
- Upload `work/aws/aws_bundle.tgz` (code + test TSVs + `aliases_full.tsv`) and
  `work/aws/run_test_blocking.sh`.
- Run it in `tmux` (~4–5 h), then download `~/cp_all_scores.tgz` and extract it
  into `output/`. That gives `output/cp_all_scores/`.
- **Terminate the instance afterwards.**
- Full click-by-click steps: plan file
  `C:\Users\Admin\.claude\plans\witty-mixing-ripple.md`.
- Fallback: if AWS isn't usable within ~1 h, run §5.6 blocking on the laptop
  after the training run.

**Frozen blocking.** The train-side dev run and the test run must use
identical:
- `blocking.py`;
- the `ertext` tokenisation functions it calls: `normalize`, `name_tokens`,
  `addr_tokens`, `name_keys`, `skeleton`, `romanize`, `canonical`;
- `work/aliases_full.tsv`;
- the flags: K 50/25, gen caps 20k/20k, df cap 200k.

Changing any of these means redoing **both** blocking runs (~3.5 h + ~4 h).
Fixes from error analysis therefore go into features and thresholds, and new
feature logic goes into *new* functions. The chance of having to break this
is ~15%: only a tokenisation bug would force it, since missed links are cheap
under F₀.₅.

**Timeline** (26 Sep; deadline 27 Sep):

| step | laptop only | with AWS |
| --- | --- | --- |
| dev blocking done | ✅ 05:54 | ✅ 05:54 |
| first real validation score | ✅ ~06:30 (0.9629) | ✅ ~06:30 (0.9629) |
| test blocking | 07:00–10:30 | on AWS, ready ~09:00 |
| fixes from error analysis | 10:30–12:00 | 07:00–09:00 |
| final model (200k entities, `--refit`) | 12:00–13:00 | 09:00–10:00 |
| test build + predict + validate | 13:00–14:00 | 10:00–11:00 |
| **first submission** | **~14:00** | **~11:00** |

### Open work queue (highest expected value first)

1. ~~Full-data dev run → first validation number~~ **Done: 0.9629 with the
   prefilter** (see above).
2. **Train on more data: done 27 Sep.** 190k train / original 10k holdout,
   macro F0.5 **0.96738221**, with the frozen baseline prefilter.
3. **Same-name ambiguity features: tested, not selected.** 0.96934548,
   only +0.00196 over item 2; below the 0.003 adoption threshold.
4. **Second-stage candidate-list model: deferred.** Prioritize the valid test
   submission; any later stacking must use out-of-fold probabilities.
5. **Final refit: done.** `work/dev/model_optimized.pkl`, all 200k entities.
   Keep `work/dev/model40k.pkl` as the frozen prefilter and fallback model.
6. ~~Test run (§5.6, blocking on AWS) → validator `PASS`~~ **Done 27 Sep**
   (see "Test run" below). Remaining: package (§9) and upload.
7. Test alias bootstrap (§5.6, optional block), especially to learn French
   abbreviations. The code exists but has never been run.
8. Per-channel K and gen caps: **not needed**, since blocking is not the
   bottleneck (oracle 0.99).

---

## 9. Submission package

```
<team_name>_submission.zip
├── output/matching_results.tsv
├── output/candidate_pairs.tsv       (the scored candidate set, predict --candidates-out)
├── code/business_entity_resolution/ (src/, README.md, requirements.txt)
└── Documentation_template.md
```

`work/` and `.venv/` are excluded. For the methodology document, the numbers to
quote are in §2 (EDA), §6 (blocking design) and §8 (results). Replace the mini
numbers with full-pool validation numbers once §8 item 2 has run.

## Classifier experiment (27 Sep, complete)

The user authorized classifier optimization while AWS test blocking runs.
`work/dev/model40k.pkl` remains the baseline and fixed stage-1 prefilter.
Blocking, `ertext.py`, aliases, and blocking settings remain unchanged.

New optional `build --ambiguity-features` appends 12 classifier-only features:
same-country S1 core-name frequencies and explicit address coverage, length,
and conflict evidence. Counts use all S1 records without ground-truth labels.
The default 47-feature build remains compatible with the baseline model.

`train --validation-from work/dev/pairs40k.npz` preserves the original 10k
validation entities when training on more data. `--fixed-prefilter` preserves
the original context filter instead of training a different filter on already
filtered pairs. The build records its prefilter SHA256, checked during training
and prediction; prediction also rejects mismatched feature schemas.

The completed comparison uses the same original 10k holdout for every model.
The baseline trained on 30k entities; the larger models trained on 190k.

`src/classifier_experiment.py` runs the audit, filtered 200k feature build
(3 workers), original-schema projection, and both models. Logs go to
`work/dev/classifier_experiment.log`; metrics to `classifier_results.json`.
The runner checks SHA256 of the frozen blocking inputs and baseline model.
Regression checks are in `tests/test_classifier.py` (all six passed).

Baseline audit reproduced **0.96285086** after filtering before assignment
(unfiltered **0.95613708**). False links fall from 704 to 401; missed links
rise from 2,560 to 2,593; false singleton merges fall from 47 to 36. The
prefilter benefit is real on this holdout. Error analysis now uses the validation IDs
saved in new model bundles, and rejects refitted models as held-out evidence.
The experiment helper also supports a post-training comparison on identical
shared-name and missing-address cohorts, including paired bootstrap intervals
for the score difference. These intervals do not account for threshold tuning
on this same validation set; scores remain development estimates.
Regression cases include early rejection of incompatible feature schemas and
prefilters. The baseline audit is also recorded in `work/runlog.txt`.
Use `classifier_experiment.py --compare-only` for the detailed cohort report.
Section 5.6 uses `work/dev/model40k.pkl` for filtering and
`work/dev/model_optimized.pkl` for prediction. Do not add `--ambiguity-features`
for the selected model, which uses the original 47 features.
The comparison also recalibrates the original model's thresholds **after**
the fixed prefilter (`work/dev/model40k_calibrated.pkl`). This cheap option
must be compared with new training, since old thresholds were tuned before
filtering. The original `model40k.pkl` is never overwritten.

**Build completed:** 200,000 entities, 964,359 retained pairs (7.0% of
13,740,903 candidates), 59 features in seven shards, 253 seconds. Retained
true links: 669,553; link recall after filtering: 0.9662. Both training runs
use 190,000 training entities and the original 10,000 validation entities.
| Experiment | Held-out macro F0.5 | Decision |
| --- | --- | --- |
| Saved baseline + prefilter | 0.96285086 | Fallback |
| Baseline, thresholds retuned after filter | 0.96285086 | No gain |
| Larger training, original 47 features | **0.96738221** | **Selected** |
| Larger training, 59 features | 0.96934548 | Extra +0.00196 is below 0.003 rule |

Post-prefilter oracle: **0.98810565**. Selected thresholds: India 0.700,
US 0.725; unseen country 0.725. Held-out model:
`work/dev/model200k_larger_baseline.pkl`. Both experiments completed in
400 seconds total. SHA256 checks
confirmed that blocking, tokenization, aliases, and the baseline model stayed
unchanged. AWS blocking outputs remain compatible.

**Selection:** threshold-only recalibration gave no gain (0.96285086).
Choose the 47-feature larger-data model: +0.00453136 versus baseline, with
paired bootstrap 95% interval [0.00328, 0.00584] on this development holdout.
False links fall **401 -> 287**, missed links **2593 -> 2420**, and false
singleton merges **36 -> 23**. Shared-core-name entities (5,247) improve
0.95282 -> 0.95736; entities with a retained empty-address candidate (2,311)
improve 0.95434 -> 0.96013. These cohort counts are after filtering, unlike
the original audit's before-filter empty-candidate cohort.

The optional 59-feature model is preserved as an experiment, not selected:
its extra gain is below 0.003 and it produces 310 false links versus 287.
`classifier_experiment.py --finalize larger_baseline` refitted the selected
classifier on all 200k entities, preserving its learned thresholds and the
original frozen prefilter. Output: `work/dev/model_optimized.pkl`.
Validation scores describe the held-out model **before** refitting; the
refitted model must not be evaluated as if those labels were still unseen.
Git ignore exceptions now retain the selected model and JSON experiment reports
for the next Git backup. No second-stage stack is being added in this run.
A saved-model smoke check (`tests/smoke_saved_model.py`) **passed** prediction,
candidate export, empty-entity output, unique target assignment, and the
organizer's validator on a 101-entity / 485-pair fixture. This does not validate
the actual test submission, which still needs AWS output and a full test run.

Reproduce from the workspace root (do not rerun blocking):

```powershell
$s = "code/business_entity_resolution/src"
.venv/Scripts/python.exe -u $s/classifier_experiment.py
.venv/Scripts/python.exe -u $s/classifier_experiment.py --compare-only
.venv/Scripts/python.exe -u $s/classifier_experiment.py --finalize larger_baseline
.venv/Scripts/python.exe -m unittest discover -s code/business_entity_resolution/tests -v
.venv/Scripts/python.exe code/business_entity_resolution/tests/smoke_saved_model.py
```

Before refitting, error analysis can use the saved 190k-training model with
`work/dev/pairs200k_filtered.npz`; it now reads the bundle's saved validation
IDs. Full-test model competition and generalization to France are not measured
by this sampled training holdout. AWS output download, test build/predict,
organizer validation, and final submission remain outstanding.

## Test run (27 Sep, complete, validator PASS)

- **AWS blocking:** 1,732,544 S1 queried (France 259,452 / India 809,986 /
  US 663,106), 118,437,763 pairs (68.4 per entity), 59 parts, 6,579 s on
  m7i-flex.large. Frozen settings. Output in `output/cp_all_scores/`.
- **Build** (`--prefilter work/dev/model40k.pkl`, 3 workers): 10,121,272 pairs
  kept (8.5%), 47 features, 1,423 s → `work/pairs_test.npz`.
- **Predict** (`work/dev/model_optimized.pkl`): thresholds France 0.725
  (unseen → conservative), India 0.700, US 0.725. 5,019,247 links with
  p ≥ 0.98 in `work/test_confident.tsv`.
- **Validator:** PASS, also with `--check-ids`.

| country | entities | empty % | mean links | sizes 0/1/2/3/4/5+ % |
| --- | --- | --- | --- | --- |
| France | 259,452 | 6.13 | 3.20 | 6.1/8.7/19.5/24.3/20.2/21.2 |
| India | 809,986 | 6.28 | 3.23 | 6.3/8.6/19.1/23.8/20.2/22.2 |
| US | 663,106 | 5.88 | 3.31 | 5.9/7.2/18.5/24.3/20.9/23.3 |
| all | 1,732,544 | 6.10 | 3.26 | 6.1/8.1/18.9/24.1/20.4/22.4 |

Train truth for comparison: 5.6% singletons, 3.46 mean. The prediction is
slightly conservative, as expected under F0.5. France behaves like the seen
countries. No test F0.5 is available locally; the dev estimate is 0.967.

**Leaderboard (27 Sep, `matching_results.tsv` only): 0.958**, i.e. 0.009 below
the dev estimate. Likely causes: France is unseen in training, the dev holdout
sits on the training distribution, and the thresholds were tuned on that same
holdout. The team's target for the next round is ≈ 0.989. Note that the dev
oracle over the retained candidates is 0.988 and over all candidates 0.990, so
that target needs gains in both the classifier and candidate recall. Leaderboard
submissions currently take only the matching file; the final zip comes later.

## Stage 2 and leaderboard tuning (27 Sep)

**Submitted (current best): leaderboard 0.965.**
- Stage 2 plus the address-number features (`src/extra_features.py`, `NUM_NAMES`),
  i.e. `work/dev/stage2_numbers.pkl`.
- Thresholds shifted +0.22, which gives the same 3.25 links per entity as the
  earlier 0.961 file.
- Reproduce with `stage2.py predict --model work/dev/stage2_numbers.pkl --method
  threshold --threshold-shift 0.22` on `work/pairs_test.npz`.
- Dev: the full holdout scores 0.9753.
  - Half B: 0.9734 → 0.9763.
  - Test-like dev2, half B: 0.9703 → 0.9724.
- Other uploads of the same model: plain +0.10 shift 0.963; the 0.961 file minus
  the links this model rejects 0.962.
- Blocking, aliases and the prefilter are unchanged (frozen).

The earlier 0.961 file was stage 2 without the number features, with
thresholds +0.10 (`work/dev/stage2.pkl`, `--threshold-shift 0.10`).

- `src/stage2.py`: stage-1 probabilities (out-of-fold on dev), list features, and
  candidate-to-candidate coherence; LightGBM on 63 features.
  - Dev holdout half B: 0.9677 → 0.9734.
  - Leaderboard: 0.958 → 0.959, and 0.961 with +0.10 thresholds.
- Threshold-shift leaderboard curve (−0.05 … +0.15): 0.958, 0.959, 0.960,
  **0.961**, 0.958.
- EM prior-shift estimate on test: candidate pairs are 58–60% true matches,
  against 69% on dev.
  - Re-blocking train with 81% of S1 queried (`work/dev2/`) reproduces the 59%
    rate.
  - The submitted model scores 0.9694 there; a model retrained on it scored
    0.9699 on dev but only 0.958 on the leaderboard (`D_realistic`).
- Leaderboard probe with France rows emptied scored 0.827, which implies
  India/US ≈ 0.962 and France ≈ 0.94 on test.
- Not adopted:
  - expected-F0.5 decoding;
  - cluster expansion (`src/expand.py`, −0.001 dev);
  - French alias bootstrap with France re-blocked (`src/country_subset.py`,
    `work/fr/`; never uploaded);
  - bigger or averaged stage-2 models and a stage-3 coherence round (±0.0006).
- Every file tried is in `output/archive/`, and each leaderboard result is logged
  in `work/runlog.txt`.

Bounded experiments. Each was run separately on both dev sets, with half-B
deltas given as old dev / dev2; blocking stayed frozen:
1. candidate-count feature ablation: −0.0017 / −0.0008, dropped;
2. joint-support coherence: −0.0001 / −0.0002, dropped;
3. **address-number evidence** (alphanumeric numbers, unit conflicts):
   **+0.0029 / +0.0021, kept**; leaderboard 0.965;
4. cross-script core names (legal suffixes are dropped before aliases, so Indic
   `प्राइवेट` → `private` stays in the core): +0.0001 / −0.0001, a tie, dropped.

## 10. Scale and memory

| stage | time (measured / estimated) | memory |
| --- | --- | --- |
| tokenise full train or test (first run, cached after) | ~12 min | ~2 GB |
| blocking queries, 2.2M S1, 5 workers | ~3–3.5 h (measured 6.8 ms/query wall, India) | shared memory-mapped index + ~0.3 GB private per worker; main ~1.4 GB |
| build, text features | ~10k pairs/s with 3 workers on mini | per part |
| train on 100k entities (~7M pairs × 47) | minutes | ~1.5 GB |

Plan around ~5 GB of *available* RAM on this machine (16 GB installed, most
committed elsewhere). Close other applications before full runs.

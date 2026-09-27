# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** White Lotus  
**Team Members:** Nidhi Singh, Priya Yadav, Anu Kumari, Prakhar Saxena  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We use a classic *block → classify → assign* pipeline, built for the 10M-record
scale and for F₀.₅, which penalises false merges twice as much as misses.
Candidates come from a **two-channel, IDF-weighted inverted index**, partitioned by
country. One channel matches the full record, the other the name only, including
concatenation keys and script-independent consonant skeletons. Each candidate pair
is scored by a **LightGBM classifier on 47 features**, after a cheap context-only
prefilter. A **stage-2 LightGBM** then re-scores each pair with its entity's whole
candidate list: probability-list statistics, *coherence* (do the other strong
candidates resemble this one?) and **address-number evidence** (unit conflicts
inside one building, alphanumeric numbers such as `600A`). We keep only pairs
above a **per-country threshold**, raised for test so the prediction keeps the
link count leaderboard feedback favoured. Each Source-2/3 record goes to at most
one Source-1 entity. No external data, network calls or pretrained models are
used. Held-out macro F₀.₅ on training data is **0.9753**; the leaderboard score of
the submitted file is **0.965**.

---

## 2. Methodology

### 2.1 Problem Analysis

| Fact (training data) | Value | Implication |
| --- | --- | --- |
| Rows S1 / S2 / S3 | 2.21M / 5.03M / 5.29M | blocking decides feasibility |
| Rows test S1 / S2 / S3 | 1.73M / 4.89M / 5.08M | test adds an unseen country, **France** |
| Cluster sizes (links per S1) | 0: 5.6%, 1: 5.4%, 2: 17%, 3: 24%, 4: 22%, 5+: 26% | mean 3.46 matches |
| Each S2/S3 ID appears in exactly one S1 cluster | 100% of 7.64M | a one-to-one assignment constraint |
| `country` agreement on true pairs | 100.00% | safe hard partition by country |
| True pairs sharing ≥ 1 name/address token | 100.00% | token blocking has a perfect ceiling |
| Non-Latin names on the S2/S3 side | 6.7% | Devanagari, Bengali, Gujarati, Odia, Tamil, Telugu, Kannada |

**Noise patterns observed.**
- *Names:* leetspeak (`8rands`, `Visi0n`), typos, token reordering, legal
  suffixes added or dropped, filler words, web-ified names
  (`securecloudservices.com`, `#ridgefellowship`), `DBA:` / `aka` prefixes,
  transliteration into Indic scripts, accent insertion, and names replaced by a
  fabricated word while the address stays intact.
- *Addresses:* abbreviations (`Rd`↔`Road`, `TX`↔`Texas`↔`महाराष्ट्र`), city variants,
  house-number mutation (`17337`→`7337`), dropped or reordered components,
  `<NULL>` placeholders, and truncation to 2–3 components.

**Metric insight.** Under macro F₀.₅, an entity missing a fraction *m* of its
links with no false positives scores ≈ 1 − 0.2·m. So recall is cheap, and a
single false merge on a true singleton costs a full 1.0. The design therefore
favours precision at every stage.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (two-stage GBDT) + constrained assignment  
**Core Innovation:** a two-channel IDF blocking index whose name channel reaches
web-ified and other-script names. These keys come from *mechanical romanisation*
using only Unicode character names, with no lookup tables. On top of that sit
*contention features*, which describe how every other Source-1 record scores the
same target, accumulated in streaming over all 1.7M queries.

Everything is language-agnostic: no rule depends on a specific country, and
`country` is treated as an open set of labels (France was handled without any
France-specific code).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used**
  - *`all` channel:* normalised name + address tokens. Aliases are canonicalised
    through a lexicon of 1,708 entries mined from training clusters, with PMI plus
    a consistency and word-frequency filter. Tokens are weighted by IDF, and each
    query keeps its top-50 candidates.
  - *`name` channel:* canonical name tokens, `#`-concatenations of core tokens
    (these reach `securecloudservices.com`), and `~` consonant skeletons after
    romanisation. With the skeletons, `शिव प्रोडक्ट्स` and `Shiva Products` both
    become `sv prtkts`. Each query keeps its top-25 candidates.
  - The two lists are merged, and every pair is scored in both channels.
  - Hard partition by country.
  - *Two token tiers:* tokens with document frequency ≤ 20k *generate* candidates
    through a batched sparse product. Commoner tokens only add score, with exact
    bound pruning. This cut query time from ~48 ms to ~11 ms without changing the
    chosen top-K much.
- **Candidate pairs generated (test):** **118,437,763** (68.4 per Source-1
  entity) over 1,732,544 queries. A stage-1 context-only prefilter then keeps
  **10,121,272** pairs (8.5%) for the full classifier.
- **How we ensured true matches were not lost**
  - Recall was measured on the **full 10.3M-record pool**, not a small subset:
    the subset overstated recall by ~4 points.
  - Blocking recall is **0.971** on 200k training entities (692,993 true links).
  - The oracle F₀.₅ over candidates, i.e. with a perfect classifier, is 0.990.
    So blocking is not the bottleneck.
  - The name channel was added specifically after analysing missed links. It
    raised recall at a similar candidate budget (75 vs 69 per entity) from 0.932 to 0.962 on the probe.
  - Larger K settings would buy ≤ +1.4 points of recall, worth ≤ +0.003 F₀.₅,
    for 2× the pairs, so they were rejected.
  - The prefilter threshold is set to keep 99.9% of the links the full model
    accepts.

---

## 4. Matching Model

**Features used (47):**
- **Name features:**
  - core, all-token and trigram Jaccard;
  - containment; sorted-token and initials equality; length ratio;
  - rapidfuzz ratio, token-set and partial ratios;
  - consonant-skeleton Jaccard, containment and ratio (script-independent);
  - compact partial match (for web-ified names).
- **Address features:** Jaccard, containment, trigram Jaccard and token-set
  similarity; house-number containment and any-match; skeleton containment.
- **Conflict and missingness features:** `num_conflict` (different numbers, e.g.
  another unit in the same building), extra and missing core name tokens,
  disjoint cores, empty address on either side, Indic script and script
  mismatch, and name-only or address-only evidence.
- **Context features (15):**
  - blocking scores, shared-token counts and ranks in both channels;
  - score ratio to the top candidate, gap to the next candidate, and the
    candidate count;
  - whether the candidate comes from Source 3;
  - **contention**: how many S1 entities claim the target, the best and
    second-best claim ratio, reverse rank, and the target margin.

**Stage-2 features (16 more, 63 in total):**
- *Probability-list features* from stage-1 probabilities: the pair's rank in its
  entity's list, its gap to the best other candidate, the entity's max, second
  max and sum of probabilities, the number of candidates with p ≥ 0.5, the list
  length, and whether the pair is the best candidate from its source (S2 or S3).
- *Coherence features:* a true cluster is a set of noisy variants of one
  business, so its S2/S3 members also resemble *each other*. For each candidate
  we compare it with the entity's other strong candidates (p ≥ 0.3, at most 8):
  max name similarity, max address similarity, probability-weighted name
  trigram similarity, the share with conflicting house numbers, and the number
  of strong candidates it agrees with. A same-building false positive resembles
  the S1 address but none of the other members.
- *Address-number evidence (6 more, 69 in total; `extra_features.py`).* The
  original `num_conflict` fires only when no number is shared, so a shared
  building number or postcode hides a different flat number. It also ignores
  alphanumeric numbers. The new features are:
  - exactly shared numbers;
  - numbers found on only one side, tolerant to the leading-digit noise
    (`17337` ≈ `7337`);
  - a **unit conflict**: some number is shared but each side also has a number
    the other lacks, e.g. flat 503 vs 508 in the same building;
  - alphanumeric match and conflict (`600A`).

  An empty address gives no evidence rather than a conflict. The unit conflict
  fires on 20% of false candidate pairs but only 2.5% of true ones.
- Stage-2 training uses **out-of-fold** stage-1 probabilities (5 folds by
  entity), so the list features are not optimistic.

**Model type:**
- Stage 1: LightGBM (500 trees, 63 leaves, learning rate 0.06), trained on 190k
  entities and refit on all 200k sampled training entities.
- A small context-only GBDT acts as the prefilter before the text features. It
  also vetoes some false merges.
- Stage 2: LightGBM with the same settings on the 69 features.

**Threshold selection method:**
- An F₀.₅ sweep over a held-out entity split, with **one threshold per country**.
  A country unseen in training (France) gets the most conservative learned one.
- **Test adjustment:** candidate pairs are true matches only ~58–60% of the time
  on test, against ~69% on training data, as estimated by EM prior-shift
  estimation (Saerens et al.) on unlabelled test predictions. Test has many
  S2/S3 records whose S1 is absent, so thresholds must be stricter on test.
  - For the model without number features, leaderboard scores for shifts of
    −0.05 / 0 / +0.05 / +0.10 / +0.15 were 0.958 / 0.959 / 0.960 / **0.961** / 0.958.
    The best was 3.25 links per entity.
  - The submitted model learned lower thresholds. It is shifted by **+0.22** so
    it predicts the same number of links (3.25 per entity): India 0.795, US and
    France 0.945.
  - At that count it scored **0.965**. The plain +0.10 shift (3.31 links) scored
    0.963, and keeping the 0.961 file while only removing links the new model
    rejects scored 0.962.
- This is followed by a **one-to-one assignment**: every S2/S3 record is kept
  only for its highest-probability S1 claimant, as the data's partition property
  requires.

---

## 5. Results & Error Analysis

**Held-out validation** (the same 10,000 training entities for every model;
blocking run on the full pool):

| Model | Macro F₀.₅ |
| --- | --- |
| Baseline (30k training entities) + prefilter | 0.9629 |
| Baseline, thresholds retuned after prefilter | 0.9629 |
| Stage 1: 190k training entities, 47 features | 0.9674 |
| 190k entities + 12 extra ambiguity features (not selected: +0.002 is below our 0.003 adoption rule) | 0.9693 |
| Stage 1 + stage 2 | 0.9726 |
| **Stage 1 + stage 2 + address-number features (submitted)** | **0.9753** |
| Oracle (perfect classifier on the retained candidates) | 0.9881 |

- **F_0.5 Score (macro):** **0.9753** on validation.
  - On the half of the holdout not used for tuning, stage 2 raised the score
    0.9677 → 0.9734, and the number features 0.9734 → 0.9763.
  - On a harder, test-like development set they raised it 0.9703 → 0.9724.
- **Leaderboard** (test, `matching_results.tsv`):

  | Submission | Score |
  | --- | --- |
  | Stage 1 | 0.958 |
  | Stage 2 | 0.959 |
  | Stage 2, +0.10 thresholds | 0.961 |
  | **Stage 2 + number features, same link count (submitted)** | **0.965** |
- **Development-to-test gap.** Test candidate lists contain more unmatched
  records than training lists. We simulated this by re-running training blocking
  with only 81% of S1 queried, which reproduced test's ~59% pair match rate. On
  that data the submitted model scored 0.9694. A model retrained on it scored
  0.9699 on dev but 0.958 on the leaderboard, so it was not used.
- **Tried and not adopted** (each measured on the same held-out entities):
  expected-F₀.₅ subset decoding (no gain over thresholds); French abbreviation
  aliases mined from confident test links (France re-blocked); cluster
  expansion by exact name keys (−0.001); larger stage-2 models, model averaging
  and a third coherence round (±0.0006); removing candidate-count features
  (−0.0017); joint name-and-address support from one candidate (−0.0001); core
  names recomputed after alias mapping (a tie).
- Stage-1 selected versus baseline error counts:

  | | Baseline | Selected |
  | --- | --- | --- |
  | False links | 401 | **287** |
  | Missed links | 2,593 | **2,420** |
  | Singletons wrongly merged | 36 | **23** |

- **Common false positives (wrong merges):**
  - the same building but a different business or unit, e.g. two firms at one
    address;
  - neighbouring house numbers (`595` vs `600A White Cliffs Dr`);
  - the same name in a different city.
- **Common false negatives (missed matches):**
  - name-only records with an empty or truncated address. The data contains many
    distinct businesses with the same name, so these are ambiguous by
    construction (`Gulf Alliance Inc.` with an empty address scored p = 0.11 and
    was a true match);
  - fabricated replacement names that leave only address evidence.

**Test submission sanity check** (no test labels, so this is a distribution
comparison only):

| Country | S1 entities | Predicted empty | Mean links |
| --- | --- | --- | --- |
| France (unseen) | 259,452 | 6.68% | 3.13 |
| India | 809,986 | 6.52% | 3.24 |
| US | 663,106 | 6.26% | 3.29 |
| All | 1,732,544 | 6.44% | 3.25 |

Training truth has 5.6% empty entities and a mean of 3.46 links. The
predictions are deliberately conservative, as F₀.₅ and the test distribution
favour. The organiser validator prints PASS, including the ID-existence check.

---

## 6. Conclusion

- A two-channel IDF blocking index gives near-perfect candidate recall at
  10M-record scale: oracle F₀.₅ 0.99 with 68 candidates per entity. With romanised
  consonant skeletons it also covers transliterated names without lookup tables.
- A 47-feature GBDT, re-scored by a list-aware stage-2 model with candidate
  coherence and address-number evidence, reaches 0.9753 macro F₀.₅ on held-out
  training data and **0.965 on the leaderboard**.
- **Key lessons:**
  - measure blocking on the full pool, not a subset;
  - conflict and contention features matter as much as similarity;
  - under F₀.₅, precision-oriented decisions (a prefilter veto, stricter
    thresholds on test) pay more than chasing recall;
  - held-out training data overstated test performance by ~0.01. The test pool
    has more unmatched records, and leaderboard feedback was needed to set the
    thresholds.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
├── src/
│   ├── ertext.py      normalisation, tokens, aliases, romanisation, skeletons, name keys
│   ├── aliases.py     alias-lexicon mining from training clusters
│   ├── blocking.py    two-channel inverted index, top-K, token cache, parallel queries
│   ├── features.py    32 text + 15 context features
│   ├── pipeline.py    build / train / predict / evaluate (stage 1)
│   ├── stage2.py      stage 2: OOF list + coherence features, training, test prediction
│   ├── extra_features.py  address-number evidence (+ cross-script core experiment)
│   ├── analyze.py     blocking recall and error decomposition
│   ├── classifier_experiment.py  model comparison runner
│   ├── expand.py      cluster-expansion experiment (not used in the submission)
│   ├── country_subset.py  one-country data subset / alias curation (France experiment)
│   ├── ambiguity.py   optional ambiguity features (experiment)
│   └── make_subset.py dev subset
├── tests/             regression tests
├── README.md          full technical notes and measured results
└── requirements.txt
```

**Reproduce the submission** (workspace root, Python 3.12; `$s = code/business_entity_resolution/src`):

```powershell
# 1. alias lexicon from training data
python $s/aliases.py --data-dir dataset/train --prefix train --ground-truth dataset/train/train_ground_truth.tsv --out work/aliases_full.tsv --max-clusters 400000
# 2. training-side blocking, features, model (README §5.2-5.4, §8)
# 3. test blocking
python $s/blocking.py --data-dir dataset/test --prefix test --out output/cp_all.tsv --no-tsv --aliases work/aliases_full.tsv --topk 50 --name-topk 25 --gen-df-cap 20000 --name-gen-df-cap 20000 --cache work/tokcache_test.npz --workers 6
# 4. test features (with the prefilter)
python $s/pipeline.py build --data-dir dataset/test --prefix test --cand-scores output/cp_all_scores --aliases work/aliases_full.tsv --prefilter work/dev/model40k.pkl --out work/pairs_test.npz --workers 3
# 5. stage 2 on dev (OOF stage-1, coherence + number features, training), then test prediction
python $s/stage2.py dev --extra-numbers work/dev/extra_features.npz --out work/dev/stage2_numbers.pkl
python $s/stage2.py predict --pairs work/pairs_test.npz --model work/dev/stage2_numbers.pkl --method threshold --threshold-shift 0.22 --out output/matching_results.tsv --candidates-out output/candidate_pairs.tsv
```

The trained models `work/dev/model_optimized.pkl` (stage 1) and
`work/dev/stage2_numbers.pkl` are in the project's Git repository. With these saved models, the `stage2.py predict`
command reproduces the submitted file byte for byte.

`output/candidate_pairs.tsv` is the candidate set the classifier actually scored,
after the prefilter.

### B. Additional Results

**Blocking recall on the full 10.3M-target pool** (3,000 S1 queries):

| Configuration | Candidates / entity | Recall | ms / query |
| --- | --- | --- | --- |
| Single channel, K=75 | 75 | 0.946 | ~33 |
| Two channels 50/25, no generation cap | 69 | 0.969 | 48 |
| **Two channels 50/25, caps 20k/20k (chosen)** | 69 | **0.962** (0.971 on the 200k run) | ~11 |
| Two channels 100/50, caps 20k/20k | 139 | 0.971 | ~11 |

**Top features by importance:** `name_len_ratio`, `addr_token_set`,
`name_tri_jacc`, `target_n_claims`, `block_score`, `rank`. The name-channel and
contention features all rank in the top 20.

**Compute:** test blocking took 1 h 50 min on a 2-vCPU / 8 GB cloud VM (AWS
m7i-flex.large). Feature build took 24 min and prediction ~3 min on a laptop with
~5 GB of available RAM.

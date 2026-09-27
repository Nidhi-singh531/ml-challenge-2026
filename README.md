# Business Entity Resolution — ML Challenge 2026

An end-to-end, high-performance machine learning pipeline designed to resolve and match multi-lingual enterprise records across three noisy sources without external databases or pretrained web models.

🏆 **Final Result:** Achieved **0.965 Macro $F_{0.5}$** on the test evaluation leaderboard.

---

## 📌 Problem Statement (PS)
In commercial data systems, business records arrive from multiple independent sources, each contributing partial, noisy fragments without shared unique keys. The goal is cross-source Entity Resolution (ER)[cite: 2]:

* **Reference Authority:** `Source 1` is the deduplicated reference source[cite: 2]. For every `Source 1` record, the pipeline identifies all corresponding entity matches across `Source 2` and `Source 3`[cite: 2].
* **Noise Patterns Handled:** Missing addresses, legal suffix discrepancies (Corp vs. Corporation, Pvt vs. Private)[cite: 2], transliterations, non-Latin Brahmic scripts[cite: 2], municipal number shifts, and co-tenant address collisions[cite: 2].
* **Target Metric:** Macro-averaged $F_{0.5}$ computed per entity (including singletons)[cite: 2]:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  Precision is weighted $2\times$ over recall[cite: 2]; false merges on singletons severely penalize the score[cite: 2].
* **Strict Constraints:** Self-contained execution, 100% language-agnostic logic, and zero external database/API lookups[cite: 2].

---

## 🗂️ Dataset & Setup

Due to file size constraints and challenge terms, the raw datasets are excluded from Git[cite: 1, 2].

* **Download Link:** [Download dataset.zip from Google Drive](https://drive.google.com/drive/folders/1mW9kGZ4xc1hNxj1mVebhn7zE2hh1bYGE?usp=sharing)
* **Setup:** Download and extract the archive directly into the project root:

ml-challenge-2026/
└── dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
├── test_source1.tsv
├── test_source2.tsv
└── test_source3.tsv


---

## 📁 Repository Overview

ml-challenge-2026/
├── dataset/                        # (Ignored) Train, test, and split data
├── output/                         # (Ignored) Generated matching_results.tsv & candidate_pairs.tsv
├── utils/
│   └── validate_submission.py      # Organizer verification tool for format/rules
├── work/                           # Models, alias mappings, run logs
│   ├── runlog.txt                  # Full experiment iteration history
│   ├── aliases_full.tsv            # Unsupervised mined alias dictionaries
│   └── dev/                        # Saved models (model40k.pkl, stage2_numbers.pkl)
├── Documentation_template.md        # Methodology write-up
├── RECOVERY.md                     # Checkpoint and artifact recovery procedures
└── code/business_entity_resolution/
├── README.md                   # Complete developer runbook & technical handoff
├── requirements.txt            # Python dependencies
├── src/                        # Complete modular pipeline source code
└── tests/                      # Schema, model integrity, and regression tests


---

## ⚙️ The 3 Pipeline Stages

         Raw Records (S1, S2, S3)
                    │
                    ▼
  ┌────────────────────────────────────┐
  │  Stage 1: Two-Channel Blocking     │
  │  - stdlib Brahmic romanization     │
  │  - Name-keys & joint tokens        │
  │  - Two-tier sparse inverted index  │
  └─────────────────┬──────────────────┘
                    │ (~69 cands/entity)
                    ▼
  ┌────────────────────────────────────┐
  │  Stage 2: Context Pre-Filtering    │
  │  - Rapid contention statistics     │
  │  - Drops 91.5% non-viable pairs    │
  │  - Retains 99.9% candidate recall  │
  └─────────────────┬──────────────────┘
                    │ (Scored pairs)
                    ▼
  ┌────────────────────────────────────┐
  │  Stage 3: Pairwise Scoring &       │
  │           Coherence Matching       │
  │  - 47 text & context features      │
  │  - Street/unit number verification │
  │  - 1-to-1 global claim assignment  │
  └─────────────────┬──────────────────┘
                    │
                    ▼
          Valid Submission TSVs

1. **Stage 1 — Two-Channel Blocking:** Combines a joint name+address index with an isolated name-key index (using consonant skeletons and prefix concatenations). A two-tier frequency cap isolates low-frequency tokens for candidate retrieval while using high-frequency tokens strictly for scoring.
2. **Stage 2 — Context Pre-Filtering:** A lightweight GBDT built on contention statistics (claims per target, runner-up score margin) weeds out massive negative pair spaces before expensive string distance metrics are computed.
3. **Stage 3 — Pairwise Scoring & Coherence Assignment:** Evaluates deep text similarity metrics alongside custom street/unit number conflict detectors. Applies per-country calibrated thresholds and strict 1-to-1 target assignment.

---

## 📊 Iterative Results

| Stage / Iteration | Dev Holdout ($F_{0.5}$) | Leaderboard ($F_{0.5}$) | Key Architectural Addition |
| :--- | :---: | :---: | :--- |
| **Baseline GBDT** | 0.9561 | — | Single index, global threshold |
| **Stage-1 Prefilter + Two-Channel** | 0.9629 | 0.958 | Name channel union + contention context features |
| **190k Scaled Classifier** | 0.9674 | 0.959 | Expanded entity pool, stable split tuning |
| **Stage-2 Coherence Model** | 0.9734 | 0.961 | Inter-candidate consistency features |
| **Stage-2 + Number Evidence (Final)** | **0.9763** | **0.965** | Alphanumeric unit and street-number conflict checks |

---

## ⚡ Quick Start

### 1. Installation
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r code/business_entity_resolution/requirements.txt
pip install rapidfuzz lightgbm
2. Predict on Test Set
PowerShell
$s = "code/business_entity_resolution/src"
python $s/stage2.py predict `
  --model work/dev/stage2_numbers.pkl `
  --method threshold `
  --threshold-shift 0.22 `
  --pairs work/pairs_test.npz `
  --source1 dataset/test/test_source1.tsv `
  --out output/matching_results.tsv `
  --candidates-out output/candidate_pairs.tsv
3. Verify Submission
Run the official challenge validator[cite: 2]:

PowerShell
python utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir dataset/test
Expected output: PASS

[cite: 2]

📖 In-Depth Technical Documentation
For full ablation logs, tokenization rules, memory-map scaling configurations, and internal experiment runlogs, see:

👉 Detailed Engineering & Technical Handoff (code/business_entity_resolution/README.md)

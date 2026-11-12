# FinMLLM — Effective Rank, Not Spectral Energy, Sets the Compression Limit

**One-line description**

> Financial LoRA adapters compress to one-sixteenth of their size (114.3 MB → 7.2 MB) with no measurable accuracy loss, because the rule everyone uses to decide how far to compress — keep 95% of the spectral energy — picks a stopping point roughly fourteen times too conservative.

---

**Paragraph description**

> Parameter-efficient fine-tuning made training cheap and left the storage bill untouched. A rank-16 LoRA adapter for a 3B model is 114.3 MB on disk, and an institution that keeps one adapter per task, per desk and per model version carries that cost on premises, because data-residency rules require it. The standard way to shrink those files is a spectral-energy rule: rank the singular directions by the share of the Frobenius norm they carry and keep those reaching 95%. This repository contains the code, the full evaluation records and the verification harness for a study showing that this rule answers the wrong question. Measuring the spectra of LoRA updates fine-tuned on the Financial PhraseBank, the effective rank of the update is **1.98** against **14.08** for a Gaussian matrix of identical shape — a ratio of 0.141. Truncating to a sixteenth of the size changes **three of 394** held-out predictions (McNemar **p = 1.000**) and, on a three-class variant, raises accuracy in every seed. The 95%-energy rule permits 1.1-fold compression where the spectrum supports sixteenfold. Across three tasks, two base models and two rank budgets, the ratio of adapter to Gaussian effective rank orders the cost of compression monotonically, and does so as a prediction rather than a post-hoc explanation. Neither quadrupling the rank budget nor moving to a 2.5× larger base model lifts effective rank above 2.5, which locates it in the task rather than in the model or the budget.

---

## Key numbers

| Quantity | Value |
|---|---|
| Effective rank (adapter / Gaussian baseline) | **1.98 / 14.08** → ratio **0.141** |
| Compression achieved | **16×** (114.3 MB → 7.2 MB), no retraining |
| Prediction changes, binary task (n = 394) | **3 of 394**, McNemar **p = 1.000** |
| Ternary task (n = 968), 3 seeds | accuracy rises, all **p ≤ 0.0001** |
| 95%-energy rule permits | **1.1×** — a factor of **14** too conservative |
| Effective rank, rank 64 budget | **2.31–2.50** (budget quadrupled, rank unchanged) |
| Effective rank, 3B → 7B | **1.94 vs 1.98** (model 2.5× larger) |
| Ratio → compression cost | 0.133 → +3.31 pp · 0.141 → +0.25 pp · 0.163 → −1.23 pp |

---

## What is in this repository

```
FinMLLM/
├── README.md                 this file
├── DEPLOYMENT.md             full deployment + reproduction manual
├── requirements.txt          pinned environment (verified on the machine that ran the study)
├── CITATION.cff
├── LICENSE                   MIT
├── verify/
│   └── verify_all.py         ⭐ re-derives all 121 published numbers from raw JSON
├── scripts/                  training, evaluation, truncation, spectral analysis, orchestration
│   ├── supp/                 supplementary experiments (masking, random direction, PARA)
│   ├── gates/                g0–g5 environment and capability gates
│   ├── manuscript/           figure and document build scripts
│   └── _deprecated/          scripts kept for the record, with reasons — do not use
├── results/                  100 evaluation records with per-item predictions
└── docs/
    ├── ERRATA.md             25 documented defects, each with symptom, root cause, fix
    ├── RESULTS_SUMMARY.md    complete experimental record
    └── LITERATURE_VERIFICATION.md
```

**Not included:** the corpus (`data/`, 268 MB), the trained adapters (`ckpts*/`, 8 GB) and the base models (21 GB). `DEPLOYMENT.md` explains how to obtain or rebuild each.

---

## Quick start

Verifying every number in the paper requires no GPU, no model weights and no corpus — only `numpy` and `scipy`:

```bash
pip install numpy==2.5.4 scipy==1.18.1
python3 verify/verify_all.py
```

Expected output:

```
✅ 全部 121 项通过 —— 论文里的数字与 raw json 完全一致。
   All 121 checks passed. Every number in the paper is traceable.
```

If that line is not green, stop — the `results/` tree and the manuscript disagree, and nothing downstream is meaningful.

---

## Reproducibility levels

| Level | What it establishes | Hardware | Time |
|---|---|---|---|
| **L0** | Every published number is recomputable from the raw records | none | 10 s |
| **L1** | 16× truncation really costs three predictions, not "none in principle" | one 24 GB GPU (CPU works, slowly) | 3–8 min per adapter |
| **L2** | The whole chain — corpus → training → truncation → evaluation — rebuilds | one 24 GB GPU | hours |

Details, exact commands and expected values for each level are in [`DEPLOYMENT.md`](DEPLOYMENT.md).

---

## Experimental scope

| Dimension | Values |
|---|---|
| Tasks | Financial PhraseBank binary (n = 394) and ternary (n = 968); FiQA-SA (n = 243) |
| Base models | Qwen2.5-3B, Qwen2.5-7B (4-bit NF4 + LoRA) |
| Nominal ranks | 16 (main), 64 (over-parameterisation control), 1/2/4/8 (truncation grid) |
| Training seeds | 42 / 123 / 2024, with the split seed fixed at 42 |
| Controls | parameter masking 40/60/80%, random-direction truncation ×5 draws, PARA global threshold ×5 |
| Evaluation | full held-out partitions, cloze prompting with null-context calibration, per-item predictions stored |

---

## A note on the error log

`docs/ERRATA.md` records **25 defects** found in this project, including a training-set
leak affecting 79% of the data in two seeds, a spectral diagnostic that was computed on
the wrong matrix, and a paired-test statistic that was transcribed with the wrong
discordant count. Every entry states the symptom, the root cause, the fix and the lesson.

Nothing is omitted. A reader who wants to know whether the authors understand where their
own pipeline can break should start there rather than here.

---

## Citation

```bibtex
@misc{fan2026adaptercompression,
  title  = {Compressing financial sentiment adapters: effective rank, not spectral energy, sets the limit},
  author = {Fan, Xiaohu},
  year   = {2026},
  note   = {Manuscript under review},
  url    = {https://github.com/RaymondXFAN/FinMLLM}
}
```

See `CITATION.cff` for the machine-readable form.

---

## Contact

**Raymond Xiaohu Fan**
School of Finance and Economics, Wuhan City Vocational College, Wuhan, Hubei, China
Corresponding author — see `CITATION.cff` for current contact details.

---

## License

Code: **MIT** (see `LICENSE`).
Evaluation records under `results/`: **CC-BY-4.0** — free to reuse for benchmark comparison.

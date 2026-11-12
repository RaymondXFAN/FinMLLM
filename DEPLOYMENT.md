# DEPLOYMENT AND REPRODUCTION MANUAL

> Complete instructions for rebuilding this project from a clean machine, and for
> re-deriving every number reported in the paper.
>
> **Author:** Raymond Xiaohu Fan · School of Finance and Economics, Wuhan City Vocational College
> **Last verified:** 2026-11-12 on a single NVIDIA RTX 4090 (24 GB), CUDA 12.4

---

## Table of contents

1. [What can be reproduced, and at what cost](#1-what-can-be-reproduced-and-at-what-cost)
2. [Hardware and operating system](#2-hardware-and-operating-system)
3. [Environment installation](#3-environment-installation)
4. [Base models](#4-base-models)
5. [Corpus preparation](#5-corpus-preparation)
6. [Level 0 — verify all published numbers (no GPU)](#6-level-0--verify-all-published-numbers-no-gpu)
7. [Level 1 — re-derive from existing adapters](#7-level-1--re-derive-from-existing-adapters)
8. [Level 2 — full retraining from the corpus](#8-level-2--full-retraining-from-the-corpus)
9. [Supplementary experiments](#9-supplementary-experiments)
10. [Spectral diagnostics (Table 3 and Table A2)](#10-spectral-diagnostics-table-3-and-table-a2)
11. [Troubleshooting](#11-troubleshooting)
12. [Directory conventions](#12-directory-conventions)
13. [Reproducibility checklist](#13-reproducibility-checklist)
14. [Environment snapshot](#14-environment-snapshot)

---

## 1. What can be reproduced, and at what cost

The project is arranged so that a reader can stop at whatever depth they need.

| Level | Question it answers | Requires | Wall clock |
|---|---|---|---|
| **L0** | Are the 121 numbers in the paper actually derivable from the stored records? | `numpy`, `scipy` — no GPU, no weights, no corpus | **10 s** |
| **L1** | Is "16× compression costs three predictions" a real measurement? | one 24 GB GPU + 114 MB adapter + 5.8 GB base model | **3–8 min** per adapter |
| **L2** | Does the whole chain rebuild — corpus, training, truncation, evaluation, spectral analysis? | one 24 GB GPU + full corpus + 21 GB of base weights | **2–4 h** per configuration |

Two things are deliberately outside the repository: the corpus (268 MB) and the trained
adapters (8 GB), plus the base models (21 GB). Sections 4 and 5 explain how to obtain or
rebuild each.

---

## 2. Hardware and operating system

| Item | Minimum | Used in the study |
|---|---|---|
| GPU | none for L0; 6 GB for 3B training | RTX 4090, 24 GB |
| GPU memory, 3B training | 6 GB | 4.2 GB peak |
| GPU memory, 7B training | 11 GB | 10.3 GB peak, 2.0 s/step |
| System RAM | 16 GB | 32 GB+ |
| Disk — code and results | 200 MB | 250 MB |
| Disk — corpus | 300 MB | 268 MB |
| Disk — adapters (all) | 8 GB | 5.0 GB (`ckpts_rv`) + 2.9 GB (`ckpts_supp`) |
| Disk — base models | 21 GB | 5.8 GB (3B) + 15 GB (7B) |
| OS | any Linux; macOS works for L0 | Ubuntu 22.04 |
| Python | 3.10 – 3.12 | 3.12.3 |
| CUDA (for L1/L2) | 12.1+ | 12.4 |

> **Disk layout warning.** On cloud instances the system disk is often small (30 GB) while
> the data disk is large. Every artifact in this project — corpus, adapters, models,
> caches — must live on the data disk. `bitsandbytes` will appear to work and then fail
> mid-training if the system disk fills up.

---

## 3. Environment installation

### 3.1 Option A — pip (recommended)

```bash
git clone https://github.com/RaymondXFAN/FinMLLM.git
cd FinMLLM

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124
```

`requirements.txt` pins the exact versions used to produce the paper. They are not
suggestions: the spectral diagnostics are sensitive to `numpy` major version, because
`np.linalg.svd` behaves differently on `float16` inputs across the 1.x/2.x boundary.

### 3.2 Option B — conda

```bash
conda create -n finmllm python=3.12 -y
conda activate finmllm
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

### 3.3 Option C — verification only (no GPU, no deep learning stack)

If you only want to run the L0 verification:

```bash
pip install numpy==2.5.4 scipy==1.18.1
python3 verify/verify_all.py
```

This is sufficient for every check in Section 6. `torch`, `transformers`, `peft` and
`bitsandbytes` are not imported by the verifier.

### 3.4 Environment variables

```bash
# Where the corpus, adapters and logs live. All scripts read this.
export PROJ=/root/autodl-tmp/FinMLLM

# Base weights. Overridable per invocation with --base.
export BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-3B

# Hugging Face mirror (use this if you are behind the GFW)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_home      # keep the cache off the system disk
```

Persist them, because a new shell will not inherit them:

```bash
cat >> ~/.bashrc <<'EOF'
export PROJ=/root/autodl-tmp/FinMLLM
export BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-3B
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_home
EOF
source ~/.bashrc
```

### 3.5 Verify the installation

```bash
python3 scripts/lib_probe.py          # prints library versions and CUDA availability
bash scripts/run_gates.sh nogpu       # gates G0–G2: environment, operators, data
```

`run_gates.sh gpu` runs G3–G5 (training smoke test, throughput profile, transfer check) and
requires a GPU. All six gates must pass before any full run; this is not optional in
practice — every failure mode in Section 11 would have been caught by a gate.

---

## 4. Base models

The study uses **Qwen2.5-3B** and **Qwen2.5-7B** in 4-bit NF4, loaded through
`bitsandbytes`. Full-precision weights can be substituted by adjusting `--base`, but the
effective ranks will shift slightly because the LoRA update then no longer absorbs
quantisation compensation (see the paper's Boundaries paragraph).

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_home
mkdir -p /root/autodl-tmp/models && cd /root/autodl-tmp/models

python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo, out in [("Qwen/Qwen2.5-3B", "Qwen2.5-3B"),
                  ("Qwen/Qwen2.5-7B", "Qwen2.5-7B")]:
    snapshot_download(repo, local_dir=out,
                      allow_patterns=["*.json", "*.safetensors", "*.model"])
    print("downloaded:", out)
PY
```

Expected disk usage: 5.8 GB (3B) and 15 GB (7B).

**Verification.** The parameter counts must match the paper (29,933,568 LoRA parameters
over a 3,085,938,688-parameter base, i.e. 0.970%):

```bash
python3 - <<'PY'
import json, struct, glob
tot = 0
for p in sorted(glob.glob("/root/autodl-tmp/models/Qwen2.5-3B/*.safetensors")):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    for k, v in hdr.items():
        if k == "__metadata__": continue
        c = 1
        for d in v["shape"]: c *= d
        tot += c
print(f"base parameters = {tot:,}")     # expect 3,085,938,688
PY
```

---

## 5. Corpus preparation

### 5.1 What is needed

Only two files, both in JSON Lines with a fixed schema:

| File | Records | Task |
|---|---|---|
| `data/unified/fpb.jsonl` | 4,840 | Financial PhraseBank, 3 classes |
| `data/unified/fiqa_sa.jsonl` | 1,213 | FiQA-SA, 3 classes, different register |

Schema, one object per line:

```json
{"id": "fpb-0", "text": "According to Gran , the company has no plans ...",
 "label": "neutral", "source": "fpb", "split": "train"}
```

`label` ∈ {`negative`, `neutral`, `positive`}. The `split` field in the stored file is a
placeholder — **the train/test partition is recomputed at load time** from `--split-seed`
(see 5.3), so editing it has no effect.

### 5.2 Rebuilding from the original sources

```bash
export PROJ=$(pwd)
python3 scripts/fetch_raw.py      # downloads from Hugging Face into data/raw/
python3 scripts/prep_data.py      # unifies into data/unified/*.jsonl
python3 scripts/supp/fix_fiqa.py  # mandatory for FiQA — see the warning below
```

Upstream datasets:

- Financial PhraseBank — `takala/financial_phrasebank`
- FiQA-SA — `pauri32/fiqa-2018` (fallback: `TheFinAI/fiqa-sentiment-classification`)

`fetch_raw.py` uses `hf_hub_download` rather than `load_dataset`, because the latter
executes dataset loading scripts that break on recent `datasets` releases.

> ⚠️ **FiQA label column.** The FiQA-SA release stores labels as **ticker symbols**
> (`AAPL`, `HCP`, …) in a column named `label`, with the sentiment in a differently named
> column. A naive loader filters everything out and yields `n = 0`.
> `scripts/supp/fix_fiqa.py` inspects value ranges rather than column names and rewrites the
> file with `-1/0/1` mapped onto `negative/neutral/positive`. **Run it before any FiQA
> evaluation.** Without it, every FiQA number will be wrong — this is ERRATA E-11.

### 5.3 The partition, and why the split seed is separate

The corpus is pooled across all published agreement subsets and re-partitioned by
`_holdout_split(rows, holdout=0.2, seed=split_seed)` with **`split_seed` fixed at 42**.

| Variant | Pool | Held out |
|---|---|---|
| binary (neutral removed) | 1,967 | **394** |
| ternary | 4,840 | **968** |
| FiQA-SA | 1,213 | **243** |

`sklearn`'s splitting uses `ceil` for the test size: `ceil(1967 × 0.2) = 394`,
`ceil(4840 × 0.2) = 968`, `ceil(1213 × 0.2) = 243`.

> ⚠️ **Do not tie the split seed to the training seed.** An earlier version of
> `train_sft.py` used one `--seed` for both, so seeds 123 and 2024 drew *different* test
> sets from the training data, with **79.1%** and **77.7%** overlap with the training
> partition. That produced accuracies up to 1.000. The corrected script is
> `rv_train_sft.py`, which exposes `--seed` and `--split-seed` separately (ERRATA E-01).

**Verification.** The class distribution of the pooled corpus must be:

```
neutral   2873   59.36%
positive  1363   28.16%
negative   604   12.48%
```

```bash
python3 - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("data/unified/fpb.jsonl")]
c = collections.Counter(r["label"] for r in rows)
print(len(rows))
for k, v in c.most_common():
    print(f"{k:9s} {v:5d}  {v/len(rows)*100:5.2f}%")
PY
```

---

## 6. Level 0 — verify all published numbers (no GPU)

This is the fastest way to establish that the repository is internally consistent.

```bash
python3 verify/verify_all.py
```

Expected:

```
✅ 全部 121 项通过 —— 论文里的数字与 raw json 完全一致。
   All 121 checks passed. Every number in the paper is traceable.
```

### 6.1 What it checks

| Block | Items | Source | Notes |
|---|---|---|---|
| Table 1 — compression–accuracy frontier | 14 | `results/rv_final/` | verified twice: table cells at seed 42, and three-seed means quoted in the text |
| Paired McNemar (b, c, Δ) | 28 | `results/rv_final/`, `results/fiqa/` | recomputed from per-item `preds`, never read from a summary |
| Table 3 — effective-rank diagnostics | 54 | `results/table3/` | 9 adapters × 6 statistics |
| Table S3 — parameter masking | 6 | `results/rv_supp/` | 40/60/80% × 2 tasks |
| Table S4 — random direction | 8 | `results/rv_supp/` | mean of 5 draws |
| PARA global threshold | 10 | `results/rv_para/` | 5 thresholds × 2 tasks |
| §4.6 cross-domain transfer | 5 | `results/fiqa_transfer/` | base / r16 / r1 / Δ / gain |

### 6.2 Optional: verify artifact sizes

If you have the adapters on disk, add:

```bash
python3 verify/verify_all.py --ckpts /path/to/ckpts_sft3
```

This adds five checks on `adapter_model.safetensors` sizes, expecting
**114.3 / 57.2 / 28.6 / 14.3 / 7.2 MB** for ranks 16 / 8 / 4 / 2 / 1 (all float32, hence
114.3 MB and not the ~57 MB a float16 store would give).

### 6.3 How the verifier is designed

The claim values live in a single block at the bottom of `verify_all.py`:

```python
CLAIM_T1 = {...}          # Table 1
CLAIM_MC = {...}          # McNemar b, c, delta
CLAIM_T3 = {...}          # Table 3
CLAIM_MASK, CLAIM_RANDDIR, CLAIM_PARA, CLAIM_TRANSFER = ...
```

The verifier never reads the manuscript. It recomputes each quantity from the raw JSON and
compares. **If a number in the paper is edited without editing this block, the check
fails** — which is the point. A hand-transcribed value cannot survive.

Tolerances follow the precision printed in the paper (two decimals for statistics, four
for accuracies). A `FAIL` on the last decimal place is a real discrepancy, not noise; the
values were verified to agree to the printed precision.

---

## 7. Level 1 — re-derive from existing adapters

This level answers the central empirical claim directly: does truncating to a sixteenth
really cost three predictions?

**Prerequisites:** Qwen2.5-3B downloaded; an adapter at `ckpts_sft3/fpb_r16`
(ternary) or the binary equivalent.

### 7.1 Truncate

```bash
export PROJ=/root/autodl-tmp/FinMLLM
export BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-3B
cd $PROJ

python3 scripts/rv_lowrank.py \
    --src ckpts_sft3/fpb_r16 \
    --rank 1 \
    --out ckpts_repro/fpb_r1
```

Takes **approximately 23 seconds**. The implementation decomposes the `r × r` core
(`ΔW = BA`, so `A` is `r × in` and `B` is `out × r`; QR both, then SVD the `r × r` product)
instead of forming and decomposing the `11008 × 2048` dense matrix, which takes 11 minutes
and saturates every core (ERRATA E-04). The result is mathematically identical — leading
`k` singular triplets, optimal in the Eckart–Young sense.

### 7.2 Evaluate

```bash
mkdir -p outputs/reports/repro

python3 scripts/rv_eval_preds.py \
    --dataset fpb --adapter ckpts_repro/fpb_r1 --classes 3 \
    --split test --holdout 0.2 \
    --out outputs/reports/repro/tri_s42_r1.json

python3 scripts/rv_eval_preds.py \
    --dataset fpb --adapter ckpts_sft3/fpb_r16 --classes 3 \
    --split test --holdout 0.2 \
    --out outputs/reports/repro/tri_s42_r16.json
```

Approximately 60 seconds each. Every evaluation stores the per-item prediction list, which
is what makes the paired test possible.

### 7.3 Compare

```bash
python3 scripts/rv_summarize.py --dir outputs/reports/repro
```

### 7.4 Expected values

Ternary (`--classes 3`, n = 968), seed 42:

| Quantity | Expected |
|---|---|
| accuracy at r = 16 | **0.7831** |
| accuracy at r = 1 | **0.8161** |
| difference (r1 − r16) | **+3.30 pp** |
| discordant counts `b` / `c` | **44 / 12** |
| McNemar p | **< 0.0001** |

Binary (`--classes 2 --binary`, n = 394), seed 42:

| Quantity | Expected |
|---|---|
| accuracy at r = 16 | **0.9873** |
| accuracy at r = 1 | **0.9898** |
| difference | **+0.25 pp** |
| `b` / `c` | **2 / 1** |
| McNemar p | **1.0000** |

Three-seed means quoted in the paper: binary **0.9873 → 0.9898**;
ternary **0.7968 → 0.8382** (mean gain **+4.13 pp**).

### 7.5 Cross-domain transfer (§4.6)

Three conditions, all evaluated on the FiQA-SA test partition (n = 243):

```bash
python3 scripts/verify_fiqa_transfer.py
```

| Condition | Expected accuracy |
|---|---|
| base, no adapter, with null-context calibration | **0.2181** |
| + PhraseBank ternary r = 16 | **0.6626** |
| + PhraseBank ternary r = 1 (16×) | **0.6749** |

Transfer gain from base to r = 16 is **+44.44 pp**; the r16 → r1 difference is
**+1.23 pp** with McNemar **p = 0.6072** — not significant. The script additionally
verifies that the 243 test items are identical to those in the archived
`results/fiqa/fiqa_s42_r16.json`, so a partition drift cannot be mistaken for a replication
failure.

> **Note.** These three numbers supersede those in an earlier draft (0.2428 / 0.6749 /
> 0.6543). The earlier values were computed before `fix_fiqa.py` rebuilt the FiQA file and
> were never written to disk. See ERRATA E-26.

---

## 8. Level 2 — full retraining from the corpus

**Prerequisites:** corpus prepared (Section 5); Qwen2.5-3B downloaded.

### 8.1 Train one configuration

```bash
export PROJ=/root/autodl-tmp/FinMLLM
cd $PROJ

nohup python3 scripts/rv_train_sft.py \
    --dataset fpb --classes 3 --rank 16 \
    --seed 42 --split-seed 42 --holdout 0.2 \
    --epochs 3 --bs 8 --lr 2e-4 \
    --out ckpts_repro/tri_r16_s42 \
    > logs/train_tri_s42.log 2>&1 &
```

Roughly **25 minutes** on a 4090.

Hyper-parameters, as reported in the paper:

| Parameter | Value |
|---|---|
| quantisation | 4-bit NF4 |
| LoRA targets | every linear projection |
| rank / alpha | 16 / 32 (scale 2.0) |
| learning rate | 2e-4, cosine schedule |
| batch size | 8 |
| epochs | 3 |
| trainable parameters | 29,933,568 (0.970% of the base) |

### 8.2 The other seeds

```bash
# serial, not parallel — see the CPU oversubscription note in Section 11
for s in 123 2024; do
  python3 scripts/rv_train_sft.py --dataset fpb --classes 3 --rank 16 \
      --seed $s --split-seed 42 --holdout 0.2 --epochs 3 \
      --out ckpts_repro/tri_r16_$s
done
```

**`--split-seed` is always 42.** Only `--seed` varies.

### 8.3 Over-parameterisation control (rank 64)

```bash
python3 scripts/rv_train_sft.py --dataset fpb --classes 3 --rank 64 \
    --seed 42 --split-seed 42 --holdout 0.2 --epochs 3 \
    --out ckpts_repro/tri_r64_s42
```

### 8.4 Then truncate, evaluate and analyse

Repeat Section 7 for each adapter, then Section 10 for the spectral statistics.

---

## 9. Supplementary experiments

### 9.1 Parameter masking

```bash
python3 scripts/rv_make_variants.py --src ckpts_sft/fpb_r16 \
    --kind mask --pcts 40,60,80 --out-root ckpts_supp/bin
```

Evaluate each variant against the `bin` and `tri` test partitions. Expected (full test
partitions, n = 394 / 968):

| Sparsity | binary | ternary |
|---|---|---|
| 40% | 0.9898 | 0.7779 |
| 60% | 0.9898 | 0.7851 |
| 80% | 0.9924 | 0.7893 |

A masked adapter keeps its full shape, so the file size does not change (114.3 MB at
every sparsity). Under a sparse encoding each non-zero costs 2 bytes of value plus 4 of
index; at 80% sparsity that is 34.3 MB against 7.2 MB for the rank-1 adapter.

### 9.2 Random-direction truncation

```bash
python3 scripts/rv_make_variants.py --src ckpts_sft/fpb_r16 \
    --kind randdir --ranks 8,4,2,1 --draws 5 --out-root ckpts_supp/bin
```

**Five draws, and the mean is what is reported.** A single draw once produced a spurious
"random beats SVD" reversal at rank 4 (0.8833); over five draws the same condition averages
0.7789 ± 0.0514. The reversal was draw variance (ERRATA E-07). Reporting one draw would
have been a serious error.

Expected means over the five draws:

| Rank | binary | ternary |
|---|---|---|
| 8 | 0.9873 | 0.7709 |
| 4 | 0.9848 | 0.7789 |
| 2 | 0.9797 | 0.7839 |
| 1 | 0.8487 | 0.6231 |

### 9.3 PARA global-threshold baseline

```bash
python3 scripts/rv_para_baseline.py --src ckpts_sft/fpb_r16 \
    --qs 40,60,80,90,98 --out-root ckpts_supp/para
```

Expected `(binary, ternary)` accuracy per threshold:

| q | rank kept | binary | ternary |
|---|---|---|---|
| 40 | 10 | 0.9898 | 0.7810 |
| 60 | 6 | 0.9898 | 0.7810 |
| 80 | 3 | 0.9924 | 0.7810 |
| 90 | 2 | 0.9924 | 0.7831 |
| 98 | 1 | 0.9898 | 0.8161 |

PARA at its most aggressive threshold reaches 0.8161 on the ternary task, versus 0.8161 for
our rank-1 truncation — the same point, but reached only after sweeping five thresholds. The
argument in the paper rests on needing no sweep, not on accuracy.

---

## 10. Spectral diagnostics (Table 3 and Table A2)

### 10.1 Table 3 — effective rank

```bash
python3 scripts/spectral_table3.py \
    --adapter ckpts_sft3/fpb_r16 \
    --tag 3b_bin_r16 \
    --out-dir outputs/reports/table3_repro
```

`scripts/spectral_table3.py` is the **only authoritative implementation** of these
statistics. `scripts/_deprecated/rv_spectral_table3.py` is kept for the record and must not
be used: it hardcodes a Gaussian baseline of 13.55 where the correct per-layer weighted mean
is 14.08 (ERRATA E-18).

Expected values for the nine Table 3 rows:

| Model | r | stable | PR | eRank | n90 | noise | ratio |
|---|---|---|---|---|---|---|---|
| 3B binary | 16 | 1.98 | 8.50 | 11.40 | 6.46 | 14.08 | 0.141 |
| 3B ternary | 16 | 1.86 | 8.02 | 11.04 | 6.02 | 14.08 | 0.133 |
| 3B FiQA | 16 | 2.28 | 9.42 | 12.04 | 7.40 | 14.08 | 0.163 |
| 3B binary | 64 | 2.33 | 22.02 | 37.06 | 13.89 | 50.09 | 0.047 |
| 3B ternary | 64 | 2.50 | 23.90 | 38.42 | 16.24 | 50.09 | 0.050 |
| 7B binary | 16 | 1.94 | 8.25 | 11.20 | 6.22 | 14.51 | 0.134 |
| 7B ternary | 16 | 1.74 | 7.54 | 10.67 | 5.42 | 14.51 | 0.120 |
| 7B binary | 64 | 2.08 | 20.73 | 36.46 | 13.20 | 52.96 | 0.039 |
| 7B ternary | 64 | 1.95 | 18.87 | 33.98 | 12.25 | 52.96 | 0.037 |

Tolerance: ±0.02 for the statistics, ±0.003 for the ratio. Larger deviations usually mean
`--base` points at the wrong model, or `numpy < 2.0`.

> **Only stable rank supports the "≈ 2" reading.** Participation ratio (8.50) and entropy
> effective rank (11.40) are both below the nominal sixteen but nowhere near two. The
> spectrum is heavy-tailed, not collapsed. Do not cite PR or eRank as evidence of low
> dimensionality.

### 10.2 Table A2 — what the cut discards

```bash
OMP_NUM_THREADS=1 python3 scripts/check_energy_drop.py ckpts_sft/fpb_r16 1
```

Expected:

| Statistic | Expected |
|---|---|
| energy discarded at the 1/16 cut | **0.4306** |
| largest discarded σ / σ₁ | **0.5199** |
| spectral gap at the cut (median) | **1.9471** |

Runs in about one minute. `OMP_NUM_THREADS=1` is not cosmetic: without it BLAS spawns one
thread per core, and 252 matrices will push a shared machine to a load average near 30.

---

## 11. Troubleshooting

Each of these was actually encountered. The symptom is quoted verbatim.

### 11.1 `can't open file 'scripts/xxx.py'`

The script `cd`s to `$PROJ` and then resolves sibling scripts relative to the working
directory. Fixed: all internal invocations use `$SCRIPT_DIR`. If you add a script, follow
that convention.

### 11.2 `size mismatch for ...`

A per-layer rank does not match PEFT's single global `r`.
`rv_make_variants.py` and `rv_para_baseline.py` emit **uniform** rank adapters for this
reason. Per-layer heterogeneous rank would require patching `LoraConfig` (ERRATA E-05).

### 11.3 `TypeError: LoraConfig.__init__() got an unexpected keyword argument 'rv_variant'`

Custom fields were being written into `adapter_config.json`. They now go to a sibling
`rv_meta.json`. PEFT validates `adapter_config.json` strictly (ERRATA E-03).

### 11.4 `pairs=0` or `n=0` on FiQA

The label column contains tickers. Run `scripts/supp/fix_fiqa.py` (Section 5.2, ERRATA E-11).

### 11.5 GPU utilisation 0%, CPU pinned, apparent hang

Two evaluation queues were running concurrently. Each 7B process already saturates eight
cores; the second queue starves the first. **Run serially.** The overnight scripts
(`rv_overnight.sh`) are serial for this reason (ERRATA E-10).

### 11.6 `FAIL *__bin` entries in a batch

An empty directory was globbed. Bash does not expand a glob that matches nothing, so the
literal pattern was passed downstream. The scripts now set `shopt -s nullglob` and pre-check
each adapter path (ERRATA E-09).

### 11.7 Truncation takes 11 minutes and every core is at 100%

You are on the dense path. Use `rv_lowrank.py`, not `make_lowrank_adapter.py`
(ERRATA E-04).

### 11.8 Spectral statistics are off by ~0.01

`numpy < 2.0`. `np.linalg.svd` differs on `float16` across the major-version boundary.
Pin `numpy==2.5.4`.

### 11.9 Accuracy reaches 1.000 on a new seed

The split seed moved with the training seed. Use `rv_train_sft.py` and pass
`--split-seed 42` explicitly (ERRATA E-01). This is the single most damaging defect in the
project's history: it inflated two of three seeds before it was caught.

### 11.10 Long job dies when the laptop lid closes

Run everything through `scripts/run_bg.sh`, which wraps `nohup` with a timestamped run
directory. A bare foreground process receives `SIGHUP` and dies.

```bash
bash scripts/run_bg.sh train "python3 scripts/rv_train_sft.py --rank 16 --seed 42"
bash scripts/status.sh --watch
```

---

## 12. Directory conventions

```
$PROJ/
├── data/
│   ├── raw/              downloaded source archives
│   └── unified/          fpb.jsonl, fiqa_sa.jsonl  ← everything reads from here
├── ckpts_sft/            base adapters (binary / ternary, rank 16)
├── ckpts_sft3/           ternary adapters used for the reported results
├── ckpts_rv/             per-seed adapters, 3B and 7B, plus FiQA
├── ckpts_supp/           mask / randdir / PARA variants
├── outputs/reports/      evaluation JSON, one file per condition
│   ├── rv_final/         ← the records the paper's numbers come from
│   ├── table3/           spectral diagnostics
│   ├── rv_supp/          supplementary conditions
│   ├── rv_para/          PARA baseline
│   └── fiqa_transfer/    §4.6 cross-domain results
├── logs/                 background job logs
└── runs/<timestamp>_<tag>/   per-run environment snapshots
```

`results/` in this repository mirrors the relevant parts of `outputs/reports/`.

---

## 13. Reproducibility checklist

Tick these in order. Each is cheap and each has caught a real defect.

- [ ] `pip install -r requirements.txt` succeeds; `python -c "import torch; print(torch.cuda.is_available())"` returns `True` for L1/L2
- [ ] `python3 verify/verify_all.py` → **121 checks passed**
- [ ] `data/unified/fpb.jsonl` has **4,840** lines; class distribution **59.36 / 28.16 / 12.48**
- [ ] `python3 scripts/supp/fix_fiqa.py` has been run; `fiqa_sa.jsonl` has **1,213** lines
- [ ] `bash scripts/run_gates.sh nogpu` passes G0–G2
- [ ] (L1) `rv_lowrank.py` on `fpb_r16 → r1` completes in ~23 s and writes a 7.2 MB adapter
- [ ] (L1) ternary r16 / r1 evaluate to **0.7831 / 0.8161**, `b=44 c=12`
- [ ] (L1) binary r16 / r1 evaluate to **0.9873 / 0.9898**, `b=2 c=1`, p = 1.0000
- [ ] (L2) a retrained rank-16 ternary adapter lands within ±0.02 of stable rank **1.86**
- [ ] (L2) `spectral_table3.py` reproduces the noise baseline **14.08** at rank 16 and **50.09** at rank 64

If a box cannot be ticked, `docs/ERRATA.md` almost certainly explains why before you have to
debug it yourself.

---

## 14. Environment snapshot

The environment in which the reported numbers were produced:

```
OS            Ubuntu 22.04 (container)
Python        3.12.3
CUDA          12.4
GPU           NVIDIA RTX 4090, 24,564 MiB
torch         2.4.1+cu124
transformers  4.46.3
peft          0.13.2
bitsandbytes  0.50.2
accelerate    0.34.2
datasets      5.1.0
numpy         2.5.4
scipy         1.18.1
pandas        3.0.6
safetensors   0.8.0
matplotlib    3.9.0
```

Anything in this list can be changed, but if a number does not reproduce, restore this set
before investigating further.

---

*Questions about reproduction: open an issue in this repository. Please include the output
of `python3 verify/verify_all.py` and your `python3 scripts/lib_probe.py` report.*

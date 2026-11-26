# OUTCOME — `2026-11-26T00-54-vjepa21-gram-rism-v2`

- **Status:** `COMPLETED` (returncode `0`, Modal container `ap-2nt0fsApbBEPxzXGMGYPZL` verified **stopped** — zero idle GPU burn)
- **Verdict:** **`ITERATE`** — **DO NOT warm-start Iter 11 from `uni_omr_v2_ijepa_best.pt` or `uni_omr_v2_ijepa_last.pt`!**
- **Platform / GPU:** Modal `NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, native `BF16`, PyTorch `2.15.0+cu132`)
- **Started / Completed (UTC):** `2026-11-25T23:15:04Z` → `2026-11-26T00:03:12Z`
- **Elapsed Time & Est. Cost:** `2880.1 s` (`48.0 min`), `~$0.64` Modal credit
- **Steps / Throughput / VRAM:** `1,500 / 1,500` steps (`5,707` RISM HW pages, `batch_size=4 × grad_accum=2 = 8`), peak `14.12 GB` VRAM, `0.58 it/s`, `oom_skips = 0`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`, `repo_type="dataset"`):**
  - Run ledger: `runs/2026-11-26T00-54-vjepa21-gram-rism-v2.json`
  - Training log: `checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/vjepa21_train_log.json` (pulled locally to [vjepa21_train_log.json](vjepa21_train_log.json) and `D:\tmp\mvd-runs\vjepa21-gram-rism-v2\vjepa21_train_log.json`)
  - Best EMA checkpoint (`step=385`): `checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/uni_omr_v2_ijepa_best.pt` (downloaded to `D:\tmp\mvd-runs\vjepa21-gram-rism-v2\uni_omr_v2_ijepa_best.pt`, `157.80 MB`)
  - Final checkpoint (`step=1500`): `checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/uni_omr_v2_ijepa_last.pt` (downloaded to `D:\tmp\mvd-runs\vjepa21-gram-rism-v2\uni_omr_v2_ijepa_last.pt`)

---

## 1. Measured Training Trajectory (`vjepa21_train_log.json`)

| Metric | Step 1 | Min (Step) | Max (Step) | Step 1500 (Final) | Notes |
|---|---:|---:|---:|---:|---|
| `loss` | `2.28573` | `0.16475` (`375`) | `2.28573` (`1`) | `0.17825` | Plateaued by step `~200` |
| `ema_loss` | `2.28573` | **`0.16820`** (`385`, best ckpt) | `2.28573` (`1`) | `0.17264` | ~80% is constant router BCE floor (`0.137`) |
| `jepa_masked` | `1.38904` | `0.00722` (`650`) | `1.38904` (`1`) | `0.01607` | Collapsed `192×` by step `175` |
| `jepa_visible` | `1.39153` | `0.00742` (`650`) | `1.39153` (`1`) | `0.01455` | Identical to `jepa_masked` (collapse signature) |
| `deep_sup` | `0.67852` | `0.00801` (`375`) | `0.67852` (`1`) | `0.01345` | Collapsed alongside final JEPA |
| `gram` | `0.01542` | `0.00004` (`175`) | `0.01542` (`1`) | `0.00055` | Anchored to random-init teacher |
| `stroke_hog` | `0.25624` | `0.04345` (`1275`) | `0.25624` (`1`) | `0.04471` | Healthy monotonic decline on `/4` `p2` |
| `router_loss` | `0.69162` | `0.67847` (`1000`) | `0.69236` (`50`) | `0.68607` | Stuck at `ln(2) ≈ 0.693` BCE entropy floor |
| `vram_gb` | `8.52` | `8.52` (`1`) | `14.12` (`950`) | `14.12` | Well inside 24 GB L4 budget (`oom_skips=0`) |

---

## 2. CPU Checkpoint Integrity & Representation Collapse Diagnostic

Verified `uni_omr_v2_ijepa_best.pt` (`step=385`) and `uni_omr_v2_ijepa_last.pt` (`step=1500`) on CPU against a freshly initialized `UniOMRv2Net(UniOMRv2Config.small_45m())` over 6 real RISM historical manuscripts (`rism-ch_1001257463.jpg` .. `rism-ch_400006313.jpg`, resized to `512px`):

- **Checkpoint integrity (`uni_omr_v2_ijepa_best.pt`):** `349` tensors, `41,334,795` params, **`NaN = 0`**, **`Inf = 0`**, `load_state_dict(strict=False)` → **`missing = 0`, `unexpected = 0`**. (2 zero-bias tensors are frozen/untrained aux heads: `staff_seg_head.head.1.bias`, `stroke_hog_head.net.1.bias`.)
- **Representation effective rank & collapse probe:**

| Checkpoint | `p2_erank` (of 128) | `p3_erank` (of 384) | `mem_erank` (of 384) | `mem_tok_std` | `within_page_cos` | `cross_page_cos` | Health |
|---|---:|---:|---:|---:|---:|---:|---|
| `random_init` (`small_45m`) | `32.5` | `33.2` | `38.0` | `0.2060` | `0.9579` | `0.9803` | Baseline |
| **`exp2_best_step385`** (`best.pt`) | `21.5` | **`2.1`** (`-93.7%`) | **`2.6`** (`-93.2%`) | `0.1115` | `0.9848` | **`0.9993`** | **SEVERE RANK COLLAPSE** |
| **`exp2_last_step1500`** (`last.pt`) | `15.2` | **`2.3`** (`-93.1%`) | `10.4` (`-72.6%`) | `0.1781` | `0.9680` | `0.9859` | **PARTIAL COLLAPSE (`p3` dead)** |

---

## Scrutiny & Anomaly Check

1. **CRITICAL ANOMALY — Severe Representation Rank Collapse in `p3` and `mem` (especially `best.pt` at step 385):**
   - Although the run completed all `1,500` steps with `0` OOMs and `0` NaN/Infs, `jepa_masked` and `jepa_visible` plummeted identically from `1.39` → `0.008` by step `175`.
   - CPU SVD probing proved **classic JEPA dimensional collapse**: at `step=385` (`uni_omr_v2_ijepa_best.pt`), `p3_erank` collapsed to **`2.1 / 384`**, `mem_erank` collapsed to **`2.6 / 384`**, and `cross_page_cos` reached **`0.9993`** (all RISM pages collapsed onto nearly the same direction).
   - Only `/4` `p2` avoided full collapse (`p2_erank = 21.5 / 128` at step 385, `15.2 / 128` at step 1500) because it was directly anchored by the non-collapsing Sobel-HOG stroke reconstruction loss (`stroke_hog`: `0.256 → 0.043`).
   - By `step=1500` (`last.pt`), `mem_erank` partially recovered (`2.6 → 10.4`), but `p3_erank` remained collapsed at `2.3 / 384` (`14×` below random init).

2. **Root Cause 1 — Random-Init DINOv3 Gram Anchoring Actively Rewards Collapse:**
   - Because Exp 2 trained from scratch (`--init-from None`), `gram_teacher` was frozen at **step-0 random weights**, whose tokens already have `within_page_cos = 0.9579` (a nearly constant Gram matrix `G_t ≈ 0.96 · 11^T`).
   - A collapsed student (`within_page_cos = 0.9848`) achieved `gram = 0.00004` (`4e-5`), meaning Gram anchoring against a random-init encoder failed to prevent collapse and instead rewarded high off-diagonal cosine similarity. Moreover, `mem_final = enc_norm(mem) + pos2d` adds fixed 2D positional encodings *before* the Gram loss, further masking feature collapse.
   - **Rule:** Gram anchoring MUST be disabled when training from scratch (`--disable-random-gram`), and only enabled when adapting a *trained* supervised checkpoint.

3. **Root Cause 2 — No Explicit Variance / Covariance Anti-Collapse Regularizer:**
   - Pure EMA-teacher JEPA with `Muon` (fast spectral updates) + `batch_size=8` from scratch collapsed faster than the EMA teacher (`decay=0.996`) could diversify features. Adding **VICReg variance + covariance regularization** (`--vicreg-var-weight 1.0 --vicreg-cov-weight 0.04`) on both `mem_post_merger` (pre-`pos2d`) and `mem_final` directly prevents `p3` and `mem` rank collapse.

4. **Root Cause 3 — Router BCE Entropy Floor (`ln(2) ≈ 0.693`) Masked Collapse & Mis-selected `best.pt`:**
   - `router_loss` used `F.binary_cross_entropy_with_logits(sal_logits_student, sigmoid(sal_logits_teacher))` instead of KL divergence. For near-zero initial logits (`p ≈ 0.5`), `BCE(0.5, 0.5) = ln(2) = 0.69315`, injecting a `0.2 × 0.686 = 0.137` constant floor (~80% of total `0.172` loss).
   - Because `ema_loss` was dominated by this floor + collapsed JEPA loss, `uni_omr_v2_ijepa_best.pt` was saved at **step 385** — the exact peak of representation collapse! Switching to `--router-loss-mode kl` (`BCE - H(p_t)`) gives a true `0.0` floor.

5. **Hardware / Platform Check:**
   - Modal `L4` (`BF16`, `torch.optim.Muon` + fused `AdamW`) ran smoothly at `0.58 it/s` (`14.12 GB` peak VRAM, `0` OOMs) and stopped cleanly after uploading all checkpoints and ledgers.

---

## 3. Exact Warm-Start Recommendation for Iter 11

> **DO NOT warm-start Iter 11 from `D:\tmp\mvd-runs\vjepa21-gram-rism-v2\uni_omr_v2_ijepa_best.pt` (or `last.pt`)!**

- `uni_omr_v2_ijepa_best.pt` (`step=385`) has a severely collapsed `p3` (`erank=2.1/384`) and `mem` (`erank=2.6/384`, `cross_page_cos=0.9993`) representation. Warm-starting a full Iter 11 run from it risks sluggish early convergence or permanent pitch/layout degradation compared to random init.
- **Recommendation for Iter 11:** Start `UniOMR v2` Iter 11 from **random init** (or wait for the anti-collapse successor below to verify `post_erank >= 15`, `mem_erank >= 25`, and positive `suite31_hw` transfer).

---

## 4. Follow-Up Action / Successor Experiment

Per `.agents/skills/experiment-finished/SKILL.md` (`ITERATE` verdict):
1. **Implemented anti-collapse fixes in `tmp-train/mvd/pretrain_ijepa_rism_v2.py`:**
   - Added `compute_vicreg_loss` (`--vicreg-var-weight`, `--vicreg-cov-weight`) applied to both `mem_post_merger` (pre-`pos2d`) and `mem_final`.
   - Added `--disable-random-gram` (disables Gram anchoring when `--init-from` is `None`).
   - Added `--router-loss-mode kl` (zero-floor KL divergence instead of `ln(2)`-floored BCE).
   - Added live `@torch.no_grad()` effective-rank telemetry (`post_erank`, `mem_erank`, `tok_std`, `within_cos`) logged directly in `vjepa21_train_log.json`.
2. **Queued DAG Successor Experiment:**
   - **Successor (Anti-Collapse Fix + Transfer Probe):** [`2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2`](../2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2/README.md) — registered in `experiments/overnight_queue/queue.json` with `depends_on: ["2026-11-26T00-54-vjepa21-gram-rism-v2"]`.
   - **Existing Empirical Probe (`ov-33`):** [`2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch`](../2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch/run_colab.py) (also `depends_on: ["2026-11-26T00-54-vjepa21-gram-rism-v2"]`) remains queued to empirically confirm Exp 2's collapsed transfer baseline vs scratch.

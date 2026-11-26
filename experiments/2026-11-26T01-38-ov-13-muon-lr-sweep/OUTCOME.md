# Outcome — `2026-11-26T01-38-ov-13-muon-lr-sweep`

- **Status:** `COMPLETED`
- **Verdict:** **`ITERATE`** *(coupled `lr=5.0e-4` won monotonically at the upper boundary with lower grad norms, but exhibited RAW greedy-decode oscillation and severe `61.8%` EMA random-init lag; follow-up [`2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep`](../2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep/PLAN.md) isolates decoupled `--muon-lr ∈ {6e-4, 1.0e-3, 1.5e-3}` with AdamW held at `3e-4` and un-lagged EMA `0.99/1`)*
- **Completed At (UTC):** `2026-11-26T02:49:01Z` (started `2026-11-26T00:46:45Z`)
- **Platform / GPU:** Google Colab `NVIDIA L4` (`24 GB`, Ada `cc=8.9`, `torch=2.14.1+cu130`, native `torch.optim.Muon`, native BF16) — verified **stopped** (`colab sessions` shows `ov-13` terminated; zero idle CU burn)
- **Elapsed Wall-Clock:** `5,752.79 s` training (`95.9 min`: `1,924.4 s` + `1,913.5 s` + `1,914.9 s`); `7,381.19 s` (`123.0 min`) total session including 6 `proxy_val` greedy decodes & HF uploads
- **Baseline Commit SHA:** `164852f`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`, `dataset`):**
  - Eval JSON: `evals/2026-11-26T01-38-ov-13-muon-lr-sweep.json` (pulled to [eval_result.json](./eval_result.json))
  - Run ledger: `runs/2026-11-26T01-38-ov-13-muon-lr-sweep.json`
  - Best EMA checkpoint (`lr_5p0e-4`): `checkpoints/2026-11-26T01-38-ov-13-muon-lr-sweep/best_ema_latest.pt`

---

## Measured Results (`800` steps/arm, `iter10_5_proxy_2k`, `seed=42`, `muon+adamw`, `dim=1280`, `eval_max_len=512`)

### 1. Proxy Validation (`64` held-out full pages, `proxy_val.jsonl`)

| Checkpoint & Metric | `lr_1p5e-4` | `lr_3p0e-4` (anchor) | `lr_5p0e-4` | Delta (`5e-4 - 3e-4`) |
|---|---:|---:|---:|---:|
| **Final-100 Train Loss** (`701–800`) ↓ | `2.66502` | `2.39667` | **`2.22052`** | **`-0.17615`** |
| **Final-100 Train CE** (`701–800`) ↓ | `1.80847` | `1.61682` | **`1.48802`** | **`-0.12880`** |
| **RAW `val_loss`** ↓ | `2.594898` | `2.289588` | **`2.142540`** | **`-0.147048`** |
| **RAW `val_ce_loss`** ↓ | `1.699101` | `1.522168` | **`1.427551`** | **`-0.094617`** |
| **RAW `val_mtp_loss`** ↓ | `2.474307` | `2.268416` | **`2.147248`** | **`-0.121168`** |
| **RAW `val_stroke_hog_loss`** ↓ | `0.026548` | `0.023099` | **`0.021897`** | **`-0.001202`** |
| **RAW `val_ser`** ↓ | `0.992525` | **`0.895229`** | `0.969529` *(see Anomaly #3)* | `+0.074300` |
| **RAW `val_mean_sample_ser`** ↓ | `1.532777` | **`1.444741`** | `1.506894` *(see Anomaly #3)* | `+0.062153` |
| **RAW `val_token_f1`** ↑ | `0.122796` | **`0.243482`** | `0.167007` *(see Anomaly #3)* | `-0.076475` |
| **RAW `val_sys_count_exact`** ↑ | `0.109375` (`7/64`) | `0.140625` (`9/64`) | **`0.171875`** (`11/64`) | **`+0.031250`** |
| **EMA `val_loss`** ↓ *(lagged)* | `2.774408` | `2.488433` | **`2.311202`** | **`-0.177231`** |
| **EMA `val_ce_loss`** ↓ *(lagged)* | `1.865178` | `1.658668` | **`1.548022`** | **`-0.110646`** |
| **EMA `val_mtp_loss`** ↓ *(lagged)* | `2.635388` | `2.438637` | **`2.315576`** | **`-0.123061`** |
| **EMA `val_stroke_hog_loss`** ↓ *(lagged)* | `0.029682` | `0.025492` | **`0.024251`** | **`-0.001241`** |
| **EMA `val_ser`** ↓ *(lagged)* | `0.996638` | `0.923590` | **`0.877651`** | **`-0.045939`** |
| **EMA `val_mean_sample_ser`** ↓ *(lagged)* | `1.547574` | `1.487527` | **`1.435824`** | **`-0.051703`** |
| **EMA `val_token_f1`** ↑ *(lagged)* | `0.111350` | `0.195254` | **`0.264290`** | **`+0.069036`** |
| **EMA `val_sys_count_exact`** ↑ *(lagged)* | `0.109375` (`7/64`) | `0.109375` (`7/64`) | `0.109375` (`7/64`) | `0.000000` |

### 2. Optimization Stability, Throughput & Memory

| Metric | `lr_1p5e-4` | `lr_3p0e-4` | `lr_5p0e-4` | Notes |
|---|---:|---:|---:|---|
| Steps `1–100` Train CE / `grad_norm_p95` | `4.0496` / `14.620` | `3.6474` / `12.446` | **`3.3787`** / **`11.357`** | Higher LR converges faster & smoother from step 1 |
| Steps `501–600` Train CE / `grad_norm_p95` | `1.9275` / `3.562` | `1.7209` / `2.448` | **`1.6171`** / **`2.381`** | Monotonic improvement at end of WSD plateau |
| Steps `701–800` Train CE / `grad_norm_p95` | `1.8085` / `3.290` | `1.6168` / `2.325` | **`1.4880`** / **`2.083`** | Cleanest cooldown at `5.0e-4` |
| Overall `grad_norm_p95` | `5.67528` | `4.12917` | **`3.65918`** | Strictly decreases with LR (`< 5.0` gate passed) |
| Wall-Clock (`800` steps) | `1,924.41 s` (`0.4157 st/s`) | `1,913.48 s` (`0.4181 st/s`) | `1,914.90 s` (`0.4178 st/s`) | Identical throughput (`~2.40 s/step`) |
| Peak VRAM Alloc / Reserved | `9.942 / 10.109 GB` | `9.942 / 10.111 GB` | `9.942 / 10.111 GB` | Identical across all 3 arms |

---

## Scrutiny & Anomaly Check

1. **Coverage & Sanity:** All `64/64` held-out validation pages were scored for both RAW and EMA checkpoints across all 3 arms (`eval_max_len=512`, `seed=42`, `optimizer_mode == "muon+adamw"` verified on all 3). Zero NaNs, zero Infs, zero OOMs, and identical `9.942 GB` peak VRAM.
2. **ANOMALY #1 — Boundary-Optimal LR with Decreasing Gradient Norms (`5.0e-4` is not yet saturated):**
   - `lr_5p0e-4` decisively beat the `3.0e-4` anchor on EMA `val_ce_loss` (`-0.1106`, `>5×` the `-0.02` gate), EMA `val_token_f1` (`+0.0690`, `~7×` the `+0.01` gate), EMA `val_ser` (`-0.0459`), RAW `val_ce_loss` (`-0.0946`), RAW `val_loss` (`-0.1470`), and RAW `val_sys_count_exact` (`+0.0313`, `11/64` vs `9/64`).
   - Remarkably, `grad_norm_p95_overall` **decreased monotonically** as LR increased (`5.675 -> 4.129 -> 3.659`), with final-100 `grad_norm_p95 = 2.083` at `5.0e-4`. Meanwhile `lr_1p5e-4` was badly under-trained (`+0.2065` EMA CE, `-0.0839` EMA F1, highest `grad_norm_p95 = 5.675`).
   - This confirms Exp 4's diagnosis: `3.0e-4` strongly under-drives Muon's 2D Transformer blocks, and the true optimum for `--muon-lr` lies **at or above `5.0e-4`**.
3. **ANOMALY #2 — Severe EMA Lag Confound (`61.8%` random step-0 init weight):**
   - With `--gpu-ema-decay 0.997 --gpu-ema-every 5` over only `800` steps (`160` EMA updates), **`0.997^160 = 0.6183` (61.8%)** of the EMA checkpoint weight is still the random step-0 initialization.
   - Consequently, EMA teacher-forced losses are severely degraded relative to RAW on every arm (`val_ce`: `+0.166`, `+0.137`, `+0.120` worse; `val_loss`: `+0.180`, `+0.199`, `+0.169` worse).
4. **ANOMALY #3 — RAW vs EMA Greedy-Decode Inversion on `lr_5p0e-4` (Coupled AdamW LR Side-Effect):**
   - Although `lr_5p0e-4` RAW achieved the lowest teacher-forced losses of the entire sweep (`val_loss=2.1425`, `val_ce=1.4276`, `val_mtp=2.1472`, `sys_count_exact=0.1719`), its greedy autoregressive decode regressed relative to `lr_3p0e-4` RAW (`val_token_f1: 0.1670` vs `0.2435`; `val_ser: 0.9695` vs `0.8952`), whereas `lr_5p0e-4` **EMA** jumped to the sweep's best greedy decode (`val_token_f1: 0.2643`, `val_ser: 0.8777`).
   - Root cause: `ov-13` swept **coupled** `--lr`, pushing AdamW's LR for embeddings, `lm_head`, norms, and 4D convs up to `5.0e-4` alongside Muon. Higher AdamW LR on the output head introduces late-step logit/exposure-bias jitter in greedy decoding that EMA smoothing suppresses — motivating **decoupling `--muon-lr`** (`> 5e-4`) while holding AdamW `--lr` at `3.0e-4`.
5. **ANOMALY #4 — Wall-Clock vs Estimate:**
   - Actual runtime was `123.0 min` (`95.9 min` training at `~2.40 s/step` + `27.1 min` for 6 `proxy_val` greedy evals & upload) vs the `30 min` estimate (`~4×` higher). Future L4 `small_45m` `dim=1280` proxy estimates should budget `~2.4 s/step` plus `~4.5 min` per `proxy_val` greedy eval.

---

## Verdict

- **Verdict: `ITERATE`**
- **Rationale:** `lr_5p0e-4` decisively beat `3.0e-4` at the upper boundary of the sweep with *lower* gradient norms (`p95=3.66`), proving `lr=3.0e-4` under-drives Muon. However, we must **not** blindly promote coupled `--lr 5.0e-4` into full Iter 12 training because (a) the optimum is still boundary-clipped (`>= 5.0e-4`), (b) coupled `5.0e-4` degraded RAW greedy decoding (`0.1670` vs `0.2435` F1), and (c) `0.997/5` EMA retained `61.8%` random step-0 init weight.

---

## Follow-Up Action / Successor Experiment

- **Queued Mandatory `ITERATE` Successor:** [`2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep`](../2026-11-26T03-31-exp4b-decoupled-muon-lr-sweep/PLAN.md) (`depends_on: ["2026-11-26T00-54-ablation-muon-vs-adamw", "2026-11-26T01-38-ov-13-muon-lr-sweep"]`, registered in `experiments/overnight_queue/queue.json`).
- **How it directly resolves all 3 `ov-13` findings:**
  1. **Probes above `5.0e-4`:** Sweeps decoupled `--muon-lr ∈ {6.0e-4, 1.0e-3, 1.5e-3}` (`2×`, `3.3×`, `5×` AdamW).
  2. **Eliminates coupled AdamW greedy-decode jitter:** Holds AdamW `--lr 3.0e-4` constant for embeddings, `lm_head`, norms, and 4D convs.
  3. **Eliminates EMA lag confound:** Uses `--gpu-ema-decay 0.99 --gpu-ema-every 1` over `1,200` steps (`0.99^1200 = 5.8e-6` init weight), enabling a direct apples-to-apples gate against Exp 4's `AdamW` RAW baseline (`val_ce=1.3769`, `token_f1=0.3609`, `sys_count_exact=0.3750`).

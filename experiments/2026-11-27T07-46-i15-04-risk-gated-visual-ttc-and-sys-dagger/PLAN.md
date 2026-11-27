# PLAN — `2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger`

## 1. Platform Chosen & Rationale

- **Platform:** Google Colab `L4` (`provider: "colab_l4"`, `gpu: "L4"`, Wave 13, 24 GB VRAM, native `BF16`).
- **Estimated Cost & Wall-Clock:** `~60 min` total (`~18 min` for Part A 125-page NorthStar zero-train eval with `<= 15%` CDR escalation + `~42 min` for Part B `2 × 1,200`-step controlled proxy training & `suite31_hw` RAW+EMA evaluation) / **`~4.5 CU`**.
- **Rationale:** Combines Pivot 3 (zero-train visual-coverage TTC on the full 125-page OMR-NorthStar benchmark using the prebuilt static Linux Rust `assembly-serialize-linux-x86_64` exporter) and Pivot 2 (`<sys>`-aligned On-Policy DAgger + Counterfactual Anti-Prior vs SFT control on `suite31_hw`) into a single BF16-safe Colab L4 run.

---

## 2. Identifiers for Follow-Up Queries

- **Experiment ID:** `2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger`
- **Short ID / Wave:** `i15-04` / Wave `13`
- **Queue Fragment:** `experiments/2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger/queue_entry.json`
- **Baseline Commit SHA:** `e426c28`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (private dataset)
  - Eval JSON: `evals/2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger.json`
  - Run Ledger: `runs/2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger.json`
- **Live Session / Log (populated on launch):**
  - Colab session: `<populated on launch>`
  - Local log: `D:\tmp\overnight-queue-logs\2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger.log`

---

## 3. Explicit Fix Matrix for All 5 `i14-14` Defects & the `bsvr = 0.997` Blind Spot

| Prior Failure (`i14-14` / `bsvr` Blind Spot) | Root Cause in `i14-14` | Exact Fix in `i15-04` (`Part A` & `Part B`) |
|---|---|---|
| **Defect 1: Miscalibrated Cascade Gates** (`96.15%` Tier-1, `94.23%` Tier-2, `7.59×` latency) | Triggered on `beat_ok_rate < 0.92` or `mean_entropy > 0.35`, firing on nearly every clean printed page. | Escalate **ONLY** on `has_loop` OR `truncated` OR `abs(n_sys - n_sys_img) >= 1` OR `attn_ink_cov < 0.80` OR `bsvr < 0.80`. Target `CDR` escalation `<= 15%`, mean latency `<= 1.45×`. |
| **Defect 2: `y_0` Discarded on Escalation** | Tier-1 MBR excluded `arm0` (`1408px`), allowing weaker views to overwrite good printed predictions. | **ALWAYS keep `y_0` (`1408px`) in the candidate pool** as index `0`. Replace `y_0` **ONLY** on a hard CFR flag (`has_loop`, `truncated`, `abs(n_sys - n_sys_img) >= 1`, `bsvr < 0.80`) OR a strict **`>= +0.08`** composite score win margin. |
| **Defect 3: Ungated Full-Encoder Mean-Pool Scribe-TTT Drift** (`-2.65 pp` printed, `-1.90 pp` HW) | 2-step encoder updates at `lr=3e-5` without rollback drifted clean printed features. | **Zero encoder TTT** (frozen encoder & decoder weights throughout Part A). |
| **Defect 4: Lower-Res MBR Bias & OOD Full-Page Tier-2 (`{1664, 1792}px`)** | Majority LCS voting over `{1152, 1280, 1536}` pulled consensus toward low-res; full-page OOD scales hurt printed F1. | No destructive LCS voting and no OOD full-page `{1664, 1792}px` decodes; candidate pool is `{y_0 @ 1408px, 1280px greedy, 1536px greedy, 1408px beam=2}` reranked directly. |
| **Defect 5: Misaligned Single-Staff Proxy Chamfer Rasterizer** (`arm3 < arm2` by `-0.55 pp`) | Rendered all multi-system notes onto a single 5-line cartoon staff and compared against full multi-system pages. | Replaced with native **decoder cross-attention spatial ink coverage (`r_attn_cov`)** + **visual system-count alignment (`r_sys_vis`)** + **length-normalized sequence logprob (`norm_lp`)**. |
| **`bsvr = 0.997` Blind Spot** | Beat-sum validity (`bsvr`) stays `~1.0` even when an entire system or measure is silently dropped. | `r_attn_cov` measures the fraction of ink-bearing visual memory tokens receiving cross-attention mass, and `r_sys_vis` checks `abs(n_sys - n_sys_img)` directly against image-space system bands. |

---

## 4. Required Data, Checkpoints & Tools

- **Baseline Checkpoint:** `checkpoints/iter13_v2_mid/uni_omr_v2_ema_latest.pt` (from `gregoreesmaa/sheet-music-converter-omr`, overridable via `UNIOMR_HF_CKPT`).
- **Part A Eval Pack:** `datasets/northstar_eval_pack_125p.tar.gz` (`OMR-NorthStar dev-v1`, 125 pages + canonical `omr_northstar` scorer).
- **Part A MusicXML Exporter:** Prebuilt static Linux Rust binary `tools/assembly-serialize-linux-x86_64` from HF (`UNIOMR_HF_ASER_BIN`), matching `eval-northstar-iter13-mid` for 100% apples-to-apples multi-staff MusicXML export parity.
- **Part B Train/Val Proxy Pool:** `datasets/iter10_5_proxy_2k.tar` (permissive license, `0` eval overlap, `2×` upsampling on dense multi-staff & HW pages).
- **Part B OOD Eval Pack:** `datasets/suite31_hw_eval.tar` (`suite31_hw_manifest.json`, 45 pieces × `clean`/`photo` = 90 pages, 31 printed + 14 real historical handwritten manuscripts).
- **License Confirmation:** 100% permissive (MIT/BSD/Apache/CC0/Public Domain); ZERO `CC-BY-NC` data.

---

## 5. Exact Experimental Design & Launch Command

### Part A — `part_a_risk_gated_visual_ttc` (Zero-Train, OMR-NorthStar 125p)

1. **`arm0_greedy`**: Standard `1408px`, `max_len=3584`, BF16, FSA grammar + MTP-4 speculative decoding + `neg_control_gate`.
2. **`arm1_risk_gated_ttc`**:
   - Computes per-page risk diagnostics on `y_0`:
     - `has_loop`: `rlvr_grpo.has_repetition_loop(raw_toks)` OR `dual_verifier.detect_sys_loop(raw_toks)`
     - `truncated`: `eos_id not in seq_ids[1:]`
     - `n_sys_img`: horizontal ink-projection band detector on the page image (system count estimate)
     - `n_sys`: `1 + raw_toks.count("<sys>")`
     - `attn_ink_cov`: fraction of ink-bearing encoder visual memory tokens covered by last-layer decoder cross-attention mass
     - `bsvr`: measure beat-sum validity rate from `rlvr_grpo._walk_parse_and_meter`
   - **Risk Gate:**
     - `hard_cfr = has_loop or truncated or (abs(n_sys - n_sys_img) >= 1) or (bsvr < 0.80)`
     - `escalate = hard_cfr or (attn_ink_cov < 0.80)`
   - **Non-escalated pages (`not escalate`):** zero extra decode calls; reuse `y_0` verbatim.
   - **Escalated pages (`escalate`):** build candidate pool `[y_0 @ 1408px, 1280px greedy, 1536px greedy, 1408px beam=2]` (zero encoder TTT). Score each candidate with:
     `score(y) = r_attn_cov(y) + r_sys_vis(y) + norm_lp_scaled(y)`
     where `r_sys_vis = max(0.0, 1.0 - 0.35 * abs(n_sys - n_sys_img))` and `norm_lp_scaled = max(0.0, 1.0 + max(-2.0, norm_lp) / 2.0)`.
     Replace `y_0` **ONLY** if `best_idx != 0` AND (`hard_cfr` OR `(best_score - y0_score) >= 0.08`).
   - Score both arms via `eval_northstar_runner.evaluate_manifest` with `n_boot=1000`, `bootstrap_seed=42`.

### Part B — `part_b_sys_dagger_probe` (`1,200` Steps/Arm, `suite31_hw` 90p)

- Both arms initialize from `HF_CKPT_PATH` (`seed=42`, `TRAIN_STEPS=1200`, `WARMUP_STEPS=100`, `GRAD_ACCUM=4`, `LR_MAX=1e-4`, WSD cosine decay `25%`, `WEIGHT_DECAY=0.04`, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`, `EMA_DECAY=0.99` every step, BF16 autocast).
- **`arm_b0_sft_control`**: `100%` standard SFT cross-entropy + MTP (`0.25`) on `iter10_5_proxy_2k`.
- **`arm_b1_sys_dagger_antiprior`**:
  - `75%` standard SFT micro-batches.
  - `25%` DAgger + Counterfactual Anti-Prior micro-batches (50/50 split):
    - **`<sys>`-Aligned On-Policy DAgger:** On multi-system pages (`>= 2` systems), select a random system boundary `k >= 1`, roll out a short on-policy model prefix (greedy/sampled up to the `k`-th `<sys>` boundary or budget cap), splice the ground-truth `<sys>` recovery suffix from system `k` onward, mask all model-generated prefix targets with **`ignore_index=-100`**, and apply **`3.0×` token loss weight** on the `<sys>` recovery header (`<sys>` + immediately following `clef-*` / `key*` / `time-*` tokens).
    - **Counterfactual Anti-Prior:** Apply vertical staff-shift / contrast-polarity visual perturbation paired with counterfactual diatonic step/header token edits (`2.0×` weight on state-header tokens) to penalize language-model hallucination over visual evidence.
- Evaluate **BOTH RAW and EMA** checkpoints of both arms on `suite31_hw` (`printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`, `all90`) to guard against short-proxy EMA lag.

### Launch Command

```bash
python3 experiments/2026-11-27T07-46-i15-04-risk-gated-visual-ttc-and-sys-dagger/run_colab.py
```

---

## 6. 5-Minute Checkpointing & Resume Plan

- **Part A:** Atomic per-page `.musicxml` / `.bekern` / `timing.json` / `ttc_meta.json` writes under `/content/vol/runs/<exp_id>/part_a/` so any interrupted run resumes cleanly at the next unfinished page.
- **Part B:** Atomic `latest.pt.tmp` → `latest.pt` (storing `step`, `model_state_dict`, `ema_state_dict`, `optimizer_state_dict`) every **`5.0` minutes** (`SAVE_EVERY_MINUTES = 5.0`), plus per-arm `arm_result.json` caching so completed arms are skipped on resume.
- **HF Sync:** Uploads `evals/<exp_id>.json` and `runs/<exp_id>.json` immediately upon completion (and after Part A finishes as a partial checkpoint).

---

## 7. Evaluation Gates & Decision Tree

### Part A (`part_a_risk_gated_visual_ttc`) Gates

- **PROMOTE:** `CDR` escalation rate `<= 15%` AND mean latency `<= 1.45×` AND printed regression `== 0.0 pp` (`>= 0.0`) AND overall NorthStar **`NS >= +3.0`** → wire Risk-Gated Visual-Coverage TTC directly into the production UniOMR inference stack.
- **ITERATE:** `NS` improves (`+1.0` to `+2.9`) or `CDR` escalation is `15%–25%` with `0.0 pp` printed regression → tighten `attn_ink_cov` threshold (`0.75` vs `0.80`) or add per-system vertical crop zoom only on `|n_sys - n_sys_img| >= 1` pages.
- **DISCARD:** Any printed regression `< 0.0 pp` OR `NS < +0.5` OR latency `> 1.8×`.

### Part B (`part_b_sys_dagger_probe`) Gates

- **PROMOTE:** **`HWM Note F1 >= +3.0 pp`** over `arm_b0_sft_control` AND **Printed Note F1 `>= -0.3 pp`** (verified on both RAW and EMA) → promote `<sys>`-aligned On-Policy DAgger + Counterfactual Anti-Prior into the Iter 15 Stage-2 training recipe.
- **ITERATE:** `HWM Note F1` lifts `+1.0` to `+2.9 pp` with Printed `>= -0.3 pp` → sweep DAgger/Anti-Prior ratio (`15%` vs `35%`) and `<sys>` header weight (`2×` vs `5×`) on full-length `2048` sequences.
- **DISCARD:** `HWM Note F1 < +0.5 pp` OR Printed Note F1 drops `< -0.5 pp` OR training instability / NaNs.

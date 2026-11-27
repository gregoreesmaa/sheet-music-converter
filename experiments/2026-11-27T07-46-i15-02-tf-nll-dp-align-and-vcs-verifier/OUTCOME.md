# OUTCOME — `2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier` (`i15-02`)

- **Status:** `COMPLETED` (`rc=0`, **Verdict: `DISCARD`**)
- **Started / Completed (UTC):** `2026-11-27T14:44:17Z` → `2026-11-27T15:13:08Z`
- **Elapsed Time:** `1731.25 s` (`28.9 min`) script runtime (`3377.8 s` total Colab launcher wall-clock including setup/download)
- **Platform / GPU / Precision:** Google Colab `NVIDIA A100-SXM4-40GB` (`torch.bfloat16` native)
- **Frozen Checkpoint:** `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` (`uni_omr_v2`, step `20000`, `missing_keys = 0`)
- **HF Artifacts:**
  - Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier.json` (pulled to [eval_hf.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier/eval_hf.json))
  - Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier.json` (pulled to [run_hf.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier/run_hf.json))

---

## 1. Measured Results

### Sanity Gate (2 Printed Clean Pages)
- `n = 2`, `dec_ok = 1.0000`, `mean_note_f1 = 1.0000` — **PASSED** (confirms canonical checkpoint load, `[0, 1]` CHW preprocessing, and BF16 decode integrity).

### Part A — Visual Consistency Score (`VCS`) Verifier (`n = 28` HW pages: 14 `clean` + 14 `photo`)

| Signal / Metric | Measured | Gate / Baseline | Status |
|---|---|---|---|
| Mean HW Strict Note F1 (`all` / `clean` / `photo`) | `0.1473` (`0.1678` / `0.1269`) | — | Reference (`iter12_v2_ema_best`) |
| **`Pearson r(VCS, Strict Note F1)`** | **`0.1616`** | `>= 0.70` | **FAIL (`-0.5384`)** |
| `Spearman rho(VCS, Strict Note F1)` | `0.1017` | — | Weak / near-noise |
| Baseline `Pearson r(ov-39 mean_logprob, Note F1)` | `-0.1283` | — | Confirmed broken (`r < 0`) |
| Component: `vcs_ink_iou` (mean `0.1514`, std `0.0782`) | `r = 0.3187` | — | Only weak positive signal |
| Component: `vcs_cross_scale_agree` (mean `0.5522`, std `0.3848`) | `r = 0.1239` | — | Broken by deterministic loops (`9/28` hit `1.000`) |
| Component: `vcs_barline_x_corr` (mean `0.5690`, std `0.1113`) | `r = 0.0079` | — | Pure noise |
| Component: `vcs_encoder_patch_cos` (mean `0.8142`, std `0.0687`) | `r = -0.0631` | — | Saturated high, negative correlation |
| `ov-39` Hallucination FP Rejection (combined / synthetic / **natural**) | `0.8750` / `1.0000` / **`0.5833`** | — | Inflated by synthetic negatives (see Scrutiny) |
| `ov-39` Hallucination AUROC (`n_valid_high_f1 = 2`) | `0.8750` | — | Underpowered (`n_pos = 2`) |

### Part B — AR-Free Teacher-Forced NLL + Cross-Attention Monotonicity DP Alignment (`n = 28` HW pages, 11 candidates/page)

| Method | Top-1 Span Acc (`all` / `clean` / `photo`) | Exact Start | Exact End | Boundary MAE (measures) | Mean Latency (`s/page`) | Gate (`>= 0.85`) |
|---|---|---|---|---|---|---|
| `free_decoded_align` (Baseline) | **`0.2143`** (`0.2857` / `0.1429`) | `0.3214` | `0.2500` | `2.2500` | `59.799` | Reference |
| `tf_nll_only` | `0.0714` (`0.0714` / `0.0714`) | `0.2500` | `0.2500` | `2.0000` | `0.317` | **FAIL** |
| **`tf_nll_plus_ca_mono_dp`** | **`0.0714`** (`0.0714` / `0.0714`) | `0.2143` | `0.3571` | `1.8214` | `0.317` | **FAIL (`-0.7786`)** |
| Delta (`dp - free_decoded`) | **`-0.1429`** | `-0.1071` | `+0.1071` | `-0.4286` | `188.6× faster` | Worse than baseline & random (`0.0909`) |

---

## 2. Scrutiny & Anomaly Check

1. **Coverage & Harness Integrity:** Full `28/28` real HW pages (`14` clean + `14` photo) scored in both Part A and Part B, `missing_keys = 0`, 2-page printed sanity gate `1.0000` Note F1, zero NaNs. Both `clean` and `photo` failed symmetrically in Part B (`0.0714` each).
2. **Part A Root Cause — Why Heuristic `VCS` Failed (`r = 0.1616`):**
   - **Deterministic cross-scale hallucination agreement:** `9/28` HW pages achieved `vcs_cross_scale_agree = 1.0000` **despite `note_f1 ∈ [0.0000, 0.1268]`** (e.g., `hw-cornelius-botschaft-d-b` photo: `note_f1 = 0.0000`, `csa = 1.0000`, `vcs = 0.6280`; `hw-hensel-im-herbste-d-b` photo: `note_f1 = 0.0714`, `csa = 1.0000`, `vcs = 0.6801`; `hw-mozart-k464/k387/k421`). When the frozen model collapses into a short/degenerate or wrong-pitch decode, it produces the *exact same* wrong output at `1280px` and `1536px`, fooling cross-scale agreement completely.
   - **Frozen encoder feature saturation:** `vcs_encoder_patch_cos` averages `0.8142` (`min = 0.5923`, `max = 0.9280`) with `r = -0.0631`; global + 4×4 pooled encoder features only measure coarse "looks like dense staff notation" rather than note-level visual grounding.
   - **Inflated `ov-39` hallucination benchmark:** Only `2/28` pages qualified as `valid_high_f1` (`hw-beethoven-das-blumchen-wunderhold-d-b` clean `0.6667` / photo `0.5812`, neither of which passed `ov-39`), while `28` easy synthetic cross-piece negatives (`100%` rejected) drove the headline `0.875` FP rejection / AUROC. On **natural** `ov-39` false positives (`n = 12`), rejection was only **`58.33%` (7/12)**, and `5/12` natural hallucinations scored **higher VCS (`0.6077–0.6801`) than the best real HW decode (`0.6057`)**.
   - **Useful negative finding:** Confirmed `ov-39` decoder `mean_logprob` has **negative** correlation (`r = -0.1283`) with real HW Note F1 (`12/28` low-F1 HW pages pass `ov-39`), proving RISM pseudo-label filtering cannot rely on `mean_logprob` + `beat_valid_frac` alone.
3. **Part B Root Cause — Why AR-Free `tf_nll` + `ca_mono_dp` Collapsed (`Top-1 = 0.0714 < 0.0909` random):**
   - **Unconditional LM-prior & length-normalization bias:** On hard OOD HW pages where visual conditioning is weak (`mean_hw_note_f1 = 0.1473`), per-token `-mean_ce(tokens | image)` is dominated by the decoder's unconditional language-model prior rather than visual evidence. Candidate windows that drop harder opening/closing measures (`[5, 43]` vs `[4, 43]`, `[5, 36]` vs `[4, 36]`, `[4, 35]` vs `[4, 36]`) or jump to simpler off-page donor measures (`[10, 20]` vs `[4, 14]` on `hw-beethoven-op135`) achieve lower mean CE than the true span.
   - **Prefix truncation at `TF_MAX_LEN = 2048`:** Long multi-system pages (30–49 measures, e.g., `hw-mozart-k464` `[4, 43]`, `hw-debussy-la-plus-que-lente` `[4, 53]`) exceed `2048` tokens, truncating the trailing measures so `end ± 1` candidates share identical prefixes.
   - **Cross-attention hooks fired (`exact_end: 0.2500 -> 0.3571`, `MAE: 2.0000 -> 1.8214`),** confirming the hook wiring worked, but `0.35 * ca_mono + 0.25 * ca_cov` could not overcome the LM-prior bias in `tf_mean_logprob`, leaving `Top-1` stuck at `0.0714` (`2/28`).

---

## 3. Verdict & Decision

- **Verdict:** **`DISCARD`** (directly matches [PLAN.md](./PLAN.md) Section 7 `DISCARD` branch: `Part A r = 0.1616 < 0.50` AND `Part B Top-1 = 0.0714 < 0.65`, worse than `free_decoded_align = 0.2143`).
- **Do NOT wire** heuristic `VCS` into RISM pseudo-label filtering, and **do NOT launch** batch `tf_nll_plus_ca_mono_dp` alignment on the `1,466` idle gold-XML archival pages.
- **Confirmed takeaways for future iterations (recorded so no compute is wasted repeating this):**
  1. Unsupervised heuristic visual-silhouette `VCS` and zero-shot `mean_ce` teacher-forced NLL do **not** work on frozen OOD HW checkpoints (`mean_hw_note_f1 = 0.1473`).
  2. If revisiting visual verification or archival gold-XML alignment later, pivot to either (a) a **learned contrastive/reward verifier** trained on `(image, correct_bekern)` vs hard-negative `(image, hallucinated/shifted_bekern)` pairs, or (b) **PMI-normalized TF-NLL** (`log P(y|x) - log P(y|blank_image)`) without `2048`-token truncation, **only after** the base HW model's visual grounding on real manuscripts is materially stronger.

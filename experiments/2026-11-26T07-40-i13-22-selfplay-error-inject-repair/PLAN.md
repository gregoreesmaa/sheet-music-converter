# Plan — `2026-11-26T07-40-i13-22-selfplay-error-inject-repair`

## 1. Scientific Design & Architecture (Zero Core-Model Edits)

- **Wrapper Module (`DraftRepairWrapper` inside `run_colab.py`):**
  - Wraps `UniOMRNet` (`uni_omr_soup.pt`, `d_model=384`, `vocab=237`).
  - Encodes a draft token sequence `y_draft` (`<= 512` tokens) using the decoder's own `tok_emb` + learned 1D positional embedding + a 2-layer causal/bidirectional Transformer block + **zero-initialized residual gate `alpha_draft`** (`~0.6M` params) to produce `draft_mem [B, L_d, d_model]`.
  - Concatenates `mem_aug = cat([visual_mem, alpha_draft * draft_mem], dim=1)` before `model.decode_tokens(...)`. At step 0 (`alpha_draft == 0`), Pass-2 is mathematically identical to Pass-1.
- **Synthetic OMR Error-Injection Operators (`corrupt_bekern_ids`):**
  - Operates directly on `bekern` token IDs (excluding `<pad>`, `<bos>`, `<eos>`, `<sys>`):
    1. **Duration drift (`30%`):** swap a duration token (`dur-4 <-> dur-8`, `dur-8 <-> dur-16`, `dur-2 <-> dur-4`) or drop/insert `dot`.
    2. **Octave / pitch slip (`25%`):** shift `oct-k -> oct-(k±1)` or swap adjacent diatonic `step-*`.
    3. **Barline cascade (`20%`):** delete a `<bar>` or duplicate an adjacent `<bar>`.
    4. **Accidental / voice error (`25%`):** drop `a1`/`a-1` or perturb `voice-k`.
  - Applied at `corrupt_rate = 0.12` per eligible token so `y_draft` looks like a realistic near-miss OMR transcription.
- **On-Policy Draft Mining (`arm2`):**
  - Every 3rd step, runs a fast `torch.no_grad()` greedy/noisy (`T=0.7`) short prefix rollout (`max_len=256`) on the current batch image and pairs that real model error draft with the ground-truth target `y*`.

## 2. Training & Evaluation Protocol

- **Dataset:** HF `datasets/iter10_5_proxy_2k.tar` (`proxy_train.jsonl`).
- **Budget:** `300` optimizer steps per arm (`grad_accum=4`, effective batch `4`, `lr=2e-5` backbone / `2e-4` draft projector, cosine decay, `BF16` on L4 / `FP16+GradScaler` on T4, `seed=42`).
- **Evaluation (`suite31_hw`, 14 HW + 12 Stratified Printed = 52 pages, `max_dim=1280`, `max_len=3840`):**
  - For each arm, evaluate:
    - `pass1_greedy`: standard `generate_mtp_spec` without draft memory.
    - `pass2_raw`: Pass-2 decoding conditioned on `[visual_mem; draft_mem(pass1)]`.
    - `pass2_rsym_gated`: accept `pass2_raw` iff `R_sym(pass2) >= R_sym(pass1)` and `r_noloop == 1.0`, else keep `pass1`.
  - Log per-piece F1, edit rate (`pass2 != pass1`), edit precision (`F1(pass2) > F1(pass1)` among edited pages), and `|alpha_draft|`.

## 3. Artifacts & Security

- Token read from `/tmp/.hf_token` and unlinked immediately.
- Uploads `evals/2026-11-26T07-40-i13-22-selfplay-error-inject-repair.json` and `runs/2026-11-26T07-40-i13-22-selfplay-error-inject-repair.json` to `gregoreesmaa/sheet-music-converter-omr`.

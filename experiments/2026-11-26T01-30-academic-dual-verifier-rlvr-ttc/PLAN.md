# Plan — Academic Experiment A (Phase 1: Dual-Verifier Gate & TTC Scaling Curve)

## 1. Platform & Rationale

- **Platform:** Google Colab **`NVIDIA L4`** (24 GB VRAM, BF16 native, `cc=8.9`, `torch 2.14.1+cu130` pre-installed) via `.\.venv\Scripts\colab.exe`.
- **Rationale:** Phase 1 is an inference + verification scaling study (48 pages × 17 rollouts = 816 sequences). L4 BF16 executes batched `G=8` KV-cached FSA rollouts in ~8–10s/page (~10–15 min total, ~1.0–1.5 CU from our ~1,479 CU pool). Zero local GPU used.

## 2. Identifiers for Follow-Up Queries

- **Colab Session Name:** `acad-a-dual-rlvr`
- **Baseline Commit SHA:** `164852f`
- **Baseline Checkpoint:** `gregoreesmaa/sheet-music-converter-omr:checkpoints/iter10_soup_baseline/uni_omr_soup.pt`
- **HF Eval Artifact:** `gregoreesmaa/sheet-music-converter-omr:evals/2026-11-26T01-30-academic-dual-verifier-rlvr-ttc.json`
- **HF Run Ledger:** `gregoreesmaa/sheet-music-converter-omr:runs/2026-11-26T01-30-academic-dual-verifier-rlvr-ttc.json`

## 3. Data & Licensing

- **Eval Dataset:** `datasets/suite31_hw_eval.tar` (`suite31_hw_manifest.json`, 45 pieces total; permissive public-domain / CC0 / CC-BY). Zero `CC-BY-NC`.
- **Slice Evaluated:** All **14 real historical HW pieces** + **10 stratified diverse printed pieces** across both `clean` and `photo` domains (**48 pages total**).

## 4. Exact Execution Protocol

1. **CPU Unit Tests (local, `$env:CUDA_VISIBLE_DEVICES=""`):**
   ```powershell
   $env:CUDA_VISIBLE_DEVICES=""
   .\.venv\Scripts\python.exe -m unittest tmp-train/mvd/test_dual_verifier.py -v
   ```
2. **Colab `L4` Execution:**
   ```powershell
   .\.venv\Scripts\colab.exe new -s acad-a-dual-rlvr --gpu L4
   .\.venv\Scripts\colab.exe upload -s acad-a-dual-rlvr <local_bundle.tar.gz> /content/bundle.tar.gz
   .\.venv\Scripts\colab.exe upload -s acad-a-dual-rlvr $env:USERPROFILE\.cache\huggingface\token /tmp/.hf_token
   .\.venv\Scripts\colab.exe exec -s acad-a-dual-rlvr -f experiments/2026-11-26T01-30-academic-dual-verifier-rlvr-ttc/run_colab.py
   # ALWAYS in finally:
   .\.venv\Scripts\colab.exe stop -s acad-a-dual-rlvr
   ```
3. **Sampling & Scoring Hyperparameters:**
   - `max_dim = 1536`, `bf16 = True`, `cudnn_sdp = False`, `mem_efficient_sdp = True`.
   - `K=1` Greedy: `eval_s31.generate_mtp_spec(..., max_len=2048, use_grammar=True, strict_grammar=False)`.
   - `K=16` Rollouts: 2 chunks × `G=8` via `rlvr_grpo.sample_rollouts(..., max_len=1536, temperature=0.7, top_p=0.92, use_grammar=True, strict_grammar=False)`.
   - Every rollout scored with `DualVerifier` ($R_{\text{sym}}$, $R_{\text{vis}}$, $R_{\text{dual}}$) AND exact Rust `assembly-serialize` + `eval-harness` **Strict Note F1**.

## 5. Checkpointing & Resilience

- Incremental per-page JSONL checkpoint saved to `/content/exp_a_partial.jsonl` after every page, and uploaded to HF on completion (plus every 16 pages) so partial progress is never lost.
- `/tmp/.hf_token` is `chmod 600` on startup and unconditionally deleted in `finally`.

## 6. Evaluation Gate & Decision Tree (Phase 1 → Phase 2)

- **Metrics:**
  - Pooled & within-page mean **Spearman $\rho$** of $R_{\text{sym}}$, $R_{\text{vis}}$, and $R_{\text{dual}}$ vs Strict Note F1 (HW vs Printed, `clean` vs `photo`).
  - **TTC Scaling Curve:** Greedy (`K=1`), Sampled (`K=1`), Best-of-4 (`K=4`), Best-of-16 (`K=16`), Oracle-of-16, and TTC slope ($\Delta\text{F1} / \log_2 K$).
- **GO (Proceed to Phase 2: 5-arm GRPO on 5,707 unlabeled RISM pages):**
  - Within-page OR pooled Spearman $\rho(R_{\text{dual}}, \text{F1}) > 0.25$ on HW and outperforms $R_{\text{sym}}$-only, **AND**
  - Best-of-16 with $R_{\text{dual}}$ improves HW Strict Note F1 over `K=1` (positive TTC slope) with Oracle-of-16 headroom $\ge +0.03$ F1.
- **ITERATE (Refine Visual Verifier before Phase 2):**
  - Oracle-of-16 headroom $\ge +0.03$ F1 exists, but $R_{\text{vis}}$ has weak/negative $\rho$ on HW → diagnose failure cases and refine $R_{\text{vis}}$ before spending GPU on 5-arm GRPO.
- **NO-GO / DISCARD:**
  - Oracle-of-16 headroom $< +0.015$ F1 (base policy has insufficient exploration diversity at `temp=0.7`) → fix base policy diversity/SFT before RLVR.

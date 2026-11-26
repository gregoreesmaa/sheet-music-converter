---
name: running-an-experiment
description: Mandatory lifecycle playbook for creating, tracking, checkpointing, evaluating, and documenting every training, pretraining, data, or ablation experiment in `experiments/YYYY-MM-DDTHH-MM-<slug>/`. Activate BEFORE launching any new local or cloud experiment (Modal, Kaggle, or Local RTX 4080 SUPER) and again when writing `OUTCOME.md` after results arrive.
---

# Running an Experiment (`experiments/`)

Every experiment in `sheet-music-converter` — whether local (`RTX 4080 SUPER`) or cloud (`Modal`, `Kaggle`) — MUST live in its own timestamped folder under `experiments/` and follow this exact lifecycle.

Cross-references:
- Cloud round-robin routing & platform rules: `.agents/skills/test-in-cloud/SKILL.md`
- Provider playbooks: `cloud-modal`, `cloud-kaggle`, `cloud-huggingface`
- Mandatory post-eval forensic loop: `.agents/skills/learn-from-suite31/SKILL.md`

---

## 1. Folder Naming Convention (Minute Precision)

Create one folder per isolated experiment under `experiments/`:

```text
experiments/YYYY-MM-DDTHH-MM-<kebab-slug>/
├── README.md      # Concise explanation, hypothesis, baseline SHA, live status
├── PLAN.md        # Platform, query identifiers, exact data/command, 5-min ckpt, decision tree
└── OUTCOME.md     # Written after results & suite31_hw eval arrive (never skipped)
```

- Timestamp uses local ISO-like minute precision with hyphens (Windows-safe, no colons):
  `YYYY-MM-DDTHH-MM-<short-slug>` — e.g., `2026-11-26T00-39-cut-rism-real-ink`.
- **1 variable per experiment:** Isolate a single architectural, optimizer, loss, or data hypothesis against a named baseline commit SHA (`git rev-parse --short HEAD`).
- **Never commit weights, logs >1 MB, or secrets** into `experiments/` — store weights/shards on Hugging Face (`gregoreesmaa/sheet-music-converter-omr`) or `D:\tmp\`, and link them from `PLAN.md` and `OUTCOME.md`.
- **Zero-Prior-Context Handoff Rule (MANDATORY):** Every experiment folder MUST be 100% self-contained so a fresh agent or human with **zero prior conversation context** can walk into `experiments/YYYY-MM-DDTHH-MM-<slug>/`, read `README.md` + `PLAN.md`, and immediately monitor, cancel, resume, evaluate, or take over the run without notice. Always place the experiment's exact launcher script (e.g., `run_modal.py` or `kaggle_kernel/`), exact status/log/resume/eval copy-paste commands, exact live cloud IDs (`app_id`, URL, kernel slug, HF repo paths), and exact baseline numbers directly inside the folder!

---

## 2. Required Files & Exact Templates

### A. `README.md` (Written BEFORE Launch)
Must clearly state:
1. **Experiment ID & Title** (`YYYY-MM-DDTHH-MM-<slug>`)
2. **Status:** `PLANNED` → `RUNNING` → `EVALUATING` → `COMPLETED (PROMOTED | ITERATE | DISCARDED)`
3. **One-Paragraph Summary & Hypothesis:** What single variable is being tested, why we expect it to help, and what baseline it compares against (exact commit SHA + checkpoint path).
4. **Quick Links:** `PLAN.md`, `OUTCOME.md` (once completed), HF repo paths (`checkpoints/<exp_id>/`, `evals/<exp_id>_suite31_hw.json`).

### B. `PLAN.md` (Written BEFORE Launch, Updated with Live Query IDs on Launch)
Must detail ALL of the following:
1. **Platform Chosen & Rationale:** `Modal (L4/A10G/A100)` | `Kaggle (2× T4, FP16-only!)` | `Local RTX 4080 SUPER (BF16)`, plus estimated wall-clock & quota/credit cost.
2. **Identifiers for Follow-Up Queries:**
   - **Modal:** profile (`gregoreesmaa1`), app name, `app_id` (`ap-...`), dashboard URL, Volume name.
   - **Kaggle:** kernel slug (`gregoreesmaa/<slug>`), kernel URL, `kaggle kernels status <slug>` command.
   - **Local:** background `taskId`, log path, `--out-dir` on `D:\tmp\`.
   - **Hugging Face:** `gregoreesmaa/sheet-music-converter-omr` paths for `runs/<exp_id>.json`, `checkpoints/<exp_id>/`, and `evals/<exp_id>_suite31_hw.json`.
3. **Required Training & Eval Data:**
   - Exact manifest(s) / HF `.tar` shard filenames + SHA-256 / sample counts + license confirmation (permissive only; ZERO `CC-BY-NC`).
   - Exact baseline checkpoint (`--init-from` / `--init-encoder-from`).
4. **Exact Launch Command & Hyperparameters:** Full reproducible CLI invocation (zero secrets!).
5. **5-Minute Checkpointing & Resume Plan (MANDATORY):**
   - Confirm `--save-every-minutes 5.0` (atomic `.tmp` → `os.replace` for both raw & EMA checkpoints).
   - Confirm cloud sync cadence (Modal `vol.commit()` every 5 min; HF Hub batched upload every 15–30 min + on new best + on exit/SIGTERM).
   - Exact resume command if preempted/timed out.
6. **Evaluation Gate & Decision Tree ("What We Do With the Result"):**
   - Exact metric(s) to measure (e.g., 45-piece `suite31_hw` `printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`, `all45_clean/photo`, plus speed/size if applicable).
   - Concrete **Promote** condition (what gets merged/wired into the next Iter), **Iterate** condition (what follow-up experiment to spawn), and **Discard** condition (when to drop cleanly).

### C. `OUTCOME.md` (Written IMMEDIATELY After Results Arrive)
Must record:
1. **Final Status & Verdict:** `PROMOTE` | `ITERATE` | `DISCARD`.
2. **Measured Results Table:** Apples-to-apples comparison against the baseline (never projected numbers — only verified on-disk measurements).
3. **Forensic Observations:** What moved, what regressed, key surprises (linking any `learn-from-suite31` scout findings).
4. **Artifacts & Reproducibility:** Exact final commit SHA, winning checkpoint path (HF &/or `D:\tmp\`), and `suite31_hw` JSON path.
5. **Next Action Taken:** Exact follow-up experiment or code promotion triggered by this outcome.
6. **Platform Learnings Amended:** Note any new quirk/gotcha discovered during the run and confirm it was immediately amended into `cloud-modal/SKILL.md`, `cloud-kaggle/SKILL.md`, or `cloud-huggingface/SKILL.md`.

---

## 3. Mandatory 5-Step Lifecycle & Git Checkpointing

1. **Create & Plan:** Create `experiments/YYYY-MM-DDTHH-MM-<slug>/` with `README.md` and `PLAN.md`, and add a row to `experiments/README.md` (or keep it current).
2. **Commit & Push BEFORE Launch:**
   ```powershell
   git add experiments/YYYY-MM-DDTHH-MM-<slug>/
   git -c user.name="Gregor Eesmaa" -c user.email="6941817+gregoreesmaa@users.noreply.github.com" commit -m "exp(<slug>): add experiment README and PLAN"
   git push origin HEAD:main
   ```
   *(If the remote branch is `master` or `main`, push cleanly so every launched experiment maps to a pushed commit SHA!)*
3. **Launch & Record Query IDs:** Launch the run with `--save-every-minutes 5.0`. If cloud launch returns a new `app_id` / kernel URL, immediately record it in `PLAN.md` (and commit/push).
4. **Mandatory Early Health Check-In (3–5 min after launch):** Within **3–5 minutes** of launching ANY local or cloud experiment, inspect live logs & GPU state (`modal app logs <app_id>`, `kaggle kernels status <slug>`, or `manage_task status` + `nvidia-smi`) to verify:
   - Zero startup/auth/import/OOM crashes or silent hangs.
   - GPU is actively running at expected precision (`BF16` on `cc >= 8`, `FP16` on T4), loss is finite and moving cleanly, and the 5-minute checkpointing loop is healthy.
   - Catching a broken run at minute 3 saves hours of wasted GPU quota!
5. **On Completion — Activate `experiment-finished` Skill Immediately:** Whenever ANY local or cloud experiment finishes, follow `.agents/skills/experiment-finished/SKILL.md`:
   - Pull & verify artifacts (`evals/<exp_id>.json`, `runs/<exp_id>.json`, checkpoints) and verify remote VMs are stopped (zero idle burn).
   - **Scrutinize results** (`## Scrutiny & Anomaly Check`) — never trust numbers blindly (check truncated `max_len`, silent FP32/CPU fallback, NaNs, domain asymmetry, missing keys).
   - Write `experiments/YYYY-MM-DDTHH-MM-<slug>/OUTCOME.md` with clear verdict (`PROMOTE` / `ITERATE` / `DISCARD`) and amend relevant provider skills immediately with any new empirical learnings.
   - **Mandatory `ITERATE` Continuation Rule:** Whenever the verdict is **`ITERATE`**, **ALWAYS immediately formulate, implement, and queue a follow-up experiment** (`depends_on: ["<this_exp_id>"]`) directly targeting the diagnosed bottleneck/asymmetry — never leave an `ITERATE` verdict dangling without a queued successor!
6. **DAG Dependency-Chain Scheduling (Never Wave Barriers):** Think of queued experiments as a **DAG of dependency chains (`depends_on: [...]`)**, NOT rigid waves. As soon as all of an experiment's predecessors have finished (`COMPLETED` locally or on HF `evals/`), it is immediately eligible to launch on the next free GPU slot.
7. **Commit & Push:**
   ```powershell
   git add experiments/ .agents/skills/
   git -c user.name="Gregor Eesmaa" -c user.email="6941817+gregoreesmaa@users.noreply.github.com" commit -m "exp(<slug>): record OUTCOME, skill updates, and follow-up DAG experiment"
   git push origin master
   ```

---

## 4. Empirical Learnings & Gotchas

- **Short-proxy EMA lag trap (`2026-11-26`, Exp 4 `ablation-muon-vs-adamw`):** Using `--gpu-ema-decay 0.997 --gpu-ema-every 5` on short from-scratch proxy runs (`<= 1,500` steps) yields only `<= 300` EMA updates, leaving `0.997^240 = 48.6%` of the EMA checkpoint at random step-0 initialization (which collapsed `AdamW` EMA `token_f1` from `0.3609` RAW to `0.0865` EMA). For short proxy runs (`<= 2,000` steps), use `--gpu-ema-decay 0.99 --gpu-ema-every 1` (at `1,200` steps `0.99^1200 = 5.8e-6`) and ALWAYS compare both RAW and EMA checkpoints in `## Scrutiny & Anomaly Check`.

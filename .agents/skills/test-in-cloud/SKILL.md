---
name: test-in-cloud
description: >-
  Master round-robin cloud experimentation orchestrator and ground rules for
  sheet-music-converter across Modal, Kaggle, Local RTX 4080 SUPER, and Hugging
  Face Private Dataset Hub. Activate whenever planning, dispatching, monitoring,
  or resuming cloud OMR training, ablations, data prep, or OOD evaluation runs.
---

# Test in Cloud — Master Round-Robin Orchestration & Ground Rules

**Trigger:** Activate this skill whenever dispatching or resuming any OMR training run, proxy ablation, or data job on cloud GPUs (`Modal` / `Kaggle`) or coordinating parallel runs alongside the local `RTX 4080 SUPER`. Always pair with the provider-specific skill (`cloud-modal`, `cloud-kaggle`, `cloud-huggingface`) and follow up every finished run with `learn-from-suite31`.

---

## Ground Rules for Using Our Models in Cloud (Non-Negotiable)

1. **Isolation (1 Variable per Trial):**
   - Every cloud trial tests **ONE** hypothesis against an exact baseline commit SHA (`git rev-parse HEAD`).
   - Commit and push all code changes **before** launching any cloud run (using `Gregor Eesmaa <6941817+gregoreesmaa@users.noreply.github.com>`).
   - Never bundle 3 unrelated changes into one cloud ablation — if it regresses or wins, you must know why.

2. **Budget & Duration Discipline:**
   - **Proxy ablations:** `2k–4k` steps (~15–30 min on Modal L4/A10G). Use these to kill bad ideas cheaply before committing overnight compute.
   - **Full runs:** `8k–36k` steps (Local 4080 SUPER overnight, Kaggle 2×T4 background, or Modal A100/H100 sprint).
   - Always pass `--save-every-minutes 5.0` locally and sync rolling checkpoints to Hugging Face every 15–30 min (and on clean exit / SIGTERM) so spot preemptions or timeouts lose `<= 5–15 min` of GPU work.

3. **Mandatory `suite31_hw` Eval Gate:**
   - Every cloud trial **must** be evaluated on **`suite31_hw`** (`45` pieces, `90` pages via `suite31_hw_manifest.json` — never the legacy 31-piece printed-only manifest).
   - Upload the resulting JSON report to HF `evals/<run_id>_suite31_hw.json` before promoting any change or launching a follow-up trial.
   - Immediately run `learn-from-suite31` on the uploaded report.

4. **Security & License Hygiene:**
   - **NEVER** hardcode or echo API tokens in scripts, configs, git commits, HF datasets, or stdout logs.
   - Always read credentials from local/secret stores: `~/.cache/huggingface/token` (or `HF_TOKEN` secret), `~/.modal.toml`, `~/.secrets/kaggle-token.txt` / `~/.kaggle/access_token`.
   - **Strictly permissive data only** (CC0 / Public Domain / CC-BY / MIT / BSD / Apache). NEVER upload or train on `CC-BY-NC` (e.g., quarantined `cvc-muscima`) or unaligned incipit sets (`rism-pdm-hw-intl`).

5. **Living Skill Amendment Rule (MANDATORY):**
   - Whenever **ANY** agent discovers a new platform quirk, CLI flag change, quota ceiling, GPU dtype gotcha, rate limit, or failure mode on Modal, Kaggle, or Hugging Face, **IMMEDIATELY amend that provider's `SKILL.md`** under `## Empirical Learnings & Gotchas` before moving on.
   - Do not leave hard-won cloud debugging lessons buried in chat logs.

---

## Round-Robin Compute Pool & Routing Table

All compute providers pull identical sharded datasets/manifests from and push checkpoints/evals back to **Hugging Face Private Dataset Hub** (`<user>/sheet-music-converter-omr`).

| Provider | Free / Cost | Hardware & Precision | Best Workload | Avoid For |
|---|---|---|---|---|
| **Modal** (`cloud-modal`) | `$30/mo` free credit | `L4` (24 GB), `A10G` (24 GB), `A100` (40/80 GB), `H100` (80 GB) — **BF16 native** | Fast `15–45 min` parallel bursts (3–5 ablations at once), `UniOMR v2` BF16/Muon trials, heavy `1536px` full-page sweeps | Idle interactive debugging; burning `$30/mo` on slow 12h runs that fit on Kaggle/Local |
| **Kaggle** (`cloud-kaggle`) | `30 hrs/wk` free | `2× T4` (32 GB total, Turing SM 7.5) — **FP16 ONLY (NO native BF16!)** | Long `2–8 h` background ablations, CPU+GPU data prep / style-transfer, FP16-compatible `v1`/`v2` runs while Modal/Local are busy | Anything requiring `--bf16` or >20 GB local scratch disk |
| **Local RTX 4080 SUPER** | `~$0.05/hr` electricity | `1× 4080 SUPER` (16 GB, Ada SM 8.9) — **BF16 native** | Primary overnight production runs (`Iter 10.5`, `Iter 11`, `Iter 12`) and fast interactive `suite31_hw` OOD evals | Parallel 4-way hyperparameter sweeps |
| **Hugging Face Hub** (`cloud-huggingface`) | `$0` private dataset storage & bandwidth | Private Dataset Repo (`datasets/`, `manifests/`, `checkpoints/`, `evals/`, `runs/`) | Single source of truth for sharded data, rolling checkpoints, run ledger, and `suite31_hw` eval JSONs | Uploading 50k tiny raw PNGs (always tar/zip into 1–2 GB shards) or per-step commits |

---

## Round-Robin Dispatch & State Tracking Protocol

To guarantee that **no two providers duplicate the same trial** and that **preempted/timed-out trials seamlessly resume on the next available provider**, coordinate every cloud run through a lightweight JSON ledger in HF `runs/<run_id>.json`:

```json
{
  "run_id": "v2-ablation-muon-lr3e4",
  "commit_sha": "abc1234def5678",
  "baseline_run": "uni-omr-iter6-clean",
  "hypothesis": "Switch decoder linear layers to Muon lr=3e-4 (1 variable vs AdamW baseline)",
  "status": "running",
  "provider": "modal-l4",
  "precision": "bf16",
  "target_steps": 4000,
  "current_step": 1500,
  "last_ckpt": "checkpoints/v2-ablation-muon-lr3e4/latest.pt",
  "eval_path": null,
  "updated_at": "2026-11-25T22:30:00Z",
  "notes": ""
}
```

### Status State Machine

`queued` → `running` → (`preempted` → `running` on next provider) → `trained` → `evaluated` → `promoted` | `rejected`

### Round-Robin Scheduling Rules

1. **Before dispatching**, pull `runs/*.json` from HF to check active trials and remaining budgets (Modal `$30/mo`, Kaggle `30 hrs/wk`, Local GPU busy/idle).
2. **Assign by fit & availability:**
   - Need BF16 + fast parallel feedback (`<= 45 min`) → **Modal** (`L4` / `A10G`, up to 3–5 concurrent).
   - Need multi-hour background throughput and model runs cleanly in `--fp16` → **Kaggle** (`2× T4`).
   - Need full overnight production run (`8k–36k` steps) or quick OOD eval → **Local 4080 SUPER**.
3. **Claim before start:** Write/update `runs/<run_id>.json` with `"status": "running"` and `"provider": "<provider>"` before starting training.
4. **Preemption / Timeout Handoff:**
   - If a Kaggle 12h kernel or Modal run stops before `target_steps`, its exit handler uploads `latest.pt` and updates `runs/<run_id>.json` (`status: "preempted"`, `current_step`, `last_ckpt`).
   - Any other provider (e.g., Local or Modal) can resume from `last_ckpt` — **except** do not resume a BF16 optimizer state into Kaggle FP16 `GradScaler` without verifying precision compatibility.
5. **Completion & Eval:**
   - Once `current_step >= target_steps`, set `status: "trained"`, run `suite31_hw` (on the same GPU if time remains, or on Local 4080 SUPER), upload `evals/<run_id>_suite31_hw.json`, set `status: "evaluated"`, and invoke `learn-from-suite31`.

---

## Standard Cloud Trial Checklist

1. **Code checkpoint:** Run unit tests locally, `git commit` + `git push`, record `git rev-parse HEAD`.
2. **Data sync:** Confirm required dataset shards (`datasets/*.tar`) and manifest (`manifests/*.jsonl`) exist on HF (`cloud-huggingface`).
3. **Register run:** Create `runs/<run_id>.json` with single-variable `hypothesis` and `commit_sha`.
4. **Launch:** Dispatch via `cloud-modal` (`modal run --detach`) or `cloud-kaggle` (`kaggle kernels push`).
5. **Monitor & collect:** Check logs/status without tight polling; fetch `evals/<run_id>_suite31_hw.json` once done.
6. **Learn & amend:** Run `learn-from-suite31`, update `progress.md`, and immediately log any new cloud quirks in the relevant provider `SKILL.md`.

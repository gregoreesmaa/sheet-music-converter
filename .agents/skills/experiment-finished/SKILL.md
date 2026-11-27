---
name: experiment-finished
description: >-
  Mandatory post-experiment completion playbook for sheet-music-converter.
  Activate IMMEDIATELY whenever ANY local or cloud experiment (Modal, Colab,
  Kaggle, or Local RTX 4080 SUPER) finishes: pull artifacts, scrutinize for
  anomalies, write OUTCOME.md, amend provider skills, and execute the
  verdict-driven action (PROMOTE, DISCARD, or MANDATORY ITERATE -> immediately
  formulate, implement, and queue a follow-up experiment in the DAG).
---

# Experiment Finished Playbook (Post-Run Scrutiny & Autonomous Continuation)

**Trigger:** Activate **IMMEDIATELY** whenever ANY experiment finishes (local RTX 4080 SUPER, Google Colab, Modal, or Kaggle), whether it succeeded, hit a gate, or failed.

---

## Mandatory 7-Step Completion Checklist

### 1. Pull & Verify All Artifacts
- **Cloud runs:** Download `evals/<exp_id>.json` and `runs/<exp_id>.json` from `gregoreesmaa/sheet-music-converter-omr` via `huggingface_hub.hf_hub_download(...)`. Verify the remote VM/container is stopped (`colab sessions` / `modal app list`) so zero idle credits or compute units are burned.
- **Local runs:** Verify final checkpoint (`uni_omr_best.pt` / `uni_omr_ema_latest.pt`) and run the full 45-piece `suite31_hw` OOD benchmark (`suite31_hw_manifest.json` + `suite31_hw_clean_gt_manifest.json`) via `learn-from-suite31` if not already scored.

### 2. Mandatory Post-Result Scrutiny (`## Scrutiny & Anomaly Check`)
Before trusting any metric or declaring a verdict, actively audit:
1. **Silent failures / partial coverage:** Were all expected pages/arms scored (`n_scored == n_pieces`, `missing_keys == []`, `decode_ok_rate == 1.0`, zero NaNs)?
2. **Apples-to-apples sanity:** Did resolution, `max_len`, val split, or metric definition silently shift compared to baseline?
3. **Domain / slice asymmetry:** Did `clean` improve while `photo` regressed, or `printed` improve while `hw` regressed? Did a hyperparameter over-adapt (`s=5` vs `s=3`) or under-move (`lr=1e-5 × 60 steps`)?
4. **Hardware / platform quirks:** Any OOM, BF16 autocast trap, concurrency limit, or missing dependency?

### 3. Write `experiments/<exp_id>/OUTCOME.md`
Every finished experiment MUST have an `OUTCOME.md` containing:
- Status, completion timestamp, platform/GPU, elapsed time, and HF artifact paths.
- Measured results table (before vs after / arm vs arm, with exact deltas).
- **`## Scrutiny & Anomaly Check`** (from Step 2).
- Clear verdict: **`PROMOTE`**, **`ITERATE`**, or **`DISCARD`**.
- Explicit **`## Follow-Up Action / Successor Experiment`** link.

### 4. Amend Living Skills Immediately
If any new cloud, hardware, PyTorch, or tooling quirk was discovered, **immediately append it** to the relevant skill's `## Empirical Learnings & Gotchas` section (`cloud-modal`, `cloud-colab`, `cloud-kaggle`, `cloud-huggingface`, or `running-an-experiment`).

### 5. Verdict-Driven Autonomous Action (CRITICAL)

| Verdict | Mandatory Immediate Action |
|---|---|
| **`PROMOTE`** | **MANDATORY IMMEDIATE IMPLEMENTATION — NEVER leave a `PROMOTE` sitting in `OUTCOME.md`!** Immediately implement and wire the winning flag, architecture, loss, data source, or inference upgrade into the **next main model iteration** (`train_uni_v2.py` / `uni_omr_v2.py` / `eval_uni_suite31.py` / companion repo AND `experiments/.../launch_iter{N+1}.ps1`), add & run CPU unit tests (`CUDA_VISIBLE_DEVICES=""`), update `todo.md` + `progress.md`, and commit & push cleanly to `origin main`. |
| **`ITERATE`** | **ALWAYS immediately formulate, implement, and queue a follow-up experiment** (`experiments/YYYY-MM-DDTHH-MM-<slug>/{README.md, PLAN.md, run_*.py}`) directly targeting the diagnosed bottleneck/asymmetry, register it in the DAG queue (`experiments/overnight_queue/queue.json` with `depends_on: ["<this_exp_id>"]`), and link the successor ID inside this experiment's `OUTCOME.md`. **NEVER leave an `ITERATE` verdict dangling without a queued successor!** |
| **`DISCARD`** | Document the exact root cause and falsified hypothesis in `OUTCOME.md` + `todo.md` so no future agent wastes compute repeating it. |

### 6. Update Ledgers & DAG Queue
- Update `experiments/README.md` (and `experiments/overnight_queue/queue.json` status to `COMPLETED` / `FAILED`) so any downstream DAG experiments whose `depends_on` predecessors are now satisfied immediately unlock and start.
- For major `suite31_hw` model runs, update `progress.md` and `todo.md` and run the 6-scout `learn-from-suite31` loop before launching the next major training iteration.
- **OMR-NorthStar trigger:** Whenever ANY `eval-northstar-*` run completes (for UniOMR or a competitor baseline), **immediately activate `learn-from-northstar`** (`.agents/skills/learn-from-northstar/SKILL.md`) to log the full NorthStar ledger, run the 6-scout CPU forensic swarm, categorize fixes (`Model` / `Export` / `Inference` / `Harness`), and feed them into the next iteration.

### 7. Git Commit & Push
- Commit `OUTCOME.md`, any newly wired `PROMOTE` implementation + unit tests, any skill amendments, and any newly queued `ITERATE` follow-up experiment (`Gregor Eesmaa <6941817+gregoreesmaa@users.noreply.github.com>`) and push to `origin main` (never `origin master`, whose legacy root carries old `.hf-cache` blobs).

# Phase 0: Repo Safety & Trainer Hardening (`T0.1` – `T0.3`)

Back to master checklist: [../todo.md](../todo.md)

Complete these tasks in exact order (`T0.1` → `T0.2` → `T0.3`). Do not skip verification commands.

---

## `[x]` Task T0.1 — Initialize Git Repository and Commit Baseline

### Objective
Initialize a local Git repository at the workspace root so no code, evaluator, or training script can be silently lost or corrupted.

### Files to Create / Modify
1. Create `.gitignore` at `C:\Users\grego\projects\sheet-music-converter\.gitignore` containing:
   ```gitignore
   .venv/
   **/target/
   **/__pycache__/
   *.pyc
   *.pt
   *.onnx
   *.tmp
   *.lock
   data-gen/score-corpus/big500/
   data-gen/score-corpus/big500-renders/
   data-gen/score-corpus/big500-degraded10/
   data-gen/score-corpus/big500-symbox/
   data-gen/score-corpus/mscx-5000/
   data-gen/score-corpus/handwritten-500/
   data-gen/score-corpus/renders-fonts/
   data-gen/score-corpus/renders-fontmatrix/
   tmp-train/mvd-runs/
   tmp-train/mvd-data/
   tmp-train/mvd-data-regen/
   tmp-train/mvd-sysdata/
   tmp-train/det-data/*.png
   tmp-train/det-data/*.jpg
   ```
2. Run `git init`, configure a local `user.name` and `user.email` if not set, stage all tracked source/doc/config files, and create the initial baseline commit.

### Verification Command
```powershell
git status --short
git log -n 1 --oneline
```

### Pass Criteria
- `git log -n 1 --oneline` prints a valid commit hash.
- `git status --short` is empty (clean working tree) and no `.pt`, `.onnx`, `.venv`, or `target/` files are tracked (`git ls-files "*.pt" "*.onnx"` outputs nothing).

---

## `[x]` Task T0.2 — Harden `tmp-train/mvd/train.py` (Single-Writer Lock, Atomic Saves, Config-Fingerprint Refusal)

### Objective
Prevent silent checkpoint corruption from concurrent processes, mid-write kills, or resumed runs with changed hyperparameters.

### Files to Read First
- `tmp-train/mvd/train.py`
- `tmp-train/mvd/test_schedule.py`

### Exact Changes to Make in `tmp-train/mvd/train.py`
1. **Single-writer lock (`<out>/.train.lock`):**
   - Add a helper function `acquire_out_lock(out_dir: Path)`:
     - Creates `<out_dir>/.train.lock` atomically using `os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)` and writes `json.dumps({"pid": os.getpid()})`.
     - If `FileExistsError` is raised, read the PID from `.train.lock`. Check if that PID is still alive on Windows (`os.kill(pid, 0)` or `ctypes.windll.kernel32.OpenProcess`). If alive (or unreadable), call `sys.exit(f"refusing: active writer lock {lock_path}")`. If stale (dead PID), remove and recreate the lock.
     - Register an `atexit` handler to unlink `.train.lock` on normal exit.
2. **Atomic `tmp + fsync + os.replace` writes for all saves:**
   - Add a helper function `atomic_torch_save(obj, path: Path)` that saves to `path.with_suffix(path.suffix + ".tmp")`, flushes and calls `os.fsync` on the file descriptor, then calls `os.replace(tmp_path, path)`.
   - Add a helper function `atomic_write_text(path: Path, text: str)` using `.tmp` + `os.fsync` + `os.replace`.
   - Replace every `torch.save(...)` and `(out / 'meta.json').write_text(...)` in `train.py` with these atomic helpers.
3. **Full config-fingerprint refusal on `--resume`:**
   - When `--resume` is passed and `meta.json` exists, verify that **all** of these keys match between `old` (`meta.json`) and current `meta`:
     `('vocab', 'pages', 'height', 'lr', 'seed', 'exclude', 'shuffle', 'no_amp', 'accum', 'trunk_lr_mult', 'warmup', 'cosine', 'sample_p')`.
   - If any key differs, call `sys.exit(f"refusing --resume: config mismatch on key {k}: old={old.get(k)} vs new={meta.get(k)}; use --init-from in a fresh --out dir")`.

### Unit Test to Create
Create `tmp-train/mvd/test_train_safety.py` testing:
1. `atomic_torch_save` writes the target file and leaves no `.tmp` file behind.
2. `acquire_out_lock` blocks a second call in the same directory while active, and cleans up on release.
3. Resuming `train.py` with a changed `--height` or `--lr` exits with non-zero status and `"refusing --resume: config mismatch"`.

### Verification Command
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tmp-train/mvd -p "test_*.py" -v
```

### Pass Criteria
- All unit tests in `tmp-train/mvd` (including `test_train_safety.py`) pass (`OK`).

---

## `[x]` Task T0.3 — Add Standing Blank-Image & Shuffled-Image Grounding Probe to `tmp-train/mvd/train.py`

### Objective
Detect encoder collapse immediately during training by measuring whether the model actually uses local image pixels vs. hallucinating from its language-model prior.

### Files to Read First
- `tmp-train/mvd/train.py` (lines 368–421)
- `tmp-train/mvd/model.py`

### Exact Changes to Make in `tmp-train/mvd/train.py`
1. Add a helper function `grounding_probe(net, eimgs, ref_ids)`:
   - Runs `hyp_real = net.greedy_kv(eimgs, max_len=len(ref_ids) + 16, constrain=True)` on the real eval image `eimgs`.
   - Creates a blank (all-zeros) tensor `blank = torch.zeros_like(eimgs)` and runs `hyp_blank = net.greedy_kv(blank, max_len=len(ref_ids) + 16, constrain=True)`.
   - Strips `EOS_ID` and `PAD_ID` from both, computes `ter_real = edit_distance(hyp_real, ref_ids) / max(1, len(ref_ids))` and `ter_blank = edit_distance(hyp_blank, ref_ids) / max(1, len(ref_ids))`, plus `blank_diff = edit_distance(hyp_real, hyp_blank) / max(1, max(len(hyp_real), 1))`.
   - Returns `(hyp_real, ter_real, ter_blank, blank_diff)`.
2. In the eval block of `train.py` (`if step % args.eval_every == 0 or step == args.steps:`):
   - Call `grounding_probe` and print `ter_blank=%.4f blank_diff=%.4f` alongside `TER=%.4f memd=%.1f`.
   - Append a JSON line per eval step to `out / 'eval_log.jsonl'` containing `{"step": step, "loss": ..., "ter": ter_real, "ter_blank": ter_blank, "blank_diff": blank_diff, "memd": memd}`.

### Unit Test to Add
In `tmp-train/mvd/test_train_safety.py`, add `test_grounding_probe_returns_metrics` using a tiny `MvdNet(vocab=len(data.VOCAB)+3, d_model=64, layers=1, heads=4, dim_ff=128)` on a synthetic `1×3×128×256` tensor.

### Verification Command
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tmp-train/mvd -p "test_*.py" -v
```

### Pass Criteria
- All unit tests pass (`OK`), and `grounding_probe` returns a 4-tuple `(hyp_real, ter_real, ter_blank, blank_diff)`.
- Commit changes to Git (`git add -A && git commit -m "Phase 0: repo safety and trainer grounding probe"`).

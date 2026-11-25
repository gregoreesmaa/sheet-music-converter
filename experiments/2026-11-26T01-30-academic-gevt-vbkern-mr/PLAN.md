# Experiment Plan — `2026-11-26T01-30-academic-gevt-vbkern-mr`

## 1. Platform & Rationale
- **Platform:** Google Colab `L4` (24 GB VRAM, `cc=8.9`, BF16 native, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `acad-c-gevt-vbkern`
- **Estimated Wall-Clock & Cost:** ~16–22 minutes total (~1.5–1.8 Compute Units out of `1,479.30` CU balance).
- **Teardown Rule:** ALWAYS execute `.\.venv\Scripts\colab.exe stop -s acad-c-gevt-vbkern` in `finally`.

## 2. Identifiers & Cloud Paths
- **Colab Session:** `acad-c-gevt-vbkern` (`--gpu L4`)
- **HF Dataset Repo:** `gregoreesmaa/sheet-music-converter-omr`
  - Input train/val shard: `datasets/iter10_5_proxy_2k.tar` (`proxy_train.jsonl` + `proxy_val.jsonl`, 64 val pages)
  - Input OOD eval shard: `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` clean + photo)
  - Output Eval JSON: `evals/2026-11-26T01-30-academic-gevt-vbkern-mr.json`
  - Output Run Ledger: `runs/2026-11-26T01-30-academic-gevt-vbkern-mr.json`
- **Baseline Commit SHA:** `164852f`
- **License Confirmation:** 100% permissive datasets only (`iter10_5_proxy_2k.tar` + `suite31_hw_eval.tar`, zero `CC-BY-NC`).

## 3. Controlled 2-Arm Ablation Design (Matched Hyperparameters)
Both arms train from scratch with **identical** architecture (`UniOMRv2Config.small_45m`, ~43M params), seed (`42`), data order, image resolution (`fullpage_max_dim=1280`), `max_seq_len=1536`, `bf16=True`, `channels_last=True`, `grad_checkpoint=True`, `batch_size=2`, `grad_accum=2` (effective batch = 4), `AdamW` (`lr=3e-4`, `weight_decay=0.05`, 100-step linear warmup + cosine decay to `1.5e-5`), `1,000` optimizer steps each:

| Arm | Tokenization | Vocab Size | Refiner Head | Target Length (suite31_hw) |
|---|---|---:|---|---:|
| **Arm 0 (Control)** | Semantic `bekern` | 237 | None | 57,485 toks (1.000x) |
| **Arm 1 (Treatment)** | Bijective `V-bekern` + Deterministic Gauge-Lifter | 294 | `MaskedRefinerHead` (`581K` params, 2-step parallel MR, loss weight `0.25`) | 42,156 toks (0.733x, -26.7%) |

## 4. Launch & Lifecycle Commands
```powershell
.\.venv\Scripts\colab.exe new -s acad-c-gevt-vbkern --gpu L4
.\.venv\Scripts\colab.exe upload -s acad-c-gevt-vbkern <bundle.tar.gz> /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s acad-c-gevt-vbkern <tmp_hf_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s acad-c-gevt-vbkern -f experiments/2026-11-26T01-30-academic-gevt-vbkern-mr/run_colab.py
# Always in finally:
.\.venv\Scripts\colab.exe stop -s acad-c-gevt-vbkern
```

## 5. Checkpointing & Resume Plan
- Each arm saves `latest.pt` every 5 minutes (atomic `.tmp` -> `os.replace`) to `/content/exp_c_out/<arm>/`.
- `run_colab.py` deletes `/tmp/.hf_token` in its `finally` block (never printed).

## 6. Evaluation Gates & Decision Tree
All metrics are scored **apples-to-apples in lifted semantic `bekern` space** (Arm 1 outputs are lifted via `vbkern.vbkern_to_bekern`):
1. **Held-out `proxy_val.jsonl` (64 pages):** Lifted semantic Token F1, LCS F1, Pitch Accuracy, Rhythm Accuracy, Exact Note Accuracy, and Gauge-Error Decomposition (`gauge_transport_errors` vs `visual_sp_errors`).
2. **Unseen-Clef / Rare-Key OOD Probe:** Stratified comparison on C-clef (`clef-C-*`) / octave-transposing (`clef-G-2-8*`) / heavy-key (`|fifths| >= 3`) passages vs standard Treble/Bass common-key passages.
3. **`suite31_hw` OOD Sample (20 pieces: 10 Printed + 10 Real HW, clean + photo = 40 pages):** Lifted semantic Token F1, Pitch Accuracy, Rhythm Accuracy, and 2-Step MR delta (`vbkern_greedy` vs `vbkern_mr2`).

### Decision Tree
- **PROMOTE:** Arm 1 achieves `>= +2.0` pts lifted semantic Token F1 or Pitch Accuracy over Arm 0 on held-out val / rare-clef OOD probe (or `>= +1.0` pt on `suite31_hw` OOD) with lower gauge-transport error share and `>= 15%` faster decoding.
- **ITERATE:** Mixed result (e.g., `V-bekern` helps rare-clef/speed while 2-step MR is neutral or needs curriculum/higher mask prob) -> isolate `V-bekern` alone vs MR schedule.
- **DISCARD:** Arm 1 underperforms Arm 0 on both val and OOD strata -> retain standard `bekern` and document why.

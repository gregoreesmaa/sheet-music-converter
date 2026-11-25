# Experiment Plan — `2026-11-26T00-55-trial-scribe-ttt-suite31-hw`

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe` (moved from queued Modal `L4` app `ap-eLJrorKeWReglJ2HRrX7IB`, which was cancelled before execution).
- **Rationale:** Zero interference with the local RTX 4080 SUPER, immediate `L4` provisioning (`~11s`) with pre-installed `torch 2.14.1+cu130`, native BF16 support, and clean auto-teardown (`colab stop -s exp6-scribe-ttt`) upon completion.
- **Measured Wall-Clock:** **`1,294.95 s` (`21.6 min`)** end-to-end on Colab `NVIDIA L4` (including HF dataset/checkpoint download + 28 pages × `[baseline + s3 TTT + s5 TTT]` with FSA grammar + MTP-4 speculative decoding).

## 2. Identifiers for Follow-Up Queries (Zero-Prior-Context Handoff)
- **Colab Session Name:** `exp6-scribe-ttt` (`NVIDIA L4`, `cc=8.9`, `bf16=True`, `torch 2.14.1+cu130` — **STOPPED / terminated** cleanly after HF upload)
- **Runner Script:** `experiments/2026-11-26T00-55-trial-scribe-ttt-suite31-hw/run_colab.py` (replaced queued Modal app `ap-eLJrorKeWReglJ2HRrX7IB`, stopped via `modal app stop -y`)
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T00-55-trial-scribe-ttt-suite31-hw.json` (**UPLOADED & VERIFIED**, 28/28 rows)
  - Run Ledger JSON: `runs/2026-11-26T00-55-trial-scribe-ttt-suite31-hw.json` (**UPLOADED**, `status=COMPLETED`)

### Copy-Paste Monitoring / Control / Download Commands
```powershell
# Download final eval JSON from Hugging Face
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T00-55-trial-scribe-ttt-suite31-hw.json', repo_type='dataset'))"
```

## 3. Required Data & Baseline Checkpoint (100% Permissive)
- **Eval Dataset:** HF `datasets/suite31_hw_eval.tar` (unpacked to `/vol/datasets/suite31_hw_eval/`), manifest `suite31/suite31_hw_manifest.json` — filtering to all **14 real historical handwritten manuscripts** across both `clean` and `photo` domains (`28` pages total, 100% permissive).
- **Baseline Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (cached at `/vol/checkpoints/iter10_soup_baseline/uni_omr_soup.pt`).
  - Measured Iter-10 `soup` baseline (`max_dim=1536`, `max_len=4096`, FSA grammar + MTP-4): `hw_clean 0.2550`, `hw_photo 0.1516`, `printed_clean 0.8895`, `printed_photo 0.8738`, `all45_clean 0.6921`.
- **Scoring Binaries:** Native Linux release builds of `converter/assembly-serialize` and `converter/eval-harness` (linked against sibling `mscz-rs`) compiled in a cached Modal image layer and wired into `eval_sys_gate.ASER` / `eval_sys_gate.EHARNESS` for exact apples-to-apples Strict Note F1 and Note SER.

## 4. Exact Launch Command & Hyperparameters
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s exp6-scribe-ttt --gpu L4
.\.venv\Scripts\colab.exe upload -s exp6-scribe-ttt "$env:TEMP\exp6_bundle.tar.gz" /content/exp6_bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s exp6-scribe-ttt "$HOME\.cache\huggingface\token" /tmp/.hf_token
try {
    .\.venv\Scripts\colab.exe exec -s exp6-scribe-ttt --timeout 0 -f experiments/2026-11-26T00-55-trial-scribe-ttt-suite31-hw/run_colab.py
} finally {
    .\.venv\Scripts\colab.exe stop -s exp6-scribe-ttt
}
```
- **Hyperparameters:**
  - `step_counts = [3, 5]` (each evaluated cleanly from `ttt.restore()` snapshot)
  - `ttt_lr = 5e-5`, `adapt_scope = "norms"` (`RMSNorm` weights in `backbone`/`enc_layers`/`enc_norm`, `3,456` params = `0.0077%`)
  - `conf_threshold = 0.85`, `cons_weight = 1.0`, `ent_weight = 1.0`, `kl_weight = 0.5`, `l2_weight = 1e-2`, `grad_clip = 1.0`
  - `max_dim = 1536`, `max_len = 4096`, `ttt_max_decode_len = 512`, `use_grammar = True`, `use_mtp_spec = True`, `bf16 = True`
  - SDPA guard: `enable_cudnn_sdp(False)` (CUTLASS/Flash only).

## 5. 5-Minute Checkpointing & Resume Plan
- Atomic JSON checkpointing (`_atomic_write_json` via `.tmp` → `os.replace`) to `/content/vol/evals/2026-11-26T00-55-trial-scribe-ttt-suite31-hw_partial.json` every `<= 5.0` minutes (and after every completed piece).
- Final eval + run ledger JSONs are uploaded to Hugging Face Hub on completion, and `/tmp/.hf_token` is purged immediately after reading and again in `finally:`.

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")
- **Primary Metrics:** `hw_clean` and `hw_photo` Strict Note F1 (and Token F1) across all 14 HW manuscripts (`before` vs `after_s3` vs `after_s5`), win/tie/loss count per piece, and median TTT wall-clock seconds/page.
- **PROMOTE:** If `delta_note_f1 >= +0.020` on `hw_clean` OR `hw_photo` (with no regression `> -0.005` on the other domain and `wins > losses`) → integrate optional `--ttt-steps` flag into `eval_uni_suite31.py` / inference CLI and run a follow-up full 45-piece check (to verify zero regression on printed scores).
- **ITERATE:** If `delta_note_f1` is modestly positive (`+0.005` to `+0.020`) or shows high variance across manuscripts → inspect per-piece `conf_tokens`, `loss_history`, and `delta_note_f1` in the JSON, and run a targeted follow-up sweeping `adapt_scope="all_rms_norms"` / `"norms_with_gn"`, `ttt_lr`, or `conf_threshold`.
- **DISCARD:** If both `s=3` and `s=5` yield `delta_note_f1 <= +0.002` or regress on `hw_clean` and `hw_photo` → document root cause in `OUTCOME.md` (e.g., too few high-confidence pseudo-tokens on hard HW pages) and shelve SCRIBE/DocTTT until base HW accuracy exceeds `0.40+`.

## 7. Measured Results (Colab `NVIDIA L4`, `1,294.95 s`, HF Verified)

**Gate Verdict:** **ITERATE** — `s=3` improves `hw_photo` by **`+0.0077`** (`6W / 5T / 3L`, median `6.8s`/page) and `hw_clean` by **`+0.0013`** (`7W / 3T / 4L`, median `6.9s`/page), whereas `s=5` over-adapts and regresses (`-0.0219` on `hw_clean`, `-0.0007` on `hw_photo`).

### Aggregate Summary (All 14 Real Historical HW Manuscripts)

| Domain | n | Baseline Note F1 | `s=3` Note F1 (`Δ`) | `s=3` W/T/L | `s=3` Med. Time | `s=5` Note F1 (`Δ`) | `s=5` W/T/L | `s=5` Med. Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hw_clean` | 14 | `0.2611` | `0.2625` (`+0.0013`) | `7 / 3 / 4` | `6.90 s` | `0.2393` (`-0.0219`) | `4 / 3 / 7` | `8.02 s` |
| `hw_photo` | 14 | `0.1916` | `0.1994` (`+0.0077`) | `6 / 5 / 3` | `6.78 s` | `0.1909` (`-0.0007`) | `6 / 3 / 5` | `7.81 s` |

### Per-Piece Strict Note F1 (`hw_clean`)

| Piece | Baseline | `s=3` (`Δ`) | `s=5` (`Δ`) | TTT Time (`s3` / `s5`) |
|---|---:|---:|---:|---:|
| `hw-beethoven-op135` | `0.0383` | `0.0157` (`-0.0226`) | `0.0144` (`-0.0239`) | 7.3s / 8.1s |
| `hw-mozart-k464` | `0.3759` | `0.3927` (`+0.0168`) | `0.3336` (`-0.0423`) | 6.5s / 7.3s |
| `hw-mozart-k387` | `0.1281` | `0.1463` (`+0.0182`) | `0.1563` (`+0.0282`) | 6.5s / 7.3s |
| `hw-mozart-k421` | `0.2081` | `0.2088` (`+0.0007`) | `0.1637` (`-0.0444`) | 6.5s / 7.4s |
| `hw-schubert-d810` | `0.0638` | `0.0608` (`-0.0030`) | `0.0639` (`+0.0002`) | 7.2s / 8.5s |
| `hw-debussy-vetement-du-blesse` | `0.2558` | `0.2317` (`-0.0240`) | `0.2306` (`-0.0252`) | 7.1s / 8.2s |
| `hw-debussy-berceuse-heroique` | `0.5390` | `0.5511` (`+0.0121`) | `0.5489` (`+0.0099`) | 6.9s / 8.0s |
| `hw-debussy-la-plus-que-lente` | `0.6932` | `0.6947` (`+0.0015`) | `0.5907` (`-0.1025`) | 6.9s / 8.1s |
| `hw-debussy-lindaraja` | `0.2685` | `0.2840` (`+0.0155`) | `0.1197` (`-0.1488`) | 6.9s / 8.2s |
| `hw-hensel-morgengruss-d-b` | `0.0395` | `0.0394` (`-0.0002`) | `0.0859` (`+0.0464`) | 6.3s / 7.4s |
| `hw-mendelssohn-herbstlied-d-b` | `0.0647` | `0.0684` (`+0.0037`) | `0.0607` (`-0.0040`) | 7.0s / 8.1s |
| `hw-beethoven-das-blumchen-wunderhold-d-b` | `0.7696` | `0.7696` (`+0.0000`) | `0.7696` (`+0.0000`) | 6.9s / 7.9s |
| `hw-cornelius-botschaft-d-b` | `0.1201` | `0.1201` (`+0.0000`) | `0.1201` (`+0.0000`) | 6.7s / 7.8s |
| `hw-hensel-im-herbste-d-b` | `0.0914` | `0.0914` (`+0.0000`) | `0.0914` (`+0.0000`) | 6.9s / 8.0s |

### Per-Piece Strict Note F1 (`hw_photo`)

| Piece | Baseline | `s=3` (`Δ`) | `s=5` (`Δ`) | TTT Time (`s3` / `s5`) |
|---|---:|---:|---:|---:|
| `hw-beethoven-op135` | `0.0775` | `0.0775` (`+0.0000`) | `0.0565` (`-0.0209`) | 6.9s / 8.1s |
| `hw-mozart-k464` | `0.1829` | `0.1772` (`-0.0058`) | `0.1661` (`-0.0168`) | 5.9s / 6.6s |
| `hw-mozart-k387` | `0.0765` | `0.0708` (`-0.0057`) | `0.0778` (`+0.0013`) | 6.5s / 7.3s |
| `hw-mozart-k421` | `0.0918` | `0.1037` (`+0.0119`) | `0.0913` (`-0.0005`) | 6.1s / 6.8s |
| `hw-schubert-d810` | `0.1200` | `0.1441` (`+0.0241`) | `0.1007` (`-0.0193`) | 7.3s / 8.6s |
| `hw-debussy-vetement-du-blesse` | `0.1718` | `0.1718` (`+0.0000`) | `0.1718` (`+0.0000`) | 7.0s / 8.1s |
| `hw-debussy-berceuse-heroique` | `0.3078` | `0.3192` (`+0.0114`) | `0.3096` (`+0.0018`) | 7.0s / 8.0s |
| `hw-debussy-la-plus-que-lente` | `0.4802` | `0.4762` (`-0.0040`) | `0.4646` (`-0.0156`) | 6.9s / 8.0s |
| `hw-debussy-lindaraja` | `0.2549` | `0.3251` (`+0.0702`) | `0.3085` (`+0.0536`) | 6.9s / 8.1s |
| `hw-hensel-morgengruss-d-b` | `0.0620` | `0.0620` (`+0.0000`) | `0.0620` (`+0.0000`) | 6.4s / 7.6s |
| `hw-mendelssohn-herbstlied-d-b` | `0.0185` | `0.0201` (`+0.0016`) | `0.0201` (`+0.0016`) | 6.9s / 8.0s |
| `hw-beethoven-das-blumchen-wunderhold-d-b` | `0.4681` | `0.4728` (`+0.0047`) | `0.4728` (`+0.0047`) | 6.7s / 7.6s |
| `hw-cornelius-botschaft-d-b` | `0.2593` | `0.2593` (`+0.0000`) | `0.2593` (`+0.0000`) | 6.5s / 7.3s |
| `hw-hensel-im-herbste-d-b` | `0.1113` | `0.1113` (`+0.0000`) | `0.1117` (`+0.0004`) | 6.3s / 7.1s |

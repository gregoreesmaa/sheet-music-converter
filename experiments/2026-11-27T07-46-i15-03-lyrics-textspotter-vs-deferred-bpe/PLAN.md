# PLAN — `2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, native BF16) via `experiments/overnight_queue/` dispatcher (also Modal L4 compatible).
- **Estimated Wall-Clock & Cost:** ~75 min (~5.5 CU).
- **Baseline Commit SHA:** `e426c28`.
- **Wave / Priority:** Wave 13, `i15-03` (Representation & Text/Lyrics Pivot).

## 2. Identifiers for Follow-Up Queries
- **Queue Spec:** [queue_entry.json](./queue_entry.json) (`id: "2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe"`, `wave: 13`, `provider: "colab_l4"`, `gpu: "L4"`).
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe.json`
  - Run Ledger: `runs/2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe.json`

## 3. Problem Statement & Controlled 2-Arm Design
OMR-NorthStar currently caps `S_expr` at a `-13.5 pp` ceiling because `text = 0.000` (no lyrics, harmony/chord symbols, or tempo/direction words emitted). We evaluate two `< 55.0 MB`-compliant representations on an identical vocal/leadsheet split (`1,500` steps/arm, `seed=42`, BF16, `grad_accum=4`, `lr=3e-4` WSD cosine cooldown, `EMA_DECAY=0.99` every step):

### Arm A: `arm_a_textspotter_head` (Parallel Decoupled Text Spotter)
- **Backbone:** `UniOMRv2Config.edge85m(grad_checkpoint=True)` (`d_model=512`, `10` dec layers).
- **Freeze Policy:** Freeze 100% of base encoder (`backbone`, `axial_layers`, `patch_merger`, `global_layers`) and base decoder (`tok_emb`, `deepstack_fusions`, `dec_layers`, `dec_norm`, `lm_head`).
- **Trainable Parameters:**
  1. `TextSpotterHead` (`~1.8M` params / `~1.15 MB` INT4): 2-layer `d_text=256` (`n_heads=4`, `d_ff=768`) cross-attention transformer block. Projects notehead/bar decoder hidden states (`512 -> 256`) as spatial-temporal queries and cross-attends directly to high-resolution `mem_post_merger` (`512 -> 256`), predicting 256-token syllable BPE text slots (`lyrics`, `harmony`, `tempo`) per anchor.
  2. `encoder.router` (unfrozen text-ink saliency MLP): allows the ink token router to retain lyric/chord/tempo ink patches above/below staves instead of pruning them as background.
- **Key Guarantee:** Because base music decoder weights are frozen, vocal `S_note` delta is structurally `0.000` (barring router perturbation, which is explicitly measured on `suite31_hw`).

### Arm B: `arm_b_deferred_sys_lyr_bpe` (Unified Deferred `<sys_lyr>` Syllable BPE)
- **Backbone:** Identical `UniOMRv2Config.edge85m(grad_checkpoint=True)` with `+256` syllable BPE tokens + structural delimiters (`<sys_lyr>`, `<bar_anchor>`, `<lyr_end>`) appended to `tok_emb` and `lm_head`.
- **Sequence Layout:** Standard bekern music tokens for each system, followed at end-of-system by a bar-anchored `<sys_lyr>` block encoding lyrics, harmony, and tempo words using the deterministic 256-token syllable BPE tokenizer.
- **Training:** `1,500` steps on the exact same vocal/leadsheet train/val split as Arm A, with identical optimizer, schedule, and EMA settings.

## 4. Data & XML Text Extraction
- **Train/Val Source:** `datasets/iter10_5_proxy_2k.tar` (100% permissive). For every page with MusicXML (`.musicxml` / `.xml`) sidecars or lyric/harmony/tempo metadata, extract bar-anchored `(lyrics, harmony, tempo)` targets; filter/upweight vocal and leadsheet pages (`2×` weight on dense vocal/chord pages) and construct a deterministic train/held-out vocal split (`seed=42`).
- **OOD Eval Source:** `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw`, 90 pages). Extract ground-truth XML `<lyric>/<text>`, `<harmony>`, and `<direction>/<words>` for every piece to compute held-out `text` F1 alongside strict vocal `S_note` F1 and decode latency.
- **Tokenizer:** Deterministic 256-token musical/lyric syllable BPE (`<pad>`, `<bos>`, `<eos>`, `<bar>`, `<lyr>`, `<harm>`, `<tempo>`, ASCII printable chars + top musical/vocal syllables such as `al-`, `le-`, `gro`, `an-`, `dan-`, `te`, `maj`, `min`, `dim`, `aug`, `the`, `and`, `ing`, `ich`, `und`, `der`, `die`, `das`, etc.).

## 5. Exact Launch Command
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\overnight_queue\run_queue_colab.py --only 2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe
```

## 6. 5-Minute Checkpointing & Resume Plan
- Atomic checkpoint save (`latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm under `/content/vol/ckpts/2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe/<arm>/`.
- Completed arm metrics are atomically cached to `arm_result.json` so preemption or Colab reconnect never repeats a finished arm.
- Final `evals/<exp_id>.json` and `runs/<exp_id>.json` are uploaded to Hugging Face (`gregoreesmaa/sheet-music-converter-omr`) immediately upon completion.

## 7. Evaluation Metrics, Gates & Decision Tree
| Metric | Target / Hard Gate | Notes |
|---|---|---|
| Held-out `text` F1 (lyrics + harmony + tempo) | **PROMOTE: `>= 0.70`** | Primary bake-off metric |
| Vocal `S_note` F1 delta vs frozen base | **Hard Gate: `>= -0.5 pp`** | `arm_a` structurally `0.000` (verify router neutrality) |
| Vocal decode latency (`mean_latency_s`) | Lower is better | Compare parallel `TextSpotterHead` vs deferred AR `<sys_lyr>` |
| Exact packed Hybrid FP16/INT4 `.bin` on `edge85m` | **Hard Gate: `< 55.0 MB`** | Real 4-bit nibbles + FP16 group-32 scales on disk |

### Decision Tree
- **PROMOTE (`arm_a_textspotter_head`):** Achieves held-out `text` F1 `>= 0.70`, vocal `S_note` delta `>= -0.5 pp` (`~0.000`), lower latency than Arm B, AND exact packed INT4 size `< 55.0 MB` → wire `TextSpotterHead` + text-ink router training into Iter 15 flagship.
- **PROMOTE (`arm_b_deferred_sys_lyr_bpe`):** Beats Arm A by `>= +0.03` held-out `text` F1 while meeting `text` F1 `>= 0.70`, vocal `S_note` delta `>= -0.5 pp`, acceptable latency (`<= 1.25×` base), AND packed INT4 size `< 55.0 MB` → adopt deferred `<sys_lyr>` BPE in Bekern v2b.
- **ITERATE:** Best arm reaches `0.45 <= text F1 < 0.70` or fails `< 55.0 MB` / `-0.5 pp` `S_note` gate → inspect failure modes (router text pruning vs anchor alignment vs BPE OOV) and queue `i15-03b` with dedicated vocal corpus (OpenScore Lieder + Choral XMLs) or tighter `d_text=192` bottleneck.
- **DISCARD:** Both arms `< 0.45` `text` F1 or `> 2.0 pp` `S_note` regression → keep music-only decoder and revisit external OCR cropper in Wave 14.

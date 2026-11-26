# Outcome — `2026-11-26T01-38-ov-19-router-saliency-keep-sweep`

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T09:12:47Z` (started `07:54:25Z`)
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`24 GB`, native `BF16`)
- **Elapsed Time:** `4,701.3 s` (`78.4 min`; `~18.4–19.2 min` per 600-step arm, `peak_vram=9.35–9.55 GB`)
- **Model & Protocol:** `UniOMR v2 small_45m` (`41.33M` total params, `InkTokenRouterV2`, `dynamic_routing=True`, `min_keep_ratio=0.15`, `max_mem_tokens=3072`), `600` steps/arm from scratch (`seed=42`, `Muon+AdamW`, `lr=3e-4`, WSD, `B=2 × GA=4`, `dim=1280`, `max_seq_len=1536`, `VAL_EVAL_MAX_LEN=512`), `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages).
- **HF Artifacts:**
  - `evals/2026-11-26T01-38-ov-19-router-saliency-keep-sweep.json` (pulled locally to [2026-11-26T01-38-ov-19-router-saliency-keep-sweep.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T01-38-ov-19-router-saliency-keep-sweep/2026-11-26T01-38-ov-19-router-saliency-keep-sweep.json))
  - `runs/2026-11-26T01-38-ov-19-router-saliency-keep-sweep.json` (pulled locally to [run_2026-11-26T01-38-ov-19-router-saliency-keep-sweep.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T01-38-ov-19-router-saliency-keep-sweep/run_2026-11-26T01-38-ov-19-router-saliency-keep-sweep.json))

## Measured Results (`proxy_val.jsonl`, 64 Full Pages)

### Per-Arm Results (EMA & Raw Weights)

| Arm | `(sal_thr, max_keep)` | Weights | `ink_rec` mean / p10 / min | Gen `ink_rec` | `comp_x` (`keep_frac`) | `val_ce` (Δ vs cons) | `val_token_f1` (Δ) | `val_ser` (Δ) | `tok/s` |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`arm_003_055`** | `(0.03, 0.55)` | **EMA** | **`0.9461`** / `0.7870` / `0.4853` | **`0.9716`** | `2.70×` (`0.407`) | `1.5108` (`0.0000`) | **`0.2438`** (`0.0000`) | **`0.8849`** (`0.0000`) | `256.4` |
| `arm_005_045` | `(0.05, 0.45)` | EMA | `0.9195` / `0.7601` / `0.4578` | `0.9497` | `3.02×` (`0.374`) | `1.5101` (`-0.0006`) | `0.1767` (`-0.0671`) | `0.9421` (`+0.0572`) | `254.4` |
| `arm_008_035` | `(0.08, 0.35)` | EMA | `0.8196` / `0.6036` / `0.3561` | `0.8537` | `3.51×` (`0.308`) | `1.5079` (`-0.0029`) | `0.2329` (`-0.0109`) | `0.8929` (`+0.0080`) | `251.3` |
| `arm_003_055` | `(0.03, 0.55)` | Raw | `0.9411` / `0.7532` / `0.4764` | `0.9720` | `2.92×` (`0.393`) | `1.2546` (`0.0000`) | `0.2012` (`0.0000`) | `0.9506` (`0.0000`) | `256.5` |
| `arm_005_045` | `(0.05, 0.45)` | Raw | `0.9145` / `0.6876` / `0.4543` | `0.9506` | `3.12×` (`0.366`) | `1.2545` (`-0.0001`) | `0.1754` (`-0.0257`) | `0.9449` (`-0.0057`) | `256.7` |
| `arm_008_035` | `(0.08, 0.35)` | Raw | `0.8184` / `0.5643` / `0.3561` | `0.8539` | `3.58×` (`0.304`) | `1.2516` (`-0.0030`) | `0.2502` (`+0.0491`) | `0.8855` (`-0.0651`) | `254.9` |

### Zero-Shot Cross-Threshold Matrix (`arm_005_045` EMA Checkpoint)

| Eval Gate `(sal_thr, max_keep)` | TF `ink_rec` mean / p10 / min | Gen `ink_rec` | TF `comp_x` (`n_out / 4665`) | `val_ce` | `val_token_f1` | `val_ser` | `tok/s` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `(0.03, 0.55)` | **`0.9468`** / `0.8092` / `0.4817` | **`0.9713`** | `2.77×` (`1900.7`) | **`1.5096`** | `0.1771` | `0.9429` | `255.2` |
| `(0.05, 0.45)` | `0.9195` / `0.7601` / `0.4578` | `0.9497` | `3.02×` (`1755.8`) | `1.5101` | `0.1767` | `0.9421` | `254.5` |
| `(0.08, 0.35)` | `0.8195` / `0.6061` / `0.3561` | `0.8536` | `3.48×` (`1450.0`) | `1.5109` | `0.1877` | `0.9296` | `254.4` |

## Scrutiny & Anomaly Check

1. **Silent failures / coverage:** Zero. All 3 trained arms and all 3 zero-shot matrix cells scored `64 / 64` val pages with zero NaNs and stable `9.35–9.55 GB` VRAM.
2. **Clear ink-recall cliff at `(0.08, 0.35)`:** Tightening `max_keep_ratio` to `0.35` drops teacher-forced `ink_recall_mean` to **`0.8196`** (`p10=0.6036`, `min=0.3561`; gen `0.8537`) — discarding up to 64% of ink tokens on dense pages and failing the `>= 0.96` gate decisively. Even `arm_005_045` drops to `0.9195` TF (`0.9497` gen). Only `arm_003_055` crosses `>= 0.96` in generation (`0.9716`, `2.70×` TF compression), while its TF `p10=0.7870` (`min=0.4853`) still warns of dense-page ink truncation.
3. **100% generation cap saturation (`VAL_EVAL_MAX_LEN = 512`) & early-training noise:** Every single arm and matrix cell produced **`total_pred_tokens = 32,768 = 64 × 512`** (`sys_count_exact = 0.1094` across all cells), proving the 600-step scratch models had not learned reliable EOS stopping yet. Because teacher-forced `val_ce` is nearly flat across arms (`1.2516–1.2546` raw; `1.5079–1.5108` EMA, spread `<= 0.003`), the non-monotonic `F1` dip on `arm_005_045` (`0.1767` vs `0.2438`/`0.2329`) reflects early-scratch repetition variance rather than a true router ceiling.
4. **EMA lag on short runs:** `GPUEmaShadow(decay=0.997, update_every=5)` executes only `120` updates in `600` steps (`0.997^120 = 0.697` random-init weight retained), causing EMA `val_ce` (`1.510`) to lag Raw `val_ce` (`1.254`) by `+0.256`.
5. **No decode speedup from tighter token pruning:** `tokens_per_sec` stayed flat (`251–256 tok/s`) from `n_out=1908` down to `1444`, because L4 batch-4 autoregressive decoding is memory-bandwidth/launch-bound rather than cross-attention FLOP-bound.

## Verdict: **`ITERATE`**

- **Eliminate `(0.08, 0.35)` and `(0.05, 0.45)`** for high-recall full-page OMR: both breach the `>= 0.96` ink-recall floor on dense pages (`p10 <= 0.76`, `min <= 0.46`) while providing **zero** decode throughput gain (`251–254` vs `256 tok/s`).
- **`arm_003_055` (`saliency_thr=0.03, max_keep_ratio=0.55`)** is the best candidate of the three (`gen ink_recall=0.9716`, `2.70×` compression, best EMA `F1=0.2438` / `SER=0.8849`), but 600 scratch steps + `max_len=512` saturation + un-warmed EMA prevented measuring true dense-page Note F1 degradation vs the `0.75` default.

## Follow-Up Action / Successor Experiment

- **Proposed `ITERATE` successor (`ov-19b`):** Evaluate `(saliency_thr, max_keep_ratio) ∈ {(0.02, 0.65), (0.03, 0.55), (0.05, 0.75 baseline)}` on a **converged / warm-started `UniOMR v2` checkpoint** (or `>= 2,500` steps with bias-corrected EMA) using full **`suite31_hw` (`max_len=3840`)** Strict Note F1, specifically auditing dense piano/orchestral pages where `ink_recall_p10` dropped below `0.80`.

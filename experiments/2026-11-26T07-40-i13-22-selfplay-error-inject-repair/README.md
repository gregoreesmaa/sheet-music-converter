# `2026-11-26T07-40-i13-22-selfplay-error-inject-repair`

- **Batch / Track:** Iter 13 Risky Bets (`batch_08`, Moonshot `i13-22`)
- **Theme:** Self-Play Synthetic-Error-Injection Critique & Repair (`SCoRe-OMR`)
- **Provider / Target:** Google Colab `L4` (`BF16`, `~30 min`, `~2.2 CU`)
- **Base Checkpoint:** `gregoreesmaa/sheet-music-converter-omr:checkpoints/iter10_soup_baseline/uni_omr_soup.pt`
- **Eval Target:** `evals/2026-11-26T07-40-i13-22-selfplay-error-inject-repair.json`

## Motivation & Plateau Addressed

On `suite31_hw`, `UniOMR` already achieves `0.749–0.764` Token F1 on historical handwritten manuscripts yet only `0.217–0.259` Strict Note F1 (`Iter 10.5`). The gap is dominated by **autoregressive exposure-bias cascades**: a single dropped augmentation dot, misread duration, octave slip, or missing `<bar>` shifts every subsequent onset and ruins the rest of the system. Standard one-shot SFT never teaches the decoder to *critique and repair* a flawed draft against the page image.

## Moonshot Hypothesis

Attaching a tiny zero-initialized **Draft Memory Projector** (`~0.6M` params, appended to visual memory `[visual_mem; draft_mem]` with zero edits to `uni_omr.py` / `uni_omr_v2.py`) and training the decoder in a **two-turn self-play repair loop** — mixing standard SFT with synthetic OMR error-injection (`duration drift`, `dot drop/insert`, `octave slip`, `<bar>` drop/dup, `accidental drop`) and on-policy draft errors — will enable a **Pass-2 self-repair decode** that catches and fixes bar-onset cascades while preserving clean passages via `R_sym` gating.

## Arms (3 Matched Arms × 300 Steps, `iter10_5_proxy_2k`, `1280px`, `BF16`, `seed=42`)

1. `arm0_sft_control`: Matched 300-step SFT continuation (no draft conditioning).
2. `arm1_selfplay_corrupt_p05`: 50% SFT + 50% synthetic OMR error-injection repair (`corrupt_rate=0.12`).
3. `arm2_selfplay_onpolicy_mix`: 40% SFT + 30% synthetic OMR corruption + 30% on-policy model draft repair.

## Kill / Promote Gates (`suite31_hw` 14 HW + 12 Stratified Printed = 52 pages, `1280px / 3840`)

- **PROMOTE:** Pass-2 beats **both** Pass-1 and `arm0_sft_control` by **`>= +0.020` HW avg Note F1** (or `>= +0.015` HW + `>= +0.005` Printed), with **`>= 60%` non-identity edit precision** (pages edited where F1 improves) and `printed_clean` drift `<= 0.005`.
- **KILL:** Pass-2 copies Pass-1 verbatim on `> 90%` of pages (identity collapse) **OR** fails to beat `arm0_sft_control` on HW avg Note F1.

# Outcome — `2026-11-26T06-25-ov-38b-scribe-ttt-budget-cons-3840` (`ov-38b`)

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T17:45:46Z` (`started_utc=2026-11-26T16:16:02Z`)
- **Platform / GPU:** Kaggle `2x Tesla T4` (`FP32` via `cc >= 8` guard; kernel `gregoreesmaa/omr-ov-38b` = `KernelWorkerStatus.COMPLETE`)
- **Elapsed Time:** `5383.73 s` (`89.7 min` total across 6 arms × 28 HW pages + top-2 arms × 20 printed safety pages)
- **Checkpoint:** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`uni_omr_v1`, step `12000`, `val_ser=0.192301`, `missing_keys=[]`, `unexpected_keys=[]`)
- **Config:** Canonical `max_dim=1280`, `max_len=3840`, `ttt_max_decode_len=512`, `conf_threshold=0.85`, `printed_safety_n=10`
- **Predecessor:** [`2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep`](../2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep/OUTCOME.md)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T06-25-ov-38b-scribe-ttt-budget-cons-3840.json`
  - Run Ledger JSON: `runs/2026-11-26T06-25-ov-38b-scribe-ttt-budget-cons-3840.json`

---

## Measured Results (Canonical `1280px / max_len=3840`)

### Handwritten Manuscripts (`suite31_hw`, `n=14` clean + `n=14` photo = 28 pages)

| Arm Key | Scope | `lr` | Steps | `ent` / `kl` | `hw_clean` F1 (delta) | `hw_clean` W/T/L | `hw_photo` F1 (delta) | `hw_photo` W/T/L | Avg HW Delta | Median TTT (T4 FP32) |
|---|---|---:|---:|---|---|---|---:|---|---:|---:|
| *Baseline (no TTT)* | — | — | `0` | — | `0.258859` (`+0.0000`) | — | `0.144459` (`+0.0000`) | — | `+0.000000` | `0.0 s` |
| `fe_lr3e-5_s2_ent1.0` | `full_encoder` | `3e-5` | `2` | `1.0 / 0.5` | `0.258185` (`-0.0007`) | `6/0/8` | `0.156427` (`+0.0120`) | `8/2/4` | `+0.005647` | `11.1 s` |
| `fe_lr1e-5_s4_ent1.0` | `full_encoder` | `1e-5` | `4` | `1.0 / 0.5` | `0.263123` (`+0.0043`) | `6/1/7` | `0.145720` (`+0.0013`) | `6/2/6` | `+0.002763` | `22.0 s` |
| `fe_lr2e-5_s3_ent1.0` | `full_encoder` | `2e-5` | `3` | `1.0 / 0.5` | `0.269457` (`+0.0106`) | `8/0/6` | `0.146410` (`+0.0020`) | `7/2/5` | `+0.006275` | `16.6 s` |
| `fe_lr3e-5_s2_ent0.0` | `full_encoder` | `3e-5` | `2` | `0.0 / 0.0` | `0.242333` (`-0.0165`) | `6/0/8` | `0.159553` (`+0.0151`) | `7/1/6` | `-0.000716` | `10.9 s` |
| `fe_lr3e-5_s2_ent0.25` | `full_encoder` | `3e-5` | `2` | `0.25 / 0.5` | `0.253509` (`-0.0054`) | `5/0/9` | `0.147498` (`+0.0030`) | `6/2/6` | `-0.001156` | `11.1 s` |
| **`bb_lr3e-5_s2_ent0.0`** | **`backbone_router`** | **`3e-5`** | **`2`** | **`0.0 / 0.0`** | **`0.264767` (`+0.005908`)** | **`8/0/6`** | **`0.171767` (`+0.027308`)** | **`10/1/3`** | **`+0.016608`** | **`10.8 s`** |

### Printed Safety Slice (`n=10` clean + `n=10` photo = 20 pages, top-2 arms)

| Arm Key | `printed_clean` F1 (delta) | `printed_clean` W/T/L | `printed_photo` F1 (delta) | `printed_photo` W/T/L | Safety Gate (`>= -0.005`) |
|---|---|---|---|---|---|
| *Baseline (no TTT)* | `0.573594` (`+0.000000`) | — | `0.559013` (`+0.000000`) | — | — |
| **`bb_lr3e-5_s2_ent0.0`** | **`0.573594` (`+0.000000`)** | **`0/10/0`** | **`0.557844` (`-0.001169`)** | **`0/9/1`** | **PASS** |
| `fe_lr2e-5_s3_ent1.0` | `0.573978` (`+0.000384`) | `1/9/0` | `0.558457` (`-0.000556`) | `1/7/2` | **PASS** |

---

## Scrutiny & Anomaly Check

1. **Coverage & integrity:** Clean — `28/28` HW pages and `20/20` printed safety pages scored at canonical `max_dim=1280, max_len=3840`, `0` OOMs, `0` NaNs, `missing_keys=[]`, `unexpected_keys=[]`, `has_eos=25/28` on both baseline and `bb_lr3e-5_s2_ent0.0`.
2. **Causal isolation confirmed (why `bb_lr3e-5_s2_ent0.0` won decisively):**
   - Pure visual consistency on `full_encoder` (`fe_lr3e-5_s2_ent0.0`) helped `hw_photo` (`+0.0151`) but **regressed `hw_clean` (`-0.0165`)** because updating `enc_layers` without KL/entropy causes semantic feature drift.
   - Restricting pure multi-view visual consistency + L2 anchor (`ent_weight=0.0, kl_weight=0.0, cons_weight=1.0, l2_weight=1e-2`) to **`backbone + router` only** (`bb_lr3e-5_s2_ent0.0`, freezing `enc_layers`) **eliminated `hw_clean` semantic drift (`-0.0165 -> +0.0059`, `8W/0T/6L`)** AND **nearly doubled `hw_photo` gain (`+0.0151 -> +0.0273`, `10W/1T/3L`)** while achieving **zero degradation (`0.0000`, `10/10` ties)** on `printed_clean` and only `-0.001169` (`9/10` ties; single loss `beethoven-grosse-fuge` `-0.0117`) on `printed_photo`.
   - Skipping `ent_weight` also eliminates the slow pre-TTT pseudo-label decode pass, cutting adaptation overhead to `10.8 s` median on T4 FP32 (and `~2–3 s` on L4 / RTX 4080 SUPER BF16).
3. **Top per-piece wins (`bb_lr3e-5_s2_ent0.0`):**
   - `hw_photo`: `hw-debussy-la-plus-que-lente` `0.0000 -> 0.2594` (`+0.2594`), `hw-debussy-lindaraja` `0.2190 -> 0.3347` (`+0.1158`), `hw-beethoven-das-blumchen-wunderhold-d-b` `+0.0367`, `hw-hensel-im-herbste-d-b` `+0.0324`, `hw-mozart-k421` `+0.0260`.
   - `hw_clean`: `hw-debussy-la-plus-que-lente` `0.5925 -> 0.7016` (`+0.1090`), `hw-debussy-lindaraja` `+0.0362`, `hw-hensel-im-herbste-d-b` `+0.0285`.
   - Notable losses: `hw-debussy-vetement-du-blesse` clean (`-0.0822`), `hw-mozart-k387` photo (`-0.0575`), `hw-beethoven-op135` photo (`-0.0514`).
4. **Hardware / platform:** Kaggle kernel `gregoreesmaa/omr-ov-38b` completed cleanly (`KernelWorkerStatus.COMPLETE`, zero lingering compute).

---

## Verdict: **`PROMOTE`**

`bb_lr3e-5_s2_ent0.0` clears **every** promotion gate in `PLAN.md`:
- `hw_photo` delta = **`+0.027308`** (`>= +0.020` threshold), **`10W / 1T / 3L`** (`wins > losses`).
- `hw_clean` delta = **`+0.005908`** (`>= -0.005`, positive!), **`8W / 0T / 6L`**.
- Printed safety = **`0.000000`** (`printed_clean`, `10/10` ties) and **`-0.001169`** (`printed_photo`, `9/10` ties), both well inside `>= -0.005`.

## Follow-Up Action / Successor Experiment

- **Code promotion target:** Wire `bb_lr3e-5_s2_ent0.0` (`scope="backbone_router"`, `lr=3e-5`, `steps=2`, `ent_weight=0.0`, `kl_weight=0.0`, `cons_weight=1.0`, `l2_weight=1e-2`) as the default `--ttt-mode backbone_cons` preset in [`tmp-train/mvd/trial_2028_ttt_adapt.py`](../../tmp-train/mvd/trial_2028_ttt_adapt.py) (with unit tests verifying frozen `enc_layers` + bit-exact CPU weight restore), and expose as an opt-in inference flag for HW/photo pages.
- **Note on `eval_uni_suite31.py`:** Keep standard `eval_uni_suite31.py` baseline unmodified (pure single-pass decode) for apples-to-apples checkpoint comparison; add `--ttt-mode backbone_cons` as an explicit opt-in flag only when clean & unit-tested.

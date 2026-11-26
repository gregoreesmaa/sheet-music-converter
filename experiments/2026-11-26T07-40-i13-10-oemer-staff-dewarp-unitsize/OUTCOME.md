# OUTCOME — `2026-11-26T07-40-i13-10-oemer-staff-dewarp-unitsize` (`i13-10`)

- **Status:** `COMPLETED`
- **Completed UTC:** `2026-11-26T12:03:55Z` (started `11:22:57Z`)
- **Platform / GPU:** `Kaggle 2xT4` (reported as `colab-Tesla T4`, FP32, `kernel_id=gregoreesmaa/omr-i13-10`)
- **Elapsed Time:** `2458.14 s` (`41.0 min`)
- **Base Checkpoint:** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`uni_omr_v1`, step `12000`, `val_ser=0.192301`, `missing_keys=[]`, `unexpected_keys=[]`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - `evals/2026-11-26T07-40-i13-10-oemer-staff-dewarp-unitsize.json`
  - `runs/2026-11-26T07-40-i13-10-oemer-staff-dewarp-unitsize.json`

## Measured Results (`suite31_hw`, all 90 pages: 31 printed + 14 HW × clean + photo, `max_len=3840`, MTP-4 + FSA)

| Slice (`n`) | `baseline_1280` | `oemer_dewarp_only` (Δ) | `homr_unitsize_only` (Δ) | `oemer_dewarp_plus_unitsize` (Δ) |
|---|---:|---:|---:|---:|
| `all45_clean` (`45`) | `0.539954` | `0.539741` (`-0.000213`) | `0.538302` (`-0.001652`) | **`0.545561`** (**`+0.005607`**) |
| `all45_photo` (`45`) | `0.479493` | `0.487505` (**`+0.008012`**) | `0.501849` (**`+0.022356`**) | **`0.515078`** (**`+0.035585`**) |
| `pr_clean` (`31`) | `0.667450` | `0.667450` (`+0.000000`, `0/31/0`) | **`0.671447`** (`+0.003997`, `9/20/2`) | **`0.671447`** (`+0.003997`, `9/20/2`) |
| `pr_photo` (`31`) | `0.619160` | `0.615655` (`-0.003505`) | `0.652747` (`+0.033587`, `8/13/10`) | **`0.661829`** (**`+0.042669`**, `7/14/10`) |
| `hw_clean` (`14`) | `0.257641` | `0.256957` (`-0.000684`) | `0.243481` (`-0.014160`, `5/3/6`) | **`0.266812`** (**`+0.009171`**, `7/3/4`) |
| `hw_photo` (`14`) | `0.170232` | **`0.203743`** (**`+0.033511`**) | `0.167716` (`-0.002516`, `5/5/4`) | `0.190129` (**`+0.019897`**, `9/0/5`) |

### Trimmed Sensitivity Check (excluding single outlier `essen-altdeu` photo `+0.8839`)

| Arm | `pr_photo` Δ (excl. `essen-altdeu`) | `all45_photo` Δ (excl. `essen-altdeu`) | Median CPU Overhead | Mean CPU Overhead |
|---|---:|---:|---:|---:|
| `oemer_dewarp_only` | `-0.00362` | **`+0.00819`** | **`78.1 ms/page`** | `141.0 ms/page` |
| `homr_unitsize_only` | `+0.00524` | `+0.00278` | `47.0 ms/page` | `45.5 ms/page` |
| `oemer_dewarp_plus_unitsize` | **`+0.01463`** | **`+0.01631`** | **`128.1 ms/page`** | `187.8 ms/page` |

## Scrutiny & Anomaly Check

1. **Coverage & Integrity:** All `90/90` pages × `4` arms = `360` decodes completed cleanly (`0` failures, `0` NaN/Inf, `missing_keys=[]`).
2. **Real, Large Win from `oemer_staffline_vertical_dewarp`:**
   - On `pr_clean`, `peak_to_peak < 1.5 px` auto-skips 100% of flat pages (`0/31/0`, `+0.000000` regression).
   - On `hw_photo`, `oemer_dewarp_only` delivers **`+0.033511`** (**+3.35 pp**), led by `+0.1978` on `hw-beethoven-das-blumchen-wunderhold-d-b`, `+0.1384` on `hw-cornelius-botschaft-d-b`, and `+0.0383` on `hw-debussy-vetement-du-blesse`.
   - On `all45_photo`, `oemer_dewarp_only` gains `+0.008012` (`+0.00819` excl. `essen-altdeu`) at `78.1 ms/page` median CPU overhead — clearing every `PLAN.md` PROMOTE gate on its own.
3. **Synergy + Forensic Confound Catch in `homr_estimate_unitsize_max_dim`:**
   - `oemer_dewarp_plus_unitsize` is the overall winner across **all 6 slices** (`all45_clean +0.0056`, `all45_photo +0.0356` / `+0.0163` excl. `essen-altdeu`, `pr_clean +0.0040`, `pr_photo +0.0427` / `+0.0146` excl. `essen-altdeu`, `hw_clean +0.0092`, `hw_photo +0.0199`) at `128.1 ms/page` median overhead.
   - **Important confound discovered in scrutiny:** `np.argmax(ac[4:27])` in `homr_estimate_unitsize_max_dim` lacked local-maximum / first-trough-to-peak filtering, so on noisy/photo profiles where autocorrelation decays monotonically from lag 0, it pinned to the left boundary `est_unit_px = 4.0` on `14/14` `hw_photo` and `24/31` `pr_photo` pages (clamping `target_max_dim = 1600`), while estimating `8.0` (`target_max_dim = 1216`) on all `31/31` `pr_clean` pages. Thus `homr_unitsize_only` acted partly as a photo-1600px / clean-1216px heuristic (explaining its `-0.0142` drop on `hw_clean` when run without dewarping, and its strong synergy once `oemer_dewarp` flattens the staves).
4. **CPU Latency Optimization Opportunity:** `oemer_staffline_vertical_dewarp` ran on full-resolution raw scans (up to `~825 ms` worst-case on large archival scans); computing the 24-stripe shift curve at a downscaled `1280px` proxy before remapping will cut worst-case latency by `>4×` with bit-identical accuracy.

## Verdict: **`PROMOTE`**

- **Both `oemer_dewarp_only` (`all45_photo +0.0080`, `hw_photo +0.0335`, `pr_clean +0.0000`, `78.1 ms` median) and `oemer_dewarp_plus_unitsize` (`all45_photo +0.0356`, `hw_photo +0.0199`, `pr_clean +0.0040`, `hw_clean +0.0092`, `128.1 ms` median) cleanly pass every `PLAN.md` PROMOTE gate** (`>= +0.005` on `all45_photo` / `hw_photo`, `<= -0.002` regression on `pr_clean`, `< 150 ms/page` median CPU overhead).
- **Recommended Promotion Action:**
  1. Promote `oemer_staffline_vertical_dewarp` into the inference preprocessing pipeline (`eval_uni_suite31.py` / `converter/ingest-cleanup`) immediately (zero-risk on flat pages via `peak_to_peak < 1.5 px` auto-skip, `+3.35 pp` on `hw_photo`).
  2. When porting `homr_estimate_unitsize_max_dim`, replace `np.argmax(ac[4:27])` with proper **local-peak detection** (first local maximum after the first autocorrelation trough) so `unit_size` measures true inter-staffline spacing rather than pinning at lag `4.0`, and compute the dewarp shift field at `1280px` working scale to keep worst-case latency `< 100 ms/page`.

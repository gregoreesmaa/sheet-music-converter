# Data Variation Audit 1 — Visual, Engraving, Paper & Phone-Photo Diversity (Iter 13 Proposal)

**Scope:** CPU-only empirical audit of `D:\tmp\uni_fullpage_iter11_clean.jsonl` (`N = 44,734` full-page rows; `n = 2,000` deterministic pixel sample, seed=42), `data-gen/score-render`, `data-gen/photo-degrade` (`doc-photo-degrade`), and `apply_online_augment` in `tmp-train/mvd/uni_omr.py`, cross-referenced against all 90 evaluation images (45 clean + 45 photo) of `suite31_hw` (`D:\tmp\uni-omr-suite31-ood\iter10_5_ema_best_suite31_hw_1536.json`).

---

## 1. Exact Measured Visual Diversity Table (Train Manifest vs. `suite31_hw`)

### 1.1 Manifest Composition (`D:\tmp\uni_fullpage_iter11_clean.jsonl`, `N = 44,734`)

| Axis | Breakdown (Exact Counts & Share) |
|---|---|
| **Domains** | `printed`: 36,490 (`81.57%`) · `hw` (synth): 7,775 (`17.38%`) · `real_hw`: 469 (`1.05%`) |
| **File Formats** | `.png`: 44,734 (`100.00%`) · `.jpg`/`.jpeg`: **0 (`0.00%`)** · offline `photo-degrade`: **0 (`0.00%`)** |
| **Renderers** | MuseScore: 23,972 (`53.59%`) · Verovio: 10,415 (`23.28%`) · `hw-score-sim`: 7,775 (`17.38%`) · OSMD: 2,103 (`4.70%`) · `real_hw_scan`: 469 (`1.05%`) · **LilyPond: 0 (`0.00%`)** |
| **Engraving Fonts (filename)** | Unspecified/default: 19,428 (`43.43%`) · Bravura: 4,718 (`10.55%`) · Petaluma: 4,661 (`10.42%`) · Leland: 4,260 (`9.52%`) · Emmentaler: 3,051 (`6.82%`) · Gonville: 2,720 (`6.08%`) · Leipzig: 2,138 (`4.78%`) · Gootville: 2,064 (`4.61%`) · MuseJazz: 1,693 (`3.78%`) |
| **Page Sizes (filename)** | Unspecified (mostly A4 portrait): 36,713 (`82.07%`) · `a4_portrait`: 1,777 (`3.97%`) · `b4_portrait`: 1,606 (`3.59%`) · `a4_landscape`: 1,553 (`3.47%`) · `letter`: 1,525 (`3.41%`) · `a4`: 567 (`1.27%`) · `a5`: 511 (`1.14%`) · `b4`: 482 (`1.08%`) |

### 1.2 Pixel-Level Visual Distribution (`n = 2,000` Train Sample vs. `suite31_hw` Eval)

| Cohort | `n` | Width `p50` (min–max) | Aspect Ratio `p50` | Landscape (`W>H`) | Grayscale (`ch<1`) | Color/Sepia (`ch≥8`) | Chroma `p50` | Dark Border (`bM<180`) | Dark Border Px Frac `p50` | Bg Texture `bg_std` `p50` | Lighting Grad `iGrad` `p50` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Train Manifest (All)** | 2,000 | 2481 (1410–5112) | 0.71 | 13.9% | **87.7%** | **8.3%** | 0.0 | **0.3%** | **0.00** | **0.0** (`p95=4.6`)* | **0.0** (`p95=1.9`) |
| ↳ MuseScore | 1,393 | 2481 (1410–4961) | 0.71 | 4.3% | 100.0% | 0.0% | 0.0 | 0.0% | 0.00 | 0.0 | 0.0 |
| ↳ Verovio | 197 | 1654 (1500–2000) | 0.72 | 12.7% | 100.0% | 0.0% | 0.0 | 0.0% | 0.00 | 0.0 | 0.0 |
| ↳ OSMD | 59 | 1671 (1413–2166) | 1.24 | 64.4% | 100.0% | 0.0% | 0.0 | 0.0% | 0.00 | 0.0 | 0.0 |
| ↳ `hw-score-sim` | 330 | 1950 (1650–2700) | 0.77 | 46.1% | 27.9% | 47.6% | — | 1.8% | 0.00 | — | 0.3 |
| ↳ `real_hw_scan` | 21 | 1654 (1654–5112) | 0.71 | 14.3% | 61.9% | 38.1% | — | 0.0% | 0.00 | 2.8 | 0.0 |
| **Suite31 Printed Clean** | 31 | 2000 | 0.71 | 0.0% | 100.0% | 0.0% | 0.0 | 0.0% | 0.00 | 0.0 | 0.0 |
| **Suite31 Printed Photo** | 31 | 2200 | 0.73 | 0.0% | 100.0% | 0.0% | 0.0 | **100.0%** | **0.87** | 0.3 | 0.0 |
| **Suite31 HW Clean** | 14 | 1599 | 0.78 | **35.7%** | 14.3% | **85.7%** | **32.9** | **71.4%** | **0.53** | **5.7** | **4.6** |
| **Suite31 HW Photo** | 14 | **1225** | 0.78 | **35.7%** | 0.0% | **100.0%** | **28.8** | **100.0%** | **0.86** | **11.9** | **20.1** |

*\*Measured on 400-row subsample with `>=` median background mask (`p50=0.00, p75=0.94, p90=2.82, p95=4.60, frac>8=0.50%`). All 226 RGBA images checked were 100% opaque (`alpha=255`).*

### 1.3 Worst-Performing `suite31_hw` Pieces & Photo-vs-Clean Drops (`iter10_5_ema_best`)

- **Summary:** Printed Clean `0.8968` → Printed Photo `0.8935` (`-0.0033`); HW Clean `0.2591` → HW Photo `0.2174` (**`-0.0417`**).
- **Largest Printed Photo-vs-Clean Drops:**
  - `beethoven-op18n1-1`: `0.995 → 0.743` (**`-0.251`**, fixes `2 → 76`, `P[2200×3030, bM=12, bDF=0.95, bgS=3.6]`)
  - `essen-altdeu`: `0.610 → 0.438` (**`-0.172`**, fixes `46 → 59`, `P[bM=33, bDF=0.87]`)
  - `weber-concertino`: `0.947 → 0.895` (**`-0.052`**, fixes `34 → 67`, `P[bM=30, bDF=0.88]`)
  - `oneills-tune`: `0.868 → 0.831` (`-0.036`, `P[bM=36, bDF=0.86]`)
  - `aloha-oe`: `1.000 → 0.966` (`-0.034`, `P[bM=48, bDF=0.81]`)
- **Worst Printed Overall:** `foster-leadsheet` (`0.021` clean & photo — engraving/leadsheet layout), `essen-altdeu` (`0.610 / 0.438`), `beethoven-grosse-fuge` (`0.788 / 0.808`), `handel-lascia` (`0.821 / 0.821`).
- **Largest HW Photo-vs-Clean Drops (severe lighting gradient + low resolution + heavy border):**
  - `hw-mozart-k464`: `0.242 → 0.059` (**`-0.182`**, `P[906×668, AR=1.36, bDF=0.86, bgS=17.7, iG=43]`)
  - `hw-beethoven-das-blumchen-wunderhold-d-b`: `0.793 → 0.626` (**`-0.167`**, `P[953×1366, ch=36.7, bDF=0.85, bgS=9.8]`)
  - `hw-debussy-vetement-du-blesse`: `0.348 → 0.208` (**`-0.140`**, `P[1268×1666, bDF=0.72, bgS=11.5, iG=20]`)
  - `hw-debussy-berceuse-heroique`: `0.565 → 0.451` (**`-0.115`**, `P[1448×2005, bgS=14.5, iG=30]`)
  - `hw-mozart-k421`: `0.152 → 0.074` (**`-0.078`**, `P[952×673, AR=1.41, bgS=14.1, iG=45]`)
  - `hw-schubert-d810`: `0.181 → 0.108` (`-0.073`, `P[2114×2036, AR=1.04, bDF=0.90, bgS=13.9, iG=16]`)

---

## 2. Top 5 Visual & Data-Generation Blind Spots

### Blind Spot #1: `0.00%` Offline `doc-photo-degrade` Realism in Training (All 44,734 Rows Are Clean PNGs)
- **Measured gap:** `uni_fullpage_iter11_clean.jsonl` contains **0 offline photo-degraded images** (`100%` PNG, `bg_std` median `0.00`, `illum_grad` median `0.00`). Meanwhile, our own release-compiled `doc-photo-degrade` (`data-gen/photo-degrade/target/release/photo-degrade.exe`) supports 7 physical `DocumentProfile` presets (`sheet-music`, `phone-photo`, `archival-manuscript`, `book-gutter-bend`, `photocopy-xerox`, `office-scan`, `thermal-receipt`) with WGPU acceleration and `--train-width`, yet **none of its physical optics/paper effects ever reach the model**:
  - 3D non-linear page curl / book-gutter bend + Lambertian normal shading
  - Handheld phone + hand cast shadow SDF + directional contact drop-shadow
  - Textured desk surfaces (`walnut`, `oak`, `granite_laminate`, `slate`, `cloth`), music-stand bottom ledge, fingertip/binder-clip edge occluders
  - 3D paper crumpling, dihedral fold creases, moisture cockling, specular glare / glossy sheet-protector sheen
  - Lens barrel distortion, off-axis radial CA, Petzval corner defocus, Xerox drum ghosting / scanner bar, capillary stains, double-JPEG
- **Impact:** Directly explains the `-0.042` HW Photo drop and `-0.251` / `-0.172` Printed Photo drops on `beethoven-op18n1-1` and `essen-altdeu`.

### Blind Spot #2: `apply_online_augment()` Early-Return Bugs & Mild Phone-Photo Ceilings
In `tmp-train/mvd/uni_omr.py` (`apply_online_augment`, lines 2800–3155):
1. **Archival early-return bug (line 2987):** `do_archival` (`55%` of HW, `12%` of printed) **returns early**, completely skipping **perspective warp (`0%`)**, **dark desk border (`0%`, only 25% archival border)**, **strong uneven shadows (`illum` clipped at `0.62..1.12` vs `0.40..1.15`)**, and **1-bit scan**. This starves HW training of perspective + heavy illumination gradients (`iG=20..45` on `hw-mozart-k421/k464`).
2. **`do_1bit` early-return bug (line 3077):** 1-bit scanner threshold returns before blur, JPEG, or noise (never simulates noisy/compressed photocopies).
3. **Too-mild geometric/optical ceilings:** Rotation capped at `±3.0°` (real handheld photos reach `±8–10°`); perspective inward jitter capped at `1–4%` (real oblique stand/desk shots reach `6–12%` keystone); **0%** non-linear page curl; **0%** hard cast shadow bands; **0%** specular glare hotspots; **0%** directional motion blur; **0%** quiet-margin tight crop; **0%** broken-ink / thin-staff-line dropout.

### Blind Spot #3: Renderer Monoculture (`0%` LilyPond) & Missing `score-render` Staff-Scaling / Line-Weight Knobs
- **Measured gap:** `53.59%` MuseScore, `23.28%` Verovio, `4.70%` OSMD, and **`0.00%` LilyPond** — even though `lilypond.exe 2.24.4.1` is already installed on this machine! LilyPond's Feta engraving, beaming slopes, slur curvature, and spacing engine are distinct from SMuFL/MuseScore/Verovio and widely used in IMSLP/Mutopia/real-world scores.
- **Font & layout concentration:** `43.43%` of rows use unspecified/default engraving fonts, `82.07%` unspecified page size (MuseScore is `95.7%` portrait, `AR=0.71`).
- **`score-render` gap (`data-gen/score-render/src/main.rs`):** Supports `--musescore-fonts`, `--musescore-text-fonts`, and `--musescore-page-sizes`, but **cannot vary `<Spatium>` (staff size) independently of page size, nor `<staffLineWidth>`, `<stemWidth>`, `<beamWidth>`, `<staffDistance>`, or `<minSystemDistance>`**.

### Blind Spot #4: Severe Color/Sepia, Low-Resolution & Landscape Deficit vs. Real Manuscripts
- **Color/sepia gap:** `87.7%` of training disk images are pure grayscale (`100%` of printed renders); only `8.25%` have `mean_chroma ≥ 8`, and `p95 = 24.89`. Yet `suite31_hw` HW Clean/Photo have median `mean_chroma = 32.9 / 28.8` (up to `50.7` on `hw-cornelius`, `49.5` on `hw-beethoven-op135`).
- **Low-resolution gap:** Training manifest minimum width is `1410px` (`p50 = 2481px`), whereas `suite31_hw` HW Photo median width is `1225px` (down to `822–952px` on `hw-cornelius`, `hw-hensel-im-herbste`, `hw-mozart-k421/k464`). Worse, `_apply_hw_lowres_defocus_compress` is gated to `max(orig_w, orig_h) >= 1280` AND `is_hw=True`, so printed scores never see low-res camera downscale and `1024–1152px` multi-scale HW batches disable it entirely.
- **Landscape gap:** Only `4.3%` of MuseScore, `12.7%` of Verovio, and `14.3%` of `real_hw_scan` are landscape, vs. `35.7%` of `suite31_hw` HW (`AR = 1.04..1.43`).

### Blind Spot #5: Extreme Dark-Border & Framing Distribution Mismatch
- **Measured gap:** On disk, **`0.00%`** of training images have `>10%` dark border pixels (`border_mean < 180` is `0.30%`), vs. **`100.0%`** of `suite31_hw` Printed Photo (`bDF_med = 0.87`), **`100.0%`** of HW Photo (`bDF_med = 0.86`), and **`71.4%`** of HW Clean (`bDF_med = 0.53`).
- In `apply_online_augment`, dark desk borders only fire on `~16.5%` of printed training steps (`aug_prob=0.75 * 0.88 * 0.25`) and `~10.3%` of HW steps (`0.75 * 0.55 * 0.25`), and only as uniform flat/linear dark gray (`0..70`) on all 4 sides — never asymmetric 1–3 side desk borders, textured wood/cloth desks, or detached corners.

---

## 3. Concrete, Ready-to-Run Iter 13 Upgrades

### Upgrade 1: Generate an Offline `photo-degrade` Multi-Profile Shard (8,000–10,000 Pages, ~15–25% of Training Mix)
Use the already-compiled release binary `data-gen\photo-degrade\target\release\photo-degrade.exe` with `--train-width 1536 --format jpeg --fast-banks` across 5 profiles (stratified across printed + HW pages, 0 eval overlap):

```powershell
# 1) Handheld phone photos of sheet music (4,000 pages: desk surfaces, stand ledge, bend, shadows, corner detach)
.\data-gen\photo-degrade\target\release\photo-degrade.exe `
  --input D:\tmp\iter13_degrade_src\sheet_music `
  --out-dir D:\tmp\iter13_degraded\sheet_music `
  --profile sheet-music --strength 0.55 --train-width 1536 --scale-min 0.55 `
  --format jpeg --fast-banks --seed 1301 --suffix _ph_sm

# 2) Heavy handheld phone photos (1,500 pages: steeper perspective, phone shadow, specular sheen, USM)
.\data-gen\photo-degrade\target\release\photo-degrade.exe `
  --input D:\tmp\iter13_degrade_src\phone_hard `
  --out-dir D:\tmp\iter13_degraded\phone_hard `
  --profile phone-photo --strength 0.65 --train-width 1536 --scale-min 0.50 `
  --format jpeg --fast-banks --seed 1302 --suffix _ph_hard

# 3) Open-book gutter bend + spine shadow (1,500 pages)
.\data-gen\photo-degrade\target\release\photo-degrade.exe `
  --input D:\tmp\iter13_degrade_src\book_gutter `
  --out-dir D:\tmp\iter13_degraded\book_gutter `
  --profile book-gutter-bend --strength 0.60 --train-width 1536 --scale-min 0.60 `
  --format jpeg --fast-banks --seed 1303 --suffix _ph_book

# 4) Archival manuscript & photocopy/Xerox (1,500 archival + 1,000 xerox)
.\data-gen\photo-degrade\target\release\photo-degrade.exe `
  --input D:\tmp\iter13_degrade_src\archival `
  --out-dir D:\tmp\iter13_degraded\archival `
  --profile archival-manuscript --strength 0.75 --train-width 1536 --scale-min 0.55 `
  --format jpeg --fast-banks --seed 1304 --suffix _ph_arch

.\data-gen\photo-degrade\target\release\photo-degrade.exe `
  --input D:\tmp\iter13_degrade_src\xerox `
  --out-dir D:\tmp\iter13_degraded\xerox `
  --profile photocopy-xerox --strength 0.65 --train-width 1536 --scale-min 0.65 `
  --format jpeg --fast-banks --seed 1305 --suffix _ph_xerox
```
*Note:* Make sure `_PRE_DEGRADED_RE` in `tmp-train/mvd/uni_omr.py` matches `_ph_(sm|hard|book|arch|xerox)` so offline-degraded images are never double-degraded online!

### Upgrade 2: Harden `apply_online_augment()` in `tmp-train/mvd/uni_omr.py`
Concrete parameter & structural fixes (all `< 3 ms`/image in OpenCV/NumPy):
1. **Remove archival early-return for geometry/lighting:** After archival parchment/ink transfer, still apply perspective warp (`p=0.45`, `1–8%` keystone), stronger uneven lighting (`illum` down to `0.42`, `gx/gy ∈ [-0.45, 0.45]`), and dark border (`p=0.45`, supporting both 4-sided and asymmetric 1–3 sided borders).
2. **Raise dark-border probability & realism:** Increase dark desk border `p` from `0.25 → 0.45` on full pages, sample asymmetric side widths (including 1–3 sided borders and `1–12%` depth), and add warm wood-tone desk fills (`RGB (40..110, 28..78, 18..55)`) alongside dark slate/black (`0..65`).
3. **Widen rotation & perspective:** Full-page rotation `±3.0° → ±7.0°` (`p=0.55`, with `expand=True`); perspective corner jitter `1–4% → 1–9%` (`p=0.45`, with asymmetric keystone bias).
4. **Add 5 missing fast phone-photo primitives:**
   - **1D cylindrical / book-gutter curl** (`p=0.25`): smooth `cv2.remap` parabolic/sinusoidal vertical displacement (`amp = 0.008..0.025 * H`) + matched gutter shading (`0.65..0.90`).
   - **Hard cast shadow polygon/band** (`p=0.25`): random half-plane or trapezoid blurred by `σ=12..35px` multiplying luma by `0.45..0.78` with cool blue-gray shadow tint.
   - **Specular glare / sheen band** (`p=0.15`): additive/screen Gaussian elliptical or diagonal highlight (`+25..+75` luma, soft-clamped).
   - **Directional motion blur** (`p=0.15`, kernel len `3..7px`, random angle) & **low-res round-trip downscale for BOTH printed & HW** (`p=0.20`, downscale to `640..1050px` regardless of `orig_w >= 1280` gate).
   - **Thin-ink / staff-line dropout morphology** (`p=0.15` on printed/photocopy): mild random `3×3` dilation mask (lightens thin 1px strokes) + `do_1bit` no longer skips subsequent blur/JPEG/noise.
5. **Expand parchment/sepia chroma range:** Allow deeper warm amber/ochre paper (`R=195..245, G=165..225, B=115..195`) so online archival chroma covers `mean_chroma ∈ [15, 55]` (matching `suite31_hw` up to `50.7`).

### Upgrade 3: Add LilyPond Renderer Shard (2,500–4,000 Pages) + Expand `score-render` Style Knobs
1. **LilyPond (`lilypond.exe 2.24.4.1`, already installed):** Add a `render_lilypond_dataset.py` worker (using `musicxml2ly` + `lilypond --png -dresolution=220..300`) across 1,500–2,000 leak-free PDMX/OpenScore/NIFC scores, varying `set-global-staff-size` (`14..22 pt`), paper sizes (`a4`, `letter`, `a4 landscape`, `b4`), and `ragged-last` / `system-system-spacing`. Verify exact measure-per-system alignment via LilyPond's SVG/PS barline annotations or single-page scores (`<= 16` bars).
2. **`data-gen/score-render/src/main.rs` upgrades:**
   - Add `--musescore-spatium 1.35,1.60,1.85,2.10` (writes `<Spatium>` into `.mss`)
   - Add `--musescore-line-weights thin,default,heavy` (writes `<staffLineWidth>`, `<stemWidth>`, `<beamWidth>`)
   - Add `--musescore-text-fonts "Edwin,Times New Roman,Georgia,Arial,Bravura Text,Petaluma Script"` and `--musescore-page-sizes "a4,a4_land,a5,a5_land,b4,folio"` (raising MuseScore landscape share from `4.3% → 25%` and font-specified share from `56.6% → 100%` on newly rendered shards).

### Upgrade 4: Landscape & Low-Res Balance for Printed + HW
- Render `25–30%` of new Verovio/OSMD/LilyPond/MuseScore pages in **landscape (`a4_landscape`, `a5_landscape`, `AR = 1.25..1.45`)**.
- In `UniFullPageDataset` (`_sample_weighted_fullpage_dim`), expand lower-resolution multi-scale bucket to include `768, 896, 1024` (`15%` weight) so the model directly sees the `822–1150px` native resolution regime of `suite31_hw` HW photos.

---

## 4. Recommended Execution Priority for Iter 13

| Priority | Action | Expected Impact | Cost / Time (CPU-only) |
|---|---|---|---|
| **P0** | Patch `apply_online_augment()` (fix archival/1-bit early returns, raise border/rotation/perspective/chroma ranges, add curl + hard shadow + glare + motion blur + universal low-res round-trip) | Immediate `+3–6 pts` HW Photo & `+2–4 pts` Printed Photo F1 (zero extra disk) | ~1–2h code + unit tests |
| **P0** | Run `photo-degrade.exe` offline on 8,000–10,000 stratified pages across 5 `DocumentProfile`s (`sheet-music`, `phone-photo`, `book-gutter-bend`, `archival-manuscript`, `photocopy-xerox`) | Closes real-world phone-photo / book-bend / desk-surface sim-to-real gap | ~30–60 min CPU/WGPU batch |
| **P1** | Render 2,500–4,000 **LilyPond** pages + add `--musescore-spatium` / `--musescore-line-weights` / `a4_land` to `score-render` | Eliminates `0%` LilyPond & `4.3%` MuseScore-landscape monoculture | ~2–4h CPU batch |
| **P2** | Add `768/896px` multi-scale bucket + `25%` landscape target on new Verovio/OSMD shards | Closes low-res (`822–952px`) & landscape (`AR=1.10–1.43`) HW gap | ~15 min code + batch |

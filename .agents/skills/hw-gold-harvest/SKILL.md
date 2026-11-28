---
name: hw-gold-harvest
description: Mandatory pipeline for matching real historical handwritten manuscripts (RISM/IIIF/archival) to digital scores and promoting verified GOLD matches into single-page OMR training data. Activate whenever running fingerprint/PAE manuscript matching OR whenever new GOLD matches are detected — GOLD matches MUST be run through this promotion pipeline before ever entering a training manifest.
---

# Real-HW GOLD Matching & Single-Page Promotion Pipeline

Real historical handwritten manuscripts are our highest-leverage training data for closing the `suite31_hw` / `OMR-NorthStar` HWM gap — **but** unverified or unsliced matches poison OMR training (e.g., multi-page labels paired with single pages, 1-staff partbooks paired with 4-staff scores, title pages, or transposed/re-arranged editions).

**Standing Rule:** Whenever GOLD matches are detected (via RISM PAE opening incipits, fingerprint retrieval, or targeted Digital→RISM IIIF API lookups), they **MUST** pass every gate of this promotion pipeline before being merged into any training manifest.

---

## Stage 1 — High-Precision GOLD Discovery (CPU)

1. **Corpus Index (`D:\tmp\hw-align\digital_opening_100k_index.pkl.gz`):**
   - Index all properly labeled digital works (`bekern-v2a-cache`, `corpus100k`, `corpus10x`, `openscore-*`, `nifc-*`).
   - **Mandatory Exclusions:** `hwsim_*`, `hw_sim_*`, `synth_*`, quarantined `corpus10x-NC` (`CC-BY-NC`), and every normalized slug in `suite31_hw_manifest.json` + `OMR-NorthStar v0.1` + `OMR-NorthStar v0.2` (plus `hw-mozart-k458`).
2. **GOLD Match Gate (Zero-Noise PAE Incipit or Attractor-Guarded Fingerprint):**
   - **Anti-ostinato / entropy filter:** Reject incipits/queries with `unique_pitches < 4` or `nonzero_interval_frac < 0.45` (eliminates repeated-note bassline/ostinato collisions).
   - **Exact notation parity:** Require `work_key == query_key`, `norm_time(work_time) == norm_time(query_time)` (allowing `2/2` (`c/`) ↔ `4/4` (`c`) equivalence), and `delta_midi == 0` (exact non-transposed pitch).
   - **Alignment & rarity thresholds:** `n_notes >= 8`, `interval_f1 >= 0.85`, `rhythm_f1 >= 0.75`, and IDF-weighted `vote_score >= 3.0`.
   - **Attractor guard (for OMR-decoded queries):** Exclude synthetic prefixes at retrieval, reject duplicate normalized decode hashes (`> 1` distinct image), and reject attractor works (`> 2` distinct query images hitting the same `work_id`).
   - **Keep SILVER out:** Empirical eyeball audits showed ~67% of SILVER (transposed/time-mismatched) hits are false positives; do **not** promote SILVER matches in bulk.

---

## Stage 2 — Visual Canvas & Physical Staff-Count Gate (Mandatory)

PAE/catalog matching identifies the *work*, **not** the downloaded canvas (a canvas can be a title page, blank verso, interior movement, or single-staff partbook).

1. **Non-music / Title-Page Rejection:** Run `neg_control_gate.is_non_music_page` + populated-staff note-ink density check. Reject title pages, covers, and blank ruled sheets immediately.
2. **Physical `visual_n_staves_per_system` Parity:** Detect physical staff lines + inter-staff vertical barline/brace continuity (`visual_n_staves`, `visual_n_systems`) independently of OMR token decodes.
   - **Hard Veto:** Require `visual_n_staves == work.n_staves` and `visual_n_systems >= 1`.
   - Rejects 1-staff Violin/Cello partbooks matched to multi-staff full scores and rejects staff-reduced digital arrangements (e.g., 6-staff MS vs 5-staff PDMX reduction).

---

## Stage 3 — Exact Single-Page `[m_start, m_end)` & `<sys>` Forced Alignment (Never Multi-Page!)

Every genuine opening manuscript page typically fits **fewer measures and different system breaks** than the digital score's page 1 (e.g., `13` bars `2+3+2+2+2+2` vs digital's `22` bars). **NEVER pair a multi-page digital label with a single manuscript page.**

1. **Forced Alignment:** Decode each Stage-2 survivor canvas with our best UniOMR checkpoint (`--no-cudnn` on Windows, `use_grammar=True`) and run `fingerprint_harvester.align_page_to_work()` against **only** its verified GOLD `work_id` to find exact single-page `[m_start, m_end)` and per-system measure breaks `[(s0, e0), (s1, e1), ...]`.
2. **Single-Page Boundary Gate:**
   - Require `len(system_breaks) == visual_n_systems` (exact system-count match),
   - Contiguous non-overlapping spans (`e_i == s_{i+1}`, each `>= 1` measure, `m_end > m_start`),
   - `align_score >= 0.45` (and optionally Teacher-Forced `PMI_pitch > 0`, `Δ_transp > 0`, and 2D cross-attention reading-order monotonicity `M_attn` when verifying interior/non-opening canvases).
3. **Exact Single-Page Label Export:**
   - Slice the digital work's MusicXML per system via `assembly-serialize.exe bekern --v2 --measures S..E` (`bekern_version="v2a"`) or per-measure bekern/kern slice (`bekern_version="v1"`).
   - Validate `0` OOV tokens against `data.VOCAB`, `PrefixState(strict=False)` grammar sanity, and `0` eval overlap.
   - Save normalized `1536px` JPEG Q95 pages + per-system `.txt` labels + `manifest.jsonl` (`domain="real_hw"`, `source="rism_gold_aligned_*"`).

---

## Stage 4 — Spot Eyeball QA & Manifest Promotion

1. **Visual Eyeball Spot-Check:** Open a sample of promoted pages (`read_file` on `.jpg` + sliced per-system `.txt`) to verify exact measure count, system count, key/clef/time, and check for manuscript-specific shorthands (e.g., beat-repeat slashes `//` vs written-out arpeggios, `8va` vs ledger lines).
2. **Promote into Training Manifest:** Only after Stages 1–4 pass, merge `D:\tmp\rism-gold-aligned-*\manifest.jsonl` into the active full-page training manifest (e.g., `build_iter14_manifest.py` / Iter 15+ manifest) and re-verify `0` eval overlap.
3. **Multi-Canvas Expansion (Optional Follow-Up):** For every verified GOLD `work_id`, fetch remaining IIIF canvases (`total_canvases`) and run Stages 2–4 locked to that known `work_id` to multiply clean real-HW pages by 5–10×.

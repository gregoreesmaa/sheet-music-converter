# `2026-11-27T07-52-i15-06-bekern3-visual-sp-chord-polyphony` — `bekern3` Core Visual-Pitch, Rhythm & Polyphony

- **Experiment ID:** `2026-11-27T07-52-i15-06-bekern3-visual-sp-chord-polyphony`
- **Short ID / Wave:** `i15-06` (Wave 13 — `bekern3` Architect 1: Core Visual-Pitch, Rhythm & Polyphony)
- **Status:** `QUEUED` (Pending merge of `queue_entry.json` into `experiments/overnight_queue/queue.json`)
- **Baseline Commit SHA:** `e426c28`
- **Baseline Architecture:** `UniOMRv2Net(small_45m)` (`~43M` params, GQA `8Q/2KV`, DeepStack, MTP-4)
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, BF16 native, est. **~65 min / ~4.8 CU**)
- **Datasets (100% permissive, 0 `CC-BY-NC`, 0 eval overlap):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`proxy_train` + `proxy_val`)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 31 printed + 14 real historical handwritten manuscripts, `clean` + `photo` = 90 pages)

---

## Summary & Hypothesis

`bekern+v2` Phase A (`vocab_size = 265`) still carries three compounding structural inefficiencies that hurt both **handwritten generalization (`HWM`)** and **MTP-4 decoding throughput**:
1. **Sounding-pitch gauge entanglement:** Emitting `<STEP> <OCT> [a±1|2]` forces the visual decoder to perform long-range mental transposition across clefs, key signatures, and bar-scoped accidentals — a single clef/key misread triggers a system-wide transposition cascade.
2. **Verbose chord & lagged-dot serialization:** Emitting `dot` *after* `<STEP> <OCT> [a±1|2]` forces lag state across 2–4 tokens, and chords repeat `chord <DUR>` on every chord tone (`quarter C 4 chord quarter E 4 chord quarter G 4` = 11 tokens for a triad).
3. **Arbitrary voice IDs & closed-set meter/tuplet OOVs:** `voice-1..voice-27` leak arbitrary MusicXML XML-ID numbering into the visual target, while closed enumerations of `time-N-D` and `tuplet-start-A-N` bloat the vocabulary and still trigger OOVs on dense archival scores.

**Hypothesis (`bekern3` Architect 1 — Core Visual-Pitch, Rhythm & Polyphony):**
Replacing `bekern+v2` with **`bekern3_core`** — combining (1) visual staff-positions `sp-14..sp+20` (+ octave-shift escape for extreme ledger lines) & 5 printed-only `acc-*` tokens with a deterministic `Ratio<i64>` (`Fraction`) pitch/accidental/beat resolver, (2) duration-first implicit chords + fused dotted durations (`quarter.`, `eighth..`, etc.), and (3) descending-`mean(sp)` polyphonic `<v>` + `skip <DUR>` with compositional rational `d0..d9 /` tuplets and meters — in a matched 2-arm `1,200`-step Colab L4 ablation on `UniOMRv2Net(small_45m)` will:
1. Pass **`>= 99.5%`** deterministic MusicXML round-trip Oracle fidelity on `suite31_hw`,
2. Achieve **`>= 30%`** exact token-length reduction across `proxy_train`, `proxy_val`, and `suite31_hw`, and
3. Deliver **`>= +3.0 pp`** `suite31_hw` handwritten (`HWM`) Note F1 gain **OR** preserve printed Note F1 (`>= -0.5 pp`) with **`>= +30%`** effective notes/sec MTP-4 speculative decoding throughput.

---

## Quick Links

- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Queue Entry (pending merge into `overnight_queue/queue.json`): [queue_entry.json](./queue_entry.json)
- Outcome *(written upon completion)*: `OUTCOME.md`
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T07-52-i15-06-bekern3-visual-sp-chord-polyphony.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-27T07-52-i15-06-bekern3-visual-sp-chord-polyphony.json`

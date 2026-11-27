# `bekern-full` Specification & Rollout Architecture

**Status:** Hardened v1.0 Specification (incorporating all 10 red-team mitigations)  
**Target Models:** `UniOMR v2` (`edge85m` `< 55 MB` WebGPU & `pro200m` server/desktop)  
**Downstream Serializer:** `mscz-rs` (`bekern` → MusicXML 3.1 / MuseScore 4 `.mscz`)  
**Vocab-Safety Foundation:** [`tmp-train/mvd/uni_omr_v2.py`](../tmp-train/mvd/uni_omr_v2.py)

---

## 1. Executive Summary & Design Goals

`bekern-full` extends UniOMR's single-stage full-page `bekern + <sys>` sequence format into a **4-layer, lossless, multi-instrument music notation representation** covering orchestral/chamber layouts, microtones, guitar tablature, percussion, overlapping expressive spanners, and vocal/choral/lead-sheet text — while strictly preserving:

1. **Zero regression on handwritten manuscripts (`HWM >= 24.83` floor)** and legacy `v1`/`v2a` benchmarks.
2. **Strict `< 55 MB` Hybrid FP16/INT4-QAT WebGPU budget** on `edge85m` (Layer 3 text adds only **`+0.34 MB`** via UTF-8 byte-level `ModeGate` dual head).
3. **Zero vocab-expansion corruption** when warm-starting from legacy `vocab_size = 265` checkpoints into expanded `bekern-full` vocabularies (`pad_id`, `bos_id`, `eos_id`, `sys_id`, `music_vocab_end` explicitly persisted in `UniOMRv2Config`).

### The 10 Red-Team Mitigations

| # | Red-Team Failure Mode | Concrete Hardened Mitigation in `bekern-full` |
| :-: | :--- | :--- |
| **1** | **Iter 14's `43.9%` false-negative poisoning & `HWM` regression** (missing expressive/text annotations in real HW corpora penalized as false negatives) | Explicit `<schema:v1>` / `<schema:v2a>` / `<schema:full>` bitmask prefix + **unannotated-layer CE loss masking** on every batch row. |
| **2** | **Hidden vocab-expansion landmine (`v-4, v-3, v-2`)** in `generate()`, `tok_dropout`, `tok_emb`, and `compute_loss` | Explicit optional `pad_id`, `bos_id`, `eos_id`, `sys_id`, `music_vocab_end` in `UniOMRv2Config` (100% backward-compatible `None` defaults for legacy checkpoints). |
| **3** | **Inline lyric interleaving breaking rhythmic FSA & note alignment** | Lyrics relocated to end-of-system `<sys_lyr v=N>` blocks with explicit note-index anchors (`@i`) + `lyr-beg/mid/end/ext`. |
| **4** | **Text vocabulary bloat blowing the `< 55 MB` `edge85m` WebGPU budget** | Physically separate **`ModeGate` dual head**: UTF-8 byte-level (`<= 256` tokens, **`+0.34 MB`**) on `edge85m`; optional `2k` BPE on `pro200m`. |
| **5** | **Hallucinated text / lyrics on noisy or handwritten scans** | **Confidence-gated text emission (`p >= 0.75`)** on `ModeGate` entry and text spans; low-confidence text falls back cleanly to music-only output. |
| **6** | **MTP-4 speculative draft corruption across music↔text head switch** | **`MTP-4` explicitly disabled on mode-switch tokens** (`<sys_lyr>`, `<harm>`, `<tempo>`, `<dir>`, `<reh>`, `<fb>`, and text-exit tokens). |
| **7** | **BD3-LM block diffusion corrupting schema headers or UTF-8/BPE text** | **`BD3-LM` strictly masked to `is_music_body`** (Layers 0–2 body tokens only; schema prefix & Layer 3 text spans are immutable). |
| **8** | **Overlapping slur / hairpin collision across polyphonic voices** | Lossless numbered spanners **`slur-1..4`** and **`hairpin-1..3`** with explicit open/close IDs. |
| **9** | **Collapsed expressive marks losing engraving fidelity** | Lossless split of `marcato`, `mordent`, `turn`, `tremolo`, `15ma`, `pedal-sost`/`pedal-uc`, `arpeggio-dir`, `trill-line`, `gliss`, `breath`, `caesura`, `fingerings`, `harmonics`. |
| **10** | **Catastrophic forgetting during warm-start into `bekern-full`** | **2-stage freeze/unfreeze** in Iter 16 (Stage 1 trains new embeddings + Layer 0–2 additions + `ModeGate` with core frozen; Stage 2 joint fine-tune gated by `HWM >= 24.83` floor). |

---

## 2. Schema Prefix & Unannotated-Layer CE Loss Masking

Every target sequence begins with `<bos>` followed immediately by a **schema prefix token** declaring which annotation layers are ground-truth supervised for that page:

| Schema Token | Layer Bitmask (`L0 L1 L2 L3`) | Meaning & Corpus Usage |
| :--- | :---: | :--- |
| `<schema:v1>` | `0 1 0 0` | Core legacy `bekern` notes/rhythms/clefs/keys/bars (real historical HW manuscripts without expressive/text GT). |
| `<schema:v2a>` | `0 1 1 0` | Core notation + Layer 2 expressive marks (augmented printed & clean synthetic corpora). |
| `<schema:full>` | `1 1 1 1` | Full 4-layer supervision (multi-part layout, extended Layer 1, lossless Layer 2, and Layer 3 text/lyrics/harmony). |

> [!IMPORTANT]
> **Partial-Layer Bitmask & Unannotated-Layer CE Loss Masking:**  
> In addition to the discrete prefix token, each training example carries a 4-bit `layer_mask` (`L0..L3`). For any token $t$ belonging to an unannotated layer ($\text{layer}(t) \notin \text{layer\_mask}$), its target CE loss and negative logit gradient are masked out (`ignore` / zeroed layer weight). This directly cures Iter 14's **`43.9%` false-negative poisoning**, where unannotated real handwritten pages taught the model to suppress valid dynamics, articulations, and slurs and regressed `HWM`.

### Vocab-Expansion Safety Contract (`UniOMRv2Config`)

Legacy checkpoints (`vocab_size = 265`) place special tokens at indices `261..264` (`PAD=261`, `BOS=262`, `EOS=263`, `SYS=264`). Appending new `bekern-full` tokens at indices `265..V-1` breaks any code assuming `pad_id = v - 4`, `bos_id = v - 3`, `eos_id = v - 2`.

`UniOMRv2Config` enforces explicit special-token persistence:
- `pad_id: Optional[int] = None`
- `bos_id: Optional[int] = None`
- `eos_id: Optional[int] = None`
- `sys_id: Optional[int] = None`
- `music_vocab_end: Optional[int] = None`

When all five are `None`, `resolved_*` falls back identically to legacy `VOCAB_SIZE` behavior. When expanded for `<schema:...>` / `bekern-full`, explicit IDs are serialized with the checkpoint so `tok_emb`, `tok_dropout`, `generate()`, and `compute_loss()` never corrupt special tokens or sample them as random dropout replacements.

---

## 3. Four-Layer Token Specification

### Layer 0 — Score Structure, Multi-Part & Navigation

Captures page-level and system-level multi-instrument layout and repeat/navigation topology:

| Category | Tokens | Semantics |
| :--- | :--- | :--- |
| **Parts & Staves** | `<part>`, `<staff-def>`, `<group>` | Declares part boundaries, staff lines/clefs/sizing, and brace/bracket/barline staff groups. |
| **Transposition** | `<transpose>` | Diatonic + chromatic transposition metadata (e.g. Bb clarinet, F horn) so written pitch round-trips to concert pitch. |
| **Measure Structure** | `<pickup>`, `<multirest>`, `mrep-1`, `mrep-2`, `mrep-4` | Anacrusis (pickup measure), numbered multi-measure rests, and 1-, 2-, and 4-measure repeat bars. |
| **Instrument Switch** | `<inst-change>` | Mid-score instrument/timbre change (e.g. flute → piccolo, muta in D). |
| **Navigation** | `D.C.`, `D.S.`, `Segno`, `Coda`, `Fine` | Structural jump & landmark markers (paired with `al Coda` / `al Fine` modifiers where present). |

### Layer 1 — Core Notation, Microtones, Beams, Percussion, Tab & Cross-Staff

Extends core `bekern` pitches/durations/chords/voices with full engraving fidelity:

| Category | Tokens | Semantics |
| :--- | :--- | :--- |
| **Arbitrary Tuplets** | `tup-open` (multi-digit `N:M`), `tup-close` | Supports arbitrary multi-digit ratios (e.g. `11:8`, `13:12`) and nested tuplets without fixed-table overflow. |
| **Microtones & Accidentals** | Quarter-tones (`q#`, `qb`, `3q#`, `3qb`), `a0`, `acc-caut`, `acc-paren` | Quarter-tone inflection, explicit natural (`a0`), cautionary accidentals, and parenthesized courtesy accidentals. |
| **Chords, Stems & Beams** | `<chord>`, `stem-up`, `stem-down`, `beam-*` | Explicit chord-stacking delimiter, stem direction, and beam topology (`beam-start`, `beam-cont`, `beam-end`, `beam-hook-l`, `beam-hook-r`). |
| **Percussion & Noteheads** | `unpitched`, `nh-*` | Unpitched staff positions + explicit notehead shapes (`nh-x`, `nh-diamond`, `nh-slash`, `nh-triangle`, `nh-circle-x`, `nh-square`). |
| **Guitar / Fretted Tab** | `str`, `fret`, `bend`, `hopo` | String index, fret number, pitch bend amount, and hammer-on / pull-off indicators. |
| **Grace & Cross-Staff** | `grace-slash`, `xstaff-N` | Explicit slashed grace (acciaccatura) vs unslashed appoggiatura, and cross-staff note/beam routing to staff `N`. |

### Layer 2 — Lossless Expressive & Articulation Split

Replaces lossy v1/v2a buckets with distinct, round-trip-lossless engraving tokens:

| Category | Tokens | Semantics |
| :--- | :--- | :--- |
| **Articulations & Ornaments** | `marcato`, `mordent` (upper/lower), `turn` (normal/inverted), `tremolo` (1/2/3-stroke & two-note) | Preserves exact ornament/articulation identity instead of collapsing into generic accent/ornament buckets. |
| **Octave & Pedal Spanners** | `15ma` (`15mb`), `pedal-sost`, `pedal-uc` | Two-octave shifts (`15ma`/`15mb`) alongside `8va`/`8vb`; sostenuto (`pedal-sost`) and una corda (`pedal-uc`) alongside sustain. |
| **Arpeggios & Glissandi** | `arpeggio-dir` (`arp-up`, `arp-down`, `arp-non`), `trill-line`, `gliss` | Directional/non-arpeggio roll marks, wavy trill extensions, and glissando/portamento lines. |
| **Numbered Overlapping Spanners** | `slur-1..4`, `hairpin-1..3` | Explicit spanner channel IDs (`open`/`close`) so concurrent cross-voice slurs and crescendo/diminuendo wedges never collide. |
| **Phrasing & Technique** | `breath`, `caesura`, `fingerings` (`fing-1..5`), `harmonics` | Breath marks, railroad-track caesuras, keyboard/string fingerings, and natural/artificial harmonic indicators. |

### Layer 3 — End-of-System Text, Lyrics, Harmony & `ModeGate` Dual Head

All natural-language / alphanumeric score text is isolated from the rhythmic music stream and decoded via a **physically separate `ModeGate` dual head**:

#### End-of-System Anchored Block Layout
- **Lyrics:** `<sys_lyr v=N>` placed at the **end of each `<sys>` block** (for verse `v=1..N`), referencing notes by zero-based note index anchors (`@i`) with syllable continuity markers **`lyr-beg`**, **`lyr-mid`**, **`lyr-end`**, and **`lyr-ext`** (melisma extender).
- **Non-Lyric Text Spans:** Anchored `<harm>` (chord symbols / Roman numerals), `<tempo>` (tempo text & metronome marks), `<dir>` (performance directions), `<reh>` (rehearsal marks), and `<fb>` (figured bass).

#### `ModeGate` Dual-Head Architecture & Inference Controls

```text
Decoder final RMSNorm h (B, T, d_model)
   │
   ├──▶ ModeGate (2-way mode router: MUSIC vs TEXT, confidence p_text)
   │       ├─ p_text < 0.75 (or non-text schema): stay in MUSIC mode (0% text hallucination)
   │       └─ p_text >= 0.75 on mode-switch token: enter TEXT head until </text_span>
   │
   ├──▶ lm_head_music (Layers 0–2 music vocab)     [MTP-4 active; BD3-LM active on is_music_body]
   └──▶ lm_head_text  (Layer 3 text vocab)         [MTP-4 DISABLED on mode-switch; BD3-LM MASKED OFF]
          ├─ edge85m: UTF-8 byte-level (<= 256)    ──▶ +0.34 MB (stays < 55 MB Hybrid FP16/INT4)
          └─ pro200m: optional 2k BPE              ──▶ higher throughput on dense vocal/choral pages
```

- **Parameter Budget (`edge85m`):** UTF-8 byte-level head (`256 × 512` input/output weights + tiny `ModeGate` linear) adds **`+0.34 MB`**, keeping `edge85m` comfortably below the `< 55 MB` WebGPU ceiling.
- **Confidence Gating (`p >= 0.75`):** Entering or continuing a Layer 3 text span requires `ModeGate` probability `p >= 0.75`; below threshold, the decoder emits cleanly without hallucinating garbled lyrics/directions on worn manuscripts.
- **`MTP-4` Mode-Switch Guard:** Multi-Token Prediction speculative drafting is **disabled** whenever the current or drafted token is a mode-switch token (`<sys_lyr>`, `<harm>`, `<tempo>`, `<dir>`, `<reh>`, `<fb>`, or span terminator), guaranteeing zero cross-head draft mismatches.
- **`BD3-LM` Masking (`is_music_body`):** Block-diffusion / iterative refinement masks strictly to `is_music_body` tokens (Layers 0–2 body), never modifying `<bos>`, `<schema:...>`, `<sys>`, or Layer 3 UTF-8/BPE text spans.

---

## 4. Rollout Roadmap (Iter 15 → Iter 16)

| Iteration | Architecture & Vocab | Active Scope | Promotion Gate |
| :--- | :--- | :--- | :--- |
| **Iter 15** | **`pro200m`** + `<schema:v1>` / `<schema:v2a>` | Adds schema prefix tokens, unannotated-layer CE loss masking, explicit `UniOMRv2Config` special-token IDs, and bundles the **20 `PROMOTE`d** ablations/fixes. Layer 3 `ModeGate` remains inactive. | Validates zero `HWM` regression, cures Iter 14's `43.9%` false-negative poisoning, and establishes the `pro200m` baseline on `suite31_hw` & OMR-NorthStar. |
| **Iter 16** | **Warm-start → `bekern-full`** (`pro200m` & distilled `edge85m`) | **2-Stage Freeze/Unfreeze:**<br>• **Stage 1 (Freeze core):** Freeze pretrained backbone/encoder/decoder music weights; train new `bekern-full` embeddings, Layer 0–2 new tokens, and `ModeGate` Layer 3 text head (`<=256` UTF-8 bytes on `edge85m`, optional `2k` BPE on `pro200m`).<br>• **Stage 2 (Unfreeze all):** Joint fine-tuning with unannotated-layer CE masking, `MTP-4` mode-switch guard, `BD3-LM` `is_music_body` mask, and `p >= 0.75` confidence gating. | Must clear **`HWM >= 24.83` floor** (hard non-negotiable gate) + positive NorthStar expressive/structural/text deltas + `< 55 MB` WebGPU export verification for `edge85m`. |

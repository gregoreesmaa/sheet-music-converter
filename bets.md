# Bets catalog — 100 novel architectural bets for sheet-music OMR

Source: bets farm, 2026-11-12. 10 research scouts + 5 expert personas +
5 data-science generalists, run through agy (pool phantoms blocked
subagents). Raw reports: `D:/tmp/bets/agy-batch*.md`
(batch 1 NCM+HTR output was clipped in transit: bets A1–A3 titles
unrecovered, A3 fragment kept).

System context at time of writing: hybrid CNN-Transformer, system
strips → bekern+ tokens (175-vocab, grammar-constrained AR decode).
100k-scale run in flight (~73k, loss ~0.17). Best verified:
degraded F1 0.395, clean 0.687, suite31-photo F1 0.240 (Gate 5: 0.975).

**Status 2026-11-13 (frozen catalog — do not rewrite; state lives in
`spikes.md`/`todo.md`):** top-12 shortlist became spikes S1–S9, ALL
DROPPED with evidence (decoder/loss/rerank shaping can't fix
recognition errors). 100k finished: photo 0.352, degraded 0.515,
clean 0.794. S10 (hand-ink data) running; agy 10× consult added
S13–S18 (bundled retrain, real-camera pairs, measure verifier,
factored tuples; VLMs/diffusion rejected).

## Ranked shortlist (controller's top 12 by payoff × cost × risk)

1. **Unconsumed-ink EOS veto** (batch1 B4, HTR ink-erasure): hard-mask EOS
   while HeatNet foreground ink remains unread right of the cursor.
   Zero-train, kills early-EOS structurally. Payoff +0.10–0.16.
2. **Contrastive sibling-margin loss** (batch1 A5, LCM-flavored): InfoNCE
   over 1-edit musical minimal pairs (dot on/off, #/b swaps) so 3-px
   glyphs stop being rounding errors. Payoff +0.09–0.14.
3. **MWER sequence training** (batch2 A5, ASR transfer): expected
   Levenshtein risk over grammar beams, 3× cost on dropped barlines /
   premature EOS. Aligns training with F1. Payoff +0.05–0.08.
4. **ILME subtraction sweep** (batch2 A2, ASR transfer): subtract the
   decoder's internal clean-score prior during beam search; zero-train
   first (1 h λ-sweep). Payoff +0.03–0.05, near-zero cost.
5. **Medusa atom-tuple heads** (batch5 A2, efficient): single-step
   chord/note-complex emission, ~3× fewer decoder steps. Speed + F1.
6. **MaxSAT confusion-lattice repair** (batch3 B2, program synthesis):
   post-decode symbolic repair (bar math, spine sync) via CP-SAT.
   No retrain. Fixes-per-page -25%.
7. **Fix-one-note auto-heal UX** (batch5 B3, product moat): human click
   at a corrupted measure root auto-corrects ~3.2 downstream tokens.
   Fixes-per-page 2.5–4×. This is the correction-UX gate made real.
8. **Fixes-per-page action loss** (batch4 B2, metric design): train on
   MuseScore keystroke-risk + bar-conservation surrogate instead of CE.
9. **Staff-indexed vertical gaze** (batch1 B2, OrigamiNet transfer):
   per-spine Gaussian attention windows replace brittle rectify as
   single point of failure. Payoff +0.11–0.17.
10. **Pruned RNN-T transducer** (batch2 A1, speech transfer): frame-
    synchronous decode, EOS structurally impossible. Bigger build,
    highest structural payoff +0.06–0.09.
11. **Test-time MLLR adapter** (batch6 ASR-20): rank-8 per-strip linear
    adapter, <25 ms/strip, absorbs unseen blur/noise per photo.
12. **MDL source-coverage decoding** (batch7 INFO-3): penalize EOS while
    encoder-column norm remains unvisited. Information-theoretic EOS
    veto, complements bet 1.

## Full catalog

### Batch 1 — NCM + HTR (10; A1–A3 titles unrecovered)
| # | Bet | Payoff |
|---|---|---|
| A1 | *(title unrecovered)* | — |
| A2 | *(title unrecovered)* | — |
| A3 | Entropy-adaptive variable-length concepts (fragment) | +0.08–0.13 |
| A4 | Geometry-as-concept prefix codes (self-rectifying NCM) | +0.10–0.16 |
| A5 | LCM contrastive sibling-margin loss for micro-glyphs | +0.09–0.14 |
| B1 | CVC-MUSCIMA ink/staff disentanglement head | +0.08–0.13 |
| B2 | OrigamiNet staff-indexed vertical gaze offsets | +0.11–0.17 |
| B3 | TrOCR diacritic-stem distillation into stride-4 | +0.09–0.15 |
| B4 | Ink-erasure cursor + unconsumed-ink EOS veto | +0.10–0.16 |
| B5 | [CAPTURE_STYLE] prefix + FiLM from params.json | +0.06–0.11 |

### Batch 2 — speech + audio-crossmodal (10)
| # | Bet | Payoff |
|---|---|---|
| A1 | Pruned stateless RNN-T transducer over encoder frames | +0.06–0.09 |
| A2 | ILME subtraction in grammar-constrained beam search | +0.03–0.05 |
| A3 | Pitch-shear + sub-band SpecAugment in early CNN maps | +0.03–0.05 |
| A4 | 2D Macaron Conformer, pitch-tall + glyph-local kernels | +0.04–0.07 |
| A5 | MWER sequence training over grammar-pruned beams | +0.05–0.08 |
| B1 | Soft-DTW chroma-onset loss on decoder logits | +0.04–0.07 |
| B2 | AMT onset/sustain prior distillation | +0.03–0.05 |
| B3 | Acoustic-tempo RoPE warp + EOS regulation | +0.05–0.08 |
| B4 | Psychoacoustic roughness/virtual-pitch loss | +0.03–0.05 |
| B5 | Cross-system CQT handoff + DTW beat-closure | +0.04–0.06 |

### Batch 3 — diff-render + grammar/synthesis (10)
| # | Bet | Payoff |
|---|---|---|
| A1 | Gaussian SMuFL splatting (`DiffKernSplat`) | +0.06–0.10 |
| A2 | Latent engraver + Langevin logit refinement | +0.07–0.12 |
| A3 | Barline-impulse structural renderer (`BarShock`) | +0.05–0.09 |
| A4 | Counterfactual SDF micro-patch compositor | +0.04–0.08 |
| A5 | EM intrinsic staff-coordinate rectification | +0.08–0.14 |
| B1 | Exact-rational polyphonic clock PDA | +0.05–0.09 |
| B2 | MaxSAT confusion-lattice repair (`KernRepair-SAT`) | +0.06–0.11 |
| B3 | Measure-boundary MCTS with spatial-progress value | +0.05–0.10 |
| B4 | Rhythmic sketch + parallel Jacobi symbolic fill | +0.06–0.11 |
| B5 | Tropical WFST + collinearity potential | +0.05–0.09 |

### Batch 4 — scribble-data + loss design (10)
| # | Bet | Payoff |
|---|---|---|
| A1 | Sigma-lognormal motor-program synthesis | +0.04–0.07 |
| A2 | Strikeout/scribble-overwrite injection | +0.05–0.08 |
| A3 | Performer pencil-markup grammar layer | +0.03–0.06 |
| A4 | Rastrum-twist + margin-crowding corruption | +0.04–0.07 |
| A5 | Capillary bridging + ISP halo physics | +0.04–0.06 |
| B1 | Banded beat-Sinkhorn metrical transport loss | +0.04–0.07 |
| B2 | Fixes-per-page action loss | +0.03–0.06, −25% fixes |
| B3 | Micro-glyph leverage (inverse-footprint focal) loss | +0.04–0.07 |
| B4 | In-grammar calibration + ink-remaining critic | +0.05–0.09 |
| B5 | Warp-equivariance + logit-JSD consistency loss | +0.05–0.08 |

### Batch 5 — efficient-arch + layout-free/UX (10)
| # | Bet | Payoff |
|---|---|---|
| A1 | Barline-anchored measure-parallel decoding | 6–9× speedup, F1+ |
| A2 | Medusa atom-tuple heads (single-step chords) | ~3.2× faster, F1+ |
| A3 | Monotonic x-windowed attention + token merging | ~3.5× faster, F1+ |
| A4 | Grammar-margin early-exit decoder | 2.4–3× CPU, F1 +0.02–0.03 |
| A5 | 2×-DPI teacher → 1× student ROI distillation | 1× latency, F1+ |
| B1 | Cursor-guided deformable 2D-RoPE full-page decoder | layout-free |
| B2 | Latent Bezier ribbon sampler (crop-free pages) | curved-page F1+ |
| B3 | Fix-one-note auto-heal measure re-decode UX | fixes 2.5–4× |
| B4 | Edit-cascade triage UI with measure forks | verify time −50% |
| B5 | Edit-trajectory LoRA-DPO + active-learning loop | +0.06–0.10/500 meas. |

### Batch 6 — personas (25; tags ENGRAVER / SIGHT-READER / HTR-VET / ASR-VET / VLM-RES)
Engraver: spacing prior, slot pooling, staircase classifier, stem/beam
bounds, staff-space caliper. Sight-reader: hand-span bias, pulse
countdown, graphic-interval pitch head, accidental working memory,
peripheral/foveal dual attention. HTR-vet: Radon deslant, DoG+Sauvola
stem, monotonic windowing, BlurPool, Chebyshev TPS augment. ASR-vet:
scheduled sampling, liftering, diverse beam, expert softmax, MLLR
adapter. VLM-res: register tokens, grounding loop, content-box RoPE,
AnyRes tile fusion, MaskFeat columns. (Full text: `D:/tmp/bets/`.)

### Batch 7 — generalists (25; OPT / BAYES / INFO / TS / RECSYS)
OPT: group-DRO, DEQ bottleneck, PCGrad surgery, entmax attention,
minimax drift perturbation. BAYES: SVGD LoRA particles, CNP page
context, Laplace-Thompson rollouts, expectile decoding, horseshoe
shrinkage. INFO: directed-info regularizer, Wyner-Ziv bottleneck, MDL
decoding, Slepian-Wolf orthogonalization, predictive InfoNCE. TS:
Takens embeddings, RevIN decomposition, matrix-profile salience,
Kalman SLDS filter, rough-path signatures. RECSYS: IPS weighting, DPP
slates, DeepFM head, LightGCN propagation, cold-start meta-embeddings.
(Full text: `D:/tmp/bets/`.)

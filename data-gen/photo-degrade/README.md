# photo-degrade

Rust CLI that turns a **clean PNG** (e.g. rendered sheet music) into a
realistic degraded **"phone photo" PNG**, for training data generation.

Degradation pipeline (in order), all scaled by `--strength 0..1`:

1. **Paper texture** — fine Gaussian grain, warm paper tint, sparse dark speckles.
2. **Page curl** — geometric warp toward a random edge (left/right) plus
   shading that deepens toward the curled edge.
3. **Perspective** — homography warp from randomly jittered frame corners
   (inverse-mapped with bilinear resampling).
4. **Blur** — Gaussian defocus (`image::imageops::blur`).
5. **Glare / shadow** — specular highlight disc, directional linear-gradient
   shadow, and corner vignette.
6. **Noise** — per-channel Gaussian sensor noise.
7. **JPEG** — in-memory JPEG encode/decode round-trip at the recorded quality.

All randomness comes from a seeded `ChaCha8Rng`, so runs are reproducible.

## Build

```sh
cargo build --release
```

Requires Rust 1.70+ and network access to fetch crates on first build
(`image`, `clap`, `rand`/`rand_chacha`, `serde`/`serde_json`).

## Usage

Single file:

```sh
photo-degrade --input clean.png --out-dir out --seed 42 --strength 0.6
```

Batch directory (non-recursive, `*.png`, sorted; file `i` uses `seed + i`
mixed with a fixed constant):

```sh
photo-degrade --input clean-dir/ --out-dir degraded/ --seed 7
```

Options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--input` | (required) | Clean PNG file or directory of PNGs |
| `--out-dir` | `out` | Output directory (created if missing) |
| `--seed` | `0` | Master seed (u64) |
| `--strength` | `0.5` | Global effect strength, clamped to [0, 1] |
| `--jpeg-quality` | sampled 30–95 | JPEG quality override, 1–100 |
| `--suffix` | `_degraded` | Degraded-image filename suffix |

## Output (paired)

For input `sonata.png` in `--out-dir out/`:

- `out/sonata_degraded.png` — the degraded image (same dimensions).
- `out/sonata_params.json` — the exact parameters used:

```json
{
  "input": "sonata.png",
  "output": "sonata_degraded.png",
  "file_seed": 42,
  "strength": 0.6,
  "width": 800,
  "height": 1100,
  "curl_side": "left",
  "curl_px": 19.3,
  "blur_sigma": 0.62,
  "noise_sigma": 4.1,
  "jpeg_quality": 63
}
```

Re-running with the same `--input`, `--seed`, and `--strength` reproduces
both files byte-approximately (PNG encoding itself is deterministic; JPEG
artifacts are a deterministic function of the seed).

## Notes

- `--strength 0` still applies a near-identity pass (tiny paper tint +
  JPEG at ~q80) so the pipeline shape stays comparable across strengths.
- Out-of-warp pixels (curl/perspective edges) are filled with paper white.
- Exit code is non-zero with a message on stderr for missing input,
  unreadable images, or unwritable output.

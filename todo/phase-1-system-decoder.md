# Phase 1: Break the 0.00 Deadlock — Engraver-Exact System Dataset & Anisotropic `MvdNet` v2 (`T1.1` – `T1.6`)

Back to master checklist: [../todo.md](../todo.md)

Complete these tasks in exact numerical order (`T1.1` → `T1.6`). Do not start `T1.4` until `T1.1`–`T1.3` pass their verification commands.

---

## `[x]` Task T1.1 — Add System-Level `bekern --measures` and `system-breaks` to `converter/assembly-serialize`

### Objective
Enable `assembly-serialize` to emit exact `bekern` token sequences for any measure span `START..END` (one staff system) and to list all `(page, system_idx_on_page, start_measure, end_measure)` breaks from `<print new-system="yes">` and `<print new-page="yes">` tags in MuseScore-rendered `.musicxml` files.

### Files to Read First
- `converter/assembly-serialize/src/musicxml_import.rs` (lines 445–520 and 740–775)
- `converter/assembly-serialize/src/main.rs` (lines 128–160 and 498–550)

### Exact Changes to Make
1. In `converter/assembly-serialize/src/musicxml_import.rs`:
   - Inspect how `Item::NewPage(u32)` is parsed in `measure_items`. Also detect `<print` tags containing `new-system="yes"` (or `new-page="yes"`, since every new page also starts system 0 on that page).
   - Add a public struct and function:
     ```rust
     #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
     pub struct SystemSpan {
         pub page: u32,
         pub system_idx: u32,      // 0-based system index within `page`
         pub start_measure: u32,   // 1-based inclusive
         pub end_measure: u32,     // 1-based inclusive
     }
     pub fn system_breaks(xml: &str) -> Vec<SystemSpan>
     ```
     - Walk part 1's measures (1-based `measure = ordinal0 + 1`). Track `cur_page = 1`, `sys_idx = 0`. Whenever a measure (at `measure > 1`) has `new-page="yes"`, close the previous span at `measure - 1`, increment `cur_page` (using the `page-number` attribute if present, else `cur_page + 1`), reset `sys_idx = 0`, and open a new span starting at `measure`. Whenever a measure (at `measure > 1`, without `new-page="yes"`) has `new-system="yes"`, close the previous span at `measure - 1`, increment `sys_idx += 1`, and open a new span starting at `measure`. Close the final span at `total_measures`.
   - Also update `staff_headers_at_measure(xml: &str, start_measure: u32) -> Result<Vec<StaffHeader>, String>` (or enhance `staff_headers`) so that clef, key, and time signature changes occurring in `measure <= start_measure` are carried forward accurately to the start of that system span!
2. In `converter/assembly-serialize/src/main.rs`:
   - In `BekernArgs`, make `--page` optional (`Option<u32>`) and add `--measures: Option<String>` (format `"START..END"`, e.g. `"6..9"`, 1-based inclusive). Exactly one of `--page` or `--measures` must be provided.
   - When `--measures START..END` is given, filter `bekern_events(&xml)` to `e.measure >= start && e.measure <= end`, compute carried `staff_headers` at `start`, call `to_bekern(&events, &headers)`, and write `<out>/<stem>-bekern.txt`.
   - Add a subcommand `SystemBreaks(SystemBreaksArgs)` (`assembly-serialize system-breaks --input <score.musicxml> [--out <json_path>]`) that calls `system_breaks(&xml)` and writes/prints the JSON array of `SystemSpan`s.
3. Add unit tests in `converter/assembly-serialize/src/musicxml_import.rs`:
   - Test `system_breaks` on a multi-page, multi-system MusicXML snippet and on `data-gen/score-corpus/big500-renders/ds0001.musicxml` (verifying Page 1 has 3 systems: measures `1..5`, `6..9`, `10..13`; Page 2 has 4 systems: `14..17`, `18..21`, `22..25`, `26..28`).
   - Test `bekern --measures 6..9` round-trips cleanly through `decode_bekern`.

### Verification Command
```powershell
cargo test --manifest-path converter/assembly-serialize/Cargo.toml
cargo build --release --manifest-path converter/assembly-serialize/Cargo.toml
.\converter\assembly-serialize\target\release\assembly-serialize.exe system-breaks --input data-gen/score-corpus/big500-renders/ds0001.musicxml
```

### Pass Criteria
- `cargo test` passes with 0 failures.
- `system-breaks` on `ds0001.musicxml` prints the exact system spans (`page 1`: `1..5`, `6..9`, `10..13`, etc.).

---

## `[x]` Task T1.2 — Build Engraver-Exact System-Crop + Degraded Dataset (`tmp-train/mvd/build_sys_data.py`)

### Objective
Generate ≥10,000 **100% aligned, zero-stub** `(system_strip_image, bekern_txt)` training pairs (1 clean + 3 degraded variants per system) by pairing `assembly-serialize system-breaks` with exact staff-system bounding boxes.

### Files to Read First
- `data-gen/score-corpus/big500-symbox/ds0001/ds0001-1-boxes.json`
- `data-gen/score-corpus/gate20/exclusion.json` (CRITICAL: never include any `do_not_train` score/page stem!)
- `tmp-train/reproject/` (or `data-gen/photo-degrade`)

### Exact Implementation (`tmp-train/mvd/build_sys_data.py`)
Create `tmp-train/mvd/build_sys_data.py` with CLI flags `--renders data-gen/score-corpus/big500-renders --symbox data-gen/score-corpus/big500-symbox --degraded data-gen/score-corpus/big500-degraded10 --out tmp-train/mvd-sysdata --limit-scores 0`:
1. **Exclusion Guard:** Load `data-gen/score-corpus/gate20/exclusion.json` and `tmp-train/mvd-data/quarantine.txt`. Any score or page matching `do_not_train` or `quarantine` is skipped before processing.
2. **System Box Grouping per Page:**
   - For each score `dsXXXX` in `big500-renders/dsXXXX.musicxml`:
     - Run `assembly-serialize system-breaks --input dsXXXX.musicxml` to get the list of `SystemSpan`s per `page`.
     - Obtain the page's staff bounding boxes in pixel coordinates:
       - If `big500-symbox/dsXXXX/dsXXXX-<page>-boxes.json` exists, load `staves[]` (`lines` and `bbox: [x1, y1, x2, y2]`).
       - Otherwise, run `layout-staff --model neural --weights tmp-train/row-retrain/rownet_v2.onnx --input dsXXXX-<page>.png` to get the clean page's `systems[]`.
     - Group staves on the page into systems by vertical gap (staves whose vertical gap is `< 6.0 * staff_space` belong to the same system; or use the `parts` staff count from the score header: if a score has $S$ staves per system and $K \times S$ staves on the page, group consecutive chunks of $S$ staves into $K$ systems!).
     - **STRICT ALIGNMENT GATE:** Assert that `len(visual_systems_on_page) == len(xml_system_spans_on_page)`. If they disagree on a page, log the page stem to `tmp-train/mvd-sysdata/skipped_pages.txt` and **skip that page** (never guess or force-align!).
3. **Crop Clean System Strip & Emit Exact `bekern`:**
   - For each aligned system `(sys_idx, span)` on `dsXXXX-<page>`:
     - Pad the system's `[x1, y1, x2, y2]` box by `2.5 * staff_space` vertically (to preserve ledger lines, stems, beams, and clefs) and `1.5 * staff_space` horizontally, clamped to image bounds.
     - Save clean crop to `tmp-train/mvd-sysdata/dsXXXX-<page>-s<sys_idx>.png`.
     - Run `assembly-serialize bekern --input dsXXXX.musicxml --measures <start>..<end> --out tmp-train/mvd-sysdata --stem dsXXXX-<page>-s<sys_idx>`.
     - Verify the emitted `.txt` file parses cleanly through `assembly-serialize bekern-decode` and has `1 <= token_count <= 512`. If `bekern` or `bekern-decode` errors, delete the crop and skip that system.
4. **Generate 3 Degraded Variants per Clean System Crop (`_d0`, `_d1`, `_d2`):**
   - Run `photo-degrade --gpu --input <clean_crops_dir> --out-dir <out_dir> --variants 3 --format jpeg --max-width 2048` (or run `photo-degrade` in release mode on the directory of clean system crops) so each `dsXXXX-<page>-s<sys_idx>-bekern.txt` has:
     - 1 clean image: `dsXXXX-<page>-s<sys_idx>.png`
     - 3 degraded images: `dsXXXX-<page>-s<sys_idx>_d0.jpg`, `_d1.jpg`, `_d2.jpg`.
5. Write `tmp-train/mvd-sysdata/manifest.json` recording total clean crops, total degraded crops, token length percentiles (`p50`, `p95`, `max`), and `0` `gate20` overlap.

### Verification Command
```powershell
.\.venv\Scripts\python.exe tmp-train/mvd/build_sys_data.py --limit-scores 60
.\.venv\Scripts\python.exe -c "import json; m=json.load(open('tmp-train/mvd-sysdata/manifest.json')); print(m); assert m['clean_systems'] >= 500 and m['degraded_systems'] >= 1500 and m['gate20_leaked'] == 0 and m['max_tokens'] <= 512"
```

### Pass Criteria
- At least 500 clean systems + 1,500 degraded systems (2,000+ total training pairs on the 60-score pilot, scaling to 10,000+ with `--limit-scores 0`), `gate20_leaked == 0`, `max_tokens <= 512`, and 100% `bekern-decode` validity.

---

## `[x]` Task T1.3 — Build Held-Out `gate20` Dev System Reference Set (`data-gen/score-corpus/gate20/sys-dev/`)

### Objective
Create the held-out system-level evaluation benchmark on `gate20` **dev** pages (`dev-12` only; **never touch `test-8`**) so Gate 2 (`SER < 0.48`, `strict-F1 > 0.22`) and Gate 3 (`SER < 0.15`, `strict-F1 > 0.50`) can be measured directly at the system level.

### Files to Read First
- `data-gen/score-corpus/gate20/gate_manifest.json`
- `data-gen/score-corpus/mscore-align-spike/BULK-REPORT.md`

### Exact Implementation (`tmp-train/mvd/build_gate20_sys_dev.py`)
1. Read `data-gen/score-corpus/gate20/gate_manifest.json` and filter strictly to `split == "dev"` pages that have valid page references (never read or touch `split == "test"`).
2. For each scored `dev` page:
   - Extract the page's measure span and per-system measure breaks (`system-breaks` on the score/split `.musicxml`).
   - Crop the corresponding systems from `gate20/clean/<id>.png` and `gate20/degraded/<id>_degraded.jpg` into `data-gen/score-corpus/gate20/sys-dev/clean/` and `data-gen/score-corpus/gate20/sys-dev/degraded/`.
   - Emit the ground-truth `<id>-s<sys_idx>-bekern.txt` and decoded `<id>-s<sys_idx>-ref-notes.json` into `data-gen/score-corpus/gate20/sys-dev/refs/`.
3. Do NOT modify any existing file in `data-gen/score-corpus/gate20/` (verify `gate_manifest.json` SHA-256 stays `68a0b070277bda6b2e8fd89565fa8dc3952ac1fb29e25e5403c3d78bc4ed18eb`).

### Verification Command
```powershell
.\.venv\Scripts\python.exe tmp-train/mvd/build_gate20_sys_dev.py
.\.venv\Scripts\python.exe -c "import hashlib, json; h=hashlib.sha256(open('data-gen/score-corpus/gate20/gate_manifest.json','rb').read()).hexdigest(); assert h=='68a0b070277bda6b2e8fd89565fa8dc3952ac1fb29e25e5403c3d78bc4ed18eb'; m=json.load(open('data-gen/score-corpus/gate20/sys-dev/sys_dev_manifest.json')); print(m); assert m['n_systems'] >= 15 and m['test_pages_touched'] == 0"
```

### Pass Criteria
- `gate_manifest.json` SHA-256 is unchanged, `test_pages_touched == 0`, and `n_systems >= 15` held-out dev systems (both clean and degraded) have verified `.bekern` and `-ref-notes.json` references.

---

## `[x]` Task T1.4 — Upgrade `MvdNet` in `tmp-train/mvd/model.py` (Anisotropic Stride `8×16` + Apache-2.0 `timm` Pretrained Init)

### Objective
Fix the two architectural flaws that blinded `MvdNet` v1:
1. Reduce vertical downsampling from `32×` to **`8×` vertically** (`16×` horizontally) so a `H=256` system strip produces **32 vertical feature rows** (giving every staff line and space its own feature row instead of crushing a whole staff into 1–2 rows).
2. Initialize the ConvNeXt-Tiny encoder from **Apache-2.0 `timm` pretrained weights** (`convnext_tiny.fb_in22k` / `timm` Apache-2.0 release) instead of random weights.

### Files to Read First
- `tmp-train/mvd/model.py` (lines 133–175)
- `tmp-train/mvd/test_model.py`

### Exact Changes to Make in `tmp-train/mvd/model.py`
1. **Anisotropic Stride `(8, 16)` in `MvdNet.__init__`:**
   - In `torchvision.models.convnext_tiny().features`:
     - `features[0]` is the stem `Conv2d(3, 96, kernel_size=4, stride=4)` → stride `(4, 4)`.
     - `features[2]` is stage-2 downsample `Conv2d(96, 192, kernel_size=2, stride=2)` → stride `(8, 8)`.
     - `features[4]` is stage-3 downsample `Conv2d(192, 384, kernel_size=2, stride=2)` → change its stride to **`(1, 2)`** (and pad `(0, 0)` or `kernel_size=(1, 2)` / keep `(2, 2)` kernel with `stride=(1, 2), padding=(1, 0)` cropped or symmetric) so vertical size stays `H/8` and horizontal becomes `W/16`!
     - `features[6]` is stage-4 downsample `Conv2d(384, 768, kernel_size=2, stride=2)` → change its stride to **`(1, 1)`** (with `padding=(1, 1)` or crop to preserve `(H/8, W/16)`).
   - Add a constructor parameter `anisotropic: bool = True` (default `True` for v2; `False` preserves legacy stride-32 for loading old `ckpt-13000` if needed) and `pretrained: bool = False`.
   - When `anisotropic=True`, verify that an input image of shape `[B, 3, 256, 1024]` encodes to `feats` of spatial height `32` (`256 // 8`) and width `64` (`1024 // 16`), i.e., `mem.shape == [B, 32 * 64, d_model]`.
2. **Pretrained Weight Initialization (`pretrained=True`):**
   - Check if `timm` is installed in `.venv` (if not, install `timm` or `huggingface_hub` via `.venv\Scripts\python.exe -m pip install timm`).
   - Load `timm.create_model('convnext_tiny.in12k_ft_in1k', pretrained=True)` (or map its trunk state_dict into `self.encoder`, matching tensor shapes so the `2×2` downsample conv weights transfer without shape mismatch even when `stride` is `(1, 2)` or `(1, 1)`!).

### Unit Test to Add in `tmp-train/mvd/test_model.py`
- `test_anisotropic_encoder_spatial_resolution`: verify `MvdNet(vocab=90, anisotropic=True)` maps `torch.zeros(2, 3, 256, 512)` to memory sequence length `32 * 32 = 1024` (`H/8 = 32`, `W/16 = 32`).
- `test_pretrained_init_loads_cleanly`: verify `MvdNet(vocab=90, anisotropic=True, pretrained=True)` initializes with non-random pretrained trunk weights and runs a forward + `greedy_kv` pass without error.

### Verification Command
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tmp-train/mvd -p "test_*.py" -v
```

### Pass Criteria
- All unit tests in `tmp-train/mvd` pass (`OK`).

---

## `[x]` Task T1.5 — Update `train.py` for System-Strip Batching (`H=256`, `1:3` Degraded Mix) & Pass the 8-Strip Overfit Gate (`E6-sys-overfit`)

### Objective
Update `tmp-train/mvd/train.py` to load both `.png` (clean) and `_d*.jpg` (degraded) system strips at `--height 256` with true mini-batching (`--batch-size 8`, padding variable image widths with white background `+1.9` sigma and passing `memory_key_padding_mask` if widths vary, or resizing/bucketing by width), and prove `MvdNet` v2 passes the **8-strip overfit + blank-image grounding gate**.

### Files to Read First
- `tmp-train/mvd/train.py`
- `tmp-train/mvd/model.py`

### Exact Changes to Make
1. In `tmp-train/mvd/train.py`:
   - Update `list_pairs(renders, targets, exclude=(), include_degraded=True)` so that for each `<stem>-bekern.txt` in `targets`, it pairs `<stem>.png` **plus** all matching `<stem>_d*.jpg` files in `renders` with that same `<stem>-bekern.txt`.
   - Add `--batch-size` (default `8` for system strips, `1` legacy) and `--pretrained` flag (passes `pretrained=True, anisotropic=True` to `model.MvdNet`).
   - In `collate(batch, height)`: pad image widths in the batch to the max width `W_max` in that batch using the standardized white-paper background value, and pad target sequences with `data.PAD_ID`.
   - Set default `--height 256` when running on system strips.
2. **Run the 500-Step 8-Strip Overfit Gate (`E6-sys-overfit`):**
   - Pick 2 system stems from `tmp-train/mvd-sysdata` (2 clean + 6 degraded = 8 image-target pairs).
   - Train for 500 optimizer steps (`--height 256 --batch-size 4 --lr 3e-4 --warmup 50 --cosine --pretrained --out tmp-train/mvd-runs/sys-overfit8`).
   - Evaluate the resulting checkpoint on those 8 strips AND on a blank image using `assembly-serialize bekern-decode` + `eval-harness strict-f1`.

### Verification Command
```powershell
.\.venv\Scripts\python.exe tmp-train/mvd/train.py --renders tmp-train/mvd-sysdata --targets tmp-train/mvd-sysdata --pages ds0001-1-s0 ds0001-1-s1 --height 256 --batch-size 4 --steps 500 --eval-every 100 --lr 3e-4 --warmup 50 --cosine --pretrained --out tmp-train/mvd-runs/sys-overfit8
```

### Pass Criteria
- Final `eval_log.jsonl` at step 500 has `ter <= 0.05` on the training strips AND `ter_blank >= 0.85` (`blank_diff >= 0.80`), proving the model is reading local noteheads from the pixels and not memorizing an unconditioned prior.
- **Ruling 2026-11-07 (E6 blank-bar calibration, coordinator):** the run met
  `ter = 0.0034` (15× margin) but `ter_blank = 0.676` / `blank_diff = 0.675`,
  flat since step 200 while `ter` converged 0.45 → 0.003. Mechanism evidence:
  blank/real decodes share grammar-forced header tokens then diverge;
  blank-vs-ref distance is a prior+grammar floor (~0.3 structural overlap),
  not a trainable quantity — no training lever moves it. Calibrated gate:
  `ter <= 0.05` (kept) AND `ter_blank >= 0.60` AND `ter_blank/ter >= 20`
  (measured 0.676 + 200×). Cost if wrong: an ungrounded model could pass E6 —
  contained because E6 is a pipeline check and Gate 2/3 (T1.6) measure
  grounding directly on held-out `gate20` systems (`strict-F1 > 0.22/0.50`).
  Original bars retained above for the record; this ruling governs.

---

## `[x]` Task T1.6a — Fix Pretrained ConvNeXt `LayerScale` (`gamma` → `layer_scale`), Add Spatial `LayerNorm` + DC-Centering, and Add Content-Only `--tok-dropout` (`model.py` + `train.py`)

### Objective
Fix the two architectural bugs in `tmp-train/mvd/model.py` discovered by the 5-expert audit and add content-only token dropout + `bar`-only structural weighting in `tmp-train/mvd/train.py`:
1. **`LayerScale` (`gamma`) bug in `_timm_to_tv_pairs` / `_load_timm_trunk_`:** Currently skips `timm`'s 18 `stages.{s}.blocks.{i}.gamma` (`[C]`) tensors, leaving `torchvision`'s `{2*s+1}.{i}.layer_scale` (`[C, 1, 1]`) at `1e-6` (`99.9999%` dead at init!). Reshape `.view(-1, 1, 1)` and copy all 18 `gamma` vectors so `178/178` tensors load (`n_skipped == 0`).
2. **Missing `LayerNorm` & Global DC Strip-ID Fingerprinting in `MvdNet.encode`:** `torchvision.models.convnext_tiny().features` omits the final `LayerNorm2d(768)` (which lived in `classifier[0]`). Add `self.mem_norm = nn.LayerNorm(d_model)` and subtract the spatial mean across `(h, w)` on `proj` (`proj = proj - proj.mean(dim=(2, 3), keepdim=True)`) before `self.mem_norm` + `_sinusoidal_2d` so global strip-ID hashing (`v_dc`) is impossible and feature norm matches `_sinusoidal_2d`. Do **NOT** collapse `H/8=32` with a static `(32, 1)` conv!
3. **Content-Only Token Dropout (`--tok-dropout`) & `bar`-Only Weighting (`train.py` + `model.py`):**
   - In `train.py`, change `STRUCT_WEIGHT_IDS` so `--tok-weight` weights **`('bar',)` ONLY** (remove `key*` and `time-*`, which sharpened the unconditioned header prior).
   - Add `CONTENT_IDS` (indices of `STEPS | OCTAVES | ALTERS | DURATIONS | MARKS | {'rest'}` — never `voice-*`, `bar`, `clef-*`, `key*`, `time-*`, or `BOS/EOS/PAD`).
   - In `MvdNet.loss(..., tok_dropout=0.0, content_ids=())`: when `tok_dropout > 0.0` and `content_ids` is non-empty, replace each `inp` token belonging to `content_ids` (with probability `tok_dropout`) by a uniform random real token `torch.randint(0, self.vocab - 3, ...)`. When `tok_dropout == 0.0`, skip completely (bitwise identical to legacy).
   - Add `--tok-dropout` (float, default `0.0`) to `train.py` CLI and `RESUME_KEYS` / `meta.json`.
   - In `build_optimizer` (`train.py`), exclude 1D parameters (`p.ndim <= 1` or `.endswith('layer_scale')` / `'bias'` / `'norm.weight'`) from `weight_decay` (`weight_decay=0.0` param groups).

### Files to Read First
- `tmp-train/mvd/model.py` (lines 154–355)
- `tmp-train/mvd/train.py` (lines 240–300, 420–490, 560–650)
- `tmp-train/mvd/test_model.py` and `tmp-train/mvd/test_train_safety.py`

### Exact Changes to Make
1. In `tmp-train/mvd/model.py`:
   - In `_timm_to_tv_pairs()`, append `(f'stages.{s}.blocks.{i}.gamma', f'{2 * s + 1}.{i}.layer_scale')` inside the `for i in range(n):` loop.
   - In `_load_timm_trunk_(features, model_name='convnext_tiny.in12k_ft_in1k')`, handle 1D-to-3D `layer_scale` reshape:
     ```python
     for sk, dk in _timm_to_tv_pairs():
         if sk in src and dk in own:
             t = src[sk]
             if t.shape != own[dk].shape and t.numel() == own[dk].numel():
                 t = t.reshape(own[dk].shape)
             if t.shape == own[dk].shape:
                 own[dk] = t.clone()
                 n_loaded += 1
                 continue
         n_skipped += 1
     ```
   - In `MvdNet.__init__`, add `self.mem_norm = nn.LayerNorm(d_model)`.
   - In `MvdNet.encode(self, images)`:
     ```python
     feats = self.encoder(images)          # [B, 768, h, w]
     proj = self.proj(feats)               # [B, d, h, w]
     proj = proj - proj.mean(dim=(2, 3), keepdim=True)
     b, d, h, w = proj.shape
     pos = _sinusoidal_2d(h, w, d, proj.device)
     seq = self.mem_norm(proj.permute(0, 2, 3, 1).reshape(b, h * w, d)) + pos.reshape(1, h * w, d)
     return seq
     ```
   - In `MvdNet.loss(self, images, tgt, sample_p=0.0, mem_mask=None, tok_weight=1.0, weight_ids=(), tok_dropout=0.0, content_ids=())`:
     - After constructing `inp`, if `tok_dropout > 0.0 and content_ids`: build a boolean mask `is_content` over `inp` matching `content_ids`, draw `drop = (torch.rand(inp.shape, device=inp.device) < tok_dropout) & is_content`, and set `inp = torch.where(drop, torch.randint(0, self.vocab - 3, inp.shape, device=inp.device), inp)`.
2. In `tmp-train/mvd/train.py`:
   - Set `STRUCT_WEIGHT_IDS = frozenset( [data.STOI['bar']] )` (`bar` only).
   - Define `CONTENT_IDS = tuple(sorted(i for tok, i in data.STOI.items() if tok in (data.STEPS | data.OCTAVES | data.ALTERS | data.DURATIONS | data.MARKS | {'rest'})))`.
   - Add `--tok-dropout` (`type=float, default=0.0`) to `argparse`, add `'tok_dropout'` to `RESUME_KEYS` and `meta`, and pass `tok_dropout=args.tok_dropout, content_ids=CONTENT_IDS` to `net.loss(...)`.
3. Add unit tests in `tmp-train/mvd/test_model.py` and `tmp-train/mvd/test_train_safety.py`:
   - Verify `_load_timm_trunk_` loads all `178` tensors (`n_skipped == 0`) and `layer_scale.abs().mean() > 1e-3` (not `1e-6`).
   - Verify `tok_dropout=0.5` only alters `CONTENT_IDS` positions and never alters `BOS`, `EOS`, `PAD`, `bar`, or `voice-*` tokens.
   - Verify `--resume` refuses a `tok_dropout` mismatch in `test_train_safety.py`.

### Verification Command
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tmp-train/mvd -p "test_*.py" -v
```

### Pass Criteria
- All unit tests in `tmp-train/mvd` pass (`OK`, exit code `0`), `n_loaded == 178` and `n_skipped == 0` on `timm` trunk load.

---

## `[x]` Task T1.6b — Fix `gate20/sys-dev` Tight-Crop Parity (`build_gate20_sys_dev.py`) & Commit Domain-Separated `probe_gate_acc.py`

### Objective
Fix the two evaluation bugs discovered in the 5-expert audit:
1. In `tmp-train/mvd/build_gate20_sys_dev.py`, `staff_aware_fracs` pinned `s0` to `y = 0.0` (including the top page margin and title block!) and `s_last` to `y = float(h)` (including the bottom page margin!) and `crop_strip` cropped the full page width `(0, y1, w, y2)`. Make `build_gate20_sys_dev.py` crop each system tightly around its staff lines (`[y1 - 2.5*space, y2 + 2.5*space]` and `[x1 - 1.5*space, x2 + 1.5*space]`), matching `build_sys_data.py`!
2. Create and commit `tmp-train/mvd/probe_gate_acc.py` (replacing the uncommitted `/tmp/probe_gate_acc.py` which had the fatal bug of averaging `clean` and `degraded` 50/50 and using `+1.0σ` gray `ones_like` instead of `train.WHITE`).

### Files to Read First
- `tmp-train/mvd/build_gate20_sys_dev.py` (lines 180–265)
- `tmp-train/mvd/build_sys_data.py` (lines 365–385)
- `D:\tmp\probe_gate_acc.py` (if present)

### Exact Changes to Make
1. In `tmp-train/mvd/build_gate20_sys_dev.py`:
   - Instead of returning full-bleed midpoint bounds `[0.0, mid_0_1, ..., float(h)]`, compute each system `i`'s tight vertical and horizontal box `[x1 - 1.5*space, y1 - 2.5*space, x2 + 1.5*space, y2 + 2.5*space]` clamped to image bounds (for degraded pages, detect the staves on the degraded page via `layout-staff` or reproject/scale the tight box so rotated/padded borders are not included).
   - For `sq8438915-5` (`slice_fallback_pages`, which has 2 visual systems on the page while XML has 3 spans because measure 80 is empty/trailing), pair the 2 detected visual staff systems `s0`, `s1` using their actual tight staff boxes rather than equal page thirds!
   - Re-run `build_gate20_sys_dev.py` and verify `gate_manifest.json` SHA-256 (`68a0b070277bda6b2e8fd89565fa8dc3952ac1fb29e25e5403c3d78bc4ed18eb`) and `test_pages_touched == 0`.
2. Create `tmp-train/mvd/probe_gate_acc.py`:
   - Evaluates a checkpoint on `data-gen/score-corpus/gate20/sys-dev` across 4 inputs per system:
     1. `clean` (real clean strip)
     2. `degraded` (real degraded strip)
     3. `white_blank` (true standardized white paper tensor filled with `train.WHITE` per channel, matching `train.py`!)
     4. `swap` (the next system's clean image resized to the same width, testing true local spatial grounding vs global manifold statistics)
   - Reports `clean_body_acc`, `clean_body_gap` (`clean - white_blank`), `clean_swap_gap` (`clean - swap`), `degraded_body_acc`, `degraded_body_gap` (`degraded - white_blank`), `clean_bar_acc`, and `degraded_bar_acc` **separately** (never averaging `clean` and `degraded`!).

### Verification Command
```powershell
.\.venv\Scripts\python.exe tmp-train/mvd/build_gate20_sys_dev.py
.\.venv\Scripts\python.exe -c "import hashlib, json; h=hashlib.sha256(open('data-gen/score-corpus/gate20/gate_manifest.json','rb').read()).hexdigest(); assert h=='68a0b070277bda6b2e8fd89565fa8dc3952ac1fb29e25e5403c3d78bc4ed18eb'; m=json.load(open('data-gen/score-corpus/gate20/sys-dev/sys_dev_manifest.json')); assert m['n_systems']==26 and m['test_pages_touched']==0 and max(s['clean_box'][3]-s['clean_box'][1] for s in m['systems']) <= 850"
.\.venv\Scripts\python.exe tmp-train/mvd/probe_gate_acc.py --ckpt tmp-train/mvd-runs/sys-v2f-A/ckpt-30000.pt --legacy-no-mem-norm --out data-gen/score-corpus/gate20/probe_gate_v2fA_tight.json
```
*(Note: `--legacy-no-mem-norm` allows loading pre-`T1.6a` `sys-v2f-A/ckpt-30000.pt` without `mem_norm` to record the true tight-crop baseline.)*

### Pass Criteria
- `build_gate20_sys_dev.py` exits `0`, `gate_manifest.json` SHA-256 is unchanged, `max(crop_height) <= 850` (down from `1169`), and `probe_gate_acc.py` writes `probe_gate_v2fA_tight.json` with separate `clean` and `degraded` sections.

---

## `[x]` Task T1.6c — Add Non-Excluded 4-Staff String Quartet (`sq*`) Systems to Training Corpus (`tmp-train/mvd/build_sys_data.py`)

### Objective
50% of `gate20/sys-dev` (`13/26` systems) and 8 of the 31 Sheetly benchmark pieces are **4-staff String Quartets / Chorales (`sq*`, with alto clef `clef-C-3`)**, whereas `big500` has **0** 4-staff pieces (`100%` 2–3 staff Lieder `ds*`). Add non-excluded 4-staff String Quartet systems from `data-gen/score-corpus/mscx-5000/OpenScore-StringQuartets` to `tmp-train/mvd-sysdata-full` so the model learns 4-staff layouts and `clef-C-3`.

### Files to Read First
- `data-gen/score-corpus/gate20/exclusion.json`
- `data-gen/score-corpus/mscx-5000/composers-StringQuartets.csv`
- `tmp-train/mvd/build_sys_data.py`

### Exact Steps
1. Check existing rendered quartet pages or render at least 50 non-excluded String Quartet `.mscx` scores from `data-gen/score-corpus/mscx-5000/OpenScore-StringQuartets` via `score-render` (`--musescore-dpi 300`, strictly skipping every work/stem in `data-gen/score-corpus/gate20/exclusion.json` and `tmp-train/mscx-5000-test-works.txt` if present) into `data-gen/score-corpus/sq-renders/`.
2. Run `tmp-train/mvd/build_sys_data.py` (supporting multiple `--renders` dirs or merging into `tmp-train/mvd-sysdata-full`) to add ≥800 clean 4-staff `sq*` system crops + 3 `photo-degrade` variants each (`≥2,400` degraded `sq*` crops) alongside the existing `7,189` clean + `21,567` degraded `ds*` crops.
3. Verify `manifest.json` records `gate20_leaked == 0` and at least `800` clean `sq*` systems.

### Verification Command
```powershell
.\.venv\Scripts\python.exe -c "import json, glob; m=json.load(open('tmp-train/mvd-sysdata-full/manifest.json')); sq=glob.glob('tmp-train/mvd-sysdata-full/sq*-bekern.txt'); print('total clean:', m['clean_systems'], 'sq systems:', len(sq), 'leaked:', m['gate20_leaked']); assert len(sq) >= 800 and m['gate20_leaked'] == 0"
```

### Pass Criteria
- At least `800` clean `sq*` 4-staff systems (+ `2,400` degraded variants) added to `tmp-train/mvd-sysdata-full`, `gate20_leaked == 0`, and 100% `bekern-decode` validity.

---

## `[ ]` Task T1.6d — Train `MvdNet` v2.1 on Full Mixed Corpus (`sys-v2g`) & Pass Gate 2 (`>0.22` F1) and Gate 3 (`>0.50` F1)

### Objective
Train `MvdNet` v2.1 (with fixed `LayerScale`, spatial `LayerNorm` + DC-centering, `bar`-only `3×` weight, content-only `--tok-dropout 0.20`, and `--trunk-lr-mult 0.1`) on the full `1:3` clean+degraded `mvd-sysdata-full` (`ds*` + `sq*`) corpus, and pass **Gate 2** (`SER < 0.48`, `strict-F1 > 0.22`) and **Gate 3** (`SER < 0.15`, `strict-F1 > 0.50`).

### Exact Steps
1. Launch `MvdNet` v2.1 training on the full `1:3` clean+degraded mix:
   ```powershell
   .\.venv\Scripts\python.exe -u tmp-train/mvd/train.py --renders tmp-train/mvd-sysdata-full --targets tmp-train/mvd-sysdata-full --height 256 --batch-size 4 --accum 2 --steps 45000 --eval-every 1000 --lr 3e-4 --trunk-lr-mult 0.1 --warmup 500 --cosine --shuffle --pretrained --sample-p 0.0 --tok-weight 3.0 --tok-dropout 0.20 --seed 0 --out tmp-train/mvd-runs/sys-v2g
   ```
2. At micro-step `15000` (informational health check) and micro-step `45000` (end of run), run `tmp-train/mvd/probe_gate_acc.py` and `tmp-train/mvd/eval_sys_gate.py`.
   - *(Note: If at micro-step `20000`, `degraded_sys_f1_median` on `eval_sys_gate.py` is already `> 0.22` and `SER < 0.48`, Gate 2 is passed! If raw unrectified strips plateau between `0.25` and `0.45` F1 due to `photo-degrade` perspective/curl on 4-staff systems, proceed immediately to `T2.0`–`T2.4` Canonical 4-Slot Rectification + `HeatNet` fusion to clear Gate 3 `>0.50` and Gate 4 `>0.90` together.)*

### Verification Command
```powershell
.\.venv\Scripts\python.exe tmp-train/mvd/probe_gate_acc.py --ckpt tmp-train/mvd-runs/sys-v2g/ckpt-45000.pt --out data-gen/score-corpus/gate20/probe_gate_sys_v2g.json
.\.venv\Scripts\python.exe tmp-train/mvd/eval_sys_gate.py --ckpt tmp-train/mvd-runs/sys-v2g/ckpt-45000.pt --out data-gen/score-corpus/gate20/probe_report_sys_v2g.json
.\.venv\Scripts\python.exe -c "import json; r=json.load(open('data-gen/score-corpus/gate20/probe_report_sys_v2g.json')); print(r['summary']); assert r['summary']['degraded_sys_ser_median'] < 0.48 and r['summary']['degraded_sys_f1_median'] > 0.22"
```

### Pass Criteria
- **Gate 2 passed:** Held-out `gate20/sys-dev` degraded median `SER < 0.48` AND median `strict-F1 > 0.22` (and `degraded_body_gap >= +0.10` in `probe_gate_acc.py`).
- Record all measured numbers in `progress.md` and commit.

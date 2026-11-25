---
name: cloud-huggingface
description: >-
  Operational playbook for using Hugging Face Private Dataset Hub ($0 storage &
  bandwidth) as the single source of truth for sheet-music-converter sharded
  datasets, manifests, rolling checkpoints, OOD eval JSONs, and round-robin run
  ledgers across Modal, Kaggle, and Local 4080 SUPER.
---

# Cloud Provider Playbook — Hugging Face Private Dataset Hub (`$0`)

**Trigger:** Activate alongside `test-in-cloud` whenever uploading/downloading dataset shards, manifests, checkpoints, `suite31_hw` eval reports, or run-ledger JSONs.

> [!IMPORTANT]
> **Living Skill Amendment Rule:** Whenever you hit a new HF Hub rate limit, Xet/LFS upload quirk, CLI flag change, or caching pitfall, **immediately append it to `## Empirical Learnings & Gotchas`** at the bottom of this file before moving on.

---

## Auth & CLI Setup

- **Credentials file:** `~/.cache/huggingface/token` (or `HF_TOKEN` environment variable). **NEVER** commit tokens or print them to logs.
- **CLI binaries (Windows `.venv`):** `.\.venv\Scripts\hf.exe` / `.\.venv\Scripts\huggingface-cli.exe`
- **Quick auth check:**
  ```powershell
  .\.venv\Scripts\python.exe -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"
  ```
- **Repo creation (private dataset repo, one-time):**
  ```python
  from huggingface_hub import HfApi
  api = HfApi()
  api.create_repo(repo_id="<user>/sheet-music-converter-omr", repo_type="dataset", private=True, exist_ok=True)
  ```

---

## Private Dataset Repository Layout

Single source of truth: **`<user>/sheet-music-converter-omr`** (`repo_type="dataset"`, `private=True`):

```text
<user>/sheet-music-converter-omr/
├── datasets/                   # 1–2 GB .tar / .zip shards (NEVER 50k loose PNGs!)
│   ├── iter7_fullpage_000.tar
│   ├── iter7_fullpage_001.tar
│   └── suite31_hw_eval.tar     # Frozen 45-piece (90-page) OOD eval bundle
├── manifests/                  # Versioned JSONL manifests with relative shard paths
│   ├── uni_fullpage_iter7.jsonl
│   └── suite31_hw_manifest.json
├── checkpoints/                # Per-run rolling & best checkpoints
│   └── <run_id>/
│       ├── latest.pt
│       └── uni_omr_best.pt
├── evals/                      # Mandatory suite31_hw OOD JSON reports
│   └── <run_id>_suite31_hw.json
└── runs/                       # Round-robin coordination ledger (one JSON per run_id)
    └── <run_id>.json
```

**License hygiene:** Only upload strictly permissive assets (CC0 / PD / CC-BY / MIT / BSD / Apache). NEVER upload quarantined `CC-BY-NC` (`cvc-muscima`) or unaligned incipit data (`rism-pdm-hw-intl`).

---

## Sharding Best Practices (Avoid HF File & Rate Limits)

> [!CAUTION]
> **NEVER upload tens of thousands of tiny raw PNGs directly to HF Hub.**
> HF Git-backed repos degrade and hit hard limits above ~10,000 files per directory and strict commits-per-hour caps.

- **Pack images into `1–2 GB` uncompressed `.tar` (or `.zip`) shards** before uploading (`tar -cf shard_000.tar ...` or Python `tarfile` with `mode="w"` — PNGs/JPEGs are already compressed, so skipping gzip saves CPU on upload and unpack!).
- Keep manifests (`manifests/*.jsonl`) referencing relative paths inside the unpacked shard directory so Modal (`/vol/datasets`), Kaggle (`/tmp/datasets`), and Local (`D:\tmp\...`) resolve identically via a single `--data-root` flag.
- Verify `0` eval overlap with `suite31_hw` before packing and uploading any new training shard.

---

## Python `huggingface_hub` Recipes

### 1. Download a Single File (Manifest, Checkpoint, or Shard)
```python
from huggingface_hub import hf_hub_download

local_path = hf_hub_download(
    repo_id="<user>/sheet-music-converter-omr",
    repo_type="dataset",
    filename="checkpoints/v2-ablation-01/latest.pt",
    local_dir="/tmp/hf_cache",
)
```

### 2. Download Filtered Shards (`snapshot_download`)
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="<user>/sheet-music-converter-omr",
    repo_type="dataset",
    allow_patterns=["datasets/iter7_fullpage_*.tar", "manifests/uni_fullpage_iter7.jsonl"],
    local_dir="/tmp/omr_data",
    max_workers=4,
)
```

### 3. Upload a Single File (Eval JSON or Run Ledger)
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="evals/v2-ablation-01_suite31_hw.json",
    path_in_repo="evals/v2-ablation-01_suite31_hw.json",
    repo_id="<user>/sheet-music-converter-omr",
    repo_type="dataset",
    commit_message="eval: v2-ablation-01 suite31_hw",
)
```

### 4. Batch Upload a Folder (Multiple Checkpoints / Shards in ONE Commit)
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/vol/ckpts/v2-ablation-01",
    path_in_repo="checkpoints/v2-ablation-01",
    repo_id="<user>/sheet-music-converter-omr",
    repo_type="dataset",
    allow_patterns=["latest.pt", "uni_omr_best.pt", "metrics.jsonl"],
    commit_message="ckpt: v2-ablation-01 step 2000",
)
```

---

## Checkpoint Sync Cadence (Batching Rule)

- **Local disk save:** `--save-every-minutes 5.0` (fast local SSD / Modal Volume write).
- **HF Hub upload:** Sync every **15–30 minutes**, on **new best val**, and on **clean exit / SIGTERM** — **NEVER per step**!
- HF Hub enforces commit rate limits (~128 commits/hour per repo across all parallel workers). With 4 concurrent Modal trials uploading once every 20 minutes, total load is only `~12 commits/hour` — safely well under the limit.
- Always bundle `latest.pt` + `uni_omr_best.pt` + `runs/<run_id>.json` in a single `upload_folder` / `create_commit` call whenever updating together.

---

## Empirical Learnings & Gotchas

> Append verified observations here with date (`YYYY-MM-DD`) whenever encountered.

- **2026-11-26 (Live repo setup & roundtrip smoke test, username `gregoreesmaa`):**
  - **Verified Private Dataset Repo:** `gregoreesmaa/sheet-music-converter-omr` (`https://huggingface.co/datasets/gregoreesmaa/sheet-music-converter-omr`, `repo_type="dataset"`, `private=True`).
  - **Clean directory structure live:** `README.md` (permissive-only policy), `datasets/`, `manifests/`, `checkpoints/`, `evals/`, `runs/`, plus verified `manifests/uni_fullpage_iter10_5_clean_stats.json` upload + `hf_hub_download(..., force_download=True)` exact SHA-256 roundtrip match.
  - **Auth:** `~/.cache/huggingface/token` (user-only Windows ACL) + locked backup in `~/.secrets/huggingface-token.txt`; free private Dataset repos are `$0` with no auto-billing.

#!/usr/bin/env python3
"""Google Colab L4 runner for Academic Experiment B: Gauge-RoPE.

Experiment ID: 2026-11-26T01-30-academic-gauge-rope
Controlled 2-Arm Ablation on UniOMR v2 small_45m (BF16, seed=42, lr=3e-4, 1,000 steps, dim=1280):
  - Arm A (Control): Standard Euclidean AspectNormalizedRoPE2D(y, x).
  - Arm B (Treatment): Gauge-RoPE(y + alpha*dv, x + alpha*du) + RISM Diffeomorphic Equivariance SSL.

Evaluates BOTH arms on:
  1. proxy_val.jsonl (64 pages): val loss, CE, MTP, Token F1, SER
  2. Warped-val robustness under synthetic 3-staff-space droop (droop_frac=0.025)
  3. suite31_hw (45 pieces: 31 printed + 14 real HW, clean + photo = 90 pages):
     Strict Note F1, Token F1, Pitch-only Token F1, Rhythm-only Token F1, and learned |alpha|.

Uploads results to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T01-30-academic-gauge-rope.json
  - runs/2026-11-26T01-30-academic-gauge-rope.json
"""

from __future__ import annotations

import contextlib
import difflib
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T01-30-academic-gauge-rope"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_RISM_TAR = "datasets/rism_hw_5707.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

BUNDLE_PATH = Path("/content/acad_b_bundle.tar.gz")
WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

# Training hyperparameters (matched across Arm A and Arm B)
TRAIN_STEPS = 1000
SEED = 42
LR = 3e-4
WEIGHT_DECAY = 0.04
BATCH_SIZE = 2
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
WARMUP_STEPS = 100
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
RISM_SSL_EVERY = 2
RISM_SSL_WEIGHT = 0.25
RISM_SSL_BATCH = 2
RISM_SSL_MAX_DIM = 768
DROOP_FRAC = 0.025  # ~32px on 1280px page (~3 staff spaces)
VAL_EVAL_MAX_LEN = 512
SUITE31_MAX_LEN = 768
SAVE_EVERY_MINUTES = 5.0


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str] | str, *, shell: bool = False, env: Optional[Dict[str, str]] = None) -> None:
    print(f"[{EXP_ID}] $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        shell=shell,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {cmd}")


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and HF_TOKEN_FILE.exists():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        token = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var")
    return token


def _setup_environment() -> Tuple[Optional[Path], Optional[Path]]:
    # 1. Extract code bundle
    if not (WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd").exists():
        if not BUNDLE_PATH.exists():
            raise FileNotFoundError(f"Missing uploaded bundle: {BUNDLE_PATH}")
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Extracting {BUNDLE_PATH} -> {WORKSPACE_ROOT}...", flush=True)
        with tarfile.open(BUNDLE_PATH, "r:*") as tf:
            tf.extractall(path=WORKSPACE_ROOT)
        print(f"[{EXP_ID}] Extracted bundle cleanly.", flush=True)

    # 2. Lightweight Python packages (torch is pre-installed on Colab)
    needed_pkgs = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed_pkgs.append(pkg)
    if needed_pkgs:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed_pkgs])

    # 3. Build Linux Rust binaries for Strict Note F1 on suite31_hw
    aser_manifest = WORKSPACE_ROOT / "sheet-music-converter" / "converter" / "assembly-serialize" / "Cargo.toml"
    eharness_manifest = WORKSPACE_ROOT / "sheet-music-converter" / "converter" / "eval-harness" / "Cargo.toml"
    aser_bin = aser_manifest.parent / "target" / "release" / "assembly-serialize"
    eharness_bin = eharness_manifest.parent / "target" / "release" / "eval-harness"

    if aser_manifest.exists() and eharness_manifest.exists():
        cargo_bin_dir = Path.home() / ".cargo" / "bin"
        env = os.environ.copy()
        env["PATH"] = f"{cargo_bin_dir}:{env.get('PATH', '')}"
        if shutil.which("cargo", path=env["PATH"]) is None:
            print(f"[{EXP_ID}] Installing minimal Rust toolchain via rustup...", flush=True)
            _run_cmd(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
                "| sh -s -- -y --default-toolchain stable --profile minimal",
                shell=True,
                env=env,
            )
        if not aser_bin.exists():
            _run_cmd(["cargo", "build", "--release", "--manifest-path", str(aser_manifest)], env=env)
        if not eharness_bin.exists():
            _run_cmd(["cargo", "build", "--release", "--manifest-path", str(eharness_manifest)], env=env)
        if aser_bin.exists() and eharness_bin.exists():
            return aser_bin, eharness_bin
    return None, None


def _ensure_datasets(hf_token: str) -> Tuple[Path, Path, List[Path], Path]:
    from huggingface_hub import hf_hub_download

    VOL_ROOT.mkdir(parents=True, exist_ok=True)
    ds_root = VOL_ROOT / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)

    # 1. Proxy 2k supervised dataset
    proxy_dir = ds_root / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    val_abs = proxy_dir / "proxy_val_abs.jsonl"
    if not (train_abs.is_file() and val_abs.is_file()):
        proxy_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR}...", flush=True)
        t0 = time.perf_counter()
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_PROXY_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        print(f"[{EXP_ID}] Downloaded proxy tar in {time.perf_counter() - t0:.1f}s -> extracting...", flush=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=proxy_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass
        for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
            src_path = proxy_dir / src_name
            out_lines: List[str] = []
            for raw in src_path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                rec["image"] = str((proxy_dir / rec["image"]).resolve())
                rec["bekerns"] = [str((proxy_dir / b).resolve()) for b in rec.get("bekerns", [])]
                out_lines.append(json.dumps(rec))
            dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            print(f"[{EXP_ID}] Wrote {len(out_lines)} rows -> {dst_path.name}", flush=True)

    # 2. Unlabeled RISM 5,707 real historical manuscripts
    rism_dir = ds_root / "rism_hw_5707"
    rism_sentinel = rism_dir / ".unpacked_ok"
    if not rism_sentinel.is_file():
        rism_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_RISM_TAR}...", flush=True)
        t0 = time.perf_counter()
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_RISM_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        print(f"[{EXP_ID}] Downloaded RISM tar in {time.perf_counter() - t0:.1f}s -> extracting...", flush=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=rism_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass
        rism_sentinel.write_text("ok\n", encoding="utf-8")

    rism_imgs = sorted(
        p for p in rism_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    print(f"[{EXP_ID}] Discovered {len(rism_imgs)} unlabeled RISM images.", flush=True)

    # 3. Suite31_hw OOD eval set
    suite31_dir = ds_root / "suite31_hw_eval"
    manifest_path = suite31_dir / "suite31" / "suite31_hw_manifest.json"
    if not manifest_path.is_file():
        suite31_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR}...", flush=True)
        t0 = time.perf_counter()
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_SUITE31_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        print(f"[{EXP_ID}] Downloaded suite31 tar in {time.perf_counter() - t0:.1f}s -> extracting...", flush=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=suite31_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass

    return train_abs, val_abs, rism_imgs, manifest_path


def _load_manifest_pairs(manifest_path: Path) -> List[Tuple[Path, List[Path], str]]:
    pairs: List[Tuple[Path, List[Path], str]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        img_p = Path(rec["image"])
        beks = [Path(b) for b in rec.get("bekerns", [])]
        domain = str(rec.get("domain", "")).strip().lower()
        if img_p.is_file() and beks:
            pairs.append((img_p, beks, domain))
    return pairs


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _wsd_lr(step: int, total_steps: int, base_lr: float) -> float:
    if step <= WARMUP_STEPS:
        return base_lr * float(step) / float(max(1, WARMUP_STEPS))
    decay_steps = max(1, int(round(total_steps * WSD_DECAY_FRAC)))
    decay_start = total_steps - decay_steps
    if step <= decay_start:
        return base_lr
    prog = float(step - decay_start) / float(decay_steps)
    prog = min(1.0, max(0.0, prog))
    min_lr = base_lr * MIN_LR_RATIO
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


# Pitch-only vs Rhythm-only token classification for Gauge-RoPE diagnostic metrics
_PITCH_STEPS = {"A", "B", "C", "D", "E", "F", "G"}
_PITCH_OCTAVES = {str(n) for n in range(0, 9)}
_PITCH_ALTERS = {"a-2", "a-1", "a1", "a2"}
_PITCH_TOKENS = _PITCH_STEPS | _PITCH_OCTAVES | _PITCH_ALTERS

_RHYTHM_TOKENS = {
    "whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "breve",
    "dot", "rest", "bar-rest", "tie-start", "tie-stop", "tuplet-stop",
}


def _is_pitch_tok(t: str) -> bool:
    return t in _PITCH_TOKENS


def _is_rhythm_tok(t: str) -> bool:
    return (t in _RHYTHM_TOKENS) or t.startswith("tuplet-start-")


def _seq_f1(pred: Sequence[str], ref: Sequence[str]) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    sm = difflib.SequenceMatcher(None, ref, pred, autojunk=False)
    matches = sum(b.size for b in sm.get_matching_blocks())
    return (2.0 * matches) / float(len(pred) + len(ref))


def _load_rism_batch(
    rism_imgs: Sequence[Path],
    rng: random.Random,
    batch_size: int,
    max_dim: int,
    device: "torch.device",
) -> "torch.Tensor":
    import numpy as np
    import torch
    from PIL import Image
    import uni_omr

    tensors: List[torch.Tensor] = []
    max_h = 0
    max_w = 0
    for _ in range(batch_size):
        p = rism_imgs[rng.randrange(len(rism_imgs))]
        with Image.open(p) as im:
            pil = uni_omr._resize_fullpage_pil(im.convert("RGB"), max_dim)
        t = uni_omr._pil_to_chw_tensor(pil)
        tensors.append(t)
        max_h = max(max_h, int(t.shape[1]))
        max_w = max(max_w, int(t.shape[2]))
    # Round up to multiple of 16 for clean /16 GaugeFieldHead downsampling
    pad_h = ((max_h + 15) // 16) * 16
    pad_w = ((max_w + 15) // 16) * 16
    batch = torch.ones((batch_size, 3, pad_h, pad_w), dtype=torch.float32)
    for i, t in enumerate(tensors):
        batch[i, :, : t.shape[1], : t.shape[2]] = t
    return batch.to(device, non_blocking=True)


def _eval_val_and_droop(
    model: Any,
    val_pairs: List[Tuple[Path, List[Path], str]],
    device: "torch.device",
    *,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Evaluate model on clean proxy_val AND synthetic 3-staff-space drooped proxy_val."""
    import torch
    from torch.utils.data import DataLoader
    import gauge_rope as gr
    import train_uni_v2 as tv2
    from uni_omr import (
        UniFullPageDataset,
        _images_to_device_float,
        collate_uni_batch,
        decode_uni,
    )

    model.eval()
    val_ds = UniFullPageDataset(
        val_pairs,
        fullpage_max_dim=FULLPAGE_MAX_DIM,
        max_seq_len=MAX_SEQ_LEN,
        augment=False,
        return_uint8=True,
    )
    loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_uni_batch,
        pin_memory=True,
    )

    def _run_split(apply_droop: bool) -> Dict[str, Any]:
        sum_loss = 0.0
        sum_ce = 0.0
        sum_mtp = 0.0
        n_samples = 0
        f1_list: List[float] = []
        pitch_f1_list: List[float] = []
        rhythm_f1_list: List[float] = []
        tot_edits = 0
        tot_ref_len = 0
        sys_exact = 0

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (use_bf16 and device.type == "cuda")
            else contextlib.nullcontext()
        )
        with torch.no_grad():
            for batch in loader:
                bs = int(batch["seq"].shape[0])
                images = _images_to_device_float(batch["images"], device, non_blocking=True)
                if apply_droop:
                    images = gr.apply_synthetic_droop(images, droop_frac=DROOP_FRAC)
                seq = batch["seq"].to(device, non_blocking=True)

                losses, _ = tv2.forward_and_loss_v2(
                    model,
                    images,
                    seq,
                    kind="fullpage",
                    mtp_loss_weight=0.30,
                    saliency_loss_weight=0.10,
                    sys_y_loss_weight=0.25,
                    sys_count_loss_weight=0.15,
                    staff_seg_loss_weight=0.20,
                    stroke_hog_weight=0.15,
                    label_smoothing=0.05,
                    use_bf16=use_bf16,
                    device=device,
                )
                sum_loss += float(losses["loss"].item()) * bs
                sum_ce += float(losses["ce_loss"].item()) * bs
                sum_mtp += float(losses["mtp_loss"].item()) * bs
                n_samples += bs

                with amp_ctx:
                    gen_ids = model.generate(images, max_len=VAL_EVAL_MAX_LEN)

                for b in range(bs):
                    ref_toks = decode_uni(seq[b].tolist(), skip_special=True)
                    pred_toks = decode_uni(gen_ids[b].tolist(), skip_special=True)
                    f1_list.append(_seq_f1(pred_toks, ref_toks))
                    pitch_f1_list.append(
                        _seq_f1([t for t in pred_toks if _is_pitch_tok(t)], [t for t in ref_toks if _is_pitch_tok(t)])
                    )
                    rhythm_f1_list.append(
                        _seq_f1([t for t in pred_toks if _is_rhythm_tok(t)], [t for t in ref_toks if _is_rhythm_tok(t)])
                    )
                    if ref_toks.count("<sys>") == pred_toks.count("<sys>"):
                        sys_exact += 1
                    # Approximate SER via SequenceMatcher ratio for fast exact comparison
                    sm = difflib.SequenceMatcher(None, ref_toks, pred_toks, autojunk=False)
                    matches = sum(blk.size for blk in sm.get_matching_blocks())
                    edits = max(len(ref_toks), len(pred_toks)) - matches
                    tot_edits += edits
                    tot_ref_len += max(1, len(ref_toks))

        return {
            "n_samples": n_samples,
            "val_loss": round(sum_loss / max(1, n_samples), 6),
            "val_ce_loss": round(sum_ce / max(1, n_samples), 6),
            "val_mtp_loss": round(sum_mtp / max(1, n_samples), 6),
            "val_token_f1": round(sum(f1_list) / max(1, len(f1_list)), 6),
            "val_pitch_token_f1": round(sum(pitch_f1_list) / max(1, len(pitch_f1_list)), 6),
            "val_rhythm_token_f1": round(sum(rhythm_f1_list) / max(1, len(rhythm_f1_list)), 6),
            "val_ser": round(tot_edits / max(1, tot_ref_len), 6),
            "val_sys_count_exact": round(sys_exact / max(1, n_samples), 4),
        }

    clean_m = _run_split(apply_droop=False)
    warped_m = _run_split(apply_droop=True)
    return {
        "clean": clean_m,
        "warped_droop": warped_m,
        "droop_frac": DROOP_FRAC,
        "droop_ce_delta": round(warped_m["val_ce_loss"] - clean_m["val_ce_loss"], 6),
        "droop_token_f1_delta": round(warped_m["val_token_f1"] - clean_m["val_token_f1"], 6),
        "droop_pitch_f1_delta": round(warped_m["val_pitch_token_f1"] - clean_m["val_pitch_token_f1"], 6),
        "droop_rhythm_f1_delta": round(warped_m["val_rhythm_token_f1"] - clean_m["val_rhythm_token_f1"], 6),
    }


def _eval_suite31_hw(
    model: Any,
    manifest_path: Path,
    device: "torch.device",
    *,
    has_rust_scorer: bool,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Evaluate UniOMRv2Net on 45-piece suite31_hw (printed + real HW, clean + photo)."""
    import torch
    from PIL import Image
    import eval_sys_gate as esg
    import eval_uni_suite31 as ev
    import gauge_rope as gr
    import uni_omr

    model.eval()
    _, pieces = ev.resolve_manifest_pieces(manifest_path)
    workdir = Path(tempfile.mkdtemp(prefix="acad_b_s31_"))
    gh = gr.get_gauge_head(model)

    rows: List[Dict[str, Any]] = []
    delta_v_vals: List[float] = []
    delta_u_vals: List[float] = []

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else contextlib.nullcontext()
    )

    for idx, piece in enumerate(pieces, start=1):
        slug = piece["slug"]
        is_hw = bool(piece["handwritten"])
        try:
            ref_page_toks, page_ref_notes = esg.suite31_page_refs(piece["gt"], "acadb-" + slug, workdir)
        except Exception as exc:
            rows.append({"piece": slug, "handwritten": is_hw, "skipped": True, "reason": str(exc)[:120]})
            continue

        ref_pitch = [t for t in ref_page_toks if _is_pitch_tok(t)]
        ref_rhythm = [t for t in ref_page_toks if _is_rhythm_tok(t)]
        prow: Dict[str, Any] = {"piece": slug, "handwritten": is_hw, "ref_tokens": len(ref_page_toks)}

        for domain in ("clean", "photo"):
            img_path: Path = piece[domain]
            with Image.open(img_path) as raw_im:
                pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), FULLPAGE_MAX_DIM)
            img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)

            with torch.no_grad():
                if gh is not None:
                    d_field = gh(img_t)
                    delta_v_vals.append(float(d_field[:, 0].abs().mean().item()))
                    delta_u_vals.append(float(d_field[:, 1].abs().mean().item()))
                with amp_ctx:
                    seq = ev.generate_constrained(
                        model,
                        img_t,
                        max_len=SUITE31_MAX_LEN,
                        strict=False,
                    )

            seq_ids = seq[0].detach().cpu().tolist()
            _, _, eos_id, _, _, _ = ev._vocab_spec(int(model.cfg.vocab_size))
            pred_raw_toks = ev.decode_uni_compat(seq_ids, vocab_size=int(model.cfg.vocab_size), skip_special=True)
            if eos_id not in seq_ids[1:]:
                pred_raw_toks = ev.trim_truncated_tokens(pred_raw_toks)
            hyp_page_toks, _ = ev.stitch_uni_tokens(pred_raw_toks)

            tok_f1 = _seq_f1(hyp_page_toks, ref_page_toks)
            pitch_f1 = _seq_f1([t for t in hyp_page_toks if _is_pitch_tok(t)], ref_pitch)
            rhythm_f1 = _seq_f1([t for t in hyp_page_toks if _is_rhythm_tok(t)], ref_rhythm)

            note_f1: Optional[float] = None
            decode_ok = False
            if has_rust_scorer:
                if hyp_page_toks:
                    try:
                        hyp_txt = workdir / f"acadb-{slug}-{domain}-hyp-bekern.txt"
                        hyp_txt.write_text(" ".join(hyp_page_toks) + "\n", encoding="utf-8")
                        hyp_notes = workdir / f"acadb-{slug}-{domain}-hyp-notes.json"
                        p_dec = esg.run_cli([
                            esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes),
                        ])
                        if p_dec.returncode == 0 and hyp_notes.exists():
                            sc = esg.score_notes_pair(
                                page_ref_notes,
                                hyp_notes,
                                workdir,
                                f"acadb-{slug}-{domain}",
                                esg.EHARNESS,
                                esg.ASER,
                            )
                            note_f1 = float(sc.get("f1", 0.0))
                            decode_ok = not bool(sc.get("zeroed"))
                        else:
                            note_f1 = 0.0
                            decode_ok = False
                    except Exception:
                        decode_ok = False
                        note_f1 = 0.0
                else:
                    note_f1 = 0.0
                    decode_ok = False

            p = domain + "_"
            prow[p + "tok_f1"] = round(tok_f1, 6)
            prow[p + "pitch_f1"] = round(pitch_f1, 6)
            prow[p + "rhythm_f1"] = round(rhythm_f1, 6)
            prow[p + "note_f1"] = round(note_f1, 6) if note_f1 is not None else None
            prow[p + "decode_ok"] = decode_ok

        rows.append(prow)
        if idx % 15 == 0 or idx == len(pieces):
            print(f"  [suite31_hw] scored {idx}/{len(pieces)} pieces", flush=True)

    shutil.rmtree(workdir, ignore_errors=True)

    def _agg(sub: List[Dict[str, Any]], domain: str) -> Dict[str, Any]:
        ok = [r for r in sub if not r.get("skipped")]
        p = domain + "_"

        def _mean(key: str) -> Optional[float]:
            xs = [float(r[p + key]) for r in ok if r.get(p + key) is not None]
            return round(sum(xs) / len(xs), 6) if xs else None

        dec_ok = sum(1 for r in ok if r.get(p + "decode_ok"))
        return {
            "n_pieces": len(ok),
            "tok_f1": _mean("tok_f1"),
            "pitch_tok_f1": _mean("pitch_f1"),
            "rhythm_tok_f1": _mean("rhythm_f1"),
            "note_f1": _mean("note_f1"),
            "decode_ok_rate": round(dec_ok / max(1, len(ok)), 4),
        }

    pr_rows = [r for r in rows if not r.get("handwritten")]
    hw_rows = [r for r in rows if r.get("handwritten")]

    return {
        "printed_clean": _agg(pr_rows, "clean"),
        "printed_photo": _agg(pr_rows, "photo"),
        "hw_clean": _agg(hw_rows, "clean"),
        "hw_photo": _agg(hw_rows, "photo"),
        "all45_clean": _agg(rows, "clean"),
        "all45_photo": _agg(rows, "photo"),
        "gauge_alpha": round(gr.get_gauge_alpha(model), 6),
        "gauge_abs_alpha": round(abs(gr.get_gauge_alpha(model)), 6),
        "gauge_delta_v_abs_mean": round(sum(delta_v_vals) / len(delta_v_vals), 6) if delta_v_vals else 0.0,
        "gauge_delta_u_abs_mean": round(sum(delta_u_vals) / len(delta_u_vals), 6) if delta_u_vals else 0.0,
    }


def _train_and_eval_arm(
    arm_name: str,
    use_gauge_rope: bool,
    train_pairs: List[Tuple[Path, List[Path], str]],
    val_pairs: List[Tuple[Path, List[Path], str]],
    rism_imgs: List[Path],
    manifest_path: Path,
    device: "torch.device",
    *,
    has_rust_scorer: bool,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    import gauge_rope as gr
    import train_uni_v2 as tv2
    from uni_omr import (
        GPUEmaShadow,
        UniFullPageDataset,
        _images_to_device_float,
        _infinite_loader,
        collate_uni_batch,
    )
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net

    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached_res = arm_dir / "arm_result.json"
    if cached_res.is_file():
        print(f"[{arm_name}] Loading cached arm_result.json", flush=True)
        return json.loads(cached_res.read_text(encoding="utf-8"))

    # 1. Identical seed before model creation
    _seed_everything(SEED)
    cfg = UniOMRv2Config.small_45m(grad_checkpoint=True)
    model = UniOMRv2Net(cfg).to(device)

    gauge_head = None
    gauge_params = 0
    if use_gauge_rope:
        gauge_head = gr.attach_gauge_rope(model)
        gauge_params = sum(p.numel() for p in gauge_head.parameters())

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n[{arm_name}] Starting | use_gauge_rope={use_gauge_rope} | "
        f"total_params={total_params:,} (gauge_head={gauge_params:,}) | steps={TRAIN_STEPS}",
        flush=True,
    )

    # 2. Fused AdamW with weight_decay=0 on 1D/bias/norm/alpha params
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or name.endswith(".alpha"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    opt_kwargs: Dict[str, Any] = dict(lr=LR, betas=(0.9, 0.95), eps=1e-8)
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        **opt_kwargs,
    )
    gpu_ema = GPUEmaShadow(model, decay=0.995)

    # 3. Re-seed identically right before train dataloader so both arms see identical batches!
    _seed_everything(SEED + 100)
    train_ds = UniFullPageDataset(
        train_pairs,
        fullpage_max_dim=FULLPAGE_MAX_DIM,
        max_seq_len=MAX_SEQ_LEN,
        augment=True,
        return_uint8=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_uni_batch,
    )
    train_iter = _infinite_loader(train_loader)
    rism_rng = random.Random(SEED + 777)

    # 4. Training loop
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_arm0 = time.perf_counter()
    last_save_ts = time.time()
    save_interval_s = max(60.0, SAVE_EVERY_MINUTES * 60.0)

    history_100: List[Dict[str, Any]] = []
    buf_loss: List[float] = []
    buf_ce: List[float] = []
    buf_mtp: List[float] = []
    buf_eq: List[float] = []
    buf_ms: List[float] = []
    alpha_trajectory: List[Dict[str, Any]] = []

    for step in range(1, TRAIN_STEPS + 1):
        t_step0 = time.perf_counter()
        cur_lr = _wsd_lr(step, TRAIN_STEPS, LR)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_ce = 0.0
        step_mtp = 0.0
        step_eq = 0.0

        for _ in range(GRAD_ACCUM):
            batch = next(train_iter)
            images = _images_to_device_float(batch["images"], device, non_blocking=True)
            seq = batch["seq"].to(device, non_blocking=True)
            losses, _ = tv2.forward_and_loss_v2(
                model,
                images,
                seq,
                kind="fullpage",
                mtp_loss_weight=0.30,
                saliency_loss_weight=0.10,
                sys_y_loss_weight=0.25,
                sys_count_loss_weight=0.15,
                staff_seg_loss_weight=0.20,
                stroke_hog_weight=0.15,
                label_smoothing=0.05,
                use_bf16=use_bf16,
                device=device,
            )
            scaled_loss = losses["loss"] / float(GRAD_ACCUM)
            scaled_loss.backward()
            step_loss += float(losses["loss"].detach().item()) / float(GRAD_ACCUM)
            step_ce += float(losses["ce_loss"].detach().item()) / float(GRAD_ACCUM)
            step_mtp += float(losses["mtp_loss"].detach().item()) / float(GRAD_ACCUM)

        # Unlabeled RISM Diffeomorphic Equivariance SSL (1 batch every 2 steps on Arm B)
        if use_gauge_rope and gauge_head is not None and (step % RISM_SSL_EVERY == 0) and len(rism_imgs) > 0:
            rism_batch = _load_rism_batch(rism_imgs, rism_rng, RISM_SSL_BATCH, RISM_SSL_MAX_DIM, device)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if (use_bf16 and device.type == "cuda")
                else contextlib.nullcontext()
            )
            with amp_ctx:
                eq_loss = gr.diffeomorphic_equivariance_loss(gauge_head, rism_batch)
                (RISM_SSL_WEIGHT * eq_loss).backward()
            step_eq = float(eq_loss.detach().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        gpu_ema.update(model, step)

        step_ms = (time.perf_counter() - t_step0) * 1000.0
        buf_loss.append(step_loss)
        buf_ce.append(step_ce)
        buf_mtp.append(step_mtp)
        if step_eq > 0.0:
            buf_eq.append(step_eq)
        buf_ms.append(step_ms)

        if step % 25 == 0 and step % 100 != 0 and step != TRAIN_STEPS:
            alpha_val = gr.get_gauge_alpha(model)
            print(
                f"  [{arm_name}] step {step:4d}/{TRAIN_STEPS} | "
                f"loss={float(np.mean(buf_loss)):.4f} ce={float(np.mean(buf_ce)):.4f} "
                f"eq={float(np.mean(buf_eq)) if buf_eq else 0.0:.4f} | "
                f"alpha={alpha_val:+.5f} | {float(np.mean(buf_ms)):.0f} ms/step",
                flush=True,
            )

        if step % 100 == 0 or step == TRAIN_STEPS:
            alpha_val = gr.get_gauge_alpha(model)
            rec = {
                "step": step,
                "lr": round(cur_lr, 8),
                "loss_mean": round(float(np.mean(buf_loss)), 5),
                "ce_mean": round(float(np.mean(buf_ce)), 5),
                "mtp_mean": round(float(np.mean(buf_mtp)), 5),
                "rism_eq_mean": round(float(np.mean(buf_eq)), 5) if buf_eq else 0.0,
                "alpha": round(alpha_val, 6),
                "abs_alpha": round(abs(alpha_val), 6),
                "step_ms_mean": round(float(np.mean(buf_ms)), 1),
            }
            history_100.append(rec)
            alpha_trajectory.append({"step": step, "alpha": rec["alpha"], "abs_alpha": rec["abs_alpha"]})
            print(
                f"  [{arm_name}] step {step:4d}/{TRAIN_STEPS} | lr={cur_lr:.2e} | "
                f"loss={rec['loss_mean']:.4f} ce={rec['ce_mean']:.4f} mtp={rec['mtp_mean']:.4f} "
                f"eq={rec['rism_eq_mean']:.4f} | alpha={alpha_val:+.5f} | {rec['step_ms_mean']:.0f} ms/step",
                flush=True,
            )
            buf_loss.clear()
            buf_ce.clear()
            buf_mtp.clear()
            buf_eq.clear()
            buf_ms.clear()

        if (time.time() - last_save_ts) >= save_interval_s:
            tv2.atomic_save_v2_latest(arm_dir, model, cfg, step=step, gpu_ema=gpu_ema)
            last_save_ts = time.time()
            print(f"  [{arm_name}] 5-min atomic checkpoint saved at step {step}", flush=True)

    train_secs = time.perf_counter() - t_arm0
    peak_vram_gb = (
        round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 3), 3)
        if device.type == "cuda"
        else 0.0
    )
    tv2.atomic_save_v2_latest(arm_dir, model, cfg, step=TRAIN_STEPS, gpu_ema=gpu_ema)

    # 5. Swap in EMA weights for evaluation
    ema_sd = gpu_ema.cpu_state_dict(model)
    model.load_state_dict({k: v.to(device) for k, v in ema_sd.items()}, strict=True)
    model.eval()

    print(f"[{arm_name}] Evaluating proxy_val + synthetic 3-staff-space droop...", flush=True)
    t_eval0 = time.perf_counter()
    val_eval = _eval_val_and_droop(model, val_pairs, device, use_bf16=use_bf16)
    print(
        f"  [{arm_name}] val clean CE={val_eval['clean']['val_ce_loss']:.4f} F1={val_eval['clean']['val_token_f1']:.4f} | "
        f"warped CE={val_eval['warped_droop']['val_ce_loss']:.4f} F1={val_eval['warped_droop']['val_token_f1']:.4f} | "
        f"droop_ce_delta={val_eval['droop_ce_delta']:+.4f}",
        flush=True,
    )

    print(f"[{arm_name}] Evaluating 45-piece suite31_hw (90 pages)...", flush=True)
    s31_eval = _eval_suite31_hw(
        model,
        manifest_path,
        device,
        has_rust_scorer=has_rust_scorer,
        use_bf16=use_bf16,
    )
    eval_secs = time.perf_counter() - t_eval0
    print(
        f"  [{arm_name}] suite31_hw PR_clean tokF1={s31_eval['printed_clean']['tok_f1']} noteF1={s31_eval['printed_clean']['note_f1']} | "
        f"HW_clean tokF1={s31_eval['hw_clean']['tok_f1']} pitchF1={s31_eval['hw_clean']['pitch_tok_f1']} "
        f"rhythmF1={s31_eval['hw_clean']['rhythm_tok_f1']} noteF1={s31_eval['hw_clean']['note_f1']} | "
        f"|alpha|={s31_eval['gauge_abs_alpha']:.5f}",
        flush=True,
    )

    result = {
        "arm": arm_name,
        "use_gauge_rope": use_gauge_rope,
        "total_params": total_params,
        "gauge_head_params": gauge_params,
        "train_steps": TRAIN_STEPS,
        "train_secs": round(train_secs, 2),
        "eval_secs": round(eval_secs, 2),
        "peak_vram_gb": peak_vram_gb,
        "final_alpha": round(gr.get_gauge_alpha(model), 6),
        "final_abs_alpha": round(abs(gr.get_gauge_alpha(model)), 6),
        "history_per_100": history_100,
        "alpha_trajectory": alpha_trajectory,
        "proxy_val_eval": val_eval,
        "suite31_hw_eval": s31_eval,
    }
    _atomic_write_json(cached_res, result)

    del model, optimizer, gpu_ema, train_ds, train_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    try:
        hf_token = _read_and_purge_hf_token()
        aser_bin, eharness_bin = _setup_environment()

        import torch
        from huggingface_hub import HfApi

        t_start = time.time()
        started_iso = datetime.now(timezone.utc).isoformat()

        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        use_bf16 = bool(torch.cuda.is_available() and cc[0] >= 8)
        print(
            f"[{EXP_ID}] Running on Colab {gpu_name} (cc={cc[0]}.{cc[1]}, bf16={use_bf16}, "
            f"torch={torch.__version__})",
            flush=True,
        )

        mvd_dir = str(WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd")
        if mvd_dir not in sys.path:
            sys.path.insert(0, mvd_dir)

        import eval_sys_gate as esg

        has_rust = bool(aser_bin is not None and eharness_bin is not None)
        if has_rust:
            esg.ASER = aser_bin
            esg.EHARNESS = eharness_bin
            print(f"[{EXP_ID}] Wired Linux Rust binaries: {aser_bin} & {eharness_bin}", flush=True)
        else:
            print(f"[{EXP_ID}] Rust binaries unavailable; Note F1 will be None.", flush=True)

        train_abs, val_abs, rism_imgs, manifest_path = _ensure_datasets(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(
            f"[{EXP_ID}] Loaded train={len(train_pairs)} val={len(val_pairs)} "
            f"rism_unlabeled={len(rism_imgs)}",
            flush=True,
        )

        arm_a = _train_and_eval_arm(
            "arm_a_control_rope2d",
            use_gauge_rope=False,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            rism_imgs=rism_imgs,
            manifest_path=manifest_path,
            device=device,
            has_rust_scorer=has_rust,
            use_bf16=use_bf16,
        )
        arm_b = _train_and_eval_arm(
            "arm_b_gauge_rope_rism_ssl",
            use_gauge_rope=True,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            rism_imgs=rism_imgs,
            manifest_path=manifest_path,
            device=device,
            has_rust_scorer=has_rust,
            use_bf16=use_bf16,
        )

        def _delta(b_val: Optional[float], a_val: Optional[float]) -> Optional[float]:
            if b_val is None or a_val is None:
                return None
            return round(float(b_val) - float(a_val), 6)

        comparison = {
            "val_clean_ce_delta_b_minus_a": _delta(
                arm_b["proxy_val_eval"]["clean"]["val_ce_loss"],
                arm_a["proxy_val_eval"]["clean"]["val_ce_loss"],
            ),
            "val_clean_tok_f1_delta_b_minus_a": _delta(
                arm_b["proxy_val_eval"]["clean"]["val_token_f1"],
                arm_a["proxy_val_eval"]["clean"]["val_token_f1"],
            ),
            "val_warped_ce_delta_b_minus_a": _delta(
                arm_b["proxy_val_eval"]["warped_droop"]["val_ce_loss"],
                arm_a["proxy_val_eval"]["warped_droop"]["val_ce_loss"],
            ),
            "val_warped_tok_f1_delta_b_minus_a": _delta(
                arm_b["proxy_val_eval"]["warped_droop"]["val_token_f1"],
                arm_a["proxy_val_eval"]["warped_droop"]["val_token_f1"],
            ),
            "droop_ce_degradation_reduction": _delta(
                arm_a["proxy_val_eval"]["droop_ce_delta"],
                arm_b["proxy_val_eval"]["droop_ce_delta"],
            ),
            "suite31_printed_clean_tok_f1_delta": _delta(
                arm_b["suite31_hw_eval"]["printed_clean"]["tok_f1"],
                arm_a["suite31_hw_eval"]["printed_clean"]["tok_f1"],
            ),
            "suite31_hw_clean_tok_f1_delta": _delta(
                arm_b["suite31_hw_eval"]["hw_clean"]["tok_f1"],
                arm_a["suite31_hw_eval"]["hw_clean"]["tok_f1"],
            ),
            "suite31_hw_clean_pitch_f1_delta": _delta(
                arm_b["suite31_hw_eval"]["hw_clean"]["pitch_tok_f1"],
                arm_a["suite31_hw_eval"]["hw_clean"]["pitch_tok_f1"],
            ),
            "suite31_hw_clean_rhythm_f1_delta": _delta(
                arm_b["suite31_hw_eval"]["hw_clean"]["rhythm_tok_f1"],
                arm_a["suite31_hw_eval"]["hw_clean"]["rhythm_tok_f1"],
            ),
            "suite31_hw_clean_note_f1_delta": _delta(
                arm_b["suite31_hw_eval"]["hw_clean"]["note_f1"],
                arm_a["suite31_hw_eval"]["hw_clean"]["note_f1"],
            ),
            "suite31_hw_photo_tok_f1_delta": _delta(
                arm_b["suite31_hw_eval"]["hw_photo"]["tok_f1"],
                arm_a["suite31_hw_eval"]["hw_photo"]["tok_f1"],
            ),
            "suite31_hw_photo_pitch_f1_delta": _delta(
                arm_b["suite31_hw_eval"]["hw_photo"]["pitch_tok_f1"],
                arm_a["suite31_hw_eval"]["hw_photo"]["pitch_tok_f1"],
            ),
            "suite31_hw_photo_rhythm_f1_delta": _delta(
                arm_b["suite31_hw_eval"]["hw_photo"]["rhythm_tok_f1"],
                arm_a["suite31_hw_eval"]["hw_photo"]["rhythm_tok_f1"],
            ),
            "suite31_hw_photo_note_f1_delta": _delta(
                arm_b["suite31_hw_eval"]["hw_photo"]["note_f1"],
                arm_a["suite31_hw_eval"]["hw_photo"]["note_f1"],
            ),
            "arm_b_learned_alpha": arm_b["final_alpha"],
            "arm_b_learned_abs_alpha": arm_b["final_abs_alpha"],
        }

        report = {
            "exp_id": EXP_ID,
            "status": "completed",
            "started_utc": started_iso,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"colab-{gpu_name}",
            "bf16": use_bf16,
            "hyperparams": {
                "preset": "small_45m",
                "steps": TRAIN_STEPS,
                "seed": SEED,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "fullpage_max_dim": FULLPAGE_MAX_DIM,
                "max_seq_len": MAX_SEQ_LEN,
                "rism_ssl_every": RISM_SSL_EVERY,
                "rism_ssl_weight": RISM_SSL_WEIGHT,
                "droop_frac": DROOP_FRAC,
            },
            "arm_a_control": arm_a,
            "arm_b_gauge_rope": arm_b,
            "comparison_b_minus_a": comparison,
        }

        eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        run_local = VOL_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(
            run_local,
            {
                "exp_id": EXP_ID,
                "status": "completed",
                "platform": f"colab-{gpu_name}",
                "started_utc": started_iso,
                "completed_utc": report["completed_utc"],
                "elapsed_secs": report["elapsed_secs"],
                "eval_path": HF_EVAL_PATH,
                "comparison_b_minus_a": comparison,
            },
        )

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload Gauge-RoPE 2-arm ablation results",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload Gauge-RoPE run ledger",
        )
        print(f"\n[{EXP_ID}] UPLOADED {HF_EVAL_PATH} & {HF_RUN_PATH} TO HF!", flush=True)
        print(json.dumps(comparison, indent=2), flush=True)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()

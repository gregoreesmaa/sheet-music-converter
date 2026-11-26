#!/usr/bin/env python3
"""Iter 13 (i13-12) on Google Colab L4:
PRAIG SMT++ Progressive System-Count Unfolding & Audiveris Non-Music Text/Lyric
Rejection Training Ablation on `UniOMR v2 small_45m` (1,200 steps/arm,
datasets/iter10_5_proxy_2k.tar, decoupled Muon LR 1e-3 / AdamW 3e-4, EMA 0.99/1).

Zero mutations to `train_uni_v2.py` or `uni_omr_v2.py`:
  - Arm A (`control_static_v2`): static full-page sampling, 1,200 steps.
  - Arm B (`smt_progressive_unfolding`): 3 chained 400-step stages ramping
    `max_systems` (<=3 -> <=6 -> <=16) via `--init-from`.
  - Arm C (`audiveris_text_clutter_reject`): 1,200 steps with 50% of training
    images augmented with synthetic non-music titles, composer credits, page
    numbers, tempo markings, and inter-staff lyric strips (unchanged GT bekern).

Evaluates all 3 arms (RAW + EMA) on both clean `proxy_val` and synthetic
text-cluttered `proxy_val_text_clutter`, then uploads eval JSON to HF.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T07-40-i13-12-smt-unfolding-text-reject-v2"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
DATASET_TAR_PATH = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

BUNDLE_CANDIDATES = (
    Path("/content/bundle.tar.gz"),
    Path("/content/ov_mvd_bundle.tar.gz"),
)
WORKSPACE_ROOT = Path("/workspace/sheet-music-converter")
VOL_ROOT = Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

ADAMW_LR = 3.0e-4
MUON_LR = 1.0e-3
DEFAULT_STEPS = 1200
DEFAULT_SEED = 42
WARMUP_STEPS = 120
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 2048
EVAL_MAX_LEN = 512
SAVE_EVERY_MINUTES = 5.0

_TITLE_POOL = (
    "SONATA No. 14 in C-sharp minor",
    "PRELUDE AND FUGUE, Op. 28",
    "NOCTURNE, Op. 9 No. 2",
    "LIEDER OHNE WORTE",
    "STRING QUARTET IN G MAJOR",
    "ETUDE TABLEAUX, Op. 33",
)
_COMPOSER_POOL = (
    "L. van Beethoven",
    "F. Chopin",
    "C. Debussy",
    "W. A. Mozart",
    "J. S. Bach",
    "R. Schumann",
)
_TEMPO_POOL = (
    "Allegro ma non troppo",
    "Andante espressivo",
    "Adagio sostenuto",
    "Moderato cantabile",
    "Vivace con brio",
)
_LYRIC_POOL = (
    "al - le Men - schen wer - den Bru - der wo dein sanf - ter Flu - gel weilt",
    "Ky - ri - e e - lei - son Chris - te e - lei - son",
    "espress.  molto legato  dim.  poco a poco cresc.  rit.",
    "Du bist die Ruh der Frie - de mild die Sehn - sucht du",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _purge_hf_token_file() -> None:
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and HF_TOKEN_FILE.exists():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        token = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    _purge_hf_token_file()
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var")
    return token


def _locate_or_extract_mvd() -> Path:
    for cand in (
        WORKSPACE_ROOT / "tmp-train" / "mvd",
        Path("/content/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
    ):
        if (cand / "train_uni_v2.py").is_file():
            return cand

    for bundle in BUNDLE_CANDIDATES:
        if bundle.is_file():
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            print(f"[{EXP_ID}] Extracting {bundle} -> {WORKSPACE_ROOT}...", flush=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORKSPACE_ROOT)
            for cand in (
                WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
                WORKSPACE_ROOT / "tmp-train" / "mvd",
                WORKSPACE_ROOT / "mvd",
            ):
                if (cand / "train_uni_v2.py").is_file():
                    return cand

    raise FileNotFoundError("Could not locate tmp-train/mvd/train_uni_v2.py")


def _ensure_deps() -> None:
    needed: List[str] = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])


def _ensure_proxy_dataset(vol_root: Path, hf_token: str) -> Tuple[Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = vol_root / "datasets" / "iter10_5_proxy_2k"
    sentinel = ds_dir / ".unpacked_ok"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"

    if sentinel.is_file() and train_abs.is_file() and val_abs.is_file():
        return ds_dir, train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] Downloading {DATASET_TAR_PATH}...", flush=True)
    local_tar = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=DATASET_TAR_PATH,
        repo_type="dataset",
        token=hf_token,
        local_dir="/tmp/hf_dl",
    )
    with tarfile.open(local_tar, "r:*") as tf:
        tf.extractall(path=ds_dir)
    try:
        os.remove(local_tar)
    except OSError:
        pass

    for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
        src_path = ds_dir / src_name
        out_lines: List[str] = []
        for raw in src_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            rec["image"] = str((ds_dir / rec["image"]).resolve())
            rec["bekerns"] = [str((ds_dir / b).resolve()) for b in rec.get("bekerns", [])]
            out_lines.append(json.dumps(rec))
        dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    sentinel.write_text(_utc_now() + "\n", encoding="utf-8")
    return ds_dir, train_abs, val_abs


def _render_text_clutter(src_img: Path, dst_img: Path, rng: random.Random) -> None:
    """Audiveris-style non-music text/header/lyric clutter overlay."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(src_img).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    def _font(sz: int) -> Any:
        for fp in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        ):
            if os.path.isfile(fp):
                try:
                    return ImageFont.truetype(fp, size=max(10, sz))
                except OSError:
                    pass
        return ImageFont.load_default()

    ink = rng.randint(15, 65)
    color = (ink, ink, ink)

    # Top title + right-aligned composer + page number
    title = rng.choice(_TITLE_POOL)
    comp = rng.choice(_COMPOSER_POOL)
    pg_num = str(rng.randint(1, 98))
    draw.text((int(w * 0.28), max(4, int(h * 0.012))), title, fill=color, font=_font(int(h * 0.022)))
    draw.text((int(w * 0.72), max(8, int(h * 0.036))), comp, fill=color, font=_font(int(h * 0.014)))
    draw.text((int(w * (0.04 if rng.random() < 0.5 else 0.93)), max(4, int(h * 0.010))), pg_num, fill=color, font=_font(int(h * 0.013)))

    # Tempo header on upper left
    draw.text((int(w * 0.08), int(h * 0.055)), rng.choice(_TEMPO_POOL), fill=color, font=_font(int(h * 0.014)))

    # 2-4 inter-staff lyric/expression lines across page height
    n_lines = rng.randint(2, 4)
    for k in range(n_lines):
        y_frac = 0.22 + 0.20 * k + rng.uniform(-0.03, 0.03)
        if 0.10 < y_frac < 0.94:
            draw.text(
                (int(w * 0.09), int(h * y_frac)),
                rng.choice(_LYRIC_POOL),
                fill=(min(110, ink + 25),) * 3,
                font=_font(int(h * 0.012)),
            )

    dst_img.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_img, format="JPEG", quality=92)


def _prepare_manifests(ds_dir: Path, train_abs: Path, val_abs: Path, seed: int) -> Dict[str, Path]:
    """Build SMT++ unfolding stage manifests and Audiveris text-clutter manifests."""
    train_rows = [json.loads(l) for l in train_abs.read_text(encoding="utf-8").splitlines() if l.strip()]
    val_rows = [json.loads(l) for l in val_abs.read_text(encoding="utf-8").splitlines() if l.strip()]

    # SMT++ progressive unfolding manifests
    sys_counts = sorted(len(r.get("bekerns", [])) for r in train_rows)
    p50 = max(3, sys_counts[len(sys_counts) // 2]) if sys_counts else 3
    s1_rows = [r for r in train_rows if len(r.get("bekerns", [])) <= 3]
    if len(s1_rows) < 64:
        s1_rows = [r for r in train_rows if len(r.get("bekerns", [])) <= p50]
    s2_rows = [r for r in train_rows if len(r.get("bekerns", [])) <= 6]
    if len(s2_rows) < len(s1_rows):
        s2_rows = list(train_rows)

    m_s1 = ds_dir / "i13_12_unfold_s1_max3.jsonl"
    m_s2 = ds_dir / "i13_12_unfold_s2_max6.jsonl"
    m_s1.write_text("\n".join(json.dumps(r) for r in s1_rows) + "\n", encoding="utf-8")
    m_s2.write_text("\n".join(json.dumps(r) for r in s2_rows) + "\n", encoding="utf-8")

    # Audiveris text-clutter manifests (50% train, 100% val clutter slice)
    clutter_dir = ds_dir / "i13_12_text_clutter_imgs"
    rng = random.Random(seed + 1312)

    m_tr_clutter = ds_dir / "i13_12_train_text_clutter.jsonl"
    if not m_tr_clutter.is_file():
        out_tr: List[str] = []
        for idx, rec in enumerate(train_rows):
            rec_copy = dict(rec)
            if rng.random() < 0.50:
                dst = clutter_dir / "train" / f"tr_{idx:05d}.jpg"
                if not dst.is_file():
                    _render_text_clutter(Path(rec["image"]), dst, rng)
                rec_copy["image"] = str(dst.resolve())
            out_tr.append(json.dumps(rec_copy))
        m_tr_clutter.write_text("\n".join(out_tr) + "\n", encoding="utf-8")

    m_va_clutter = ds_dir / "i13_12_val_text_clutter.jsonl"
    if not m_va_clutter.is_file():
        out_va: List[str] = []
        for idx, rec in enumerate(val_rows):
            rec_copy = dict(rec)
            dst = clutter_dir / "val" / f"va_{idx:05d}.jpg"
            if not dst.is_file():
                _render_text_clutter(Path(rec["image"]), dst, rng)
            rec_copy["image"] = str(dst.resolve())
            out_va.append(json.dumps(rec_copy))
        m_va_clutter.write_text("\n".join(out_va) + "\n", encoding="utf-8")

    print(
        f"[{EXP_ID}] Manifests ready: s1={len(s1_rows)}, s2={len(s2_rows)}, "
        f"full={len(train_rows)}, val_clutter={len(val_rows)}",
        flush=True,
    )
    return {
        "train_full": train_abs,
        "unfold_s1": m_s1,
        "unfold_s2": m_s2,
        "train_clutter": m_tr_clutter,
        "val_clean": val_abs,
        "val_clutter": m_va_clutter,
    }


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


def _eval_ckpt(ckpt_path: Path, val_manifest: Path, device: Any) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    import train_uni_v2 as tv2
    from uni_omr import (
        UniFullPageDataset,
        _images_to_device_float,
        collate_uni_batch,
        eval_dataset,
    )
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    raw_cfg = ck.get("cfg") or ck.get("config") or {}
    valid = set(UniOMRv2Config.__dataclass_fields__.keys())
    cfg_kwargs = {k: v for k, v in dict(raw_cfg).items() if k in valid}
    for tk in ("backbone_channels", "backbone_blocks", "deepstack_layers"):
        if tk in cfg_kwargs and isinstance(cfg_kwargs[tk], list):
            cfg_kwargs[tk] = tuple(cfg_kwargs[tk])
    cfg = UniOMRv2Config(**cfg_kwargs)
    model = UniOMRv2Net(cfg).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    val_ds = UniFullPageDataset(
        _load_manifest_pairs(val_manifest),
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
    sum_ce = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            bs = int(batch["seq"].shape[0])
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
                use_bf16=True,
                device=device,
            )
            sum_ce += float(losses["ce_loss"].item()) * bs
            n_samples += bs

    gen = eval_dataset(
        model,
        val_ds,
        device=device,
        max_len=EVAL_MAX_LEN,
        batch_size=2,
        use_bf16=True,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "n": n_samples,
        "val_ce_loss": round(sum_ce / max(1, n_samples), 6),
        "val_ser": gen["ser"],
        "val_token_f1": gen["token_f1"],
        "val_sys_count_exact": gen["sys_count_exact"],
    }


def _run_train_stage(
    mvd_dir: Path,
    *,
    manifest: Path,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
    init_from: Optional[Path] = None,
) -> float:
    cmd = [
        sys.executable,
        "-u",
        str(mvd_dir / "train_uni_v2.py"),
        "--preset", "small_45m",
        "--curriculum", "fullpage",
        "--fullpage-manifest", str(manifest),
        "--max-systems-per-page", "16",
        "--stratified-fullpage",
        "--fullpage-max-dim", str(FULLPAGE_MAX_DIM),
        "--max-seq-len", str(MAX_SEQ_LEN),
        "--batch-size", "2",
        "--grad-accum", "4",
        "--optimizer", "muon",
        "--muon-lr", str(MUON_LR),
        "--lr", str(ADAMW_LR),
        "--lr-schedule", "wsd",
        "--warmup-steps", str(warmup_steps),
        "--wsd-decay-frac", str(WSD_DECAY_FRAC),
        "--min-lr-ratio", str(MIN_LR_RATIO),
        "--gpu-ema-decay", "0.99",
        "--gpu-ema-every", "1",
        "--grad-checkpoint",
        "--bf16",
        "--steps", str(steps),
        "--eval-every", str(steps + 1),
        "--log-every", "50",
        "--save-every-minutes", str(SAVE_EVERY_MINUTES),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
    ]
    if init_from is not None:
        cmd.extend(["--init-from", str(init_from)])

    t0 = time.perf_counter()
    print(f"[{EXP_ID}] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"train_uni_v2.py failed (rc={rc}) for {out_dir.name}")
    return round(time.perf_counter() - t0, 2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    try:
        started_utc = _utc_now()
        hf_token = _read_and_purge_hf_token()
        _ensure_deps()
        mvd_dir = _locate_or_extract_mvd()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        import torch
        from huggingface_hub import HfApi

        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        assert torch.cuda.is_available(), "CUDA GPU required"
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        print(f"[{EXP_ID}] GPU={gpu_name} (cc={cc[0]}.{cc[1]}) | steps={args.steps}", flush=True)

        ds_dir, train_abs, val_abs = _ensure_proxy_dataset(VOL_ROOT, hf_token)
        manifests = _prepare_manifests(ds_dir, train_abs, val_abs, seed=args.seed)

        out_root = VOL_ROOT / "ckpts" / EXP_ID
        out_root.mkdir(parents=True, exist_ok=True)
        partial_path = out_root / f"{EXP_ID}_partial.json"
        arms_out: Dict[str, Any] = {}
        if partial_path.is_file():
            arms_out = dict(json.loads(partial_path.read_text(encoding="utf-8")).get("arms", {}))

        # Arm A: control_static_v2
        if "control_static_v2" not in arms_out:
            arm_dir = out_root / "control_static_v2"
            wall = _run_train_stage(
                mvd_dir,
                manifest=manifests["train_full"],
                out_dir=arm_dir,
                steps=args.steps,
                seed=args.seed,
                warmup_steps=WARMUP_STEPS,
            )
            arms_out["control_static_v2"] = {
                "wall_time_s": wall,
                "val_clean_raw": _eval_ckpt(arm_dir / "uni_omr_v2_latest.pt", manifests["val_clean"], device),
                "val_clean_ema": _eval_ckpt(arm_dir / "uni_omr_v2_ema_latest.pt", manifests["val_clean"], device),
                "val_clutter_ema": _eval_ckpt(arm_dir / "uni_omr_v2_ema_latest.pt", manifests["val_clutter"], device),
            }
            _atomic_write_json(partial_path, {"exp_id": EXP_ID, "arms": arms_out})

        # Arm B: smt_progressive_unfolding (3 x 400 steps: <=3 -> <=6 -> <=16 systems)
        if "smt_progressive_unfolding" not in arms_out:
            s_steps = max(50, args.steps // 3)
            rem_steps = args.steps - 2 * s_steps
            b1 = out_root / "smt_unfold_s1"
            b2 = out_root / "smt_unfold_s2"
            b3 = out_root / "smt_progressive_unfolding"
            w1 = _run_train_stage(mvd_dir, manifest=manifests["unfold_s1"], out_dir=b1, steps=s_steps, seed=args.seed, warmup_steps=min(80, s_steps // 4))
            w2 = _run_train_stage(mvd_dir, manifest=manifests["unfold_s2"], out_dir=b2, steps=s_steps, seed=args.seed + 1, warmup_steps=30, init_from=b1 / "uni_omr_v2_latest.pt")
            w3 = _run_train_stage(mvd_dir, manifest=manifests["train_full"], out_dir=b3, steps=rem_steps, seed=args.seed + 2, warmup_steps=30, init_from=b2 / "uni_omr_v2_latest.pt")
            arms_out["smt_progressive_unfolding"] = {
                "wall_time_s": round(w1 + w2 + w3, 2),
                "stage_steps": [s_steps, s_steps, rem_steps],
                "val_clean_raw": _eval_ckpt(b3 / "uni_omr_v2_latest.pt", manifests["val_clean"], device),
                "val_clean_ema": _eval_ckpt(b3 / "uni_omr_v2_ema_latest.pt", manifests["val_clean"], device),
                "val_clutter_ema": _eval_ckpt(b3 / "uni_omr_v2_ema_latest.pt", manifests["val_clutter"], device),
            }
            _atomic_write_json(partial_path, {"exp_id": EXP_ID, "arms": arms_out})

        # Arm C: audiveris_text_clutter_reject
        if "audiveris_text_clutter_reject" not in arms_out:
            arm_dir = out_root / "audiveris_text_clutter_reject"
            wall = _run_train_stage(
                mvd_dir,
                manifest=manifests["train_clutter"],
                out_dir=arm_dir,
                steps=args.steps,
                seed=args.seed,
                warmup_steps=WARMUP_STEPS,
            )
            arms_out["audiveris_text_clutter_reject"] = {
                "wall_time_s": wall,
                "val_clean_raw": _eval_ckpt(arm_dir / "uni_omr_v2_latest.pt", manifests["val_clean"], device),
                "val_clean_ema": _eval_ckpt(arm_dir / "uni_omr_v2_ema_latest.pt", manifests["val_clean"], device),
                "val_clutter_ema": _eval_ckpt(arm_dir / "uni_omr_v2_ema_latest.pt", manifests["val_clutter"], device),
            }
            _atomic_write_json(partial_path, {"exp_id": EXP_ID, "arms": arms_out})

        ctrl = arms_out["control_static_v2"]
        deltas = {
            k: {
                "clean_ema_delta_ce": round(v["val_clean_ema"]["val_ce_loss"] - ctrl["val_clean_ema"]["val_ce_loss"], 6),
                "clean_ema_delta_f1": round(v["val_clean_ema"]["val_token_f1"] - ctrl["val_clean_ema"]["val_token_f1"], 6),
                "clutter_ema_delta_f1": round(v["val_clutter_ema"]["val_token_f1"] - ctrl["val_clutter_ema"]["val_token_f1"], 6),
            }
            for k, v in arms_out.items()
        }
        finished_utc = _utc_now()
        eval_payload = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "provider": "colab_l4",
            "gpu": gpu_name,
            "started_utc": started_utc,
            "completed_utc": finished_utc,
            "steps_per_arm": args.steps,
            "seed": args.seed,
            "arms": arms_out,
            "deltas_vs_control": deltas,
        }
        eval_local = out_root / f"{EXP_ID}_eval.json"
        run_local = out_root / f"{EXP_ID}_run.json"
        _atomic_write_json(eval_local, eval_payload)
        _atomic_write_json(
            run_local,
            {
                "exp_id": EXP_ID,
                "status": "COMPLETED",
                "started_utc": started_utc,
                "completed_utc": finished_utc,
                "gpu": gpu_name,
                "hf_eval_path": HF_EVAL_PATH,
                "deltas_vs_control": deltas,
            },
        )
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload SMT++ unfolding + Audiveris text rejection eval JSON",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark COMPLETED",
        )
        print(f"[{EXP_ID}] COMPLETED | deltas={json.dumps(deltas)}", flush=True)
        return 0
    finally:
        _purge_hf_token_file()


if __name__ == "__main__":
    raise SystemExit(main())

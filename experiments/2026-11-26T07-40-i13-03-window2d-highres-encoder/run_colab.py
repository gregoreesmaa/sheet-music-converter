#!/usr/bin/env python3
"""Google Colab L4 runner — Iter 13 Academic Exp 03: Qwen2.5-VL Shifted 2D Window Attention at /8.

Experiment ID: 2026-11-26T07-40-i13-03-window2d-highres-encoder
Papers:
  - Bai et al. (Alibaba Qwen), "Qwen2.5-VL Technical Report" (arXiv:2502.13923, Feb 2025)
  - Wei et al. (DeepSeek-AI), "DeepSeek-OCR: Contexts Optical Compression" (2025)
  - Li et al., "TrOMR: Transformer-Based Polyphonic OMR" (ICASSP 2024, arXiv:2308.09370)

Tests replacing or pairing the high-resolution /8 AxialEncoderLayer stack with
Qwen2.5-VL shifted 2D local window attention + 2D-RoPE on UniOMR v2 small_45m.

Arms (4 single-variable controlled arms, seed=42, 300 steps each):
  - axial_control:             2x AxialEncoderLayer (baseline)
  - win2d_8x8_shifted:         2x shifted 8x8 2D-window attention + 2D-RoPE
  - win2d_8x16_staff_strip:    2x shifted 8x16 staff-strip 2D-window attention + 2D-RoPE
  - hybrid_win8x16_then_axial: L0=Window2D(8x16), L1=AxialEncoderLayer

Security:
  - Reads HF_TOKEN from /tmp/.hf_token (chmod 600) or env HF_TOKEN, never prints it.
  - Always deletes /tmp/.hf_token in finally.
  - Uploads evals/<exp_id>.json and runs/<exp_id>.json to gregoreesmaa/sheet-music-converter-omr.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXP_ID = "2026-11-26T07-40-i13-03-window2d-highres-encoder"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORK_ROOT = Path("/content/i13_03")
OUT_ROOT = WORK_ROOT / "out" / EXP_ID

STEPS_PER_ARM = 300
SEED = 42
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 3e-4
WARMUP_STEPS = 45
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
EVAL_MAX_LEN = 512
SAVE_EVERY_MINUTES = 5.0

ARMS: List[Dict[str, Any]] = [
    {"name": "axial_control", "topology": "axial"},
    {"name": "win2d_8x8_shifted", "topology": "win2d_8x8"},
    {"name": "win2d_8x16_staff_strip", "topology": "win2d_8x16"},
    {"name": "hybrid_win8x16_then_axial", "topology": "hybrid_8x16_axial"},
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_hf_token() -> str:
    if HF_TOKEN_FILE.is_file():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        tok = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        raise RuntimeError("HF token missing: provide /tmp/.hf_token or HF_TOKEN env var")
    return tok


def _purge_hf_token() -> None:
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass


def _ensure_deps() -> None:
    needed = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])


def _locate_or_unpack_mvd() -> Path:
    candidates = [
        Path.cwd() / "tmp-train" / "mvd",
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        WORK_ROOT / "mvd",
        WORK_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
    ]
    for cand in candidates:
        if (cand / "train_uni_v2.py").is_file() and (cand / "uni_omr_v2.py").is_file():
            return cand

    for bundle in (Path("/content/bundle.tar.gz"), Path("/content/mvd_bundle.tar.gz")):
        if bundle.is_file():
            WORK_ROOT.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORK_ROOT)
            for p in WORK_ROOT.rglob("train_uni_v2.py"):
                return p.parent

    raise FileNotFoundError(
        "Could not locate tmp-train/mvd/train_uni_v2.py. Upload /content/bundle.tar.gz "
        "containing tmp-train/mvd before running."
    )


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = WORK_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"
    if train_abs.is_file() and val_abs.is_file():
        return train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR} from {HF_REPO_ID}...", flush=True)
    tar_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PROXY_TAR,
        repo_type="dataset",
        token=hf_token,
        local_dir="/tmp/hf_dl",
    )
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=ds_dir)
    try:
        os.remove(tar_path)
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
        print(f"[{EXP_ID}] Prepared {dst_path.name} ({len(out_lines)} rows)", flush=True)

    print(f"[{EXP_ID}] Dataset ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return train_abs, val_abs


def _upload_json_to_hf(hf_token: str, local_path: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"{EXP_ID}: upload {path_in_repo}",
    )
    print(f"[{EXP_ID}] Uploaded {path_in_repo} -> {HF_REPO_ID}", flush=True)


def _patch_model_with_win2d(model: Any, topology: str) -> Any:
    """Replace model.encoder.axial_layers with shifted 2D local window attention layers."""
    if topology == "axial":
        return model

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import uni_omr_v2

    class Window2DEncoderLayer(nn.Module):
        """Qwen2.5-VL style shifted 2D local window attention with local 2D-RoPE.

        Matches AxialEncoderLayer.forward(x, row_cos, row_sin, col_cos, col_sin)
        signature so HybridAxialMergerEncoder & grad_checkpoint work unmodified.
        """

        def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            win_h: int = 8,
            win_w: int = 16,
            shift_h: int = 0,
            shift_w: int = 0,
            dropout: float = 0.1,
            rope_base: float = 10000.0,
        ) -> None:
            super().__init__()
            self.win_h = int(win_h)
            self.win_w = int(win_w)
            self.shift_h = int(shift_h)
            self.shift_w = int(shift_w)
            head_dim = d_model // n_heads
            self.norm1 = uni_omr_v2.RMSNorm(d_model)
            self.attn = uni_omr_v2._SelfAttention(d_model, n_heads)
            self.win_rope2d = uni_omr_v2.AspectNormalizedRoPE2D(
                head_dim=head_dim,
                base=rope_base,
                coord_scale=float(max(win_h, win_w)),
                rope_jitter=0.0,
            )
            self.norm2 = uni_omr_v2.RMSNorm(d_model)
            self.ffn = uni_omr_v2.SwiGLU(d_model, d_ff, dropout=dropout)
            self.drop = nn.Dropout(dropout)

        def forward(
            self,
            x: torch.Tensor,
            row_cos: torch.Tensor,
            row_sin: torch.Tensor,
            col_cos: torch.Tensor,
            col_sin: torch.Tensor,
        ) -> torch.Tensor:
            B, H, W, D = x.shape
            h_in = self.norm1(x)
            pad_h = (self.win_h - (H % self.win_h)) % self.win_h
            pad_w = (self.win_w - (W % self.win_w)) % self.win_w
            if pad_h or pad_w:
                h_in = F.pad(h_in.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h), mode="replicate").permute(0, 2, 3, 1)
            Hp, Wp = h_in.shape[1], h_in.shape[2]

            do_shift = (self.shift_h > 0 or self.shift_w > 0) and (Hp > self.win_h or Wp > self.win_w)
            sh = self.shift_h if Hp > self.win_h else 0
            sw = self.shift_w if Wp > self.win_w else 0
            if do_shift:
                h_in = torch.roll(h_in, shifts=(-sh, -sw), dims=(1, 2))

            nh, nw = Hp // self.win_h, Wp // self.win_w
            # Partition into (B * nh * nw, win_h * win_w, D)
            wins = (
                h_in.reshape(B, nh, self.win_h, nw, self.win_w, D)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(B * nh * nw, self.win_h * self.win_w, D)
            )
            cos_g, sin_g, _, _ = self.win_rope2d._get_grid_cos_sin(
                self.win_h, self.win_w, wins.device, wins.dtype
            )
            cos_w = cos_g.unsqueeze(0).unsqueeze(0)
            sin_w = sin_g.unsqueeze(0).unsqueeze(0)
            wins_out = self.attn(wins, cos_w, sin_w)

            out = (
                wins_out.reshape(B, nh, nw, self.win_h, self.win_w, D)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(B, Hp, Wp, D)
            )
            if do_shift:
                out = torch.roll(out, shifts=(sh, sw), dims=(1, 2))
            if pad_h or pad_w:
                out = out[:, :H, :W, :]

            x = x + self.drop(out)
            x = x + self.drop(self.ffn(self.norm2(x)))
            return x

    cfg = model.cfg
    d_model = int(cfg.d_model)
    n_heads = int(cfg.n_heads)
    axial_d_ff = int(round(d_model * cfg.axial_ffn_mult))
    dropout = float(cfg.dropout)
    rope_base = float(cfg.rope_base)

    if topology == "win2d_8x8":
        layers = [
            Window2DEncoderLayer(d_model, n_heads, axial_d_ff, 8, 8, 0, 0, dropout, rope_base),
            Window2DEncoderLayer(d_model, n_heads, axial_d_ff, 8, 8, 4, 4, dropout, rope_base),
        ]
    elif topology == "win2d_8x16":
        layers = [
            Window2DEncoderLayer(d_model, n_heads, axial_d_ff, 8, 16, 0, 0, dropout, rope_base),
            Window2DEncoderLayer(d_model, n_heads, axial_d_ff, 8, 16, 4, 8, dropout, rope_base),
        ]
    elif topology == "hybrid_8x16_axial":
        layers = [
            Window2DEncoderLayer(d_model, n_heads, axial_d_ff, 8, 16, 0, 0, dropout, rope_base),
            uni_omr_v2.AxialEncoderLayer(d_model, n_heads, axial_d_ff, dropout=dropout),
        ]
    else:
        raise ValueError(f"Unknown topology: {topology}")

    model.encoder.axial_layers = nn.ModuleList(layers)
    return model


def _summarize_arm(arm: Dict[str, Any], res: Dict[str, Any], elapsed_sec: float, peak_vram_mb: float) -> Dict[str, Any]:
    hist = res.get("history", [])
    losses = [float(r["loss"]) for r in hist[-25:]] if hist else []
    eval_hist = res.get("eval_history", [])
    last_eval = eval_hist[-1] if eval_hist else {}
    fp_val = last_eval.get("fullpage_val", {})
    ema_fp_val = (last_eval.get("ema_eval") or {}).get("fullpage_val", {})
    steps_done = int(res.get("steps", len(hist)))

    return {
        "arm": arm["name"],
        "topology": str(arm["topology"]),
        "steps": steps_done,
        "elapsed_sec": round(elapsed_sec, 2),
        "steps_per_sec": round(steps_done / max(1e-6, elapsed_sec), 4),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "final_25step_loss": round(sum(losses) / max(1, len(losses)), 4) if losses else None,
        "val_fullpage": fp_val,
        "ema_val_fullpage": ema_fp_val,
        "val_token_f1": fp_val.get("token_f1"),
        "val_ser": fp_val.get("ser"),
        "val_sys_count_exact": fp_val.get("sys_count_exact"),
        "ema_val_token_f1": ema_fp_val.get("token_f1"),
        "ema_val_ser": ema_fp_val.get("ser"),
        "params": res.get("params"),
    }


def _run_arm(
    train_uni_v2: Any,
    arm: Dict[str, Any],
    train_manifest: Path,
    val_manifest: Path,
) -> Dict[str, Any]:
    import torch
    import uni_omr

    arm_dir = OUT_ROOT / arm["name"]
    done_file = arm_dir / "arm_done.json"
    if done_file.is_file():
        print(f"[{EXP_ID}] Skipping completed arm {arm['name']} (found {done_file})", flush=True)
        return json.loads(done_file.read_text(encoding="utf-8"))

    arm_dir.mkdir(parents=True, exist_ok=True)
    parser = train_uni_v2.build_arg_parser()
    argv = [
        "--preset", "small_45m",
        "--optimizer", "muon",
        "--lr", str(LR),
        "--weight-decay", "0.01",
        "--lr-schedule", "wsd",
        "--warmup-steps", str(WARMUP_STEPS),
        "--wsd-decay-frac", "0.25",
        "--min-lr-ratio", "0.05",
        "--steps", str(STEPS_PER_ARM),
        "--batch-size", str(BATCH_SIZE),
        "--grad-accum", str(GRAD_ACCUM),
        "--curriculum", "fullpage",
        "--fullpage-manifest", str(train_manifest),
        "--fullpage-max-dim", str(FULLPAGE_MAX_DIM),
        "--max-seq-len", str(MAX_SEQ_LEN),
        "--keep-ratio", "0.40",
        "--dynamic-routing",
        "--max-mem-tokens", "10240",
        "--stroke-hog-weight", "0.15",
        "--gpu-ema-decay", "0.997",
        "--save-every-minutes", str(SAVE_EVERY_MINUTES),
        "--eval-every", str(STEPS_PER_ARM),
        "--eval-max-len", str(EVAL_MAX_LEN),
        "--val-count", "0",
        "--prefetch-threads", "4",
        "--num-workers", "0",
        "--log-every", "25",
        "--grad-checkpoint",
        "--bf16",
        "--seed", str(SEED),
        "--out-dir", str(arm_dir),
    ]
    args = parser.parse_args(argv)
    args.multi_scale = False

    val_pairs = uni_omr.discover_fullpage_pairs([], manifest=str(val_manifest))
    orig_split = train_uni_v2.split_train_val
    orig_net_cls = train_uni_v2.UniOMRv2Net

    def _fixed_val_split(pairs: Any, val_count: int = 0, seed: int = 42) -> Tuple[Any, Any]:
        return list(pairs), list(val_pairs)

    def _patched_net_factory(cfg: Any) -> Any:
        net = orig_net_cls(cfg)
        return _patch_model_with_win2d(net, str(arm["topology"]))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_uni_v2.split_train_val = _fixed_val_split
    train_uni_v2.UniOMRv2Net = _patched_net_factory
    try:
        print(
            f"\n[{EXP_ID}] === Starting arm={arm['name']} "
            f"(topology={arm['topology']}, steps={STEPS_PER_ARM}) ===",
            flush=True,
        )
        res = train_uni_v2.run_train_v2(args)
    finally:
        train_uni_v2.split_train_val = orig_split
        train_uni_v2.UniOMRv2Net = orig_net_cls

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    else:
        peak_vram_mb = 0.0
    elapsed = time.perf_counter() - t0

    summary = _summarize_arm(arm, res, elapsed, peak_vram_mb)
    _atomic_write_json(done_file, summary)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> int:
    t_start = time.perf_counter()
    started_at = _utc_now()
    hf_token: Optional[str] = None
    arm_results: List[Dict[str, Any]] = []
    status = "RUNNING"
    error_msg: Optional[str] = None

    try:
        hf_token = _read_hf_token()
        _ensure_deps()
        mvd_dir = _locate_or_unpack_mvd()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        import torch
        import train_uni_v2

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        print(f"[{EXP_ID}] Device={gpu_name} | torch={torch.__version__}", flush=True)

        train_manifest, val_manifest = _ensure_proxy_dataset(hf_token)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)

        for arm in ARMS:
            summary = _run_arm(train_uni_v2, arm, train_manifest, val_manifest)
            arm_results.append(summary)

            eval_payload = {
                "exp_id": EXP_ID,
                "status": "RUNNING" if len(arm_results) < len(ARMS) else "COMPLETED",
                "started_at": started_at,
                "updated_at": _utc_now(),
                "gpu": gpu_name,
                "steps_per_arm": STEPS_PER_ARM,
                "seed": SEED,
                "arms": arm_results,
            }
            eval_local = OUT_ROOT / "eval.json"
            _atomic_write_json(eval_local, eval_payload)
            _upload_json_to_hf(hf_token, eval_local, HF_EVAL_PATH)

        status = "COMPLETED"
    except Exception as exc:
        status = "FAILED"
        error_msg = f"{type(exc).__name__}: {exc}"
        print(f"[{EXP_ID}] ERROR: {error_msg}", flush=True)
        raise
    finally:
        total_sec = time.perf_counter() - t_start
        best_arm = max(
            arm_results,
            key=lambda r: float(r.get("ema_val_token_f1") or r.get("val_token_f1") or -1.0),
            default=None,
        )
        run_payload = {
            "exp_id": EXP_ID,
            "status": status,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_sec": round(total_sec, 2),
            "n_arms_completed": len(arm_results),
            "best_arm": best_arm["arm"] if best_arm else None,
            "best_val_token_f1": (
                best_arm.get("ema_val_token_f1") or best_arm.get("val_token_f1")
                if best_arm
                else None
            ),
            "eval_path": HF_EVAL_PATH,
            "error": error_msg,
        }
        run_local = OUT_ROOT / "run.json"
        _atomic_write_json(run_local, run_payload)
        if hf_token:
            try:
                _upload_json_to_hf(hf_token, run_local, HF_RUN_PATH)
            except Exception as up_exc:
                print(f"[{EXP_ID}] WARN: failed to upload run.json: {up_exc}", flush=True)
        _purge_hf_token()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

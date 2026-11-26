#!/usr/bin/env python3
"""Google Colab L4 runner — Iter 13 Academic Exp 01: DeepSeek-OCR 2 Visual Causal Flow Queries.

Experiment ID: 2026-11-26T07-40-i13-01-visual-causal-flow-encoder
Paper: Wei, Sun & Li (DeepSeek-AI), "DeepSeek-OCR 2: Visual Causal Flow" (arXiv:2601.20552, Jan 2026).

Tests injecting M learnable Visual Causal Flow (VCF) queries inside
HybridAxialMergerEncoder.global_layers with DeepEncoder-V2's hybrid attention mask
(visual tokens attend bidirectionally to visual tokens; causal flow queries attend
to all visual tokens + causally to earlier flow queries) on UniOMR v2 small_45m.

Arms (4 single-variable controlled arms, seed=42, 300 steps each):
  - control_raster:   M=0   (exact UniOMR v2 baseline)
  - vcf_append_64:    M=64  appended to DeepStack mem_levels
  - vcf_append_128:   M=128 appended to DeepStack mem_levels
  - vcf_compress_256: M=256 replacing raw raster memory (pure optical compression)

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
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXP_ID = "2026-11-26T07-40-i13-01-visual-causal-flow-encoder"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORK_ROOT = Path("/content/i13_01")
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
    {"name": "control_raster", "n_flow_queries": 0, "vcf_mode": "none"},
    {"name": "vcf_append_64", "n_flow_queries": 64, "vcf_mode": "append"},
    {"name": "vcf_append_128", "n_flow_queries": 128, "vcf_mode": "append"},
    {"name": "vcf_compress_256", "n_flow_queries": 256, "vcf_mode": "replace"},
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


def _patch_model_with_vcf(model: Any, n_flow_queries: int, vcf_mode: str) -> Any:
    """Attach DeepSeek-OCR 2 Visual Causal Flow queries to HybridAxialMergerEncoder."""
    if n_flow_queries <= 0 or vcf_mode == "none":
        return model

    import torch
    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint as _grad_ckpt
    import uni_omr_v2

    enc = model.encoder
    d_model = int(enc.cfg.d_model)
    enc.vcf_queries = nn.Parameter(torch.randn(1, n_flow_queries, d_model) * 0.02)
    enc.vcf_pos = nn.Parameter(torch.randn(1, n_flow_queries, d_model) * 0.02)
    enc.vcf_n_queries = int(n_flow_queries)
    enc.vcf_mode = str(vcf_mode)
    enc._vcf_mask_cache = OrderedDict()

    def _get_hybrid_mask(n_vis: int, n_flow: int, device: torch.device) -> torch.Tensor:
        key = (n_vis, n_flow, device)
        cached = enc._vcf_mask_cache.get(key)
        if cached is not None:
            enc._vcf_mask_cache.move_to_end(key)
            return cached
        total = n_vis + n_flow
        mask = torch.zeros((total, total), device=device, dtype=torch.bool)
        mask[:n_vis, :n_vis] = True
        mask[n_vis:, :n_vis] = True
        mask[n_vis:, n_vis:] = torch.tril(torch.ones((n_flow, n_flow), device=device, dtype=torch.bool))
        mask = mask.unsqueeze(0).unsqueeze(0)
        enc._vcf_mask_cache[key] = mask
        while len(enc._vcf_mask_cache) > 16:
            enc._vcf_mask_cache.popitem(last=False)
        return mask

    def _masked_global_layer_forward(
        layer: Any,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = layer.norm1(x)
        sa = layer.self_attn
        B, L, _ = h.shape
        q, k, v = sa.qkv_proj(h).chunk(3, dim=-1)
        q = sa.q_norm(q.view(B, L, sa.n_heads, sa.head_dim)).transpose(1, 2)
        k = sa.k_norm(k.view(B, L, sa.n_heads, sa.head_dim)).transpose(1, 2)
        v = v.view(B, L, sa.n_heads, sa.head_dim).transpose(1, 2)
        q, k = uni_omr_v2.apply_rope(q, k, rope_cos, rope_sin)
        out = uni_omr_v2._sdpa(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = sa.o_proj(out.transpose(1, 2).reshape(B, L, sa.d_model))
        x = x + layer.drop(out)
        x = x + layer.drop(layer.ffn(layer.norm2(x)))
        return x

    def _vcf_encoder_forward(
        feat_2d: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        keep_ratio_override: Optional[float] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, Any]]:
        _, _, H_p, W_p = feat_2d.shape
        x = feat_2d.permute(0, 2, 3, 1).contiguous()

        row_cos, row_sin = enc.axial_rope(W_p, device=x.device, dtype=x.dtype)
        col_cos, col_sin = enc.axial_rope(H_p, device=x.device, dtype=x.dtype)

        use_ckpt = enc.training and enc.cfg.grad_checkpoint and torch.is_grad_enabled()
        for layer in enc.axial_layers:
            if use_ckpt:
                x = _grad_ckpt(layer, x, row_cos, row_sin, col_cos, col_sin, use_reentrant=False)
            else:
                x = layer(x, row_cos, row_sin, col_cos, col_sin)

        merged = enc.patch_merger(x)
        merged_2d = merged.permute(0, 3, 1, 2)

        routed_tokens, routed_coords, router_info = enc.router(
            merged_2d, images=images, keep_ratio_override=keep_ratio_override, feat_nhwc=merged
        )
        router_info["axial_grid_hw"] = (H_p, W_p)

        mem_post_merger_vis = routed_tokens
        grid_hw = router_info["grid_hw"]
        kept_idx = router_info["kept_idx"]
        n_bg = int(router_info["n_bg_summary"])
        is_full_grid = bool(router_info.get("is_full_grid", False))
        pos2d = enc.pos2d_proj(
            enc._sinusoidal_pos2d_indexed(
                kept_idx, grid_hw=grid_hw, n_bg_summary=n_bg, dtype=routed_tokens.dtype, is_full_grid=is_full_grid
            )
        )
        cos2d, sin2d = enc.rope2d.forward_indexed(
            kept_idx,
            grid_hw=grid_hw,
            n_bg_summary=n_bg,
            dtype=routed_tokens.dtype,
            coords=routed_coords,
            is_full_grid=is_full_grid,
        )
        mem_vis = routed_tokens + pos2d
        B = mem_vis.shape[0]

        n_reg = int(enc.cfg.n_register_tokens) if enc.register_tokens is not None else 0
        if n_reg > 0 and enc.register_tokens is not None:
            reg = enc.register_tokens.expand(B, -1, -1).to(mem_vis.dtype)
            mem_vis = torch.cat([reg, mem_vis], dim=1)
            ones_r = torch.ones((B, 1, n_reg, cos2d.shape[-1]), device=cos2d.device, dtype=cos2d.dtype)
            zeros_r = torch.zeros((B, 1, n_reg, sin2d.shape[-1]), device=sin2d.device, dtype=sin2d.dtype)
            cos2d = torch.cat([ones_r, cos2d], dim=2)
            sin2d = torch.cat([zeros_r, sin2d], dim=2)

        n_vis = mem_vis.shape[1]
        n_flow = enc.vcf_n_queries
        flow_init = (enc.vcf_queries + enc.vcf_pos).expand(B, -1, -1).to(mem_vis.dtype)
        mem = torch.cat([mem_vis, flow_init], dim=1)

        ones_f = torch.ones((B, 1, n_flow, cos2d.shape[-1]), device=cos2d.device, dtype=cos2d.dtype)
        zeros_f = torch.zeros((B, 1, n_flow, sin2d.shape[-1]), device=sin2d.device, dtype=sin2d.dtype)
        cos_all = torch.cat([cos2d, ones_f], dim=2)
        sin_all = torch.cat([sin2d, zeros_f], dim=2)

        hybrid_mask = _get_hybrid_mask(n_vis, n_flow, mem.device)

        n_global = len(enc.global_layers)
        mid_idx = max(0, (n_global // 2) - 1)
        vis_mid = mem_post_merger_vis
        flow_mid = flow_init
        for i, layer in enumerate(enc.global_layers):
            if use_ckpt and (mem.shape[0] * mem.shape[1] > 12000):
                mem = _grad_ckpt(
                    _masked_global_layer_forward, layer, mem, cos_all, sin_all, hybrid_mask, use_reentrant=False
                )
            else:
                mem = _masked_global_layer_forward(layer, mem, cos_all, sin_all, hybrid_mask)
            if i == mid_idx:
                vis_mid = mem[:, n_reg:n_vis, :]
                flow_mid = mem[:, n_vis:, :]

        vis_final = enc.enc_norm(mem[:, n_reg:n_vis, :]) + pos2d
        flow_final = enc.enc_norm(mem[:, n_vis:, :]) + enc.vcf_pos.expand(B, -1, -1).to(mem.dtype)

        if enc.vcf_mode == "replace":
            mem_levels = (flow_init, flow_mid, flow_final)
            router_info["n_out"] = n_flow
        else:
            mem_post_ext = torch.cat([mem_post_merger_vis, flow_init], dim=1)
            mem_mid_ext = torch.cat([vis_mid, flow_mid], dim=1)
            mem_final_ext = torch.cat([vis_final, flow_final], dim=1)
            mem_levels = (mem_post_ext, mem_mid_ext, mem_final_ext)
            router_info["n_out"] = int(mem_final_ext.shape[1])

        return mem_levels, router_info

    enc.forward = _vcf_encoder_forward  # type: ignore[method-assign]
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
        "n_flow_queries": int(arm["n_flow_queries"]),
        "vcf_mode": str(arm["vcf_mode"]),
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
        return _patch_model_with_vcf(net, int(arm["n_flow_queries"]), str(arm["vcf_mode"]))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_uni_v2.split_train_val = _fixed_val_split
    train_uni_v2.UniOMRv2Net = _patched_net_factory
    try:
        print(
            f"\n[{EXP_ID}] === Starting arm={arm['name']} "
            f"(n_flow_queries={arm['n_flow_queries']}, vcf_mode={arm['vcf_mode']}, steps={STEPS_PER_ARM}) ===",
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

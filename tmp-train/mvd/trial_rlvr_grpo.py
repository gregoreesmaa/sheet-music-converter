#!/usr/bin/env python3
"""Standalone post-SFT RLVR (K=4 GRPO) value trial runner for MvdNet.

Loads an existing ``MvdNet`` checkpoint (e.g. ``exp2B @ 20k``), freezes a
``ref_model`` copy as the KL anchor (``--kl-beta 0.08``), and runs ``--steps``
of ``unsup_bar_math.grpo_bar_math_step`` (Tier-1 bar-math + Tier-2 visual
reward) over a mix of unlabeled printed + real HW/RISM strips (``--strip-dirs``).

Logs per-step / rolling ``mean_reward``, ``mean_valid_bar_ratio``, ``kl_loss``,
``pg_loss``, and ``active_frac``, and saves the RLVR-polished checkpoint plus
JSON summary metrics to ``--out-dir``.

Supports ``--dry-run`` on CPU with a tiny dummy ``MvdNet`` to verify end-to-end
execution without requiring a GPU or production checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

_MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = _MVD_DIR.parent.parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import data as mvd_data  # noqa: E402
import model as mvd_model  # noqa: E402
import train as mvd_train  # noqa: E402
from unsup_bar_math import grpo_bar_math_step  # noqa: E402

try:
    import verifier_3tier as _v3t  # noqa: E402
except Exception:  # pragma: no cover
    _v3t = None


IMG_EXTS = frozenset((".png", ".jpg", ".jpeg", ".webp", ".bmp"))


def discover_unlabeled_strips(
    strip_dirs: Sequence[str],
    *,
    max_per_dir: int = 10000,
    seed: int = 42,
) -> List[Path]:
    """Recursively collect image strips from directories or JSONL manifests."""
    found: List[Path] = []
    for raw in strip_dirs:
        p = Path(raw)
        if not p.exists():
            continue
        dir_paths: List[Path] = []
        if p.is_file() and p.suffix.lower() == ".jsonl":
            root = p.parent
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                for key in ("image_path", "strip_path", "path", "file"):
                    if rec.get(key):
                        cand = Path(str(rec[key]))
                        if not cand.is_absolute():
                            cand = root / cand
                        if cand.exists():
                            dir_paths.append(cand)
                        break
                else:
                    for rel in rec.get("image_paths", []) or []:
                        cand = Path(str(rel))
                        if not cand.is_absolute():
                            cand = root / cand
                        if cand.exists():
                            dir_paths.append(cand)
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
                    # Skip diagnostic/params thumbnails if any.
                    if fp.name.endswith("_params.json"):
                        continue
                    dir_paths.append(fp)
                    if len(dir_paths) >= max_per_dir:
                        break
        found.extend(dir_paths[:max_per_dir])

    rng = random.Random(seed)
    rng.shuffle(found)
    return found


def make_tier2_visual_scorer() -> Optional[Callable[[torch.Tensor, Sequence[str]], float]]:
    """Build a fast Tier-2 visual consistency scorer for GRPO rewards.

    Uses ``verifier_3tier._to_obs_gray`` + ``_fallback_ink_density_score`` so
    Tier-2 visual reward runs in <1 ms on CPU/GPU tensors with zero external
    rendering subprocesses in the inner RLVR loop.
    """
    if _v3t is None:
        return None
    to_gray = getattr(_v3t, "_to_obs_gray", None)
    ink_score = getattr(_v3t, "_fallback_ink_density_score", None)
    if not callable(to_gray) or not callable(ink_score):
        return None

    def _scorer(strip_t: torch.Tensor, toks: Sequence[str]) -> float:
        gray = to_gray(strip_t)
        return float(ink_score(gray, toks))

    return _scorer


def load_mvdnet_from_ckpt(
    ckpt_path: Path,
    *,
    stride: str = "8x16",
    device: torch.device,
) -> tuple[mvd_model.MvdNet, Dict[str, Any]]:
    """Load an ``MvdNet`` checkpoint with automatic architecture flag detection."""
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    sd = state.get("model", state) if isinstance(state, dict) else {}

    w = sd.get("encoder.0.0.weight")
    extra_chans = max(0, int(w.shape[1]) - 3) if (w is not None and getattr(w, "ndim", 0) >= 2) else 0
    has_foveal = any(k.startswith("foveal_proj.") for k in sd)
    has_comer = any(k.startswith("comer_bias.") for k in sd)
    has_gdcm = any(k.startswith("gdcm_head.") for k in sd)
    has_s18 = any(k.startswith("s18_head.") for k in sd)
    has_rope2d = any(k.startswith("rope2d.") for k in sd)

    net = mvd_model.MvdNet(
        vocab=len(mvd_data.VOCAB) + 3,
        anisotropic=True,
        stride=stride,
        extra_chans=extra_chans,
        foveal_tap=has_foveal,
        use_comer=has_comer,
        use_gdcm=has_gdcm,
        use_s18=has_s18,
        rope_2d=has_rope2d,
    )
    mvd_train.load_weights_compat(net, state)
    net.to(device)
    meta = {
        "extra_chans": extra_chans,
        "stride": stride,
        "foveal_tap": has_foveal,
        "use_comer": has_comer,
        "use_gdcm": has_gdcm,
        "use_s18": has_s18,
        "rope_2d": has_rope2d,
    }
    return net, meta


def load_strip_batch(
    paths: Sequence[Path],
    *,
    height: int,
    in_chans: int,
    max_width: int = 1536,
) -> Optional[torch.Tensor]:
    """Load and right-pad a batch of strip images to ``[B, C, H, W_max]``."""
    tensors: List[torch.Tensor] = []
    max_w = 0
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            ow, oh = im.size
            nw = max(32, min(int(max_width), int(round(ow * (height / max(1, oh))))))
            im = im.resize((nw, int(height)), Image.BOX)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            t = (torch.from_numpy(arr).permute(2, 0, 1) - 0.85) / 0.25
            if in_chans > 3:
                pad_c = torch.zeros((in_chans - 3, int(height), nw), dtype=torch.float32)
                t = torch.cat([t, pad_c], dim=0)
            tensors.append(t)
            max_w = max(max_w, nw)
        except Exception:
            continue
    if not tensors:
        return None
    padded = []
    for t in tensors:
        pw = max_w - int(t.shape[2])
        if pw > 0:
            t = F.pad(t, (0, pw, 0, 0), value=0.6)
        padded.append(t)
    return torch.stack(padded, dim=0)


def _create_dry_run_strips(tmp_dir: Path, n: int = 4) -> List[Path]:
    paths: List[Path] = []
    for i in range(n):
        p = tmp_dir / f"dry_strip_{i}.png"
        arr = np.full((64, 160, 3), 240, dtype=np.uint8)
        for r in range(16, 49, 8):
            arr[r : r + 1, 8:152, :] = 35
        # Add deterministic notehead blobs so ink-density scorer gets signal.
        for c in range(24 + i * 4, 140, 22):
            arr[24 + (c % 16) : 29 + (c % 16), c : c + 6, :] = 20
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Post-SFT RLVR (K=4 GRPO) value trial runner for MvdNet."
    )
    ap.add_argument("--ckpt", default=None, help="Path to base SFT MvdNet .pt checkpoint.")
    ap.add_argument(
        "--strip-dirs",
        nargs="+",
        default=[],
        help="Directories or JSONL manifests of unlabeled printed + HW/RISM strips.",
    )
    ap.add_argument("--out-dir", default=None, help="Output directory for RLVR ckpt + JSON summary.")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-beams", type=int, default=4, help="GRPO group size K (default 4).")
    ap.add_argument("--kl-beta", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight", type=float, default=1.0, help="Loss multiplier.")
    ap.add_argument("--w-bm", type=float, default=0.75, help="Tier-1 bar-math reward weight.")
    ap.add_argument("--w-vis", type=float, default=0.25, help="Tier-2 visual reward weight.")
    ap.add_argument("--w-deg", type=float, default=0.50, help="Anti-degeneracy penalty weight.")
    ap.add_argument("--no-visual-reward", action="store_true", help="Disable Tier-2 visual reward.")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--stride", default="8x16", choices=("8x16", "4x8"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Fast CPU smoke test with a tiny dummy MvdNet and synthetic strips.",
    )
    return ap


def run_rlvr_trial(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the RLVR GRPO value trial and return summary metrics."""
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    tmp_ctx: Optional[tempfile.TemporaryDirectory] = None
    try:
        if args.dry_run:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="rlvr_grpo_dryrun_")
            tmp_p = Path(tmp_ctx.name)
            dev = torch.device("cpu")
            net = mvd_model.MvdNet(
                vocab=len(mvd_data.VOCAB) + 3,
                d_model=32,
                layers=1,
                heads=4,
                dim_ff=64,
                dropout=0.0,
                anisotropic=True,
                pretrained=False,
            ).to(dev)
            arch_meta = {"dry_run_tiny_net": True, "extra_chans": 0}
            strip_paths = _create_dry_run_strips(tmp_p, n=4)
            out_dir = Path(args.out_dir) if args.out_dir else (tmp_p / "out")
            steps = min(int(args.steps), 2)
            height = 64
            max_len = min(int(args.max_len), 24)
            use_amp = False
        else:
            if not args.ckpt:
                raise ValueError("--ckpt is required unless --dry-run is set")
            if not args.strip_dirs:
                raise ValueError("--strip-dirs is required unless --dry-run is set")
            dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            net, arch_meta = load_mvdnet_from_ckpt(
                Path(args.ckpt), stride=args.stride, device=dev
            )
            strip_paths = discover_unlabeled_strips(args.strip_dirs, seed=int(args.seed))
            if not strip_paths:
                raise ValueError(f"No strip images found in --strip-dirs: {args.strip_dirs}")
            out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent / "rlvr-out"
            steps = int(args.steps)
            height = int(args.height)
            max_len = int(args.max_len)
            use_amp = (not args.no_amp) and (dev.type == "cuda")

        out_dir.mkdir(parents=True, exist_ok=True)

        # Freeze KL anchor copy.
        ref_model = copy.deepcopy(net).to(dev).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        visual_scorer = None if args.no_visual_reward else make_tier2_visual_scorer()
        w_vis = 0.0 if visual_scorer is None else float(args.w_vis)

        opt = torch.optim.AdamW(net.parameters(), lr=float(args.lr), weight_decay=0.01)
        in_chans = int(getattr(net, "in_chans", 3 + int(arch_meta.get("extra_chans", 0))))
        net.train()

        history: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        n_imgs = len(strip_paths)
        bs = max(1, int(args.batch_size))

        for step in range(1, steps + 1):
            step_t0 = time.perf_counter()
            idxs = rng.integers(0, n_imgs, size=bs)
            batch_paths = [strip_paths[int(i)] for i in idxs]
            x = load_strip_batch(batch_paths, height=height, in_chans=in_chans)
            if x is None:
                continue
            x = x.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                step_loss, stats = grpo_bar_math_step(
                    net,
                    ref_model,
                    x,
                    id2tok=mvd_data.VOCAB,
                    num_beams=int(args.num_beams),
                    kl_beta=float(args.kl_beta),
                    max_len=max_len,
                    visual_scorer=visual_scorer,
                    w_bm=float(args.w_bm),
                    w_vis=w_vis,
                    w_deg=float(args.w_deg),
                )
                loss = step_loss * float(args.weight)

            did_step = False
            if (
                float(stats.get("active_frac", 0.0)) > 0.0
                and torch.isfinite(loss)
                and loss.requires_grad
            ):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                did_step = True

            step_ms = (time.perf_counter() - step_t0) * 1000.0
            row = {
                "step": step,
                "loss": round(float(loss.detach().item()), 6),
                "pg_loss": round(float(stats.get("pg_loss", 0.0)), 6),
                "kl_loss": round(float(stats.get("kl_loss", 0.0)), 6),
                "mean_reward": round(float(stats.get("mean_reward", 0.0)), 6),
                "mean_valid_bar_ratio": round(float(stats.get("mean_valid_bar_ratio", 0.0)), 6),
                "active_frac": round(float(stats.get("active_frac", 0.0)), 4),
                "degenerate_frac": round(float(stats.get("degenerate_frac", 0.0)), 4),
                "did_step": did_step,
                "ms": round(step_ms, 2),
            }
            history.append(row)

            if step == 1 or step % max(1, int(args.log_every)) == 0 or step == steps:
                print(
                    f"[trial_rlvr_grpo] step={step}/{steps} "
                    f"reward={row['mean_reward']:.4f} "
                    f"valid_bar={row['mean_valid_bar_ratio']:.4f} "
                    f"kl={row['kl_loss']:.4f} "
                    f"active={row['active_frac']:.2%} "
                    f"ms={row['ms']:.1f}",
                    flush=True,
                )

        # Save RLVR-polished checkpoint (tmp + rename for crash safety).
        ckpt_out = out_dir / "ckpt-rlvr-final.pt"
        ckpt_tmp = out_dir / "ckpt-rlvr-final.pt.tmp"
        torch.save(
            {
                "model": net.state_dict(),
                "rlvr_meta": {
                    "base_ckpt": str(args.ckpt) if args.ckpt else "dry_run_dummy",
                    "steps": steps,
                    "num_beams": int(args.num_beams),
                    "kl_beta": float(args.kl_beta),
                    "w_bm": float(args.w_bm),
                    "w_vis": w_vis,
                    "w_deg": float(args.w_deg),
                },
            },
            str(ckpt_tmp),
        )
        ckpt_tmp.replace(ckpt_out)

        summary = {
            "trial": "rlvr_grpo_mvdnet",
            "dry_run": bool(args.dry_run),
            "device": str(dev),
            "arch_meta": arch_meta,
            "n_strip_pool": n_imgs,
            "steps_ran": len(history),
            "config": {
                "num_beams": int(args.num_beams),
                "kl_beta": float(args.kl_beta),
                "lr": float(args.lr),
                "w_bm": float(args.w_bm),
                "w_vis": w_vis,
                "w_deg": float(args.w_deg),
                "visual_reward_enabled": visual_scorer is not None,
            },
            "metrics": {
                "mean_reward": round(statistics.mean(r["mean_reward"] for r in history), 6) if history else 0.0,
                "mean_valid_bar_ratio": round(statistics.mean(r["mean_valid_bar_ratio"] for r in history), 6) if history else 0.0,
                "mean_kl_loss": round(statistics.mean(r["kl_loss"] for r in history), 6) if history else 0.0,
                "mean_active_frac": round(statistics.mean(r["active_frac"] for r in history), 4) if history else 0.0,
                "steps_updated": sum(1 for r in history if r["did_step"]),
                "elapsed_secs": round(time.perf_counter() - t0, 2),
            },
            "ckpt_out": str(ckpt_out),
            "history": history,
        }

        summary_out = out_dir / "rlvr_summary.json"
        summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    summary = run_rlvr_trial(args)
    m = summary["metrics"]
    print(
        f"[trial_rlvr_grpo] DONE dry_run={summary['dry_run']} "
        f"steps={summary['steps_ran']} reward={m['mean_reward']:.4f} "
        f"valid_bar={m['mean_valid_bar_ratio']:.4f} kl={m['mean_kl_loss']:.4f} "
        f"elapsed={m['elapsed_secs']:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

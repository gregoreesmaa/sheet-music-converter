#!/usr/bin/env python3
"""Colab A100 / Local RTX 4080 SUPER runner — Iter 15 Exp i15-47: DeYO Conservation-Gated RLVR (GRPO).

Experiment ID: 2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr
Cross-field origin: Sim2Real (DeYO ICLR 2024 PLPD stability) + HMER/Tables/TikZ/Drift
(critic-free GRPO / RLVR with deterministic musical conservation rewards).

Controlled 2-arm proxy on edge85m + ax0_gl6 (600 total steps/arm, BF16, muon+adamw, seed=42):
  - Arm 0 (sft_ctrl):  600-step SFT baseline (muon+adamw, QAT at step 420, EMA 0.99).
  - Arm 1 (deyo_rlvr): 450-step SFT + 150-step short-horizon conservation-gated DeYO
                       pseudo-label consistency (PLPD stability gate on patch-perturbed views)
                       + critic-free GRPO (G=4 KV-cached rollouts) rewarded by exact syntax +
                       bar-duration conservation + anti-loop + token/note edit similarity
                       (`rlvr_grpo.compute_omr_rlvr_reward`).

Measures per arm:
  - raw_val_ce, ema_val_ce, int4_dequant_val_ce
  - val_rlvr_reward, val_r_parse, val_r_meter (bar conservation), val_r_note (edit similarity)
  - step_ms, train_elapsed_sec, peak_vram_mb, exact packed_int4_mb (< 55.0 MB budget)

Includes 1-arm HF partial checkpoint sync + startup auto-resume and Windows cuDNN 9.10 safety.
Security: reads HF_TOKEN from /tmp/.hf_token (chmod 600), HF_TOKEN env, or ~/.cache/huggingface/token;
never prints it, always unlinks /tmp/.hf_token in finally.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXP_ID = "2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr"
EXP_CODE = "i15-47"
ARMS = ("sft_ctrl", "deyo_rlvr")
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_CKPT_PREFIX = f"checkpoints/{EXP_ID}"

HF_TOKEN_FILE = Path("/tmp/.hf_token")


def _default_work_root() -> Path:
    env_root = os.environ.get("OMR_WORK_ROOT", "").strip()
    if env_root:
        return Path(env_root) / "i15_47"
    if Path("/vol").is_dir():
        return Path("/vol/i15_47")
    if Path("/content").is_dir():
        return Path("/content/i15_47")
    if sys.platform == "win32":
        return Path("D:/tmp/omr_work/i15_47")
    return Path("/tmp/i15_47")


WORK_ROOT = _default_work_root()

SEED = 42
TRAIN_STEPS = 600
RLVR_START_STEP = 451
WARMUP_STEPS = 60
QAT_START_STEP = 420
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR_MAX = 3e-4
WEIGHT_DECAY = 0.04
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
EMA_DECAY = 0.99
INT4_GROUP_SIZE = 32
HARD_BUDGET_MB = 55.0
SAVE_EVERY_MINUTES = 5.0

# DeYO + GRPO Phase-2 hyperparameters
GRPO_G = 4
GRPO_ROLLOUT_MAX_LEN = 192
GRPO_TEMPERATURE = 0.8
GRPO_CLIP_EPS = 0.2
GRPO_KL_BETA = 0.02
DEYO_PLPD_MIN = 0.02
LOSS_W_SFT = 0.50
LOSS_W_DEYO = 0.25
LOSS_W_GRPO = 0.25
VAL_ROLLOUT_PAGES = 16


# ---------------------------------------------------------------------------
# Infrastructure Helpers
# ---------------------------------------------------------------------------

def _configure_backend_safety() -> None:
    import torch

    if sys.platform == "win32":
        torch.backends.cudnn.enabled = False
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)


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
    if tok:
        return tok
    cache_tok = Path.home() / ".cache" / "huggingface" / "token"
    if cache_tok.is_file():
        tok = cache_tok.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    raise RuntimeError("HF token missing: provide /tmp/.hf_token, HF_TOKEN env var, or ~/.cache/huggingface/token")


def _purge_hf_token() -> None:
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass


def _locate_mvd() -> Path:
    candidates = [
        Path.cwd() / "tmp-train" / "mvd",
        Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path("/workspace/mvd"),
        Path("/content/mvd"),
        WORK_ROOT / "mvd",
    ]
    for cand in candidates:
        if (cand / "uni_omr_v2.py").is_file() and (cand / "rlvr_grpo.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    for bundle in (Path("/content/bundle.tar.gz"), Path("/content/mvd_bundle.tar.gz")):
        if bundle.is_file():
            WORK_ROOT.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORK_ROOT)
            for p in WORK_ROOT.rglob("uni_omr_v2.py"):
                if str(p.parent) not in sys.path:
                    sys.path.insert(0, str(p.parent))
                return p.parent
    raise FileNotFoundError("Could not locate tmp-train/mvd/uni_omr_v2.py")


def _ensure_deps() -> None:
    needed = []
    for mod, pkg in (("huggingface_hub", "huggingface_hub>=0.26"), ("PIL", "pillow"), ("numpy", "numpy")):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _wsd_mult(step: int, total_steps: int, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    decay_steps = max(1, int(round(total_steps * WSD_DECAY_FRAC)))
    decay_start = total_steps - decay_steps
    if step <= decay_start:
        return 1.0
    prog = min(1.0, max(0.0, float(step - decay_start) / float(decay_steps)))
    return MIN_LR_RATIO + 0.5 * (1.0 - MIN_LR_RATIO) * (1.0 + math.cos(math.pi * prog))


def _upload_json_to_hf(hf_token: str, local_path: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"{EXP_CODE}({EXP_ID}): upload {path_in_repo}",
    )
    print(f"[{EXP_ID}] Uploaded {path_in_repo} -> {HF_REPO_ID}", flush=True)


def _try_resume_from_hf(hf_token: str) -> Dict[str, Dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    recovered: Dict[str, Dict[str, Any]] = {}
    dl_dir = WORK_ROOT / "hf_dl" / "resume"
    dl_dir.mkdir(parents=True, exist_ok=True)

    try:
        ev_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_EVAL_PATH,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(dl_dir),
        )
        ev_data = json.loads(Path(ev_path).read_text(encoding="utf-8"))
        for arm_name, arm_res in (ev_data.get("arms") or {}).items():
            if arm_name in ARMS and isinstance(arm_res, dict) and "ema_val_ce" in arm_res:
                recovered[arm_name] = arm_res
    except Exception:
        pass

    for arm_name in ARMS:
        if arm_name in recovered:
            continue
        try:
            p = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"{HF_CKPT_PREFIX}/{arm_name}_arm_result.json",
                repo_type="dataset",
                token=hf_token,
                local_dir=str(dl_dir),
            )
            arm_res = json.loads(Path(p).read_text(encoding="utf-8"))
            if isinstance(arm_res, dict) and "ema_val_ce" in arm_res:
                recovered[arm_name] = arm_res
        except Exception:
            pass

    for arm_name, arm_res in recovered.items():
        cached = WORK_ROOT / "ckpts" / arm_name / "arm_result.json"
        if not cached.is_file():
            _atomic_write_json(cached, arm_res)
        print(f"[{EXP_ID}] Auto-resumed completed arm '{arm_name}' from HF Private Hub", flush=True)

    return recovered


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = WORK_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"
    if train_abs.is_file() and val_abs.is_file():
        return train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = WORK_ROOT / "hf_dl"
    dl_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR} from {HF_REPO_ID}...", flush=True)
    tar_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PROXY_TAR,
        repo_type="dataset",
        token=hf_token,
        local_dir=str(dl_dir),
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


def _load_manifest_pairs(manifest_path: Path) -> List[Tuple[Path, List[Path], str, float]]:
    pairs: List[Tuple[Path, List[Path], str, float]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        img_p = Path(rec["image"])
        beks = [Path(b) for b in rec.get("bekerns", [])]
        domain = str(rec.get("domain", "")).strip().lower()
        if img_p.is_file() and beks:
            is_dense_or_hw = (len(beks) >= 4) or ("hw" in domain) or ("hand" in domain) or ("debussy" in domain)
            weight = 2.0 if is_dense_or_hw else 1.0
            pairs.append((img_p, beks, domain, weight))
    return pairs


def _encode_bekerns_full(beks: List[Path]) -> List[int]:
    import uni_omr

    toks: List[str] = []
    for idx, bp in enumerate(beks):
        if idx > 0:
            toks.append(uni_omr.SYS_TOKEN)
        raw = bp.read_text(encoding="utf-8", errors="ignore").strip()
        if raw:
            toks.extend(raw.split())
    return uni_omr.encode_uni(toks)


def _pil_to_tensor(img_pil: Any, max_dim: int) -> "torch.Tensor":
    import numpy as np
    import torch
    from PIL import Image

    rgb = img_pil.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, float(max_dim) / float(max(w, h)))
    nw = max(32, int(round(w * scale / 16.0)) * 16)
    nh = max(32, int(round(h * scale / 16.0)) * 16)
    if (nw, nh) != (w, h):
        rgb = rgb.resize((nw, nh), Image.BICUBIC)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return (t - 0.5) / 0.5


def build_model(grad_checkpoint: bool = True, tiny: bool = False) -> Any:
    import uni_omr_v2

    if tiny:
        cfg = uni_omr_v2.UniOMRv2Config(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            n_axial_layers=0,
            n_global_layers=2,
            n_dec_layers=2,
            ffn_mult=2.0,
            axial_ffn_mult=2.0,
            backbone_channels=(16, 32, 48, 64),
            backbone_blocks=(1, 1, 1, 1),
            backbone_expand=2.0,
            router_hidden=32,
            max_mem_tokens=256,
            mtp_k=4,
            grad_checkpoint=grad_checkpoint,
        )
    else:
        cfg = uni_omr_v2.UniOMRv2Config.edge85m(
            n_axial_layers=0,
            n_global_layers=6,
            grad_checkpoint=grad_checkpoint,
        )
    return uni_omr_v2.UniOMRv2Net(cfg)


# ---------------------------------------------------------------------------
# DeYO PLPD Stability Perturbation + Conservation-Gated GRPO (G=4)
# ---------------------------------------------------------------------------

def _make_deyo_perturbed_views(imgs: "torch.Tensor", patch_size: int = 16) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Produce:
      1. `img_mild`: slight photometric/noise view for Sim2Real consistency target;
      2. `img_destroyed`: DeYO patch-shuffled view (40% of 16x16 patches permuted) to measure PLPD.
    """
    import torch

    # Mild photometric + Gaussian noise view (preserves notation)
    gain = 1.0 + 0.08 * (torch.rand((imgs.shape[0], 1, 1, 1), device=imgs.device) * 2.0 - 1.0)
    bias = 0.05 * (torch.rand((imgs.shape[0], 1, 1, 1), device=imgs.device) * 2.0 - 1.0)
    noise = 0.03 * torch.randn_like(imgs)
    img_mild = (imgs * gain + bias + noise).clamp(-1.0, 1.0)

    # Patch-destroyed view for DeYO PLPD measurement
    B, C, H, W = imgs.shape
    ph = max(8, min(patch_size, H // 2))
    pw = max(8, min(patch_size, W // 2))
    gh, gw = H // ph, W // pw
    if gh >= 2 and gw >= 2:
        crop = imgs[:, :, : gh * ph, : gw * pw].clone()
        patches = crop.view(B, C, gh, ph, gw, pw).permute(0, 2, 4, 1, 3, 5).reshape(B, gh * gw, C, ph, pw)
        n_p = gh * gw
        n_shuf = max(2, int(round(0.40 * n_p)))
        for b in range(B):
            idx = torch.randperm(n_p, device=imgs.device)[:n_shuf]
            perm = idx[torch.randperm(n_shuf, device=imgs.device)]
            patches[b, idx] = patches[b, perm]
        rebuilt = patches.view(B, gh, gw, C, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(B, C, gh * ph, gw * pw)
        img_destroyed = imgs.clone()
        img_destroyed[:, :, : gh * ph, : gw * pw] = rebuilt
    else:
        img_destroyed = imgs.flip(-1)
    return img_mild, img_destroyed


def compute_deyo_plpd_consistency_loss(
    model: Any,
    imgs: "torch.Tensor",
    tgt_in: "torch.Tensor",
    clean_logits: "torch.Tensor",
    use_bf16: bool,
) -> Tuple["torch.Tensor", float]:
    """DeYO (ICLR 2024) PLPD-gated symmetric KL consistency loss.

    Computes Pseudo-Label Probability Difference (PLPD) = p_clean(y*) - p_destroyed(y*).
    Tokens with PLPD > DEYO_PLPD_MIN are genuinely visually grounded (destroying image patches
    drops their confidence), so we enforce symmetric KL consistency against a mildly perturbed view
    only on visually grounded tokens, preventing collapse on hallucinated/language-prior tokens.
    """
    import torch
    import torch.nn.functional as F
    import uni_omr

    img_mild, img_destroyed = _make_deyo_perturbed_views(imgs)
    with torch.no_grad():
        with torch.amp.autocast(imgs.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            mem_dest, _, _ = model.encode_image(img_destroyed, return_aux=False)
            logits_dest, _, _ = model.decode_tokens(
                tgt_in, mem_dest, return_mtp=False, return_aux=False, apply_tok_dropout=False
            )
        p_clean = F.softmax(clean_logits.detach().float(), dim=-1)
        p_dest = F.softmax(logits_dest.detach().float(), dim=-1)
        top_p_clean, top_ids = p_clean.max(dim=-1)
        top_p_dest = p_dest.gather(-1, top_ids.unsqueeze(-1)).squeeze(-1)
        plpd = (top_p_clean - top_p_dest).clamp(min=0.0)  # (B, T)
        valid = (tgt_in != uni_omr.PAD_ID).float()
        gate = ((plpd > DEYO_PLPD_MIN).float() * (1.0 + plpd)) * valid

    with torch.amp.autocast(imgs.device.type, dtype=torch.bfloat16, enabled=use_bf16):
        mem_mild, _, _ = model.encode_image(img_mild, return_aux=False)
        logits_mild, _, _ = model.decode_tokens(
            tgt_in, mem_mild, return_mtp=False, return_aux=False, apply_tok_dropout=False
        )

    p_c = F.softmax(clean_logits.float(), dim=-1)
    log_p_c = F.log_softmax(clean_logits.float(), dim=-1)
    p_m = F.softmax(logits_mild.float(), dim=-1)
    log_p_m = F.log_softmax(logits_mild.float(), dim=-1)

    sym_kl = 0.5 * ((p_c.detach() * (log_p_c.detach() - log_p_m)).sum(dim=-1) + (p_m.detach() * (log_p_m.detach() - log_p_c)).sum(dim=-1))
    denom = gate.sum().clamp(min=1.0)
    loss = (sym_kl * gate).sum() / denom
    mean_plpd = float((plpd * valid).sum().item() / valid.sum().clamp(min=1.0).item())
    return loss, mean_plpd


def _sample_grpo_rollouts(
    model: Any,
    img_1: "torch.Tensor",
    g: int,
    max_len: int,
    temperature: float,
    use_bf16: bool,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Single-encoder-pass batched KV-cached sampling of G rollouts for 1 page."""
    import torch
    import torch.nn.functional as F
    import uni_omr

    was_training = model.training
    model.eval()
    device = img_1.device
    with torch.no_grad():
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            mem, _, _ = model.encode_image(img_1, return_aux=False)
            if isinstance(mem, (tuple, list)):
                mem_g = tuple(t.expand(g, -1, -1).contiguous() for t in mem)
            else:
                mem_g = mem.expand(g, -1, -1).contiguous()
            past_kvs = model.precompute_cross_kv(mem_g)

        cur_ids = torch.full((g, 1), uni_omr.BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(g, dtype=torch.bool, device=device)
        seqs = [cur_ids]
        old_lps = []

        for _ in range(max_len):
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits, _, past_kvs = model.decode_tokens(
                    cur_ids, memory=None, return_mtp=False, return_aux=False,
                    past_kvs=past_kvs, use_cache=True, apply_tok_dropout=False,
                )
            step_logits = logits[:, -1, :].float() / max(1e-3, float(temperature))
            log_probs = F.log_softmax(step_logits, dim=-1)
            probs = torch.exp(log_probs)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            tok_lp = log_probs.gather(-1, next_tok.unsqueeze(-1)).squeeze(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, uni_omr.PAD_ID), next_tok)
            tok_lp = torch.where(finished, torch.zeros_like(tok_lp), tok_lp)
            seqs.append(next_tok.unsqueeze(1))
            old_lps.append(tok_lp.unsqueeze(1))
            finished = finished | (next_tok == uni_omr.EOS_ID)
            if bool(finished.all()):
                break
            cur_ids = next_tok.unsqueeze(1)

    if was_training:
        model.train()
    return torch.cat(seqs, dim=1), torch.cat(old_lps, dim=1)


def compute_grpo_conservation_loss(
    model: Any,
    ref_model: Any,
    img_1: "torch.Tensor",
    ref_ids_1: List[int],
    use_bf16: bool,
    g: int = GRPO_G,
    max_len: int = GRPO_ROLLOUT_MAX_LEN,
) -> Tuple["torch.Tensor", Dict[str, float]]:
    """Sample G=4 rollouts, score with deterministic rlvr_grpo.compute_omr_rlvr_reward,
    and compute clipped GRPO surrogate + k3 KL penalty against frozen step-450 ref_model."""
    import torch
    import torch.nn.functional as F
    import rlvr_grpo
    import uni_omr

    rollout_seqs, old_lps = _sample_grpo_rollouts(
        model, img_1, g=g, max_len=max_len, temperature=GRPO_TEMPERATURE, use_bf16=use_bf16
    )
    ref_toks = uni_omr.decode_uni(ref_ids_1, skip_special=True)

    rewards: List[float] = []
    r_meters: List[float] = []
    r_notes: List[float] = []
    for i in range(g):
        pred_toks = uni_omr.decode_uni(rollout_seqs[i].tolist(), skip_special=True)
        rw = rlvr_grpo.compute_omr_rlvr_reward(pred_toks, ref_toks)
        rewards.append(float(rw["reward"]))
        r_meters.append(float(rw["r_meter"]))
        r_notes.append(float(rw["r_note"]))

    r_t = torch.tensor(rewards, dtype=torch.float32, device=img_1.device)
    adv = (r_t - r_t.mean()) / (r_t.std(unbiased=False) + 1e-6)  # (G,)

    tgt_in = rollout_seqs[:, :-1]
    tgt_out = rollout_seqs[:, 1:]
    valid = (tgt_out != uni_omr.PAD_ID).float()  # (G, T)

    # Current policy log-probs (differentiable)
    with torch.amp.autocast(img_1.device.type, dtype=torch.bfloat16, enabled=use_bf16):
        mem, _, _ = model.encode_image(img_1, return_aux=False)
        if isinstance(mem, (tuple, list)):
            mem_g = tuple(t.expand(g, -1, -1) for t in mem)
        else:
            mem_g = mem.expand(g, -1, -1)
        cur_logits, _, _ = model.decode_tokens(
            tgt_in, mem_g, return_mtp=False, return_aux=False, apply_tok_dropout=False
        )
    cur_lps = F.log_softmax(cur_logits.float(), dim=-1).gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)

    # Frozen reference policy log-probs for k3 KL penalty
    with torch.no_grad():
        with torch.amp.autocast(img_1.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            rmem, _, _ = ref_model.encode_image(img_1, return_aux=False)
            if isinstance(rmem, (tuple, list)):
                rmem_g = tuple(t.expand(g, -1, -1) for t in rmem)
            else:
                rmem_g = rmem.expand(g, -1, -1)
            ref_logits, _, _ = ref_model.decode_tokens(
                tgt_in, rmem_g, return_mtp=False, return_aux=False, apply_tok_dropout=False
            )
        ref_lps = F.log_softmax(ref_logits.float(), dim=-1).gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)

    log_ratio = (cur_lps - old_lps.detach()).clamp(-10.0, 10.0)
    ratio = torch.exp(log_ratio)
    adv_tok = adv.unsqueeze(1).expand_as(ratio)
    surr1 = ratio * adv_tok
    surr2 = ratio.clamp(1.0 - GRPO_CLIP_EPS, 1.0 + GRPO_CLIP_EPS) * adv_tok
    pg_loss = -torch.minimum(surr1, surr2)

    # Unbiased k3 KL estimator: exp(ref - cur) - (ref - cur) - 1
    kl_diff = (ref_lps - cur_lps).clamp(-10.0, 10.0)
    k3_kl = torch.exp(kl_diff) - kl_diff - 1.0

    tok_loss = pg_loss + GRPO_KL_BETA * k3_kl
    loss = (tok_loss * valid).sum() / valid.sum().clamp(min=1.0)
    stats = {
        "reward": round(float(r_t.mean().item()), 4),
        "r_meter": round(sum(r_meters) / max(1, len(r_meters)), 4),
        "r_note": round(sum(r_notes) / max(1, len(r_notes)), 4),
    }
    return loss, stats


# ---------------------------------------------------------------------------
# Packing, Validation CE & Rollout Reward Eval
# ---------------------------------------------------------------------------

def _pack_and_verify_hybrid_int4(
    model: Any,
    out_bin: Path,
    group_size: int = INT4_GROUP_SIZE,
    tiny: bool = False,
) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    infer_model = build_model(grad_checkpoint=False, tiny=tiny).cpu().eval()
    infer_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True)
    infer_model.strip_aux_heads()
    infer_model.mtp_heads = nn.ModuleList()

    int4_roots = (
        infer_model.encoder.axial_layers,
        infer_model.encoder.patch_merger,
        infer_model.encoder.global_layers,
        infer_model.deepstack_fusions,
        infer_model.dec_layers,
        infer_model.lm_head,
    )
    int4_ids = {id(m) for r in int4_roots for m in r.modules() if isinstance(m, nn.Linear)}
    int4_names = {f"{n}.weight" if n else "weight" for n, m in infer_model.named_modules() if id(m) in int4_ids}

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    dequant_sd: Dict[str, torch.Tensor] = {}
    with open(out_bin, "wb") as f:
        for name, param in infer_model.named_parameters():
            t = param.detach().float()
            if name in int4_names:
                flat = t.reshape(-1)
                orig_n = flat.numel()
                pad = (group_size - (orig_n % group_size)) % group_size
                if pad:
                    flat = torch.nn.functional.pad(flat, (0, pad))
                g = flat.reshape(-1, group_size)
                amax = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                scale = (amax / 7.0).to(torch.float16)
                q = torch.clamp(torch.round(g / scale.float()), -8, 7).to(torch.int8)
                u = (q.to(torch.int16) + 8).to(torch.uint8).reshape(-1)
                if u.numel() % 2 == 1:
                    u = torch.nn.functional.pad(u, (0, 1))
                packed = (u[0::2] | (u[1::2] << 4)).numpy()
                f.write(packed.tobytes())
                f.write(scale.squeeze(-1).numpy().tobytes())
                dq = (q.float() * scale.float()).reshape(-1)[:orig_n].reshape(t.shape)
                dequant_sd[name] = dq.to(param.dtype)
            else:
                fp16_np = t.to(torch.float16).numpy()
                f.write(fp16_np.tobytes())
                dequant_sd[name] = t.to(torch.float16).to(param.dtype)

    disk_bytes = out_bin.stat().st_size
    disk_mb = disk_bytes / 1_000_000.0
    infer_model.load_state_dict(dequant_sd, strict=False)
    return {
        "disk_bytes": int(disk_bytes),
        "packed_int4_mb": round(disk_mb, 3),
        "passes_55mb_budget": bool(disk_mb < HARD_BUDGET_MB),
        "dequant_model": infer_model,
    }


def _eval_val_ce(
    model: Any,
    val_pairs: List[Tuple[Path, List[Path], str, float]],
    device: "torch.device",
    use_bf16: bool,
) -> float:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr

    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for img_p, beks, _, _ in val_pairs:
            ids = _encode_bekerns_full(beks)
            if len(ids) < 3:
                continue
            inp = _pil_to_tensor(Image.open(img_p).convert("RGB"), FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
            tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                mem, _, _ = model.encode_image(inp, return_aux=False)
                logits, _, _ = model.decode_tokens(
                    tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID
                )
            losses.append(float(loss.item()))
    return round(sum(losses) / max(1, len(losses)), 5)


def _eval_val_rlvr_metrics(
    model: Any,
    val_pairs: List[Tuple[Path, List[Path], str, float]],
    device: "torch.device",
    use_bf16: bool,
    n_pages: int = VAL_ROLLOUT_PAGES,
) -> Dict[str, float]:
    """Greedy-decode up to n_pages of proxy_val and score with exact rlvr_grpo.compute_omr_rlvr_reward."""
    import torch
    from PIL import Image
    import rlvr_grpo
    import uni_omr

    model.eval()
    rws, parses, meters, notes = [], [], [], []
    with torch.no_grad():
        for img_p, beks, _, _ in val_pairs[:n_pages]:
            ids = _encode_bekerns_full(beks)
            if len(ids) < 3:
                continue
            inp = _pil_to_tensor(Image.open(img_p).convert("RGB"), FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                seq = model.generate(inp, max_len=min(384, len(ids) + 64))
            pred_toks = uni_omr.decode_uni(seq[0].tolist(), skip_special=True)
            ref_toks = uni_omr.decode_uni(ids, skip_special=True)
            rw = rlvr_grpo.compute_omr_rlvr_reward(pred_toks, ref_toks)
            rws.append(float(rw["reward"]))
            parses.append(float(rw["r_parse"]))
            meters.append(float(rw["r_meter"]))
            notes.append(float(rw["r_note"]))
    return {
        "val_rlvr_reward": round(sum(rws) / max(1, len(rws)), 5),
        "val_r_parse": round(sum(parses) / max(1, len(parses)), 5),
        "val_r_meter": round(sum(meters) / max(1, len(meters)), 5),
        "val_r_note": round(sum(notes) / max(1, len(notes)), 5),
    }


def _collate_microbatch(
    samples: List[Tuple[Path, List[Path], str, float]],
    device: "torch.device",
) -> Optional[Tuple["torch.Tensor", "torch.Tensor", List[List[int]]]]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr

    imgs: List[torch.Tensor] = []
    seqs: List[List[int]] = []
    for img_p, beks, _, _ in samples:
        ids = _encode_bekerns_full(beks)
        if len(ids) < 3:
            continue
        imgs.append(_pil_to_tensor(Image.open(img_p).convert("RGB"), FULLPAGE_MAX_DIM))
        seqs.append(ids[:MAX_SEQ_LEN])
    if not imgs:
        return None
    max_h = max(t.shape[1] for t in imgs)
    max_w = max(t.shape[2] for t in imgs)
    padded_imgs = [F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1]), value=1.0) for t in imgs]
    max_t = max(len(s) for s in seqs)
    tgt = torch.full((len(seqs), max_t), uni_omr.PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        tgt[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return torch.stack(padded_imgs, dim=0).to(device), tgt.to(device), seqs


def _train_arm(
    arm_name: str,
    *,
    train_pairs: List[Tuple[Path, List[Path], str, float]],
    val_pairs: List[Tuple[Path, List[Path], str, float]],
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    import train_uni_v2
    import uni_omr_v2

    _seed_everything(SEED)
    arm_dir = WORK_ROOT / "ckpts" / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached = arm_dir / "arm_result.json"
    if cached.is_file():
        print(f"[{EXP_ID}] Reusing cached result for {arm_name}", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))

    model = build_model(grad_checkpoint=True).to(device)
    psummary = model.param_summary(int4_block_size=INT4_GROUP_SIZE)
    qat = uni_omr_v2.QATInt4ControllerV2(model, block_size=INT4_GROUP_SIZE)
    opt = train_uni_v2.build_optimizer_v2(
        model,
        optimizer_name="muon",
        lr=LR_MAX,
        weight_decay=WEIGHT_DECAY,
        device_type=device.type,
        no_decay_1d=True,
    )
    if device.type == "cuda" and opt.mode != "muon+adamw":
        raise RuntimeError(f"[{EXP_ID}][{arm_name}] Silent fallback detected: expected muon+adamw, got {opt.mode}")
    print(f"[{EXP_ID}][{arm_name}] Optimizer verified: mode={opt.mode}", flush=True)

    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ref_model: Optional[Any] = None
    weights = [p[3] for p in train_pairs]
    rng = random.Random(SEED)
    start_step = 1
    loss_hist: List[float] = []
    step_ms_hist: List[float] = []
    grpo_reward_hist: List[float] = []
    plpd_hist: List[float] = []

    latest_pt = arm_dir / "latest.pt"
    if latest_pt.is_file():
        ck = torch.load(str(latest_pt), map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        ema_state = {k: v.to(device) for k, v in ck["ema_state_dict"].items()}
        if "opt_state_dicts" in ck and len(ck["opt_state_dicts"]) == len(opt.optimizers):
            for o_obj, sd in zip(opt.optimizers, ck["opt_state_dicts"]):
                o_obj.load_state_dict(sd)
        if "rng_state" in ck:
            rng.setstate(ck["rng_state"])
        start_step = int(ck["step"]) + 1
        loss_hist = list(ck.get("loss_hist", []))
        step_ms_hist = list(ck.get("step_ms_hist", []))
        if start_step > QAT_START_STEP:
            qat.enable()
        if arm_name == "deyo_rlvr" and "ref_state_dict" in ck and ck["ref_state_dict"] is not None:
            ref_model = build_model(grad_checkpoint=False).to(device).eval()
            ref_model.load_state_dict({k: v.to(device) for k, v in ck["ref_state_dict"].items()}, strict=True)
            for p in ref_model.parameters():
                p.requires_grad_(False)
        print(f"[{EXP_ID}][{arm_name}] Resumed local checkpoint at step {start_step}/{TRAIN_STEPS}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_arm0 = time.perf_counter()
    t_last_save = time.time()

    model.train()
    for step in range(start_step, TRAIN_STEPS + 1):
        if step == QAT_START_STEP:
            qat.enable()
            print(f"[{EXP_ID}][{arm_name}] Enabled QATInt4ControllerV2 at step {step}", flush=True)

        if arm_name == "deyo_rlvr" and step == RLVR_START_STEP and ref_model is None:
            ref_model = build_model(grad_checkpoint=False).to(device).eval()
            ref_model.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()}, strict=True)
            for p in ref_model.parameters():
                p.requires_grad_(False)
            print(f"[{EXP_ID}][{arm_name}] Snapshotted frozen step-450 ref_model; entering Phase 2 DeYO+GRPO", flush=True)

        mult = _wsd_mult(step, TRAIN_STEPS, WARMUP_STEPS)
        opt.set_lr(mult)

        t_step0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        in_phase2 = (arm_name == "deyo_rlvr") and (step >= RLVR_START_STEP)

        for ga_idx in range(GRAD_ACCUM):
            batch_samples = rng.choices(train_pairs, weights=weights, k=BATCH_SIZE)
            collated = _collate_microbatch(batch_samples, device)
            if collated is None:
                continue
            inp, tgt, raw_seqs = collated
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(inp, tgt[:, :-1], return_aux=False, return_mtp=True)
                sft_loss = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)["loss"]

            if in_phase2:
                deyo_loss, mean_plpd = compute_deyo_plpd_consistency_loss(
                    model, inp, tgt[:, :-1], out["logits"], use_bf16=use_bf16
                )
                plpd_hist.append(mean_plpd)
                # Run G=4 GRPO on first microbatch of each step to bound wall-clock & VRAM
                if ga_idx == 0 and ref_model is not None:
                    grpo_l, gstats = compute_grpo_conservation_loss(
                        model, ref_model, inp[:1], raw_seqs[0], use_bf16=use_bf16,
                        g=GRPO_G, max_len=GRPO_ROLLOUT_MAX_LEN,
                    )
                    grpo_reward_hist.append(gstats["reward"])
                    comb = LOSS_W_SFT * sft_loss + LOSS_W_DEYO * deyo_loss + LOSS_W_GRPO * grpo_l
                else:
                    comb = (LOSS_W_SFT + LOSS_W_GRPO) * sft_loss + LOSS_W_DEYO * deyo_loss
                loss = comb / GRAD_ACCUM
            else:
                loss = sft_loss / GRAD_ACCUM

            loss.backward()
            step_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_state[k].mul_(EMA_DECAY).add_(v, alpha=1.0 - EMA_DECAY)
                else:
                    ema_state[k].copy_(v)

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_ms_hist.append((time.perf_counter() - t_step0) * 1000.0)
        loss_hist.append(step_loss)

        if step % 100 == 0 or step == 1 or step == RLVR_START_STEP:
            mean_l = sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:]))
            mean_ms = sum(step_ms_hist[-50:]) / max(1, len(step_ms_hist[-50:]))
            extra = ""
            if grpo_reward_hist:
                extra = f" grpo_R={sum(grpo_reward_hist[-20:])/len(grpo_reward_hist[-20:]):.3f}"
            print(
                f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} "
                f"loss={mean_l:.4f} step_ms={mean_ms:.1f}{extra} elapsed={time.perf_counter() - t_arm0:.1f}s",
                flush=True,
            )

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ck = arm_dir / "latest.pt.tmp"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema_state,
                    "ref_state_dict": ref_model.state_dict() if ref_model is not None else None,
                    "opt_state_dicts": [o.state_dict() for o in opt.optimizers],
                    "rng_state": rng.getstate(),
                    "loss_hist": loss_hist,
                    "step_ms_hist": step_ms_hist,
                },
                tmp_ck,
            )
            os.replace(tmp_ck, latest_pt)
            t_last_save = time.time()

    train_elapsed_sec = time.perf_counter() - t_arm0
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0) if device.type == "cuda" else 0.0
    )

    if ref_model is not None:
        del ref_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    qat.disable()
    raw_val_ce = _eval_val_ce(model, val_pairs, device, use_bf16)

    model.load_state_dict(ema_state, strict=True)
    ema_val_ce = _eval_val_ce(model, val_pairs, device, use_bf16)
    rlvr_eval = _eval_val_rlvr_metrics(model, val_pairs, device, use_bf16, n_pages=VAL_ROLLOUT_PAGES)

    packed_info = _pack_and_verify_hybrid_int4(
        model, arm_dir / f"{arm_name}_hybrid_int4.bin", INT4_GROUP_SIZE
    )
    dq_model = packed_info.pop("dequant_model").to(device)
    int4_dequant_val_ce = _eval_val_ce(dq_model, val_pairs, device, use_bf16)

    steady_ms = step_ms_hist[10:] if len(step_ms_hist) > 10 else step_ms_hist
    result: Dict[str, Any] = {
        "arm": arm_name,
        "optimizer_mode": opt.mode,
        "steps": TRAIN_STEPS,
        "param_summary": psummary,
        "packed_int4_mb": packed_info["packed_int4_mb"],
        "packed_int4_verification": packed_info,
        "train_loss_tail50": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5),
        "raw_val_ce": raw_val_ce,
        "ema_val_ce": ema_val_ce,
        "int4_dequant_val_ce": int4_dequant_val_ce,
        **rlvr_eval,
        "step_ms": round(sum(steady_ms) / max(1, len(steady_ms)), 2),
        "train_elapsed_sec": round(train_elapsed_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }
    if grpo_reward_hist:
        result["train_grpo_reward_tail20"] = round(sum(grpo_reward_hist[-20:]) / len(grpo_reward_hist[-20:]), 4)
        result["train_mean_plpd"] = round(sum(plpd_hist) / max(1, len(plpd_hist)), 4)

    _atomic_write_json(cached, result)
    del model, dq_model, opt, ema_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# CPU Smoke Test
# ---------------------------------------------------------------------------

def run_smoke_test() -> int:
    """CPU smoke test: 2 SFT steps + 2 DeYO+GRPO (G=4) steps on tiny model + edge85m packed_int4_mb."""
    _locate_mvd()
    import torch
    import train_uni_v2
    import uni_omr

    device = torch.device("cpu")
    _seed_everything(SEED)

    model = build_model(grad_checkpoint=False, tiny=True).to(device)
    opt = train_uni_v2.build_optimizer_v2(
        model, optimizer_name="muon", lr=LR_MAX, weight_decay=WEIGHT_DECAY,
        device_type="cpu", allow_cpu_muon=True, no_decay_1d=True,
    )
    assert opt.mode == "muon+adamw", f"Expected muon+adamw, got {opt.mode}"
    model.train()

    sft_losses = []
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        imgs = torch.randn(1, 3, 64, 96, device=device)
        tgt = torch.randint(4, min(64, uni_omr.VOCAB_SIZE), (1, 14), device=device)
        tgt[:, 0] = uni_omr.BOS_ID
        out = model(imgs, tgt[:, :-1], return_aux=False, return_mtp=True)
        loss = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        sft_losses.append(round(float(loss.item()), 4))

    # Snapshot frozen ref_model and run 2 Phase-2 DeYO + GRPO (G=4) steps
    ref_model = build_model(grad_checkpoint=False, tiny=True).to(device).eval()
    ref_model.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()}, strict=True)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    p2_losses = []
    p2_rewards = []
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        imgs = torch.randn(1, 3, 64, 96, device=device)
        tgt = torch.randint(4, min(64, uni_omr.VOCAB_SIZE), (1, 14), device=device)
        tgt[:, 0] = uni_omr.BOS_ID
        out = model(imgs, tgt[:, :-1], return_aux=False, return_mtp=True)
        sft_l = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)["loss"]
        deyo_l, mean_plpd = compute_deyo_plpd_consistency_loss(model, imgs, tgt[:, :-1], out["logits"], use_bf16=False)
        grpo_l, gstats = compute_grpo_conservation_loss(
            model, ref_model, imgs[:1], tgt[0].tolist(), use_bf16=False, g=4, max_len=12
        )
        total_l = LOSS_W_SFT * sft_l + LOSS_W_DEYO * deyo_l + LOSS_W_GRPO * grpo_l
        assert torch.isfinite(total_l), f"Non-finite Phase-2 loss: {total_l}"
        total_l.backward()
        opt.step()
        p2_losses.append(round(float(total_l.item()), 4))
        p2_rewards.append(gstats["reward"])

    with tempfile.TemporaryDirectory() as tmpd:
        m_edge = build_model(grad_checkpoint=False, tiny=False)
        pack_edge = _pack_and_verify_hybrid_int4(
            m_edge, Path(tmpd) / "edge85m.bin", INT4_GROUP_SIZE, tiny=False
        )
        pack_edge.pop("dequant_model")
        assert pack_edge["passes_55mb_budget"], f"Exceeded 55 MB: {pack_edge['packed_int4_mb']}"
        del m_edge

    print(f"[{EXP_CODE}][SMOKE] SFT 2-step losses: {sft_losses}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] Phase-2 DeYO+GRPO(G=4) 2-step losses: {p2_losses} | rewards: {p2_rewards}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] edge85m packed_int4_mb={pack_edge['packed_int4_mb']} MB (< {HARD_BUDGET_MB} MB)", flush=True)
    print(f"[{EXP_CODE}][SMOKE] PASS!", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{EXP_ID} runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run CPU smoke test and exit")
    args = parser.parse_args()
    if args.smoke_test:
        return run_smoke_test()

    t_start = time.perf_counter()
    started_at = _utc_now()
    hf_token: Optional[str] = None
    status = "RUNNING"
    error_msg: Optional[str] = None
    arm_results: Dict[str, Dict[str, Any]] = {}

    try:
        hf_token = _read_hf_token()
        _ensure_deps()
        _locate_mvd()
        _configure_backend_safety()

        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        print(
            f"[{EXP_ID}] Device={gpu_name} | bf16={use_bf16} | cudnn={torch.backends.cudnn.enabled} | torch={torch.__version__}",
            flush=True,
        )

        arm_results.update(_try_resume_from_hf(hf_token))

        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(f"[{EXP_ID}] Loaded {len(train_pairs)} train / {len(val_pairs)} val", flush=True)

        for arm_name in ARMS:
            res = _train_arm(
                arm_name,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                device=device,
                use_bf16=use_bf16,
            )
            arm_results[arm_name] = res

            arm_json_local = WORK_ROOT / "ckpts" / arm_name / "arm_result.json"
            _upload_json_to_hf(hf_token, arm_json_local, f"{HF_CKPT_PREFIX}/{arm_name}_arm_result.json")

            eval_payload: Dict[str, Any] = {
                "exp_id": EXP_ID,
                "exp_code": EXP_CODE,
                "status": "RUNNING" if len(arm_results) < len(ARMS) else "COMPLETED",
                "started_at": started_at,
                "updated_at": _utc_now(),
                "gpu": gpu_name,
                "bf16": use_bf16,
                "steps_per_arm": TRAIN_STEPS,
                "seed": SEED,
                "arms": arm_results,
            }
            if "sft_ctrl" in arm_results and "deyo_rlvr" in arm_results:
                c = arm_results["sft_ctrl"]
                t = arm_results["deyo_rlvr"]
                eval_payload["delta_deyo_rlvr_minus_sft_ctrl"] = {
                    "ema_val_ce": round(t["ema_val_ce"] - c["ema_val_ce"], 5),
                    "int4_dequant_val_ce": round(t["int4_dequant_val_ce"] - c["int4_dequant_val_ce"], 5),
                    "val_rlvr_reward": round(t["val_rlvr_reward"] - c["val_rlvr_reward"], 5),
                    "val_r_meter": round(t["val_r_meter"] - c["val_r_meter"], 5),
                    "val_r_note": round(t["val_r_note"] - c["val_r_note"], 5),
                    "step_ms": round(t["step_ms"] - c["step_ms"], 2),
                }
            eval_local = WORK_ROOT / "out" / "eval.json"
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
        run_payload = {
            "exp_id": EXP_ID,
            "exp_code": EXP_CODE,
            "status": status,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_sec": round(total_sec, 2),
            "n_arms_completed": len(arm_results),
            "eval_path": HF_EVAL_PATH,
            "error": error_msg,
        }
        run_local = WORK_ROOT / "out" / "run.json"
        _atomic_write_json(run_local, run_payload)
        if hf_token:
            try:
                _upload_json_to_hf(hf_token, run_local, HF_RUN_PATH)
            except Exception as up_exc:
                print(f"[{EXP_ID}] WARN: failed to upload run ledger: {up_exc}", flush=True)
        _purge_hf_token()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Iter 13 Proposal i13-09: ParetoQ Mixed-Bit (INT3/INT2/Ternary FFN) & AWQ Edge QAT.

Experiment ID: 2026-11-26T07-40-i13-09-paretoq-awq-edge-qat
Academic basis:
  - ParetoQ (Liu et al., Meta AI, NeurIPS 2025, arXiv:2502.02631): unified sub-4-bit
    QAT scaling showing 3-bit / 2-bit / 1.58-bit ternary FFN matrices achieve a
    strictly superior accuracy-vs-MB Pareto frontier for edge deployment.
  - AWQ (Lin et al., MLSys 2024 Best Paper): smooth per-channel activation-salience
    scaling (s_j = mean(|X_j|)^alpha) protects salient channels without SpinQuant R1's
    Hadamard spike smearing under weight-only W4A16 quantization (Exp 8 finding).

Controlled 5-arm QAT sweep from 1 shared 600-step trained UniOMRv2Net(small_45m):
  1. int4_uniform_qat            (baseline: W4A16 block_size=32 everywhere)
  2. int4_awq_scaled_qat         (AWQ alpha=0.5 channel scaling + W4A16 QAT)
  3. paretoq_w3_ffn_w4_attn      (3-bit SwiGLU FFN + 4-bit attn/lm_head)
  4. paretoq_w2_ffn_w4_attn      (2-bit ParetoQ SwiGLU FFN + 4-bit attn/lm_head)
  5. paretoq_ternary_ffn_w4_attn (1.58-bit BitNet ternary FFN + 4-bit attn/lm_head)

Uploads evals/<EXP_ID>.json and runs/<EXP_ID>.json to HF Private Dataset Hub.
Zero edits to train_uni_v2.py or uni_omr_v2.py.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import difflib
import gc
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T07-40-i13-09-paretoq-awq-edge-qat"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_CANDIDATES = (
    Path("/content/bundle.tar.gz"),
    Path("/content/ov_bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
)
WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")

SEED = 42
BASE_STEPS = 600
BASE_LR = 3e-4
QAT_STEPS = 200
QAT_LR = 1e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
VAL_EVAL_MAX_LEN = 512
KL_MAX_PAGES = 24
AWQ_CALIB_PAGES = 32
AWQ_ALPHA = 0.5
BLOCK_SIZE = 32
SAVE_EVERY_MINUTES = 5.0

ARMS: List[Dict[str, Any]] = [
    {"name": "int4_uniform_qat", "ffn_mode": "int4", "use_awq": False},
    {"name": "int4_awq_scaled_qat", "ffn_mode": "int4", "use_awq": True},
    {"name": "paretoq_w3_ffn_w4_attn", "ffn_mode": "int3", "use_awq": False},
    {"name": "paretoq_w2_ffn_w4_attn", "ffn_mode": "int2", "use_awq": False},
    {"name": "paretoq_ternary_ffn_w4_attn", "ffn_mode": "ternary", "use_awq": False},
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and TOKEN_PATH.exists():
        try:
            os.chmod(TOKEN_PATH, 0o600)
        except OSError:
            pass
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    if TOKEN_PATH.exists():
        try:
            TOKEN_PATH.unlink()
        except OSError:
            pass
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var")
    return token


def _setup_environment() -> Path:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    VOL_ROOT.mkdir(parents=True, exist_ok=True)

    candidates = [
        WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
        Path("/content/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
    ]
    mvd_dir = next((c for c in candidates if (c / "uni_omr_v2.py").is_file()), None)
    if mvd_dir is None:
        for b in BUNDLE_CANDIDATES:
            if b.is_file():
                print(f"[{EXP_ID}] Extracting {b} -> {WORKSPACE_ROOT}...", flush=True)
                with tarfile.open(b, "r:*") as tf:
                    tf.extractall(path=WORKSPACE_ROOT)
                mvd_dir = next((c for c in candidates if (c / "uni_omr_v2.py").is_file()), None)
                if mvd_dir is not None:
                    break
    if mvd_dir is None:
        raise FileNotFoundError("Could not locate tmp-train/mvd/uni_omr_v2.py")

    needed = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)
    return mvd_dir


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    proxy_dir = VOL_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    val_abs = proxy_dir / "proxy_val_abs.jsonl"
    if not (train_abs.is_file() and val_abs.is_file()):
        proxy_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO, filename=HF_PROXY_TAR, repo_type="dataset",
            token=hf_token, local_dir="/tmp/hf_dl",
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=proxy_dir)
        Path(tar_p).unlink(missing_ok=True)
        for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
            out_lines: List[str] = []
            for raw in (proxy_dir / src_name).read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                rec["image"] = str((proxy_dir / rec["image"]).resolve())
                rec["bekerns"] = [str((proxy_dir / b).resolve()) for b in rec.get("bekerns", [])]
                out_lines.append(json.dumps(rec))
            dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return train_abs, val_abs


def _load_manifest_pairs(manifest_path: Path) -> List[Tuple[Path, List[Path]]]:
    pairs: List[Tuple[Path, List[Path]]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        img_p = Path(rec["image"])
        beks = [Path(b) for b in rec.get("bekerns", [])]
        if img_p.is_file() and beks:
            pairs.append((img_p, beks))
    return pairs


def _load_batch(
    pairs: Sequence[Tuple[Path, List[Path]]],
    rng: random.Random,
    batch_size: int,
    device: Any,
) -> Tuple[Any, Any, Any]:
    from PIL import Image
    import torch
    import uni_omr

    imgs: List[Any] = []
    seqs: List[List[int]] = []
    for _ in range(batch_size):
        img_p, beks = rng.choice(pairs)
        with Image.open(img_p) as im:
            rgb = im.convert("RGB")
        imgs.append(uni_omr._pil_to_chw_tensor(uni_omr._resize_fullpage_pil(rgb, FULLPAGE_MAX_DIM)))
        toks: List[str] = []
        for idx, bp in enumerate(beks):
            if idx > 0:
                toks.append(uni_omr.SYS_TOKEN)
            toks.extend(bp.read_text(encoding="utf-8").split())
        ids = [uni_omr.BOS_ID] + [
            uni_omr.UNI_TOKEN_TO_ID[t] for t in toks if t in uni_omr.UNI_TOKEN_TO_ID
        ] + [uni_omr.EOS_ID]
        seqs.append(ids[:MAX_SEQ_LEN])

    max_h = max(im.shape[1] for im in imgs)
    max_w = max(im.shape[2] for im in imgs)
    img_batch = torch.ones((batch_size, 3, max_h, max_w), dtype=torch.float32)
    for i, im in enumerate(imgs):
        img_batch[i, :, : im.shape[1], : im.shape[2]] = im

    max_l = max(len(s) for s in seqs)
    tok_batch = torch.full((batch_size, max_l), uni_omr.PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        tok_batch[i, : len(s)] = torch.tensor(s, dtype=torch.long)

    return img_batch.to(device), tok_batch[:, :-1].to(device), tok_batch[:, 1:].to(device)


def _fake_quant_group(w: Any, mode: str, block_size: int = BLOCK_SIZE) -> Any:
    """Differentiable STE group-wise fake quantizer for INT4 / INT3 / ParetoQ INT2 / Ternary."""
    import torch
    import torch.nn.functional as F

    orig_shape = w.shape
    w_f = w.float()
    flat = w_f.reshape(-1)
    rem = flat.numel() % block_size
    if rem != 0:
        flat = F.pad(flat, (0, block_size - rem))
    groups = flat.view(-1, block_size)

    if mode == "int4":
        qmax = 7.0
        scale = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / qmax
        q = torch.clamp(torch.round(groups / scale), -qmax, qmax) * scale
    elif mode == "int3":
        qmax = 3.0
        scale = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / qmax
        q = torch.clamp(torch.round(groups / scale), -qmax, qmax) * scale
    elif mode == "int2":
        # ParetoQ 2-bit symmetric 4-level {-1.5, -0.5, +0.5, +1.5} * scale
        scale = groups.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
        normed = groups / scale
        q_idx = torch.clamp(torch.floor(normed + 2.0), 0.0, 3.0) - 1.5
        q = q_idx * scale
    elif mode == "ternary":
        # BitNet b1.58 / ParetoQ ternary {-1, 0, +1} * mean(|w|)
        scale = groups.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
        q = torch.clamp(torch.round(groups / scale), -1.0, 1.0) * scale
    else:
        raise ValueError(f"Unknown quant mode: {mode}")

    q_flat = q.reshape(-1)[: w_f.numel()].view(orig_shape).to(w.dtype)
    return w + (q_flat - w).detach()


class _MixedBitQATController:
    """Attach STE fake quantization (and optional AWQ per-channel scaling) via forward_pre_hooks."""

    def __init__(
        self,
        model: Any,
        *,
        ffn_mode: str,
        awq_scales: Optional[Dict[str, Any]] = None,
        block_size: int = BLOCK_SIZE,
    ) -> None:
        import torch.nn as nn

        self.handles: List[Any] = []
        self.ffn_params = 0
        self.attn_params = 0
        self.ffn_mode = ffn_mode
        self.block_size = block_size

        target_roots = {
            "encoder.axial_layers",
            "encoder.patch_merger",
            "encoder.global_layers",
            "deepstack_fusions",
            "dec_layers",
            "lm_head",
        }
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if not any(name == r or name.startswith(r + ".") for r in target_roots):
                continue
            is_ffn = (".w_gate_up" in name) or (".w_down" in name) or (".ffn." in name)
            mode = ffn_mode if is_ffn else "int4"
            if is_ffn:
                self.ffn_params += mod.weight.numel()
            else:
                self.attn_params += mod.weight.numel()
            s_vec = awq_scales.get(name) if awq_scales else None

            def _make_hook(m_mode: str, ch_scale: Optional[Any]):
                def _pre_hook(m: Any, inp: Tuple[Any, ...]):
                    w = m.weight
                    if ch_scale is not None:
                        s = ch_scale.to(device=w.device, dtype=w.dtype)
                        w_eff = _fake_quant_group(w * s.unsqueeze(0), m_mode, block_size) / s.unsqueeze(0)
                    else:
                        w_eff = _fake_quant_group(w, m_mode, block_size)
                    m._orig_weight_tmp = m.weight
                    m._parameters["weight"] = w_eff
                    return inp
                return _pre_hook

            def _post_hook(m: Any, _inp: Any, out: Any):
                if hasattr(m, "_orig_weight_tmp"):
                    m._parameters["weight"] = m._orig_weight_tmp
                    delattr(m, "_orig_weight_tmp")
                return out

            self.handles.append(mod.register_forward_pre_hook(_make_hook(mode, s_vec)))
            self.handles.append(mod.register_forward_hook(_post_hook))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def estimate_webgpu_mb(self, model: Any) -> Dict[str, float]:
        ps = model.param_summary(int4_block_size=self.block_size)
        bits_map = {"int4": 4.0, "int3": 3.0, "int2": 2.0, "ternary": 1.585}
        ffn_b = bits_map[self.ffn_mode]
        scale_bytes = 2.0 / float(self.block_size)
        ffn_bytes = self.ffn_params * (ffn_b / 8.0 + scale_bytes)
        attn_bytes = self.attn_params * (4.0 / 8.0 + scale_bytes)
        fp16_bytes = ps["fp16_keep_params"] * 2.0
        small_mb = (fp16_bytes + attn_bytes + ffn_bytes) / 1e6
        # Scale factor from measured Exp 8 ratio (edge85m 50.735 / small_45m 28.038 = 1.8095)
        edge85_mb = small_mb * (50.735 / 28.038)
        return {
            "small_45m_webgpu_mb": round(small_mb, 3),
            "projected_edge85m_webgpu_mb": round(edge85_mb, 3),
            "ffn_params": int(self.ffn_params),
            "attn_params": int(self.attn_params),
        }


def _calibrate_awq_scales(
    model: Any,
    train_pairs: Sequence[Tuple[Path, List[Path]]],
    device: Any,
    amp_dtype: Any,
) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    act_sums: Dict[str, Any] = {}
    act_counts: Dict[str, int] = {}
    handles: List[Any] = []

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            def _make_h(nm: str):
                def _hook(_m: Any, inp: Tuple[Any, ...], _out: Any) -> None:
                    x = inp[0].detach().float().abs().reshape(-1, inp[0].shape[-1]).mean(dim=0)
                    if nm not in act_sums:
                        act_sums[nm] = x
                        act_counts[nm] = 1
                    else:
                        act_sums[nm] = act_sums[nm] + x
                        act_counts[nm] += 1
                return _hook
            handles.append(mod.register_forward_hook(_make_h(name)))

    rng = random.Random(SEED + 99)
    model.eval()
    with torch.no_grad():
        for _ in range(max(1, AWQ_CALIB_PAGES // BATCH_SIZE)):
            imgs, inp_ids, _ = _load_batch(train_pairs, rng, BATCH_SIZE, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                model(imgs, inp_ids)

    for h in handles:
        h.remove()

    scales: Dict[str, Any] = {}
    for nm, s in act_sums.items():
        mean_abs = (s / float(max(1, act_counts[nm]))).clamp(min=1e-4)
        sc = mean_abs.pow(AWQ_ALPHA)
        sc = sc / sc.mean().clamp(min=1e-4)
        scales[nm] = sc.clamp(0.25, 4.0).cpu()
    return scales


def _eval_val_metrics(
    model: Any,
    val_pairs: Sequence[Tuple[Path, List[Path]]],
    device: Any,
    amp_dtype: Any,
    ref_logits_cache: Optional[List[Any]] = None,
) -> Tuple[Dict[str, float], List[Any]]:
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import uni_omr

    model.eval()
    ce_list: List[float] = []
    f1_list: List[float] = []
    kl_list: List[float] = []
    mse_list: List[float] = []
    top1_list: List[float] = []
    captured_logits: List[Any] = []

    with torch.no_grad():
        for idx, (img_p, beks) in enumerate(val_pairs[:KL_MAX_PAGES]):
            with Image.open(img_p) as im:
                rgb = im.convert("RGB")
            img_t = uni_omr._pil_to_chw_tensor(uni_omr._resize_fullpage_pil(rgb, FULLPAGE_MAX_DIM)).unsqueeze(0).to(device)
            toks: List[str] = []
            for bi, bp in enumerate(beks):
                if bi > 0:
                    toks.append(uni_omr.SYS_TOKEN)
                toks.extend(bp.read_text(encoding="utf-8").split())
            ids = [uni_omr.BOS_ID] + [
                uni_omr.UNI_TOKEN_TO_ID[t] for t in toks if t in uni_omr.UNI_TOKEN_TO_ID
            ] + [uni_omr.EOS_ID]
            ids = ids[:VAL_EVAL_MAX_LEN]
            t_in = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            t_out = torch.tensor([ids[1:]], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(img_t, t_in)
            logits = out["logits"].float()
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), t_out.reshape(-1), ignore_index=uni_omr.PAD_ID)
            ce_list.append(float(ce.item()))

            pred_ids = logits.argmax(dim=-1)[0].tolist()
            sm = difflib.SequenceMatcher(None, ids[1:], pred_ids, autojunk=False)
            matches = sum(b.size for b in sm.get_matching_blocks())
            f1_list.append(2.0 * matches / max(1, len(ids[1:]) + len(pred_ids)))

            if ref_logits_cache is None:
                captured_logits.append(logits.cpu())
            else:
                ref_l = ref_logits_cache[idx].to(device)
                p_ref = F.softmax(ref_l, dim=-1)
                log_p_ref = F.log_softmax(ref_l, dim=-1)
                log_p_q = F.log_softmax(logits, dim=-1)
                kl = (p_ref * (log_p_ref - log_p_q)).sum(dim=-1).mean()
                kl_list.append(float(kl.item()))
                mse_list.append(float(F.mse_loss(logits, ref_l).item()))
                top1_list.append(float((logits.argmax(-1) == ref_l.argmax(-1)).float().mean().item()))

    res = {
        "val_ce_loss": round(sum(ce_list) / max(1, len(ce_list)), 5),
        "val_token_f1": round(sum(f1_list) / max(1, len(f1_list)), 5),
    }
    if ref_logits_cache is not None:
        res["logit_kl_vs_fp32"] = round(sum(kl_list) / max(1, len(kl_list)), 6)
        res["logit_mse_vs_fp32"] = round(sum(mse_list) / max(1, len(mse_list)), 6)
        res["top1_token_agree"] = round(sum(top1_list) / max(1, len(top1_list)), 5)
    return res, captured_logits


def main() -> None:
    t0 = time.perf_counter()
    try:
        hf_token = _read_and_purge_hf_token()
        mvd_dir = _setup_environment()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        import torch
        from huggingface_hub import HfApi
        import uni_omr_v2 as v2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
        amp_dtype = torch.bfloat16 if (device.type == "cuda" and cc[0] >= 8) else (
            torch.float16 if device.type == "cuda" else None
        )
        use_scaler = (device.type == "cuda" and cc[0] < 8)

        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)

        ckpt_dir = VOL_ROOT / "ckpts" / EXP_ID
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        base_ckpt = ckpt_dir / "base_fp_600.pt"

        cfg = v2.UniOMRv2Config.small_45m(grad_checkpoint=True)
        base_model = v2.UniOMRv2Net(cfg).to(device)
        if base_ckpt.is_file():
            base_model.load_state_dict(torch.load(base_ckpt, map_location="cpu", weights_only=True))
        else:
            random.seed(SEED)
            torch.manual_seed(SEED)
            rng = random.Random(SEED)
            opt = torch.optim.AdamW(base_model.parameters(), lr=BASE_LR, weight_decay=0.04)
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
            base_model.train()
            last_save = time.perf_counter()
            for step in range(1, BASE_STEPS + 1):
                lr = BASE_LR * min(1.0, step / 40.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / BASE_STEPS)))
                for pg in opt.param_groups:
                    pg["lr"] = lr
                opt.zero_grad(set_to_none=True)
                for _ in range(GRAD_ACCUM):
                    imgs, inp_ids, tgt_ids = _load_batch(train_pairs, rng, BATCH_SIZE, device)
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                        out = base_model(imgs, inp_ids)
                        loss_d = base_model.compute_loss(out, tgt_ids, images=imgs)
                        loss = loss_d["loss"] / float(GRAD_ACCUM)
                    scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if step % 100 == 0 or step == BASE_STEPS:
                    print(f"[{EXP_ID}] base step {step}/{BASE_STEPS} loss={loss_d['loss'].item():.4f}", flush=True)
                if (time.perf_counter() - last_save) >= SAVE_EVERY_MINUTES * 60.0:
                    tmp = base_ckpt.with_suffix(".pt.tmp")
                    torch.save(base_model.state_dict(), tmp)
                    os.replace(tmp, base_ckpt)
                    last_save = time.perf_counter()
            tmp = base_ckpt.with_suffix(".pt.tmp")
            torch.save(base_model.state_dict(), tmp)
            os.replace(tmp, base_ckpt)
            del opt

        fp32_metrics, ref_logits = _eval_val_metrics(base_model, val_pairs, device, amp_dtype=None)
        awq_scales = _calibrate_awq_scales(base_model, train_pairs, device, amp_dtype)
        base_state = copy.deepcopy(base_model.state_dict())
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        arm_results: Dict[str, Any] = {}
        for arm in ARMS:
            arm_res_p = ckpt_dir / arm["name"] / "arm_result.json"
            if arm_res_p.is_file():
                arm_results[arm["name"]] = json.loads(arm_res_p.read_text(encoding="utf-8"))
                continue

            model = v2.UniOMRv2Net(cfg).to(device)
            model.load_state_dict(base_state, strict=True)
            ctrl = _MixedBitQATController(
                model,
                ffn_mode=arm["ffn_mode"],
                awq_scales=(awq_scales if arm["use_awq"] else None),
                block_size=BLOCK_SIZE,
            )
            ptq_metrics, _ = _eval_val_metrics(model, val_pairs, device, amp_dtype=amp_dtype, ref_logits_cache=ref_logits)

            rng = random.Random(SEED + 7)
            opt = torch.optim.AdamW(model.parameters(), lr=QAT_LR, weight_decay=0.02)
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
            model.train()
            t_q = time.perf_counter()
            for step in range(1, QAT_STEPS + 1):
                lr = QAT_LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / QAT_STEPS)))
                for pg in opt.param_groups:
                    pg["lr"] = lr
                opt.zero_grad(set_to_none=True)
                for _ in range(GRAD_ACCUM):
                    imgs, inp_ids, tgt_ids = _load_batch(train_pairs, rng, BATCH_SIZE, device)
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                        out = model(imgs, inp_ids)
                        loss_d = model.compute_loss(out, tgt_ids, images=imgs)
                        loss = loss_d["loss"] / float(GRAD_ACCUM)
                    scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if step % 50 == 0 or step == QAT_STEPS:
                    print(f"[{EXP_ID}] {arm['name']} QAT {step}/{QAT_STEPS} loss={loss_d['loss'].item():.4f}", flush=True)

            qat_metrics, _ = _eval_val_metrics(model, val_pairs, device, amp_dtype=amp_dtype, ref_logits_cache=ref_logits)
            size_est = ctrl.estimate_webgpu_mb(model)
            ctrl.remove()

            qat_metrics["f1_retention_vs_fp32"] = round(
                qat_metrics["val_token_f1"] / max(1e-6, fp32_metrics["val_token_f1"]), 5
            )
            ptq_metrics["f1_retention_vs_fp32"] = round(
                ptq_metrics["val_token_f1"] / max(1e-6, fp32_metrics["val_token_f1"]), 5
            )
            res = {
                "arm": arm,
                "qat_wall_sec": round(time.perf_counter() - t_q, 2),
                "size_estimate": size_est,
                "ptq_0step": ptq_metrics,
                "qat_200step": qat_metrics,
            }
            _atomic_write_json(arm_res_p, res)
            arm_results[arm["name"]] = res
            del model, opt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        payload = {
            "exp_id": EXP_ID,
            "completed_at": _utc_now(),
            "wall_sec": round(time.perf_counter() - t0, 2),
            "fp32_reference": fp32_metrics,
            "arms": arm_results,
        }
        out_eval = VOL_ROOT / "eval_out" / f"{EXP_ID}.json"
        out_run = VOL_ROOT / "eval_out" / f"{EXP_ID}_run.json"
        _atomic_write_json(out_eval, payload)
        _atomic_write_json(out_run, {"exp_id": EXP_ID, "status": "COMPLETED", "completed_at": payload["completed_at"], "wall_sec": payload["wall_sec"]})

        api = HfApi(token=hf_token)
        api.upload_file(path_or_fileobj=str(out_eval), path_in_repo=HF_EVAL_PATH, repo_id=HF_REPO, repo_type="dataset")
        api.upload_file(path_or_fileobj=str(out_run), path_in_repo=HF_RUN_PATH, repo_id=HF_REPO, repo_type="dataset")
        print(f"[{EXP_ID}] Uploaded {HF_EVAL_PATH} and {HF_RUN_PATH} successfully!", flush=True)
    finally:
        if TOKEN_PATH.exists():
            try:
                TOKEN_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()

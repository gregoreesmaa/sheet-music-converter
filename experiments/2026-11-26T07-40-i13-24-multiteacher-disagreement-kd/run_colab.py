#!/usr/bin/env python3
"""Iter 13 Moonshot i13-24 on Google Colab L4 / Kaggle T4:
Disagreement-Gated Multi-Teacher Ensemble KD + Unlabeled RISM Consensus Span
Distillation (MT-DKD) into UniOMR v2 small_45m.

Teachers (HF verified):
  - T1: checkpoints/iter10_soup_baseline/uni_omr_soup.pt
  - T2: checkpoints/iter10_5_best/uni_omr_ema_best.pt

Compares 3 matched 400-step arms on iter10_5_proxy_2k (1280px, BF16, seed=42):
  - arm0_single_teacher_kd: T1-only G-JSD KD (alpha=0.50, T=2.0)
  - arm1_multi_teacher_ent_disagree_kd: T1+T2 entropy-inverse ensemble + JSD disagreement weighting
  - arm2_mt_kd_plus_rism_consensus: arm1 + every-4th-step unlabeled RISM consensus span KD

Uploads to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T07-40-i13-24-multiteacher-disagreement-kd.json
  - runs/2026-11-26T07-40-i13-24-multiteacher-disagreement-kd.json
"""

from __future__ import annotations

from datetime import datetime, timezone
import difflib
import gc
import importlib
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
import xml.etree.ElementTree as ET

EXP_ID = "2026-11-26T07-40-i13-24-multiteacher-disagreement-kd"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_T1_CKPT = "checkpoints/iter10_soup_baseline/uni_omr_soup.pt"
HF_T2_CKPT = "checkpoints/iter10_5_best/uni_omr_ema_best.pt"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_RISM_TAR = "datasets/rism_hw_5707.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_CANDIDATES = (
    Path("/content/ov_bundle.tar.gz"),
    Path("/content/bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
)
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

SEED = 42
ARM_STEPS = 400
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 3.0e-4
WEIGHT_DECAY = 0.04
WARMUP_STEPS = 40
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
KD_TEMP = 2.0
KD_ALPHA = 0.50
TAU_ENT = 0.50
LAMBDA_DIS = 2.0
RISM_EVERY = 4
RISM_CONF_THRESH = 0.70
RISM_KD_WEIGHT = 0.25
EVAL_MAX_LEN = 1536

# (arm_id, use_multi_teacher, use_rism_consensus)
ARMS: List[Tuple[str, bool, bool]] = [
    ("arm0_single_teacher_kd", False, False),
    ("arm1_multi_teacher_ent_disagree_kd", True, False),
    ("arm2_mt_kd_plus_rism_consensus", True, True),
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
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    mvd_dir = WORK_ROOT / "sheet-music-converter" / "tmp-train" / "mvd"
    if not mvd_dir.exists():
        alt = Path("/workspace/sheet-music-converter/tmp-train/mvd")
        if alt.exists():
            mvd_dir = alt
        else:
            bundle = next((b for b in BUNDLE_CANDIDATES if b.exists()), None)
            if bundle is None:
                raise FileNotFoundError(f"Missing bundle in {BUNDLE_CANDIDATES}")
            print(f"[{EXP_ID}] Extracting {bundle} -> {WORK_ROOT}...", flush=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORK_ROOT)

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
        importlib.invalidate_caches()
    return mvd_dir


def _extract_tar_once(tar_path: Path, dest_dir: Path) -> None:
    sentinel = dest_dir / ".extracted"
    if sentinel.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=dest_dir)
    sentinel.write_text(_utc_now() + "\n", encoding="utf-8")


def _ensure_assets(hf_token: str) -> Tuple[Path, Path, Path, Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    t1_path = CACHE_ROOT / HF_T1_CKPT
    if not t1_path.exists():
        hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", filename=HF_T1_CKPT,
            local_dir=str(CACHE_ROOT), token=hf_token,
        )

    t2_path = CACHE_ROOT / HF_T2_CKPT
    if not t2_path.exists():
        hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", filename=HF_T2_CKPT,
            local_dir=str(CACHE_ROOT), token=hf_token,
        )

    proxy_dir = CACHE_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    val_abs = proxy_dir / "proxy_val_abs.jsonl"
    if not (train_abs.exists() and val_abs.exists()):
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO, repo_type="dataset", filename=HF_PROXY_TAR,
                local_dir=str(CACHE_ROOT / "tars"), token=hf_token,
            )
        )
        _extract_tar_once(tar_p, proxy_dir)
        tar_p.unlink(missing_ok=True)
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

    rism_dir = CACHE_ROOT / "datasets" / "rism_hw"
    if not (rism_dir / ".extracted").exists():
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO, repo_type="dataset", filename=HF_RISM_TAR,
                local_dir=str(CACHE_ROOT / "tars"), token=hf_token,
            )
        )
        _extract_tar_once(tar_p, rism_dir)
        tar_p.unlink(missing_ok=True)

    s31_dir = CACHE_ROOT / "datasets" / "suite31_hw_eval"
    if not (s31_dir / ".extracted").exists():
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO, repo_type="dataset", filename=HF_SUITE31_TAR,
                local_dir=str(CACHE_ROOT / "tars"), token=hf_token,
            )
        )
        _extract_tar_once(tar_p, s31_dir)
        tar_p.unlink(missing_ok=True)

    manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json under {s31_dir}")
    return t1_path, t2_path, train_abs, val_abs, rism_dir, manifests[0]


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


def _wsd_lr(step: int, total_steps: int, base_lr: float) -> float:
    if step <= WARMUP_STEPS:
        return base_lr * float(step) / float(max(1, WARMUP_STEPS))
    decay_steps = max(1, int(round(total_steps * WSD_DECAY_FRAC)))
    decay_start = total_steps - decay_steps
    if step <= decay_start:
        return base_lr
    prog = min(1.0, max(0.0, float(step - decay_start) / float(decay_steps)))
    return base_lr * (MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * 0.5 * (1.0 + math.cos(math.pi * prog)))


def _map_student_ids_to_teacher(tgt_in: Any, s_vocab: int, t_vocab: int) -> Any:
    import torch

    if s_vocab == t_vocab:
        return tgt_in
    s_base = s_vocab - 4
    t_base = t_vocab - 4
    t_pad, t_bos, t_eos, t_sys = t_base, t_base + 1, t_base + 2, t_base + 3
    s_pad, s_bos, s_eos, s_sys = s_base, s_base + 1, s_base + 2, s_base + 3
    out = tgt_in.clone()
    new_base = (tgt_in >= t_base) & (tgt_in < s_base)
    out = torch.where(new_base, torch.full_like(out, t_pad), out)
    out = torch.where(tgt_in == s_pad, torch.full_like(out, t_pad), out)
    out = torch.where(tgt_in == s_bos, torch.full_like(out, t_bos), out)
    out = torch.where(tgt_in == s_eos, torch.full_like(out, t_eos), out)
    out = torch.where(tgt_in == s_sys, torch.full_like(out, t_sys), out)
    return out


def _align_logits(s_logits: Any, t_logits: Any) -> Tuple[Any, Any]:
    import torch

    s_v = int(s_logits.shape[-1])
    t_v = int(t_logits.shape[-1])
    if s_v == t_v:
        return s_logits, t_logits
    n_base = min(s_v - 4, t_v - 4)
    return (
        torch.cat([s_logits[..., :n_base], s_logits[..., s_v - 4 :]], dim=-1),
        torch.cat([t_logits[..., :n_base], t_logits[..., t_v - 4 :]], dim=-1),
    )


def _gjsd_per_token(p_target: Any, log_p_student: Any) -> Any:
    import torch

    p_s = torch.exp(log_p_student)
    m = 0.5 * (p_target + p_s).clamp(min=1e-8)
    log_m = torch.log(m)
    log_p_t = torch.log(p_target.clamp(min=1e-8))
    kl_t = (p_target * (log_p_t - log_m)).sum(dim=-1)
    kl_s = (p_s * (log_p_student - log_m)).sum(dim=-1)
    return 0.5 * (kl_t + kl_s)


def _compute_multiteacher_kd(
    s_logits: Any,
    t1_logits: Any,
    t2_logits: Optional[Any],
    valid_mask: Any,
) -> Tuple[Any, float]:
    import torch
    import torch.nn.functional as F

    s_al, t1_al = _align_logits(s_logits.float(), t1_logits.float())
    log_ps = F.log_softmax(s_al / KD_TEMP, dim=-1)
    p1 = F.softmax(t1_al / KD_TEMP, dim=-1)

    if t2_logits is None:
        tok_kd = _gjsd_per_token(p1, log_ps) * (KD_TEMP * KD_TEMP)
        denom = valid_mask.float().sum().clamp(min=1.0)
        return (tok_kd * valid_mask.float()).sum() / denom, 0.0

    _, t2_al = _align_logits(s_al, t2_logits.float())
    p2 = F.softmax(t2_al / KD_TEMP, dim=-1)

    h1 = -(p1 * torch.log(p1.clamp(min=1e-8))).sum(dim=-1)
    h2 = -(p2 * torch.log(p2.clamp(min=1e-8))).sum(dim=-1)
    ent_stack = torch.stack([-h1 / TAU_ENT, -h2 / TAU_ENT], dim=-1)
    w = F.softmax(ent_stack, dim=-1)
    p_ens = w[..., 0:1] * p1 + w[..., 1:2] * p2

    # JSD disagreement between T1 and T2 in [0, 1]
    m12 = 0.5 * (p1 + p2).clamp(min=1e-8)
    log_m12 = torch.log(m12)
    jsd12 = 0.5 * (
        (p1 * (torch.log(p1.clamp(min=1e-8)) - log_m12)).sum(dim=-1)
        + (p2 * (torch.log(p2.clamp(min=1e-8)) - log_m12)).sum(dim=-1)
    )
    dis_norm = (jsd12 / math.log(2.0)).clamp(0.0, 1.0)
    gamma = 1.0 + LAMBDA_DIS * dis_norm

    vm = valid_mask.float()
    denom = vm.sum().clamp(min=1.0)
    mean_gamma = ((gamma * vm).sum() / denom).clamp(min=1e-6)
    gamma_norm = gamma / mean_gamma

    tok_kd = _gjsd_per_token(p_ens, log_ps) * (KD_TEMP * KD_TEMP)
    loss = (tok_kd * gamma_norm * vm).sum() / denom
    mean_dis = float(((dis_norm * vm).sum() / denom).detach().item())
    return loss, mean_dis


def _seq_f1(pred: Sequence[str], ref: Sequence[str]) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    sm = difflib.SequenceMatcher(None, ref, pred, autojunk=False)
    matches = sum(blk.size for blk in sm.get_matching_blocks())
    return float(2.0 * matches) / float(len(pred) + len(ref))


def _eval_proxy_val(
    model: Any,
    val_pairs: List[Tuple[Path, List[Path], str]],
    device: Any,
    *,
    pad_id: int,
    amp_dtype: Any,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from uni_omr import UniFullPageDataset, _images_to_device_float, decode_uni

    model.eval()
    val_ds = UniFullPageDataset(
        val_pairs,
        fullpage_max_dim=FULLPAGE_MAX_DIM,
        max_seq_len=MAX_SEQ_LEN,
        augment=False,
        return_uint8=True,
    )
    ce_vals: List[float] = []
    f1_vals: List[float] = []
    with torch.no_grad():
        for i in range(len(val_ds)):
            img_u8, tgt_ids = val_ds[i][0], val_ds[i][1]
            img_t = _images_to_device_float(img_u8.unsqueeze(0), device)
            tgt_t = tgt_ids.unsqueeze(0).to(device)
            inp_t, lbl_t = tgt_t[:, :-1], tgt_t[:, 1:]
            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(img_t, inp_t, return_aux=False, return_mtp=False)
            else:
                out = model(img_t, inp_t, return_aux=False, return_mtp=False)
            logits = out["logits"].float()
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), lbl_t.reshape(-1), ignore_index=pad_id
            )
            ce_vals.append(float(ce.item()))

            if i < 24:
                gen = model.generate(img_t, max_len=512)
                pred_toks = decode_uni(gen[0].detach().cpu().tolist(), skip_special=True)
                ref_toks = decode_uni(tgt_ids.tolist(), skip_special=True)
                f1_vals.append(_seq_f1(pred_toks, ref_toks))

    return {
        "n_val": len(ce_vals),
        "val_ce_mean": round(sum(ce_vals) / max(1, len(ce_vals)), 6),
        "val_token_f1_mean": round(sum(f1_vals) / max(1, len(f1_vals)), 6),
    }


def _extract_musicxml_notes(xml_path: Path) -> List[Tuple[str, int, str, str, str]]:
    notes: List[Tuple[str, int, str, str, str]] = []
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return notes

    def _strip(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    for elem in root.iter():
        if _strip(elem.tag) != "note":
            continue
        is_rest = is_grace = False
        step = octave = alter = dur = ""
        n_dots = 0
        for child in elem:
            ctag = _strip(child.tag)
            if ctag == "rest":
                is_rest = True
            elif ctag == "grace":
                is_grace = True
            elif ctag == "pitch":
                for pc in child:
                    ptag = _strip(pc.tag)
                    text = (pc.text or "").strip()
                    if ptag == "step":
                        step = text.upper()
                    elif ptag == "octave":
                        octave = text
                    elif ptag == "alter":
                        try:
                            av = int(round(float(text)))
                            if av != 0:
                                alter = f"a{av}"
                        except ValueError:
                            pass
            elif ctag == "type":
                dur = (child.text or "").strip().lower()
            elif ctag == "dot":
                n_dots += 1
        if not is_rest and not is_grace and step and octave and dur:
            notes.append((dur, n_dots, step, octave, alter))
    return notes


def _select_sweep_pieces(pieces: Sequence[Dict[str, Any]], n_printed: int = 12) -> List[Dict[str, Any]]:
    hw = [p for p in pieces if p["handwritten"]]
    pr = [p for p in pieces if not p["handwritten"]]
    return hw + sorted(pr, key=lambda x: x["slug"])[:n_printed]


def _eval_suite31_subset(
    model: Any,
    vocab_size: int,
    pieces: Sequence[Dict[str, Any]],
    device: Any,
    *,
    amp_dtype: Any,
) -> Dict[str, Any]:
    from PIL import Image
    import torch
    import eval_uni_suite31 as eval_s31
    import rlvr_grpo
    import uni_omr

    _, _, eos_id, _, _, _ = eval_s31._vocab_spec(int(vocab_size))
    model.eval()
    buckets: Dict[str, List[float]] = {
        "printed_clean": [], "printed_photo": [], "hw_clean": [], "hw_photo": [],
    }
    for piece in pieces:
        ref_notes = _extract_musicxml_notes(piece["gt"])
        pfx = "hw_" if piece["handwritten"] else "printed_"
        for dom in ("clean", "photo"):
            with Image.open(piece[dom]) as raw_im:
                pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), FULLPAGE_MAX_DIM)
            img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
            with torch.no_grad():
                if amp_dtype is not None and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        seq = model.generate(img_t, max_len=EVAL_MAX_LEN)
                else:
                    seq = model.generate(img_t, max_len=EVAL_MAX_LEN)
            s_ids = seq[0].detach().cpu().tolist()
            has_eos = eos_id in s_ids[1:]
            pred_raw = eval_s31.decode_uni_compat(s_ids, vocab_size=int(vocab_size), skip_special=True)
            if not has_eos:
                pred_raw = eval_s31.trim_truncated_tokens(pred_raw)
            hyp_toks, _ = eval_s31.stitch_uni_tokens(pred_raw)
            hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
            buckets[pfx + dom].append(float(rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes)))

    return {
        k: round(sum(v) / max(1, len(v)), 6) for k, v in buckets.items()
    }


def main() -> None:
    started_utc = _utc_now()
    t_start = time.perf_counter()
    hf_token = _read_and_purge_hf_token()
    mvd_dir = _setup_environment()
    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))

    from PIL import Image
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from huggingface_hub import HfApi
    import eval_uni_suite31 as eval_s31
    import uni_omr
    import uni_omr_v2

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
        amp_dtype = (
            torch.bfloat16 if (device.type == "cuda" and cc[0] >= 8)
            else (torch.float16 if device.type == "cuda" else None)
        )
        host_pfx = "kaggle" if Path("/kaggle/working").exists() else "colab"
        platform_str = f"{host_pfx}-{gpu_name}"

        t1_path, t2_path, train_manifest, val_manifest, rism_dir, s31_manifest = _ensure_assets(hf_token)
        train_pairs = _load_manifest_pairs(train_manifest)
        val_pairs = _load_manifest_pairs(val_manifest)
        rism_images = sorted(
            p for p in rism_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        _, _all_s31_pieces = eval_s31.resolve_manifest_pieces(s31_manifest)
        s31_pieces = _select_sweep_pieces(_all_s31_pieces, n_printed=12)

        api = HfApi(token=hf_token)
        ledger_tmp = CACHE_ROOT / f"{EXP_ID}_run.json"
        _atomic_write_json(
            ledger_tmp,
            {
                "exp_id": EXP_ID,
                "status": "RUNNING",
                "started_utc": started_utc,
                "platform": platform_str,
                "arms": [a[0] for a in ARMS],
            },
        )
        try:
            api.upload_file(
                path_or_fileobj=str(ledger_tmp), path_in_repo=HF_RUN_PATH,
                repo_id=HF_REPO, repo_type="dataset",
                commit_message=f"{EXP_ID}: mark RUNNING on {platform_str}",
            )
        except Exception as exc:
            print(f"[{EXP_ID}] Warning: initial HF ledger upload failed: {exc}", flush=True)

        t1_model, t1_cfg, _ = eval_s31.load_uni_checkpoint(t1_path, device=device)
        t1_model.eval()
        for p in t1_model.parameters():
            p.requires_grad_(False)

        t2_model, t2_cfg, _ = eval_s31.load_uni_checkpoint(t2_path, device=device)
        t2_model.eval()
        for p in t2_model.parameters():
            p.requires_grad_(False)

        s_vocab = int(uni_omr.VOCAB_SIZE)
        pad_id = int(uni_omr.PAD_ID)

        arms_out: Dict[str, Any] = {}
        arm0_secs: Optional[float] = None

        for arm_id, use_mt, use_rism in ARMS:
            print(f"[{EXP_ID}] === {arm_id} (multi_teacher={use_mt}, rism_cons={use_rism}) ===", flush=True)
            random.seed(SEED)
            torch.manual_seed(SEED)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(SEED)

            v2_cfg = uni_omr_v2.UniOMRv2Config(vocab_size=s_vocab)
            student = uni_omr_v2.UniOMRv2Net(v2_cfg).to(device)
            optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

            train_ds = uni_omr.UniFullPageDataset(
                train_pairs,
                fullpage_max_dim=FULLPAGE_MAX_DIM,
                max_seq_len=MAX_SEQ_LEN,
                augment=True,
                return_uint8=True,
            )
            loader = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                collate_fn=uni_omr.collate_uni_batch,
                drop_last=True,
            )
            data_iter = iter(loader)
            rism_rng = random.Random(SEED + 99)

            t_arm = time.perf_counter()
            dis_hist: List[float] = []
            rism_tok_yield: List[int] = []
            loss_hist: List[float] = []

            for step in range(1, ARM_STEPS + 1):
                cur_lr = _wsd_lr(step, ARM_STEPS, LR)
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr

                student.train()
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0

                for _ in range(GRAD_ACCUM):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        batch = next(data_iter)
                    imgs_u8, tgt_ids = batch[0], batch[1]
                    imgs = uni_omr._images_to_device_float(imgs_u8, device)
                    tgt = tgt_ids.to(device)
                    inp_s, lbl_s = tgt[:, :-1], tgt[:, 1:]
                    valid_mask = lbl_s.ne(pad_id)

                    inp_t1 = _map_student_ids_to_teacher(inp_s, s_vocab, int(t1_cfg.vocab_size))
                    inp_t2 = _map_student_ids_to_teacher(inp_s, s_vocab, int(t2_cfg.vocab_size))

                    with torch.no_grad():
                        if amp_dtype is not None and device.type == "cuda":
                            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                                t1_out = t1_model(imgs, inp_t1, return_aux=False, return_mtp=False)
                                t2_out = (
                                    t2_model(imgs, inp_t2, return_aux=False, return_mtp=False)
                                    if use_mt else None
                                )
                        else:
                            t1_out = t1_model(imgs, inp_t1, return_aux=False, return_mtp=False)
                            t2_out = (
                                t2_model(imgs, inp_t2, return_aux=False, return_mtp=False)
                                if use_mt else None
                            )

                    if amp_dtype is not None and device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            s_out = student(imgs, inp_s, return_aux=False, return_mtp=False)
                            s_logits = s_out["logits"]
                            ce_loss = F.cross_entropy(
                                s_logits.reshape(-1, s_logits.size(-1)),
                                lbl_s.reshape(-1),
                                ignore_index=pad_id,
                                label_smoothing=0.02,
                            )
                            kd_loss, mean_dis = _compute_multiteacher_kd(
                                s_logits,
                                t1_out["logits"],
                                t2_out["logits"] if t2_out is not None else None,
                                valid_mask,
                            )
                            loss = ((1.0 - KD_ALPHA) * ce_loss + KD_ALPHA * kd_loss) / GRAD_ACCUM
                    else:
                        s_out = student(imgs, inp_s, return_aux=False, return_mtp=False)
                        s_logits = s_out["logits"]
                        ce_loss = F.cross_entropy(
                            s_logits.reshape(-1, s_logits.size(-1)),
                            lbl_s.reshape(-1),
                            ignore_index=pad_id,
                            label_smoothing=0.02,
                        )
                        kd_loss, mean_dis = _compute_multiteacher_kd(
                            s_logits,
                            t1_out["logits"],
                            t2_out["logits"] if t2_out is not None else None,
                            valid_mask,
                        )
                        loss = ((1.0 - KD_ALPHA) * ce_loss + KD_ALPHA * kd_loss) / GRAD_ACCUM

                    if use_mt:
                        dis_hist.append(mean_dis)
                    if torch.isfinite(loss):
                        if scaler.is_enabled():
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        step_loss += float(loss.detach().item())

                # Unlabeled RISM multi-teacher consensus span distillation
                if use_rism and rism_images and (step % RISM_EVERY == 0):
                    r_path = rism_images[rism_rng.randrange(len(rism_images))]
                    try:
                        with Image.open(r_path) as raw_im:
                            pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), FULLPAGE_MAX_DIM)
                        r_img = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
                        with torch.no_grad():
                            if amp_dtype is not None and device.type == "cuda":
                                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                                    draft_seq = t2_model.generate(r_img, max_len=256)
                            else:
                                draft_seq = t2_model.generate(r_img, max_len=256)
                        if draft_seq.shape[1] >= 24:
                            r_in = draft_seq[:, :-1]
                            with torch.no_grad():
                                if amp_dtype is not None and device.type == "cuda":
                                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                                        r_t1 = t1_model(r_img, r_in, return_aux=False, return_mtp=False)["logits"]
                                        r_t2 = t2_model(r_img, r_in, return_aux=False, return_mtp=False)["logits"]
                                else:
                                    r_t1 = t1_model(r_img, r_in, return_aux=False, return_mtp=False)["logits"]
                                    r_t2 = t2_model(r_img, r_in, return_aux=False, return_mtp=False)["logits"]
                            _, r_t1_al = _align_logits(r_t1.float(), r_t1.float())
                            _, r_t2_al = _align_logits(r_t1_al, r_t2.float())
                            p1_r = F.softmax(r_t1_al, dim=-1)
                            p2_r = F.softmax(r_t2_al, dim=-1)
                            p_ens_r = 0.5 * (p1_r + p2_r)
                            agree = p1_r.argmax(dim=-1).eq(p2_r.argmax(dim=-1))
                            conf_ok = p_ens_r.max(dim=-1).values >= RISM_CONF_THRESH
                            cons_mask = agree & conf_ok
                            n_cons = int(cons_mask.sum().item())
                            rism_tok_yield.append(n_cons)
                            if n_cons >= 16:
                                if amp_dtype is not None and device.type == "cuda":
                                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                                        r_s = student(r_img, r_in, return_aux=False, return_mtp=False)["logits"]
                                        r_kd, _ = _compute_multiteacher_kd(r_s, r_t1, r_t2, cons_mask)
                                        r_loss = RISM_KD_WEIGHT * r_kd
                                else:
                                    r_s = student(r_img, r_in, return_aux=False, return_mtp=False)["logits"]
                                    r_kd, _ = _compute_multiteacher_kd(r_s, r_t1, r_t2, cons_mask)
                                    r_loss = RISM_KD_WEIGHT * r_kd
                                if torch.isfinite(r_loss):
                                    if scaler.is_enabled():
                                        scaler.scale(r_loss).backward()
                                    else:
                                        r_loss.backward()
                    except Exception:
                        pass

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()

                loss_hist.append(step_loss)
                if step == 1 or step % 50 == 0 or step == ARM_STEPS:
                    print(
                        f"[{arm_id}] step {step:03d}/{ARM_STEPS} loss={step_loss:.4f} "
                        f"({time.perf_counter() - t_arm:.1f}s)",
                        flush=True,
                    )

            train_secs = time.perf_counter() - t_arm
            if arm0_secs is None:
                arm0_secs = train_secs

            val_res = _eval_proxy_val(student, val_pairs, device, pad_id=pad_id, amp_dtype=amp_dtype)
            s31_res = _eval_suite31_subset(student, s_vocab, s31_pieces, device, amp_dtype=amp_dtype)

            arms_out[arm_id] = {
                "use_multi_teacher": use_mt,
                "use_rism_consensus": use_rism,
                "train_secs": round(train_secs, 2),
                "wallclock_ratio_vs_arm0": round(train_secs / max(1e-6, arm0_secs), 3),
                "mean_teacher_jsd": round(sum(dis_hist) / max(1, len(dis_hist)), 5) if dis_hist else 0.0,
                "mean_rism_consensus_tokens": (
                    round(sum(rism_tok_yield) / max(1, len(rism_tok_yield)), 2) if rism_tok_yield else 0.0
                ),
                "mean_last50_loss": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5),
                "proxy_val": val_res,
                "suite31_hw_subset": s31_res,
            }

            del student, optimizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        payload = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_utc,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.perf_counter() - t_start, 2),
            "platform": platform_str,
            "teachers": {"T1": HF_T1_CKPT, "T2": HF_T2_CKPT},
            "arms": arms_out,
        }
        eval_local = CACHE_ROOT / "evals" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, payload)
        api.upload_file(
            path_or_fileobj=str(eval_local), path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO, repo_type="dataset",
            commit_message=f"{EXP_ID}: upload completed eval JSON",
        )
        _atomic_write_json(
            ledger_tmp,
            {
                "exp_id": EXP_ID,
                "status": "COMPLETED",
                "started_utc": started_utc,
                "completed_utc": payload["completed_utc"],
                "elapsed_secs": payload["elapsed_secs"],
                "platform": platform_str,
                "hf_eval_path": HF_EVAL_PATH,
            },
        )
        api.upload_file(
            path_or_fileobj=str(ledger_tmp), path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO, repo_type="dataset",
            commit_message=f"{EXP_ID}: mark COMPLETED",
        )
        print(f"[{EXP_ID}] COMPLETED and uploaded to {HF_EVAL_PATH}", flush=True)
    finally:
        if TOKEN_PATH.exists():
            try:
                TOKEN_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()

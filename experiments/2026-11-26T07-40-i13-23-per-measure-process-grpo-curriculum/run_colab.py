#!/usr/bin/env python3
"""Iter 13 Moonshot i13-23 on Google Colab L4 / Kaggle T4:
Token-Aligned Per-Measure Process-Reward GRPO (PR-GRPO) + Online Hard-Example
Onset-Drift Curriculum on unlabeled RISM archival manuscripts.

Compares 3 matched 120-update arms (G=6, max_len=512, kl_beta=0.10):
  - arm0_page_grpo_uniform: standard page-level scalar GRPO + uniform RISM sampling
  - arm1_measure_pr_grpo_uniform: per-measure process-reward GRPO + first-offending-measure blame
  - arm2_measure_pr_grpo_curriculum: per-measure PR-GRPO + variance-weighted hard-example curriculum

Uploads to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T07-40-i13-23-per-measure-process-grpo-curriculum.json
  - runs/2026-11-26T07-40-i13-23-per-measure-process-grpo-curriculum.json
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
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

EXP_ID = "2026-11-26T07-40-i13-23-per-measure-process-grpo-curriculum"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_PATH = "checkpoints/iter10_soup_baseline/uni_omr_soup.pt"
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
UPDATES = 120
GROUP_SIZE = 6
ROLLOUT_MAX_LEN = 512
KL_BETA = 0.10
CLIP_EPS = 0.20
LR = 1.0e-5
TEMPERATURE = 0.85
TOP_P = 0.95
TRAIN_MAX_DIM = 1024
EVAL_MAX_DIM = 1280
EVAL_MAX_LEN = 3840

# (arm_id, use_process_reward, use_curriculum)
ARMS: List[Tuple[str, bool, bool]] = [
    ("arm0_page_grpo_uniform", False, False),
    ("arm1_measure_pr_grpo_uniform", True, False),
    ("arm2_measure_pr_grpo_curriculum", True, True),
]

_DUR_BEATS: Dict[str, float] = {
    "dur-1": 4.0,
    "dur-2": 2.0,
    "dur-4": 1.0,
    "dur-8": 0.5,
    "dur-16": 0.25,
    "dur-32": 0.125,
    "dur-64": 0.0625,
    "dur-whole": 4.0,
    "dur-half": 2.0,
    "dur-quarter": 1.0,
    "dur-eighth": 0.5,
    "dur-16th": 0.25,
    "dur-32nd": 0.125,
}


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


def _ensure_assets(hf_token: str) -> Tuple[Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    ckpt_path = CACHE_ROOT / HF_CKPT_PATH
    if not ckpt_path.exists():
        hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", filename=HF_CKPT_PATH,
            local_dir=str(CACHE_ROOT), token=hf_token,
        )

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
    return ckpt_path, rism_dir, manifests[0]


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
    by_stratum: Dict[str, List[Dict[str, Any]]] = {}
    for p in sorted(pr, key=lambda x: x["slug"]):
        by_stratum.setdefault(str(p.get("stratum", "other")), []).append(p)
    strata = sorted(by_stratum.keys())
    chosen_pr: List[Dict[str, Any]] = []
    idx = 0
    while len(chosen_pr) < min(int(n_printed), len(pr)):
        progressed = False
        for s in strata:
            bucket = by_stratum[s]
            if idx < len(bucket) and len(chosen_pr) < int(n_printed):
                chosen_pr.append(bucket[idx])
                progressed = True
        if not progressed:
            break
        idx += 1
    return hw + chosen_pr


def _compute_token_process_rewards(
    seq_ids: Sequence[int],
    id_to_tok: Sequence[str],
    page_reward: float,
) -> List[float]:
    """Compute dense token-aligned per-measure process rewards with first-offending-measure blame."""
    toks = [id_to_tok[t] if 0 <= t < len(id_to_tok) else "" for t in seq_ids]
    n = len(toks)
    if n == 0:
        return []

    bar_len = 4.0
    spans: List[Tuple[int, int, float]] = []
    span_start = 0
    voice_cur = "v1"
    voice_sums: Dict[str, float] = {}
    last_dur = 0.0

    for i, tok in enumerate(toks):
        if tok.startswith("time-"):
            parts = tok.split("-")
            if len(parts) >= 3:
                try:
                    num = float(parts[1])
                    den = float(parts[2])
                    if num > 0 and den > 0:
                        bar_len = num * (4.0 / den)
                except ValueError:
                    pass
        elif tok.startswith("voice-"):
            voice_cur = tok
        elif tok in _DUR_BEATS:
            last_dur = _DUR_BEATS[tok]
            voice_sums[voice_cur] = voice_sums.get(voice_cur, 0.0) + last_dur
        elif tok == "dot" and last_dur > 0.0:
            voice_sums[voice_cur] = voice_sums.get(voice_cur, 0.0) + 0.5 * last_dur
            last_dur *= 0.5
        elif tok in ("<bar>", "<sys>", "<eos>") or tok.startswith("bar-"):
            if voice_sums:
                errs = [abs(v - bar_len) / max(0.25, bar_len) for v in voice_sums.values()]
                mean_err = sum(errs) / len(errs)
                r_bar = 1.0 if max(errs) <= 1e-3 else max(0.0, 1.0 - min(1.0, mean_err))
            else:
                r_bar = float(page_reward)
            spans.append((span_start, i + 1, r_bar))
            span_start = i + 1
            voice_sums = {}
            voice_cur = "v1"
            last_dur = 0.0

    if span_start < n:
        spans.append((span_start, n, float(page_reward)))

    # First-offending-measure blame + downstream decay
    first_bad_idx: Optional[int] = None
    for idx, (_, _, r_bar) in enumerate(spans):
        if r_bar < 0.5:
            first_bad_idx = idx
            break

    tok_rewards = [float(page_reward)] * n
    for idx, (s, e, r_bar) in enumerate(spans):
        shaped = r_bar
        if first_bad_idx is not None:
            if idx == first_bad_idx:
                shaped = max(-0.5, r_bar - 0.35)
            elif idx > first_bad_idx:
                shaped = r_bar * (0.85 ** (idx - first_bad_idx))
        blended = 0.65 * shaped + 0.35 * float(page_reward)
        for pos in range(s, e):
            tok_rewards[pos] = blended
    return tok_rewards


def _pr_grpo_step(
    model: Any,
    ref_model: Any,
    img_t: Any,
    id_to_tok: Sequence[str],
    *,
    pad_id: int,
    eos_id: int,
    use_process_reward: bool,
    amp_dtype: Any,
) -> Tuple[Any, Dict[str, float]]:
    import torch
    import torch.nn.functional as F
    import eval_uni_suite31 as eval_s31
    import rlvr_grpo

    model.eval()
    with torch.no_grad():
        rollouts = rlvr_grpo.sample_rollouts(
            model,
            img_t,
            group_size=GROUP_SIZE,
            max_len=ROLLOUT_MAX_LEN,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            use_grammar=True,
            strict_grammar=False,
        )

    seqs = rollouts.sequences  # [G, T]
    old_lp = rollouts.old_log_probs.detach()  # [G, T-1]
    mask = rollouts.action_mask.float().detach()  # [G, T-1]
    g_sz, seq_t = seqs.shape
    act_len = seq_t - 1

    page_rewards: List[float] = []
    r_rhythms: List[float] = []
    r_noloops: List[float] = []
    tok_reward_mat = torch.zeros((g_sz, act_len), dtype=torch.float32, device=seqs.device)

    for i in range(g_sz):
        s_ids = seqs[i].detach().cpu().tolist()
        has_eos = eos_id in s_ids[1:]
        pred_raw = eval_s31.decode_uni_compat(s_ids, vocab_size=len(id_to_tok), skip_special=True)
        struct = rlvr_grpo.compute_unlabeled_structural_reward(
            pred_raw, vocab=id_to_tok, terminated_eos=has_eos
        )
        r_page = float(struct["reward"])
        page_rewards.append(r_page)
        r_rhythms.append(float(struct["r_rhythm"]))
        r_noloops.append(float(struct["r_noloop"]))

        if use_process_reward:
            t_rews = _compute_token_process_rewards(s_ids[1:], id_to_tok, r_page)
            t_rews = t_rews[:act_len]
            if len(t_rews) < act_len:
                t_rews.extend([r_page] * (act_len - len(t_rews)))
            tok_reward_mat[i] = torch.tensor(t_rews, dtype=torch.float32, device=seqs.device)
        else:
            tok_reward_mat[i].fill_(r_page)

    valid_vals = tok_reward_mat[mask > 0.5]
    if valid_vals.numel() < 2 or float(valid_vals.std().item()) < 1e-6:
        zero_loss = torch.tensor(0.0, device=seqs.device, requires_grad=True)
        return zero_loss, {
            "loss": 0.0,
            "kl_loss": 0.0,
            "mean_reward": sum(page_rewards) / max(1, len(page_rewards)),
            "std_reward": float(torch.tensor(page_rewards).std().item()) if len(page_rewards) > 1 else 0.0,
            "mean_r_rhythm": sum(r_rhythms) / max(1, len(r_rhythms)),
            "mean_r_noloop": sum(r_noloops) / max(1, len(r_noloops)),
            "clip_frac": 0.0,
            "nonzero_adv": 0.0,
        }

    adv = (tok_reward_mat - valid_vals.mean()) / (valid_vals.std() + 1e-6)
    adv = adv.clamp(-4.0, 4.0)

    inp_ids = seqs[:, :-1]
    tgt_ids = seqs[:, 1:]

    model.train()
    if amp_dtype is not None and seqs.device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            vis_mem, _, _ = model.encode_image(img_t)
            mem_exp = vis_mem.expand(g_sz, -1, -1)
            dec_out = model.decode_tokens(inp_ids, mem_exp, return_mtp=False)
            logits = dec_out["logits"]
    else:
        vis_mem, _, _ = model.encode_image(img_t)
        mem_exp = vis_mem.expand(g_sz, -1, -1)
        dec_out = model.decode_tokens(inp_ids, mem_exp, return_mtp=False)
        logits = dec_out["logits"]

    new_lp = F.log_softmax(logits.float(), dim=-1).gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        if amp_dtype is not None and seqs.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                ref_mem, _, _ = ref_model.encode_image(img_t)
                ref_exp = ref_mem.expand(g_sz, -1, -1)
                ref_out = ref_model.decode_tokens(inp_ids, ref_exp, return_mtp=False)
                ref_logits = ref_out["logits"]
        else:
            ref_mem, _, _ = ref_model.encode_image(img_t)
            ref_exp = ref_mem.expand(g_sz, -1, -1)
            ref_out = ref_model.decode_tokens(inp_ids, ref_exp, return_mtp=False)
            ref_logits = ref_out["logits"]
        ref_lp = F.log_softmax(ref_logits.float(), dim=-1).gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)

    log_ratio = (new_lp - old_lp.float()).clamp(-10.0, 10.0)
    ratio = torch.exp(log_ratio)
    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
    pg_loss = -torch.minimum(surr1, surr2)

    # Unbiased k3 KL penalty
    kl_diff = (ref_lp - new_lp).clamp(-10.0, 10.0)
    kl_tok = torch.exp(kl_diff) - kl_diff - 1.0

    denom = mask.sum().clamp(min=1.0)
    loss = ((pg_loss + KL_BETA * kl_tok) * mask).sum() / denom
    kl_mean = float(((kl_tok * mask).sum() / denom).detach().item())
    clip_frac = float((((ratio - 1.0).abs() > CLIP_EPS).float() * mask).sum().item() / denom.item())

    return loss, {
        "loss": float(loss.detach().item()),
        "kl_loss": kl_mean,
        "mean_reward": sum(page_rewards) / max(1, len(page_rewards)),
        "std_reward": float(torch.tensor(page_rewards).std().item()) if len(page_rewards) > 1 else 0.0,
        "mean_r_rhythm": sum(r_rhythms) / max(1, len(r_rhythms)),
        "mean_r_noloop": sum(r_noloops) / max(1, len(r_noloops)),
        "clip_frac": clip_frac,
        "nonzero_adv": 1.0,
    }


def _evaluate_pieces_python(
    model: Any,
    cfg: Any,
    pieces: Sequence[Dict[str, Any]],
    device: Any,
    *,
    tag: str,
    amp_dtype: Any,
) -> Dict[str, Any]:
    from PIL import Image
    import torch
    import eval_uni_suite31 as eval_s31
    import rlvr_grpo
    import uni_omr

    _, _, eos_id, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))
    model.eval()
    rows: List[Dict[str, Any]] = []
    t_all = time.perf_counter()

    for idx, piece in enumerate(pieces, start=1):
        ref_notes = _extract_musicxml_notes(piece["gt"])
        prow: Dict[str, Any] = {
            "piece": piece["slug"],
            "handwritten": bool(piece["handwritten"]),
            "stratum": piece.get("stratum", ""),
        }
        for domain in ("clean", "photo"):
            with Image.open(piece[domain]) as raw_im:
                pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), EVAL_MAX_DIM)
            img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
            with torch.no_grad():
                if amp_dtype is not None and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        seq, _ = eval_s31.generate_mtp_spec(
                            model, img_t, max_len=EVAL_MAX_LEN, use_grammar=True,
                            strict_grammar=False, return_stats=True,
                        )
                else:
                    seq, _ = eval_s31.generate_mtp_spec(
                        model, img_t, max_len=EVAL_MAX_LEN, use_grammar=True,
                        strict_grammar=False, return_stats=True,
                    )
            seq_ids = seq[0].detach().cpu().tolist()
            has_eos = eos_id in seq_ids[1:]
            pred_raw = eval_s31.decode_uni_compat(
                seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
            )
            if not has_eos:
                pred_raw = eval_s31.trim_truncated_tokens(pred_raw)
            hyp_toks, _ = eval_s31.stitch_uni_tokens(pred_raw)
            struct = rlvr_grpo.compute_unlabeled_structural_reward(
                pred_raw, vocab=id_to_tok, terminated_eos=has_eos
            )
            hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
            note_f1 = rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes)

            pfx = domain + "_"
            prow[pfx + "note_f1_py"] = round(float(note_f1), 6)
            prow[pfx + "r_rhythm"] = round(float(struct["r_rhythm"]), 4)
            prow[pfx + "r_noloop"] = round(float(struct["r_noloop"]), 4)
            prow[pfx + "has_eos"] = bool(has_eos)

        rows.append(prow)
        if idx % 10 == 0 or idx == len(pieces):
            print(f"[eval:{tag}] {idx:02d}/{len(pieces):02d} ({time.perf_counter() - t_all:.1f}s)", flush=True)

    def _agg(sub: Sequence[Dict[str, Any]], dom: str) -> Dict[str, Any]:
        p = dom + "_"
        n = max(1, len(sub))
        return {
            "n": len(sub),
            "note_f1_py_mean": round(sum(r[p + "note_f1_py"] for r in sub) / n, 6),
            "r_rhythm_mean": round(sum(r[p + "r_rhythm"] for r in sub) / n, 4),
            "r_noloop_mean": round(sum(r[p + "r_noloop"] for r in sub) / n, 4),
            "eos_rate": round(sum(1 for r in sub if r[p + "has_eos"]) / n, 4),
        }

    pr_rows = [r for r in rows if not r["handwritten"]]
    hw_rows = [r for r in rows if r["handwritten"]]
    summary = {
        "printed_clean": _agg(pr_rows, "clean"),
        "printed_photo": _agg(pr_rows, "photo"),
        "hw_clean": _agg(hw_rows, "clean"),
        "hw_photo": _agg(hw_rows, "photo"),
    }
    return {"summary": summary, "pieces": rows, "elapsed_secs": round(time.perf_counter() - t_all, 2)}


def main() -> None:
    started_utc = _utc_now()
    t_start = time.perf_counter()
    hf_token = _read_and_purge_hf_token()
    mvd_dir = _setup_environment()
    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))

    from PIL import Image
    import torch
    from huggingface_hub import HfApi
    import eval_uni_suite31 as eval_s31
    import uni_omr

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

        ckpt_path, rism_dir, s31_manifest = _ensure_assets(hf_token)
        rism_images = sorted(
            p for p in rism_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not rism_images:
            raise RuntimeError(f"No RISM images found under {rism_dir}")

        _, all_pieces = eval_s31.resolve_manifest_pieces(s31_manifest)
        sweep_pieces = _select_sweep_pieces(all_pieces, n_printed=12)

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

        model, cfg, _ = eval_s31.load_uni_checkpoint(ckpt_path, device=device)
        ref_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ref_model = copy.deepcopy(model).to(device).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        pad_id, _, eos_id, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))

        baseline_eval = _evaluate_pieces_python(
            model, cfg, sweep_pieces, device, tag="baseline", amp_dtype=amp_dtype
        )

        arms_out: Dict[str, Any] = {}
        for arm_id, use_pr, use_curr in ARMS:
            print(
                f"[{EXP_ID}] === {arm_id} (process_reward={use_pr}, curriculum={use_curr}) ===",
                flush=True,
            )
            rng = random.Random(SEED)
            torch.manual_seed(SEED)
            model.load_state_dict({k: v.to(device) for k, v in ref_state_cpu.items()})
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
            scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

            pool = list(rism_images)
            rng.shuffle(pool)
            pool = pool[: min(600, len(pool))]
            priorities: Dict[int, float] = {}
            seen_pages: set[int] = set()
            next_unseen = 0

            t_arm = time.perf_counter()
            history: List[Dict[str, float]] = []
            warmup = max(1, UPDATES // 10)

            for step in range(1, UPDATES + 1):
                if step <= warmup:
                    cur_lr = LR * (step / warmup)
                else:
                    prog = (step - warmup) / max(1, UPDATES - warmup)
                    cur_lr = LR * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * prog)))
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr

                if use_curr and step > 30 and priorities and rng.random() < 0.70:
                    keys = list(priorities.keys())
                    weights = [math.exp(priorities[k] / 0.15) for k in keys]
                    pidx = rng.choices(keys, weights=weights, k=1)[0]
                else:
                    pidx = next_unseen % len(pool)
                    next_unseen += 1
                seen_pages.add(pidx)

                try:
                    with Image.open(pool[pidx]) as raw_im:
                        pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), TRAIN_MAX_DIM)
                    img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
                except Exception:
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss, stats = _pr_grpo_step(
                    model, ref_model, img_t, id_to_tok,
                    pad_id=pad_id, eos_id=eos_id,
                    use_process_reward=use_pr, amp_dtype=amp_dtype,
                )

                if use_curr:
                    priorities[pidx] = (
                        stats["std_reward"] * (1.0 - stats["mean_r_rhythm"]) + 0.05
                    )

                if torch.isfinite(loss) and loss.requires_grad and stats["nonzero_adv"] > 0.5:
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if math.isfinite(float(gn.detach().item())):
                            scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if math.isfinite(float(gn.detach().item())):
                            optimizer.step()
                        else:
                            optimizer.zero_grad(set_to_none=True)

                history.append(stats)
                if step == 1 or step % 20 == 0 or step == UPDATES:
                    print(
                        f"[{arm_id}] step {step:03d}/{UPDATES} R={stats['mean_reward']:.3f} "
                        f"rhy={stats['mean_r_rhythm']:.3f} kl={stats['kl_loss']:.4f} "
                        f"({time.perf_counter() - t_arm:.1f}s)",
                        flush=True,
                    )

            arm_eval = _evaluate_pieces_python(
                model, cfg, sweep_pieces, device, tag=arm_id, amp_dtype=amp_dtype
            )
            nonzero_frac = sum(h["nonzero_adv"] for h in history) / max(1, len(history))
            arms_out[arm_id] = {
                "use_process_reward": use_pr,
                "use_curriculum": use_curr,
                "curriculum_unique_pages": len(seen_pages),
                "nonzero_adv_frac": round(nonzero_frac, 4),
                "final_kl": round(sum(h["kl_loss"] for h in history[-20:]) / max(1, len(history[-20:])), 5),
                "max_kl": round(max((h["kl_loss"] for h in history), default=0.0), 5),
                "final_mean_reward": round(sum(h["mean_reward"] for h in history[-20:]) / max(1, len(history[-20:])), 4),
                "train_secs": round(time.perf_counter() - t_arm, 2),
                "eval": arm_eval,
            }

        payload = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_utc,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.perf_counter() - t_start, 2),
            "platform": platform_str,
            "baseline_ckpt": HF_CKPT_PATH,
            "baseline_eval": baseline_eval,
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

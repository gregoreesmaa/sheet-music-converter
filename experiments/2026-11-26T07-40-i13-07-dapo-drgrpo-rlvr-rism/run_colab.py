#!/usr/bin/env python3
"""Iter 13 Proposal i13-07: DAPO + Dr. GRPO Unbiased RLVR on Unlabeled RISM.

Experiment ID: 2026-11-26T07-40-i13-07-dapo-drgrpo-rlvr-rism
Academic basis:
  - DAPO (Yu et al., ByteDance Seed / Tsinghua AIR, arXiv:2503.14476, 2025):
    Clip-Higher (eps_low=0.20, eps_high=0.28) + Dynamic Sampling (filter/resample
    zero-variance std(R) < 1e-3 groups) + token-level PG loss.
  - Dr. GRPO (Liu et al., Sea AI Lab / NUS, arXiv:2503.20783, 2025):
    Remove std(R) denominator from advantage (A_i = R_i - mean(R)) and use
    constant-length normalization (max_len) to eliminate difficulty & length bias.

Controlled 4-arm single-variable objective ablation on unlabeled RISM HW pages:
  1. vanilla_grpo        (std-norm adv, symmetric clip 0.20/0.20, no dyn-filter)
  2. dr_grpo             (unbiased R-mean adv, symmetric clip 0.20/0.20, const max_len norm)
  3. dapo_clip_dyn       (std-norm adv, clip 0.20/0.28, dynamic sampling std>=1e-3)
  4. dapo_drgrpo_hybrid  (unbiased R-mean adv, clip 0.20/0.28, dynamic sampling, const norm)

Uploads evals/<EXP_ID>.json and runs/<EXP_ID>.json to HF Private Dataset Hub.
Zero edits to train_uni_v2.py or uni_omr_v2.py.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
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
import xml.etree.ElementTree as ET

EXP_ID = "2026-11-26T07-40-i13-07-dapo-drgrpo-rlvr-rism"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter10_5_best/uni_omr_ema_best.pt",
    "checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
)
HF_RISM_TAR = "datasets/rism_hw_5707.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_CANDIDATES = (
    Path("/content/bundle.tar.gz"),
    Path("/content/ov_bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
)
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

UPDATES = 120
GROUP_SIZE = 8
KL_BETA = 0.06
ROLLOUT_MAX_LEN = 512
ROLLOUT_MAX_DIM = 1024
EVAL_MAX_DIM = 1280
EVAL_MAX_LEN = 3840
TEMPERATURE = 0.85
TOP_P = 0.95
LR_MAX = 1.0e-5
LR_MIN = 2.0e-6
WARMUP_UPDATES = 12
SEED = 1307
SAVE_EVERY_MINUTES = 5.0
DYN_STD_MIN = 1e-3
DYN_MAX_RESAMPLE = 2

ARMS: List[Dict[str, Any]] = [
    {
        "name": "vanilla_grpo",
        "div_by_std": True,
        "clip_low": 0.20,
        "clip_high": 0.20,
        "dynamic_sampling": False,
        "const_len_norm": False,
    },
    {
        "name": "dr_grpo",
        "div_by_std": False,
        "clip_low": 0.20,
        "clip_high": 0.20,
        "dynamic_sampling": False,
        "const_len_norm": True,
    },
    {
        "name": "dapo_clip_dyn",
        "div_by_std": True,
        "clip_low": 0.20,
        "clip_high": 0.28,
        "dynamic_sampling": True,
        "const_len_norm": False,
    },
    {
        "name": "dapo_drgrpo_hybrid",
        "div_by_std": False,
        "clip_low": 0.20,
        "clip_high": 0.28,
        "dynamic_sampling": True,
        "const_len_norm": True,
    },
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

    candidates = [
        WORK_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
    ]
    mvd_dir: Optional[Path] = next((c for c in candidates if (c / "rlvr_grpo.py").is_file()), None)
    if mvd_dir is None:
        for b in BUNDLE_CANDIDATES:
            if b.is_file():
                print(f"[{EXP_ID}] Extracting {b} -> {WORK_ROOT}...", flush=True)
                with tarfile.open(b, "r:*") as tf:
                    tf.extractall(path=WORK_ROOT)
                mvd_dir = next((c for c in candidates if (c / "rlvr_grpo.py").is_file()), None)
                if mvd_dir is not None:
                    break
    if mvd_dir is None:
        raise FileNotFoundError("Could not locate tmp-train/mvd/rlvr_grpo.py")

    needed = []
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
            needed.append(pkg)
    if needed:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)
    return mvd_dir


def _extract_tar_once(tar_path: Path, dest_dir: Path) -> None:
    sentinel = dest_dir / ".extracted"
    if sentinel.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=dest_dir)
    sentinel.write_text(_utc_now() + "\n", encoding="utf-8")


def _ensure_assets(hf_token: str) -> Tuple[Path, str, Path, Path]:
    from huggingface_hub import hf_hub_download

    ckpt_path: Optional[Path] = None
    ckpt_hf_used = ""
    for cand in HF_CKPT_CANDIDATES:
        local_p = CACHE_ROOT / cand
        if local_p.is_file():
            ckpt_path = local_p
            ckpt_hf_used = cand
            break
        try:
            hf_hub_download(
                repo_id=HF_REPO, repo_type="dataset", filename=cand,
                local_dir=str(CACHE_ROOT), token=hf_token,
            )
            ckpt_path = local_p
            ckpt_hf_used = cand
            break
        except Exception as exc:
            print(f"[{EXP_ID}] Checkpoint candidate {cand} unavailable ({exc}), trying fallback...", flush=True)
    if ckpt_path is None:
        raise RuntimeError(f"Failed to download any checkpoint from {HF_CKPT_CANDIDATES}")

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
    return ckpt_path, ckpt_hf_used, rism_dir, manifests[0]


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


def _dapo_drgrpo_loss(
    new_log_probs: Any,
    old_log_probs: Any,
    ref_log_probs: Any,
    advantages: Any,
    action_mask: Any,
    *,
    clip_low: float,
    clip_high: float,
    kl_beta: float,
    const_len_norm: bool,
    max_len: int,
) -> Dict[str, Any]:
    import torch

    new_lp = new_log_probs.float()
    old_lp = old_log_probs.detach().float()
    ref_lp = ref_log_probs.detach().float()
    adv = advantages.detach().float().unsqueeze(-1)
    mask = action_mask.to(dtype=torch.float32)

    log_ratio = (new_lp - old_lp).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_low), 1.0 + float(clip_high))

    surr1 = ratio * adv
    surr2 = clipped_ratio * adv
    pg_per_tok = -torch.minimum(surr1, surr2)

    ref_log_diff = (ref_lp - new_lp).clamp(-20.0, 20.0)
    kl_per_tok = torch.exp(ref_log_diff) - ref_log_diff - 1.0

    tok_denom = mask.sum().clamp(min=1.0)
    if const_len_norm:
        n_seqs = float(max(1, mask.shape[0] * mask.shape[1]))
        pg_denom = n_seqs * float(max(1, max_len))
        pg_loss = (pg_per_tok * mask).sum() / pg_denom
    else:
        pg_loss = (pg_per_tok * mask).sum() / tok_denom

    kl_loss = (kl_per_tok * mask).sum() / tok_denom
    total_loss = pg_loss + float(kl_beta) * kl_loss

    with torch.no_grad():
        clipped = ((ratio - clipped_ratio).abs() > 1e-7).to(torch.float32)
        clip_frac = (clipped * mask).sum() / tok_denom
        neg_lp_entropy_proxy = (-(new_lp) * mask).sum() / tok_denom

    return {
        "loss": total_loss,
        "pg_loss": pg_loss,
        "kl_loss": kl_loss,
        "clip_frac": clip_frac,
        "entropy_proxy": neg_lp_entropy_proxy,
    }


def _evaluate_pieces_python(
    model: Any,
    cfg: Any,
    pieces: Sequence[Dict[str, Any]],
    device: Any,
    *,
    tag: str,
    max_dim: int = EVAL_MAX_DIM,
    max_len: int = EVAL_MAX_LEN,
    amp_dtype: Any = None,
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

    for piece in pieces:
        slug = piece["slug"]
        is_hw = bool(piece["handwritten"])
        ref_notes = _extract_musicxml_notes(piece["gt"])
        prow: Dict[str, Any] = {"slug": slug, "handwritten": is_hw}
        for dom in ("clean", "photo"):
            img_p = piece[dom]
            with Image.open(img_p) as im:
                rgb = im.convert("RGB")
            img_t = uni_omr.prepare_image_tensor(rgb, max_dim=max_dim).unsqueeze(0).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                seq = model.generate(img_t, max_len=max_len, use_grammar=True)
            ids = seq[0].tolist()
            toks: List[str] = []
            for tid in ids[1:]:
                if tid == eos_id:
                    break
                toks.append(id_to_tok.get(int(tid), ""))
            hyp_notes = rlvr_grpo.extract_pitch_dur_notes(toks)
            note_f1 = rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes)
            prow[f"{dom}_note_f1_py"] = round(float(note_f1), 6)
            prow[f"{dom}_n_toks"] = len(toks)
        rows.append(prow)

    def _mean(sub: List[Dict[str, Any]], key: str) -> float:
        vals = [float(r[key]) for r in sub if key in r]
        return round(sum(vals) / max(1, len(vals)), 6)

    hw_rows = [r for r in rows if r["handwritten"]]
    pr_rows = [r for r in rows if not r["handwritten"]]
    return {
        "tag": tag,
        "wall_sec": round(time.perf_counter() - t_all, 2),
        "n_pieces": len(rows),
        "hw_clean_note_f1": _mean(hw_rows, "clean_note_f1_py"),
        "hw_photo_note_f1": _mean(hw_rows, "photo_note_f1_py"),
        "printed_clean_note_f1": _mean(pr_rows, "clean_note_f1_py"),
        "printed_photo_note_f1": _mean(pr_rows, "photo_note_f1_py"),
        "per_piece": rows,
    }


def _run_arm(
    arm: Dict[str, Any],
    base_state: Dict[str, Any],
    cfg: Any,
    rism_paths: Sequence[Path],
    eval_pieces: Sequence[Dict[str, Any]],
    device: Any,
    amp_dtype: Any,
    use_scaler: bool,
    arm_dir: Path,
) -> Dict[str, Any]:
    from PIL import Image
    import torch
    import rlvr_grpo
    import uni_omr

    arm_result_path = arm_dir / "arm_result.json"
    if arm_result_path.is_file():
        print(f"[{EXP_ID}] Arm {arm['name']} already complete; loading cached result.", flush=True)
        return json.loads(arm_result_path.read_text(encoding="utf-8"))

    arm_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = uni_omr.UniOMRNet(cfg).to(device)
    model.load_state_dict(base_state, strict=True)
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    rng = random.Random(SEED)
    t_arm = time.perf_counter()
    last_save = time.perf_counter()
    rewards_log: List[float] = []
    kl_log: List[float] = []
    clip_log: List[float] = []
    ent_log: List[float] = []
    len_log: List[float] = []
    zero_var_skipped = 0

    for upd in range(1, UPDATES + 1):
        if upd <= WARMUP_UPDATES:
            lr = LR_MAX * float(upd) / float(max(1, WARMUP_UPDATES))
        else:
            prog = float(upd - WARMUP_UPDATES) / float(max(1, UPDATES - WARMUP_UPDATES))
            lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        chosen_rollouts = None
        chosen_rewards = None
        chosen_img = None
        max_tries = (1 + DYN_MAX_RESAMPLE) if arm["dynamic_sampling"] else 1
        for attempt in range(max_tries):
            img_p = rng.choice(rism_paths)
            with Image.open(img_p) as im:
                rgb = im.convert("RGB")
            img_t = uni_omr.prepare_image_tensor(rgb, max_dim=ROLLOUT_MAX_DIM).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                rollouts = rlvr_grpo.sample_rollouts(
                    model,
                    img_t,
                    group_size=GROUP_SIZE,
                    max_len=ROLLOUT_MAX_LEN,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    use_grammar=True,
                )
            r_vals: List[float] = []
            for g in range(GROUP_SIZE):
                toks = rollouts.decoded_tokens[0][g]
                res = rlvr_grpo.compute_unlabeled_structural_reward(toks, vocab=uni_omr.UNI_TOKEN_TO_ID)
                r_vals.append(float(res["reward"]))
            r_t = torch.tensor([r_vals], dtype=torch.float32, device=device)
            r_std = float(r_t.std(unbiased=False).item())
            chosen_rollouts = rollouts
            chosen_rewards = r_t
            chosen_img = img_t
            if (not arm["dynamic_sampling"]) or (r_std >= DYN_STD_MIN):
                break
            zero_var_skipped += 1

        assert chosen_rollouts is not None and chosen_rewards is not None and chosen_img is not None
        mean_r = chosen_rewards.mean(dim=-1, keepdim=True)
        std_r = chosen_rewards.std(dim=-1, unbiased=False, keepdim=True)
        if arm["div_by_std"]:
            adv = (chosen_rewards - mean_r) / (std_r + 1e-6)
        else:
            adv = chosen_rewards - mean_r

        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            new_lp = rlvr_grpo.compute_sequence_log_probs(model, chosen_img, chosen_rollouts.seq_ids)
            with torch.no_grad():
                ref_lp = rlvr_grpo.compute_sequence_log_probs(ref_model, chosen_img, chosen_rollouts.seq_ids)
            loss_dict = _dapo_drgrpo_loss(
                new_lp,
                chosen_rollouts.old_log_probs,
                ref_lp,
                adv,
                chosen_rollouts.action_mask,
                clip_low=arm["clip_low"],
                clip_high=arm["clip_high"],
                kl_beta=KL_BETA,
                const_len_norm=arm["const_len_norm"],
                max_len=ROLLOUT_MAX_LEN,
            )

        scaler.scale(loss_dict["loss"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        rewards_log.append(float(mean_r.item()))
        kl_log.append(float(loss_dict["kl_loss"].item()))
        clip_log.append(float(loss_dict["clip_frac"].item()))
        ent_log.append(float(loss_dict["entropy_proxy"].item()))
        len_log.append(float(chosen_rollouts.action_mask.sum(dim=-1).mean().item()))

        if upd % 20 == 0 or upd == UPDATES:
            print(
                f"[{EXP_ID}] {arm['name']} upd {upd}/{UPDATES} "
                f"R={rewards_log[-1]:.3f} kl={kl_log[-1]:.4f} ent={ent_log[-1]:.3f} "
                f"len={len_log[-1]:.1f} skip0={zero_var_skipped}",
                flush=True,
            )

        if (time.perf_counter() - last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ckpt = arm_dir / "latest.pt.tmp"
            torch.save({"upd": upd, "model": model.state_dict()}, tmp_ckpt)
            os.replace(tmp_ckpt, arm_dir / "latest.pt")
            last_save = time.perf_counter()

    eval_res = _evaluate_pieces_python(
        model, cfg, eval_pieces, device, tag=arm["name"], amp_dtype=amp_dtype
    )
    out = {
        "arm": arm,
        "train_wall_sec": round(time.perf_counter() - t_arm, 2),
        "mean_reward": round(sum(rewards_log) / max(1, len(rewards_log)), 5),
        "mean_kl": round(sum(kl_log) / max(1, len(kl_log)), 5),
        "mean_clip_frac": round(sum(clip_log) / max(1, len(clip_log)), 5),
        "mean_entropy_proxy": round(sum(ent_log) / max(1, len(ent_log)), 5),
        "mean_rollout_len": round(sum(len_log) / max(1, len(len_log)), 2),
        "zero_var_skipped": int(zero_var_skipped),
        "eval": eval_res,
    }
    _atomic_write_json(arm_result_path, out)
    del model, ref_model, opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    t0 = time.perf_counter()
    hf_token = ""
    try:
        hf_token = _read_and_purge_hf_token()
        mvd_dir = _setup_environment()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        import torch
        from huggingface_hub import HfApi
        import eval_uni_suite31 as eval_s31
        import rlvr_grpo
        import uni_omr

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
        amp_dtype = torch.bfloat16 if (device.type == "cuda" and cc[0] >= 8) else (
            torch.float16 if device.type == "cuda" else None
        )
        use_scaler = (device.type == "cuda" and cc[0] < 8)
        print(f"[{EXP_ID}] device={device} cc={cc} amp_dtype={amp_dtype}", flush=True)

        ckpt_path, ckpt_hf_used, rism_dir, manifest_path = _ensure_assets(hf_token)
        rism_paths = rlvr_grpo.collect_unlabeled_image_paths(rism_dir)
        all_pieces = eval_s31.load_suite31_manifest(manifest_path)
        eval_pieces = _select_sweep_pieces(all_pieces, n_printed=12)

        raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_dict = raw_ckpt.get("cfg", raw_ckpt.get("config", {}))
        cfg = uni_omr.UniOMRConfig(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict
        base_state = raw_ckpt.get("ema_model", raw_ckpt.get("model", raw_ckpt))

        base_model = uni_omr.UniOMRNet(cfg).to(device)
        base_model.load_state_dict(base_state, strict=True)
        baseline_eval = _evaluate_pieces_python(
            base_model, cfg, eval_pieces, device, tag="baseline", amp_dtype=amp_dtype
        )
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        arm_results: Dict[str, Any] = {}
        for arm in ARMS:
            arm_dir = CACHE_ROOT / "ckpts" / EXP_ID / arm["name"]
            res = _run_arm(
                arm, base_state, cfg, rism_paths, eval_pieces,
                device, amp_dtype, use_scaler, arm_dir,
            )
            arm_results[arm["name"]] = res

        payload = {
            "exp_id": EXP_ID,
            "completed_at": _utc_now(),
            "wall_sec": round(time.perf_counter() - t0, 2),
            "base_checkpoint": ckpt_hf_used,
            "gpu_cc": list(cc),
            "amp_dtype": str(amp_dtype),
            "hyperparams": {
                "updates": UPDATES,
                "group_size": GROUP_SIZE,
                "kl_beta": KL_BETA,
                "rollout_max_len": ROLLOUT_MAX_LEN,
                "eval_max_dim": EVAL_MAX_DIM,
                "eval_max_len": EVAL_MAX_LEN,
                "lr_max": LR_MAX,
                "seed": SEED,
            },
            "baseline": baseline_eval,
            "arms": arm_results,
        }
        out_eval = CACHE_ROOT / "eval_out" / f"{EXP_ID}.json"
        out_run = CACHE_ROOT / "eval_out" / f"{EXP_ID}_run.json"
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

#!/usr/bin/env python3
"""Iter 13 Exp i13-06 on Google Colab L4:
MEMO Marginal Multi-View Entropy, SAR Sharpness-Aware (SAM) Perturbation, and
OV-39 Rhythm-Reliability Gated Test-Time Adaptation (TTT) at canonical
max_dim=1280, max_len=3840 on all 14 HW manuscripts (28 pages) + 10-piece
printed safety slice (20 pages) for top-2 arms.

4 matched full_encoder TTT arms (lr=3e-5, steps=2, l2_weight=0.1, cons_weight=1.0):
  1. arm_ctrl_ov38    : ov-38 winning baseline (single-view ent=1.0, kl=0.5)
  2. arm_memo_marginal: MEMO marginal entropy H(0.5*(p_c+p_a)) + 0.5*JSD(p_c,p_a) + 0.5*KL
  3. arm_sar_sam      : SAR 1-step sharpness-aware SAM perturbation (rho=0.01) + ent=0.5, kl=0.5
  4. arm_rhythm_gated : OV-39 reliability gate (mean_logprob>=-0.35 & beat_valid>=0.70 & r_noloop==1.0)
                        enabling ent=0.5, kl=0.5 on reliable pages, pure visual cons on unreliable pages

Expects:
  - /content/bundle.tar.gz (or pre-extracted workspace)
  - /tmp/.hf_token (Hugging Face token; chmod 600, deleted immediately and in finally)
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

EXP_ID = "2026-11-26T07-40-i13-06-memo-sar-rhythm-gated-ttt"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_PATH = "checkpoints/iter10_soup_baseline/uni_omr_soup.pt"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

ARMS: List[Dict[str, Any]] = [
    {"key": "arm_ctrl_ov38", "mode": "ctrl", "lr": 3e-5, "steps": 2, "ent_w": 1.0, "kl_w": 0.5, "sam_rho": 0.0},
    {"key": "arm_memo_marginal", "mode": "memo", "lr": 3e-5, "steps": 2, "ent_w": 0.5, "kl_w": 0.5, "sam_rho": 0.0},
    {"key": "arm_sar_sam", "mode": "sar", "lr": 3e-5, "steps": 2, "ent_w": 0.5, "kl_w": 0.5, "sam_rho": 0.01},
    {"key": "arm_rhythm_gated", "mode": "rhythm_gated", "lr": 3e-5, "steps": 2, "ent_w": 0.5, "kl_w": 0.5, "sam_rho": 0.0},
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _purge_hf_token() -> None:
    if TOKEN_PATH.exists():
        try:
            TOKEN_PATH.unlink()
        except OSError:
            pass


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and TOKEN_PATH.exists():
        try:
            os.chmod(TOKEN_PATH, 0o600)
        except OSError:
            pass
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    _purge_hf_token()
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var")
    return token


def _setup_environment() -> Path:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    mvd_dir = WORK_ROOT / "sheet-music-converter" / "tmp-train" / "mvd"
    if not mvd_dir.exists():
        alt_mvd = Path("/workspace/sheet-music-converter/tmp-train/mvd")
        if alt_mvd.exists():
            mvd_dir = alt_mvd
        else:
            bundle = BUNDLE_PATH if BUNDLE_PATH.exists() else Path("/content/repo_bundle.tar.gz")
            if not bundle.exists():
                raise FileNotFoundError(f"Missing bundle at {BUNDLE_PATH}")
            print(f"[{EXP_ID}] Extracting {bundle} -> {WORK_ROOT}...", flush=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORK_ROOT)

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


def _ensure_assets(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ckpt_path = CACHE_ROOT / HF_CKPT_PATH
    if not ckpt_path.exists():
        print(f"[{EXP_ID}] Downloading {HF_CKPT_PATH}...", flush=True)
        hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", filename=HF_CKPT_PATH,
            local_dir=str(CACHE_ROOT), token=hf_token,
        )

    s31_dir = CACHE_ROOT / "datasets" / "suite31_hw_eval"
    sentinel = s31_dir / ".extracted"
    if not sentinel.exists():
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR}...", flush=True)
        tar_p = Path(hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", filename=HF_SUITE31_TAR,
            local_dir=str(CACHE_ROOT / "tars"), token=hf_token,
        ))
        s31_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=s31_dir)
        sentinel.write_text(_utc_now() + "\n", encoding="utf-8")
        tar_p.unlink(missing_ok=True)

    manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json under {s31_dir}")
    return ckpt_path, manifests[0]


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


def _select_full_encoder_params(model: Any) -> List[Tuple[str, Any]]:
    selected: List[Tuple[str, Any]] = []
    seen: set[int] = set()
    for root_name in ("backbone", "router", "enc_layers", "enc_norm"):
        root_mod = getattr(model, root_name, None)
        if root_mod is None:
            continue
        for pname, p in root_mod.named_parameters():
            if id(p) not in seen:
                seen.add(id(p))
                selected.append((f"{root_name}.{pname}", p))
    if not selected:
        raise RuntimeError("No full_encoder parameters found on model")
    return selected


def _decode_and_score_py(
    model: Any,
    cfg: Any,
    img_t: Any,
    ref_notes: Sequence[Tuple[str, int, str, str, str]],
    *,
    max_len: int,
    use_bf16: bool,
    device: Any,
    eval_s31: Any,
    rlvr_grpo: Any,
) -> Dict[str, Any]:
    import torch

    _, _, eos_id, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        if use_bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                seq, mtp_stats = eval_s31.generate_mtp_spec(
                    model, img_t, max_len=max_len, use_grammar=True,
                    strict_grammar=False, return_stats=True,
                )
        else:
            seq, mtp_stats = eval_s31.generate_mtp_spec(
                model, img_t, max_len=max_len, use_grammar=True,
                strict_grammar=False, return_stats=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.perf_counter() - t0)

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
    return {
        "note_f1_py": round(float(note_f1), 6),
        "r_rhythm": round(float(struct["r_rhythm"]), 4),
        "r_noloop": round(float(struct["r_noloop"]), 4),
        "has_eos": bool(has_eos),
        "n_tokens": len(pred_raw),
        "decode_secs": round(elapsed, 3),
        "mtp_acc": round(float(mtp_stats["acceptance_rate"]), 4),
    }


def _compute_reliability_gate(
    model: Any,
    cfg: Any,
    img_t: Any,
    pseudo_cache: Optional[Tuple[Any, Any, Any, Any, int]],
    *,
    eval_s31: Any,
    rlvr_grpo: Any,
) -> Dict[str, Any]:
    """Compute OV-39 winning sequence-level reliability gate:
    mean_logprob >= -0.35 AND beat_valid_frac >= 0.70 AND r_noloop == 1.0.
    """
    if pseudo_cache is None:
        return {"reliable": False, "mean_logprob": -10.0, "beat_valid_frac": 0.0, "r_noloop": 0.0}
    inp_ids, _conf_mask, valid_mask, ref_log_probs, n_conf = pseudo_cache
    if n_conf <= 0 or inp_ids.shape[1] < 2:
        return {"reliable": False, "mean_logprob": -10.0, "beat_valid_frac": 0.0, "r_noloop": 0.0}

    _, _, eos_id, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))
    seq_ids = inp_ids[0].detach().cpu().tolist()
    has_eos = eos_id in seq_ids[1:]
    pred_raw = eval_s31.decode_uni_compat(
        seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
    )
    if not has_eos:
        pred_raw = eval_s31.trim_truncated_tokens(pred_raw)
    struct = rlvr_grpo.compute_unlabeled_structural_reward(
        pred_raw, vocab=id_to_tok, terminated_eos=has_eos
    )
    # Approximate sequence mean_logprob from top-1 token log-probs in ref_log_probs
    top_lp = ref_log_probs.max(dim=-1).values
    mean_lp = float(((top_lp * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)).item())
    beat_v = float(struct["r_rhythm"])
    noloop = float(struct["r_noloop"])
    reliable = bool(mean_lp >= -0.35 and beat_v >= 0.70 and noloop >= 0.999 and len(pred_raw) >= 24)
    return {
        "reliable": reliable,
        "mean_logprob": round(mean_lp, 4),
        "beat_valid_frac": round(beat_v, 4),
        "r_noloop": round(noloop, 4),
    }


def main() -> None:
    t0 = time.perf_counter()
    started_utc = _utc_now()
    max_dim = 1280
    max_len = 3840
    ttt_max_decode_len = 512
    conf_threshold = 0.85
    cons_weight = 1.0
    l2_weight = 0.1
    grad_clip = 1.0
    printed_safety_n = 10
    seed = 42

    try:
        hf_token = _read_and_purge_hf_token()
        mvd_dir = _setup_environment()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        from huggingface_hub import HfApi
        from PIL import Image
        import torch
        import torch.nn.functional as F
        import eval_uni_suite31 as eval_s31
        import rlvr_grpo
        import trial_2028_ttt_adapt as ttt_mod
        import uni_omr

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        use_bf16 = bool(device.type == "cuda" and cc[0] >= 8)
        env_tag = "kaggle" if Path("/kaggle").exists() else ("colab" if Path("/content").exists() else "cloud")
        platform_str = f"{env_tag}-{gpu_name}"
        print(f"[{EXP_ID}] {platform_str} (cc={cc[0]}.{cc[1]}, bf16={use_bf16})", flush=True)

        api = HfApi(token=hf_token)
        ckpt_path, manifest_path = _ensure_assets(hf_token)
        model, cfg, ckpt_meta = eval_s31.load_uni_checkpoint(ckpt_path, device=device)
        _, all_pieces = eval_s31.resolve_manifest_pieces(manifest_path)
        hw_pieces = [p for p in all_pieces if p["handwritten"]]
        printed_pieces = [p for p in all_pieces if not p["handwritten"]][:printed_safety_n]

        named_enc = _select_full_encoder_params(model)
        enc_ids = {id(p) for _, p in named_enc}
        enc_list = [p for _, p in named_enc]
        full_enc_snap_cpu = {nm: p.detach().cpu().clone() for nm, p in named_enc}

        @torch.no_grad()
        def _restore_all() -> None:
            for nm, p in named_enc:
                p.copy_(full_enc_snap_cpu[nm].to(device=p.device, dtype=p.dtype))
                p.grad = None

        pseudo_helper = ttt_mod.TTTSession(
            model=model, ttt_lr=1e-5, adapt_scope="norms",
            conf_threshold=conf_threshold, use_bf16=use_bf16,
            strict_grammar=False, max_decode_len=ttt_max_decode_len, seed=seed,
        )

        ledger_tmp = CACHE_ROOT / "run_ledger.json"
        _atomic_write_json(ledger_tmp, {
            "exp_id": EXP_ID,
            "status": "RUNNING",
            "started_utc": started_utc,
            "platform": platform_str,
            "arms": [a["key"] for a in ARMS],
            "max_len": max_len,
        })
        try:
            api.upload_file(
                path_or_fileobj=str(ledger_tmp),
                path_in_repo=HF_RUN_PATH,
                repo_id=HF_REPO,
                repo_type="dataset",
                commit_message=f"{EXP_ID}: mark RUNNING on {platform_str}",
            )
        except Exception as exc:
            print(f"[{EXP_ID}] Warning: initial HF ledger upload failed: {exc}", flush=True)

        def _compute_step_loss(
            arm: Dict[str, Any],
            img_t: Any,
            img_aug: Any,
            pseudo_cache: Optional[Tuple[Any, Any, Any, Any, int]],
            is_reliable: bool,
        ) -> Any:
            if use_bf16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    mem_c, _, rinfo_c = model.encode_image(img_t, return_aux=False)
                    mem_a, _, rinfo_a = model.encode_image(img_aug, return_aux=False)
            else:
                mem_c, _, rinfo_c = model.encode_image(img_t, return_aux=False)
                mem_a, _, rinfo_a = model.encode_image(img_aug, return_aux=False)

            loss_cons, _, _ = ttt_mod.compute_multiview_consistency_loss(mem_c, rinfo_c, mem_a, rinfo_a)
            loss_dec = torch.zeros((), device=device, dtype=torch.float32)

            mode = arm["mode"]
            ent_w = float(arm["ent_w"])
            kl_w = float(arm["kl_w"])
            if mode == "rhythm_gated" and not is_reliable:
                ent_w, kl_w = 0.0, 0.0

            if (ent_w > 0.0 or kl_w > 0.0) and pseudo_cache is not None:
                inp_ids, conf_mask, valid_mask, ref_log_probs, n_conf = pseudo_cache
                if n_conf > 0:
                    if use_bf16 and device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits_c, _ = model.decode_tokens(
                                inp_ids, memory=mem_c, return_mtp=False,
                                use_cache=False, apply_tok_dropout=False,
                            )
                            logits_a = (
                                model.decode_tokens(
                                    inp_ids, memory=mem_a, return_mtp=False,
                                    use_cache=False, apply_tok_dropout=False,
                                )[0]
                                if mode == "memo"
                                else None
                            )
                    else:
                        logits_c, _ = model.decode_tokens(
                            inp_ids, memory=mem_c, return_mtp=False,
                            use_cache=False, apply_tok_dropout=False,
                        )
                        logits_a = (
                            model.decode_tokens(
                                inp_ids, memory=mem_a, return_mtp=False,
                                use_cache=False, apply_tok_dropout=False,
                            )[0]
                            if mode == "memo"
                            else None
                        )

                    log_pc = F.log_softmax(logits_c.float(), dim=-1)
                    pc = log_pc.exp()
                    if mode == "memo" and logits_a is not None:
                        log_pa = F.log_softmax(logits_a.float(), dim=-1)
                        pa = log_pa.exp()
                        p_bar = (0.5 * (pc + pa)).clamp(min=1e-7)
                        log_pbar = p_bar.log()
                        ent_marg = -(p_bar * log_pbar).sum(dim=-1)
                        jsd_tok = 0.5 * (
                            (pc * (log_pc - log_pbar)).sum(dim=-1)
                            + (pa * (log_pa - log_pbar)).sum(dim=-1)
                        ).clamp(min=0.0)
                        loss_ent = ((ent_marg + 0.5 * jsd_tok) * conf_mask).sum() / conf_mask.sum().clamp(min=1.0)
                    else:
                        ent_tok = -(pc * log_pc).sum(dim=-1)
                        loss_ent = (ent_tok * conf_mask).sum() / conf_mask.sum().clamp(min=1.0)

                    ref_p = ref_log_probs.exp()
                    kl_tok = (ref_p * (ref_log_probs - log_pc)).sum(dim=-1).clamp(min=0.0)
                    loss_kl = (kl_tok * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
                    loss_dec = ent_w * loss_ent + kl_w * loss_kl

            loss_l2 = torch.stack([
                (p.float() - full_enc_snap_cpu[nm].to(device=device, dtype=torch.float32)).pow(2).mean()
                for nm, p in named_enc
            ]).mean()
            return cons_weight * loss_cons + loss_dec + l2_weight * loss_l2

        def _run_arm_on_page(
            arm: Dict[str, Any],
            img_t: Any,
            ref_notes: Sequence[Tuple[str, int, str, str, str]],
            before_f1: float,
            pseudo_cache: Optional[Tuple[Any, Any, Any, Any, int]],
            is_reliable: bool,
        ) -> Dict[str, Any]:
            _restore_all()
            for p in model.parameters():
                p.requires_grad_(id(p) in enc_ids)
            optimizer = torch.optim.AdamW(enc_list, lr=float(arm["lr"]), betas=(0.9, 0.99), weight_decay=0.0)
            t_traj = time.perf_counter()
            oom_hit = False
            sam_rho = float(arm["sam_rho"])

            with torch.enable_grad():
                for step_idx in range(1, int(arm["steps"]) + 1):
                    model.eval()
                    optimizer.zero_grad(set_to_none=True)
                    img_aug = ttt_mod.make_jittered_view(img_t, step_idx=step_idx - 1, seed=seed)
                    try:
                        loss = _compute_step_loss(arm, img_t, img_aug, pseudo_cache, is_reliable)
                        if not bool(torch.isfinite(loss)):
                            continue
                        loss.backward()

                        if sam_rho > 0.0:
                            # SAR 1-step sharpness-aware ascent -> gradient at w+e -> restore -> step
                            grad_norm = torch.sqrt(
                                sum((p.grad.detach().float().pow(2).sum() for p in enc_list if p.grad is not None), torch.zeros((), device=device))
                                + 1e-12
                            )
                            scale = sam_rho / grad_norm
                            eps_list: List[Optional[Any]] = []
                            with torch.no_grad():
                                for p in enc_list:
                                    if p.grad is not None:
                                        e = (p.grad.detach().float() * scale).to(dtype=p.dtype)
                                        p.add_(e)
                                        eps_list.append(e)
                                    else:
                                        eps_list.append(None)
                            optimizer.zero_grad(set_to_none=True)
                            loss_sam = _compute_step_loss(arm, img_t, img_aug, pseudo_cache, is_reliable)
                            if bool(torch.isfinite(loss_sam)):
                                loss_sam.backward()
                            with torch.no_grad():
                                for p, e in zip(enc_list, eps_list):
                                    if e is not None:
                                        p.sub_(e)

                        gn = torch.nn.utils.clip_grad_norm_(enc_list, max_norm=grad_clip)
                        if math.isfinite(float(gn.item())):
                            optimizer.step()
                        else:
                            optimizer.zero_grad(set_to_none=True)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        oom_hit = True
                        break

            ttt_secs = round(time.perf_counter() - t_traj, 3)
            scored = _decode_and_score_py(
                model, cfg, img_t, ref_notes,
                max_len=max_len, use_bf16=use_bf16, device=device,
                eval_s31=eval_s31, rlvr_grpo=rlvr_grpo,
            )
            scored["ttt_secs"] = ttt_secs
            scored["oom_hit"] = bool(oom_hit)
            scored["delta_note_f1_py"] = round(scored["note_f1_py"] - before_f1, 6)
            _restore_all()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return scored

        partial_path = CACHE_ROOT / "evals" / f"{EXP_ID}_partial.json"
        hw_rows: List[Dict[str, Any]] = []
        pr_rows: List[Dict[str, Any]] = []
        done_hw: set[Tuple[str, str]] = set()
        done_pr: set[Tuple[str, str]] = set()
        if partial_path.exists():
            try:
                prev = json.loads(partial_path.read_text(encoding="utf-8"))
                hw_rows = prev.get("hw_rows", [])
                pr_rows = prev.get("printed_rows", [])
                done_hw = {(r["piece"], r["domain"]) for r in hw_rows}
                done_pr = {(r["piece"], r["domain"]) for r in pr_rows}
            except Exception:
                pass

        for idx, piece in enumerate(hw_pieces, start=1):
            slug = piece["slug"]
            ref_notes = _extract_musicxml_notes(piece["gt"])
            for domain in ("clean", "photo"):
                if (slug, domain) in done_hw:
                    continue
                with Image.open(piece[domain]) as raw_im:
                    pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
                img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)

                _restore_all()
                before = _decode_and_score_py(
                    model, cfg, img_t, ref_notes,
                    max_len=max_len, use_bf16=use_bf16, device=device,
                    eval_s31=eval_s31, rlvr_grpo=rlvr_grpo,
                )
                pseudo_cache = pseudo_helper._build_pseudo_cache(img_t)
                rel_info = _compute_reliability_gate(
                    model, cfg, img_t, pseudo_cache, eval_s31=eval_s31, rlvr_grpo=rlvr_grpo
                )

                arms_out: Dict[str, Any] = {}
                for arm in ARMS:
                    arms_out[arm["key"]] = _run_arm_on_page(
                        arm, img_t, ref_notes, before["note_f1_py"], pseudo_cache, rel_info["reliable"]
                    )
                hw_rows.append({
                    "piece": slug,
                    "domain": domain,
                    "before": before,
                    "reliability": rel_info,
                    "arms": arms_out,
                })
                done_hw.add((slug, domain))
                _atomic_write_json(partial_path, {"exp_id": EXP_ID, "hw_rows": hw_rows, "printed_rows": pr_rows})
                print(
                    f"[hw {idx:02d}/{len(hw_pieces):02d} {domain}] {slug} base={before['note_f1_py']:.4f} "
                    f"rel={rel_info['reliable']} | "
                    + " ".join(f"{a['key']}={arms_out[a['key']]['delta_note_f1_py']:+.4f}" for a in ARMS),
                    flush=True,
                )

        def _summarize(rows: Sequence[Dict[str, Any]], arm_keys: Sequence[str]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for dom in ("clean", "photo"):
                sub = [r for r in rows if r["domain"] == dom]
                if not sub:
                    continue
                base_mean = round(sum(r["before"]["note_f1_py"] for r in sub) / len(sub), 6)
                dom_arms: Dict[str, Any] = {}
                for ak in arm_keys:
                    vals = [r["arms"][ak]["note_f1_py"] for r in sub if ak in r["arms"]]
                    deltas = [r["arms"][ak]["delta_note_f1_py"] for r in sub if ak in r["arms"]]
                    if not vals:
                        continue
                    dom_arms[ak] = {
                        "note_f1_py_mean": round(sum(vals) / len(vals), 6),
                        "delta_mean": round(sum(deltas) / len(deltas), 6),
                        "wins": sum(1 for d in deltas if d > 1e-6),
                        "ties": sum(1 for d in deltas if abs(d) <= 1e-6),
                        "losses": sum(1 for d in deltas if d < -1e-6),
                    }
                out[dom] = {"n": len(sub), "baseline_note_f1_py_mean": base_mean, "arms": dom_arms}
            return out

        all_arm_keys = [a["key"] for a in ARMS]
        hw_summary = _summarize(hw_rows, all_arm_keys)

        def _hw_avg_delta(ak: str) -> float:
            dc = hw_summary.get("clean", {}).get("arms", {}).get(ak, {}).get("delta_mean", 0.0)
            dp = hw_summary.get("photo", {}).get("arms", {}).get(ak, {}).get("delta_mean", 0.0)
            return 0.5 * (float(dc) + float(dp))

        top2_keys = sorted(all_arm_keys, key=_hw_avg_delta, reverse=True)[:2]
        top2_arms = [a for a in ARMS if a["key"] in top2_keys]
        print(f"[{EXP_ID}] Running printed safety check on top-2 arms: {top2_keys}", flush=True)

        for idx, piece in enumerate(printed_pieces, start=1):
            slug = piece["slug"]
            ref_notes = _extract_musicxml_notes(piece["gt"])
            for domain in ("clean", "photo"):
                if (slug, domain) in done_pr:
                    continue
                with Image.open(piece[domain]) as raw_im:
                    pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
                img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
                _restore_all()
                before = _decode_and_score_py(
                    model, cfg, img_t, ref_notes,
                    max_len=max_len, use_bf16=use_bf16, device=device,
                    eval_s31=eval_s31, rlvr_grpo=rlvr_grpo,
                )
                pseudo_cache = pseudo_helper._build_pseudo_cache(img_t)
                rel_info = _compute_reliability_gate(
                    model, cfg, img_t, pseudo_cache, eval_s31=eval_s31, rlvr_grpo=rlvr_grpo
                )
                arms_out = {
                    arm["key"]: _run_arm_on_page(
                        arm, img_t, ref_notes, before["note_f1_py"], pseudo_cache, rel_info["reliable"]
                    )
                    for arm in top2_arms
                }
                pr_rows.append({"piece": slug, "domain": domain, "before": before, "reliability": rel_info, "arms": arms_out})
                done_pr.add((slug, domain))
                _atomic_write_json(partial_path, {"exp_id": EXP_ID, "hw_rows": hw_rows, "printed_rows": pr_rows})

        pr_summary = _summarize(pr_rows, top2_keys)
        report = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_utc,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.perf_counter() - t0, 2),
            "platform": platform_str,
            "bf16": use_bf16,
            "ckpt_meta": ckpt_meta,
            "max_dim": max_dim,
            "max_len": max_len,
            "arms": ARMS,
            "top2_printed_safety_arms": top2_keys,
            "hw_summary": hw_summary,
            "printed_safety_summary": pr_summary,
            "hw_rows": hw_rows,
            "printed_rows": pr_rows,
        }
        eval_local = CACHE_ROOT / "evals" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(ledger_tmp, {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "platform": platform_str,
            "elapsed_secs": report["elapsed_secs"],
            "hw_summary": hw_summary,
            "printed_safety_summary": pr_summary,
        })

        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload MEMO/SAR/Rhythm-Gated TTT eval",
        )
        api.upload_file(
            path_or_fileobj=str(ledger_tmp),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): mark COMPLETED",
        )
        print(f"[{EXP_ID}] COMPLETE & UPLOADED TO HF!", flush=True)
    finally:
        _purge_hf_token()


if __name__ == "__main__":
    main()

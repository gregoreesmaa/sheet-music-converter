#!/usr/bin/env python3
"""Overnight Exp 38 (Batch 09, Wave 7) on Google Colab L4:
SCRIBE/DocTTT Per-Manuscript TTA Sweep:
  steps in {2, 4, 8} x lr in {1e-5, 3e-5, 1e-4} x scope in {encoder_norm_only, full_encoder}
on all 14 real historical handwritten manuscripts in `suite31_hw` (clean + photo = 28 pages).

Uses cumulative stepping (0 -> 2 -> 4 -> 8) per (scope, lr) trajectory and caches the
pre-TTT fast constrained pseudo-decode once per page so all 18 grid cells finish in
~26-32 min on Colab L4 with bit-exact (atol=0) weight restoration between trajectories.

Expects:
  - /content/bundle.tar.gz (or pre-extracted /content/workspace/sheet-music-converter/tmp-train/mvd)
  - /tmp/.hf_token (Hugging Face token; chmod 600, deleted immediately and in finally)
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

EXP_ID = "2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_PATH = "checkpoints/iter10_soup_baseline/uni_omr_soup.pt"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

STEP_COUNTS = [2, 4, 8]
LRS = [1e-5, 3e-5, 1e-4]
SCOPES = ["encoder_norm_only", "full_encoder"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
        alt_mvd = Path("/workspace/sheet-music-converter/tmp-train/mvd")
        if alt_mvd.exists():
            mvd_dir = alt_mvd
        else:
            if not BUNDLE_PATH.exists():
                raise FileNotFoundError(f"Missing bundle at {BUNDLE_PATH}")
            print(f"[{EXP_ID}] Extracting {BUNDLE_PATH} -> {WORK_ROOT}...", flush=True)
            with tarfile.open(BUNDLE_PATH, "r:*") as tf:
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
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=HF_CKPT_PATH,
            local_dir=str(CACHE_ROOT),
            token=hf_token,
        )

    s31_dir = CACHE_ROOT / "datasets" / "suite31_hw_eval"
    sentinel = s31_dir / ".extracted"
    if not sentinel.exists():
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR}...", flush=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=HF_SUITE31_TAR,
                local_dir=str(CACHE_ROOT / "tars"),
                token=hf_token,
            )
        )
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


def _select_scope_params(model: Any, scope: str, uni_omr_mod: Any) -> List[Tuple[str, Any]]:
    selected: List[Tuple[str, Any]] = []
    seen: set[int] = set()
    if scope == "encoder_norm_only":
        for root_name in ("backbone", "enc_layers", "enc_norm"):
            root_mod = getattr(model, root_name, None)
            if root_mod is None:
                continue
            for sub_name, mod in root_mod.named_modules():
                if isinstance(mod, uni_omr_mod.RMSNorm):
                    for pname, p in mod.named_parameters(recurse=False):
                        if id(p) not in seen:
                            seen.add(id(p))
                            full = f"{root_name}.{sub_name}.{pname}" if sub_name else f"{root_name}.{pname}"
                            selected.append((full, p))
    elif scope == "full_encoder":
        for root_name in ("backbone", "router", "enc_layers", "enc_norm"):
            root_mod = getattr(model, root_name, None)
            if root_mod is None:
                continue
            for pname, p in root_mod.named_parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    selected.append((f"{root_name}.{pname}", p))
    else:
        raise ValueError(f"Unknown scope: {scope}")
    if not selected:
        raise RuntimeError(f"No parameters matched scope={scope}")
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
    hyp_toks, n_sys = eval_s31.stitch_uni_tokens(pred_raw)
    struct = rlvr_grpo.compute_unlabeled_structural_reward(
        pred_raw, vocab=id_to_tok, terminated_eos=has_eos
    )
    hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
    note_f1 = rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes)
    return {
        "note_f1_py": round(float(note_f1), 6),
        "struct_reward": round(float(struct["reward"]), 6),
        "r_rhythm": round(float(struct["r_rhythm"]), 4),
        "r_noloop": round(float(struct["r_noloop"]), 4),
        "has_eos": bool(has_eos),
        "n_sys": int(n_sys),
        "n_hyp_notes": len(hyp_notes),
        "elapsed_secs": round(elapsed, 3),
        "mtp_acc": round(float(mtp_stats["acceptance_rate"]), 4),
    }


def _summarize_grid(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for domain in ("clean", "photo"):
        drows = [r for r in rows if r["domain"] == domain]
        base_vals = [float(r["before"]["note_f1_py"]) for r in drows]
        base_mean = round(sum(base_vals) / len(base_vals), 6) if base_vals else None
        cells: Dict[str, Any] = {}
        for scope in SCOPES:
            for lr in LRS:
                for s in STEP_COUNTS:
                    ckey = f"{scope}_lr{lr:.0e}_s{s}"
                    aft_vals = [
                        float(r["cells"][ckey]["note_f1_py"])
                        for r in drows
                        if ckey in r.get("cells", {})
                    ]
                    rhy_vals = [
                        float(r["cells"][ckey]["r_rhythm"])
                        for r in drows
                        if ckey in r.get("cells", {})
                    ]
                    times = [
                        float(r["cells"][ckey]["ttt_secs"])
                        for r in drows
                        if ckey in r.get("cells", {})
                    ]
                    wins = sum(
                        1 for r in drows
                        if ckey in r.get("cells", {})
                        and float(r["cells"][ckey]["note_f1_py"]) > float(r["before"]["note_f1_py"]) + 1e-6
                    )
                    losses = sum(
                        1 for r in drows
                        if ckey in r.get("cells", {})
                        and float(r["cells"][ckey]["note_f1_py"]) < float(r["before"]["note_f1_py"]) - 1e-6
                    )
                    aft_mean = round(sum(aft_vals) / len(aft_vals), 6) if aft_vals else None
                    cells[ckey] = {
                        "scope": scope,
                        "lr": lr,
                        "steps": s,
                        "n": len(aft_vals),
                        "after_note_f1_py": aft_mean,
                        "delta_note_f1_py": round(aft_mean - base_mean, 6)
                        if (aft_mean is not None and base_mean is not None)
                        else None,
                        "r_rhythm_mean": round(sum(rhy_vals) / len(rhy_vals), 4) if rhy_vals else None,
                        "wins": wins,
                        "ties": len(aft_vals) - wins - losses,
                        "losses": losses,
                        "median_ttt_secs": round(float(statistics.median(times)), 3) if times else None,
                    }
        summary[domain] = {"n": len(drows), "before_note_f1_py": base_mean, "cells": cells}
    return summary


def run_sweep(
    max_dim: int = 1280,
    max_len: int = 1536,
    ttt_max_decode_len: int = 512,
    conf_threshold: float = 0.85,
    cons_weight: float = 1.0,
    ent_weight: float = 1.0,
    kl_weight: float = 0.5,
    l2_weight: float = 1e-2,
    grad_clip: float = 1.0,
    save_every_minutes: float = 5.0,
    seed: int = 2028,
) -> Dict[str, Any]:
    restore_fn = None
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

        t_start = time.perf_counter()
        started_utc = _utc_now()
        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
        use_bf16 = bool(device.type == "cuda" and cc[0] >= 8)
        platform_str = f"colab-{gpu_name}"
        print(f"[{EXP_ID}] {platform_str} (cc={cc[0]}.{cc[1]}, bf16={use_bf16})", flush=True)

        api = HfApi(token=hf_token)
        ckpt_path, manifest_path = _ensure_assets(hf_token)
        model, cfg, ckpt_meta = eval_s31.load_uni_checkpoint(ckpt_path, device=device)
        _, all_pieces = eval_s31.resolve_manifest_pieces(manifest_path)
        hw_pieces = [p for p in all_pieces if p["handwritten"]]

        # Precompute parameter lists + CPU snapshots for both scopes
        total_params = sum(p.numel() for p in model.parameters())
        scope_params: Dict[str, List[Tuple[str, Any]]] = {
            sc: _select_scope_params(model, sc, uni_omr) for sc in SCOPES
        }
        full_enc_snap_cpu = {
            name: p.detach().cpu().clone() for name, p in scope_params["full_encoder"]
        }

        @torch.no_grad()
        def _restore_all() -> None:
            for name, p in scope_params["full_encoder"]:
                p.copy_(full_enc_snap_cpu[name].to(device=p.device, dtype=p.dtype))
                p.grad = None

        restore_fn = _restore_all

        # Helper TTTSession only for its pre-TTT fast constrained pseudo-decode
        pseudo_helper = ttt_mod.TTTSession(
            model=model, ttt_lr=1e-5, adapt_scope="norms",
            conf_threshold=conf_threshold, use_bf16=use_bf16,
            strict_grammar=False, max_decode_len=ttt_max_decode_len, seed=seed,
        )

        # Upload RUNNING ledger
        ledger_tmp = CACHE_ROOT / "run_ledger.json"
        _atomic_write_json(
            ledger_tmp,
            {
                "exp_id": EXP_ID,
                "status": "RUNNING",
                "started_utc": started_utc,
                "platform": platform_str,
                "grid": {"steps": STEP_COUNTS, "lrs": LRS, "scopes": SCOPES},
                "n_hw_pieces": len(hw_pieces),
            },
        )
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

        partial_path = CACHE_ROOT / "evals" / f"{EXP_ID}_partial.json"
        rows: List[Dict[str, Any]] = []
        done_keys: set[Tuple[str, str]] = set()
        if partial_path.exists():
            try:
                prev = json.loads(partial_path.read_text(encoding="utf-8"))
                if isinstance(prev.get("rows"), list):
                    rows = prev["rows"]
                    done_keys = {(r["piece"], r["domain"]) for r in rows}
                    print(f"[{EXP_ID}] Resuming: {len(done_keys)}/28 pages already done", flush=True)
            except Exception:
                pass

        last_save_t = time.perf_counter()
        max_steps = max(STEP_COUNTS)

        for idx, piece in enumerate(hw_pieces, start=1):
            slug = piece["slug"]
            ref_notes = _extract_musicxml_notes(piece["gt"])

            for domain in ("clean", "photo"):
                if (slug, domain) in done_keys:
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

                # Cache pre-TTT fast constrained pseudo-decode once per page
                inp_ids, conf_mask, valid_mask, ref_log_probs, n_confident = (
                    pseudo_helper._fast_constrained_pseudo_decode(img_t, max_len=ttt_max_decode_len)
                )

                cell_results: Dict[str, Any] = {}
                for scope in SCOPES:
                    named_p = scope_params[scope]
                    adapted_ids = {id(p) for _, p in named_p}
                    adapted_list = [p for _, p in named_p]

                    for lr in LRS:
                        _restore_all()
                        for p in model.parameters():
                            p.requires_grad_(id(p) in adapted_ids)
                        optimizer = torch.optim.AdamW(
                            adapted_list, lr=float(lr), betas=(0.9, 0.99), weight_decay=0.0
                        )
                        t_traj = time.perf_counter()
                        oom_hit = False

                        with torch.enable_grad():
                            for step_idx in range(1, max_steps + 1):
                                model.eval()
                                optimizer.zero_grad(set_to_none=True)
                                img_aug = ttt_mod.make_jittered_view(img_t, step_idx=step_idx - 1, seed=seed)
                                try:
                                    if use_bf16 and device.type == "cuda":
                                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                            mem_c, _, rinfo_c = model.encode_image(img_t, return_aux=False)
                                            mem_a, _, rinfo_a = model.encode_image(img_aug, return_aux=False)
                                    else:
                                        mem_c, _, rinfo_c = model.encode_image(img_t, return_aux=False)
                                        mem_a, _, rinfo_a = model.encode_image(img_aug, return_aux=False)

                                    loss_cons, _, _ = ttt_mod.compute_multiview_consistency_loss(
                                        mem_c, rinfo_c, mem_a, rinfo_a
                                    )
                                    loss_ent = torch.zeros((), device=device, dtype=torch.float32)
                                    loss_kl = torch.zeros((), device=device, dtype=torch.float32)
                                    if n_confident > 0 and ent_weight > 0.0:
                                        if use_bf16 and device.type == "cuda":
                                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                                logits, _ = model.decode_tokens(
                                                    inp_ids, memory=mem_c, return_mtp=False,
                                                    use_cache=False, apply_tok_dropout=False,
                                                )
                                        else:
                                            logits, _ = model.decode_tokens(
                                                inp_ids, memory=mem_c, return_mtp=False,
                                                use_cache=False, apply_tok_dropout=False,
                                            )
                                        log_p = F.log_softmax(logits.float(), dim=-1)
                                        p_cur = log_p.exp()
                                        ent_tok = -(p_cur * log_p).sum(dim=-1)
                                        loss_ent = (ent_tok * conf_mask).sum() / conf_mask.sum().clamp(min=1.0)
                                        if kl_weight > 0.0:
                                            ref_p = ref_log_probs.exp()
                                            kl_tok = (ref_p * (ref_log_probs - log_p)).sum(dim=-1).clamp(min=0.0)
                                            loss_kl = (kl_tok * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

                                    loss_l2 = torch.zeros((), device=device, dtype=torch.float32)
                                    if l2_weight > 0.0:
                                        l2_terms = [
                                            (p.float() - full_enc_snap_cpu[nm].to(device=device, dtype=torch.float32)).pow(2).mean()
                                            for nm, p in named_p
                                        ]
                                        loss_l2 = torch.stack(l2_terms).mean()

                                    total_loss = (
                                        cons_weight * loss_cons
                                        + ent_weight * loss_ent
                                        + kl_weight * loss_kl
                                        + l2_weight * loss_l2
                                    )
                                    if bool(torch.isfinite(total_loss)):
                                        total_loss.backward()
                                        gn = torch.nn.utils.clip_grad_norm_(adapted_list, max_norm=grad_clip)
                                        if math.isfinite(float(gn.item())):
                                            optimizer.step()
                                        else:
                                            optimizer.zero_grad(set_to_none=True)
                                except torch.cuda.OutOfMemoryError:
                                    torch.cuda.empty_cache()
                                    oom_hit = True
                                    break

                                if step_idx in STEP_COUNTS:
                                    ckey = f"{scope}_lr{lr:.0e}_s{step_idx}"
                                    ttt_secs = round(time.perf_counter() - t_traj, 3)
                                    scored = _decode_and_score_py(
                                        model, cfg, img_t, ref_notes,
                                        max_len=max_len, use_bf16=use_bf16, device=device,
                                        eval_s31=eval_s31, rlvr_grpo=rlvr_grpo,
                                    )
                                    scored["ttt_secs"] = ttt_secs
                                    scored["delta_note_f1_py"] = round(
                                        scored["note_f1_py"] - before["note_f1_py"], 6
                                    )
                                    cell_results[ckey] = scored

                        if oom_hit:
                            print(f"[{EXP_ID}] OOM on {slug}[{domain}] scope={scope} lr={lr:.0e}", flush=True)
                        _restore_all()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                row = {
                    "piece": slug,
                    "domain": domain,
                    "n_confident_tokens": int(n_confident),
                    "before": before,
                    "cells": cell_results,
                }
                rows.append(row)
                done_keys.add((slug, domain))

                best_cell = max(
                    cell_results.items(),
                    key=lambda kv: kv[1]["note_f1_py"],
                    default=("none", {"note_f1_py": before["note_f1_py"]}),
                )
                print(
                    f"[{idx:02d}/{len(hw_pieces):02d}] {slug:34s} [{domain:5s}] "
                    f"base={before['note_f1_py']:.4f} | best={best_cell[0]}:{best_cell[1]['note_f1_py']:.4f} "
                    f"(conf={n_confident})",
                    flush=True,
                )

                if (time.perf_counter() - last_save_t) >= save_every_minutes * 60.0 or domain == "photo":
                    _atomic_write_json(
                        partial_path,
                        {
                            "exp_id": EXP_ID,
                            "status": "RUNNING",
                            "summary": _summarize_grid(rows),
                            "rows": rows,
                        },
                    )
                    last_save_t = time.perf_counter()

        summary = _summarize_grid(rows)
        # Identify best cell by average clean+photo delta
        best_cell_key = None
        best_avg_delta = -1e9
        for scope in SCOPES:
            for lr in LRS:
                for s in STEP_COUNTS:
                    ckey = f"{scope}_lr{lr:.0e}_s{s}"
                    dc = summary["clean"]["cells"][ckey]["delta_note_f1_py"]
                    dp = summary["photo"]["cells"][ckey]["delta_note_f1_py"]
                    if dc is not None and dp is not None:
                        avg = 0.5 * (dc + dp)
                        if avg > best_avg_delta:
                            best_avg_delta = avg
                            best_cell_key = ckey

        total_secs = round(time.perf_counter() - t_start, 2)
        final_report = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_utc,
            "completed_utc": _utc_now(),
            "elapsed_secs": total_secs,
            "platform": platform_str,
            "base_ckpt": HF_CKPT_PATH,
            "base_ckpt_meta": ckpt_meta,
            "scope_param_counts": {
                sc: {
                    "count": sum(p.numel() for _, p in scope_params[sc]),
                    "fraction": round(sum(p.numel() for _, p in scope_params[sc]) / max(1, total_params), 6),
                }
                for sc in SCOPES
            },
            "config": {
                "steps": STEP_COUNTS,
                "lrs": LRS,
                "scopes": SCOPES,
                "max_dim": max_dim,
                "max_len": max_len,
                "ttt_max_decode_len": ttt_max_decode_len,
                "conf_threshold": conf_threshold,
            },
            "best_cell_key": best_cell_key,
            "best_avg_delta_note_f1_py": round(best_avg_delta, 6) if best_cell_key else None,
            "summary": summary,
            "rows": rows,
        }
        final_local = CACHE_ROOT / "evals" / f"{EXP_ID}.json"
        _atomic_write_json(final_local, final_report)

        api.upload_file(
            path_or_fileobj=str(final_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload SCRIBE/DocTTT 18-cell sweep eval JSON",
        )
        _atomic_write_json(
            ledger_tmp,
            {
                "exp_id": EXP_ID,
                "status": "COMPLETED",
                "started_utc": started_utc,
                "completed_utc": final_report["completed_utc"],
                "elapsed_secs": total_secs,
                "platform": platform_str,
                "best_cell_key": best_cell_key,
                "best_avg_delta_note_f1_py": final_report["best_avg_delta_note_f1_py"],
                "hf_eval_path": HF_EVAL_PATH,
            },
        )
        api.upload_file(
            path_or_fileobj=str(ledger_tmp),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark COMPLETED",
        )
        print(f"[{EXP_ID}] COMPLETED in {total_secs}s | best_cell={best_cell_key}", flush=True)
        return final_report
    finally:
        if restore_fn is not None:
            try:
                restore_fn()
            except Exception:
                pass
        if TOKEN_PATH.exists():
            try:
                TOKEN_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    run_sweep()

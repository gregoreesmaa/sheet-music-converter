#!/usr/bin/env python3
"""Experiment 5: Label-Free Unlabeled-RISM Structural-Validity RLVR (GRPO) on Modal L4.

Tests 1-variable isolation: applying 120 updates of label-free structural-validity
GRPO (G=4, kl_beta=0.04, unlabeled_frac=1.0) via ``tmp-train/mvd/rlvr_grpo.py``
on 5,707 unlabeled real RISM handwritten pages starting from the Iter-10
``uni_omr_soup.pt`` baseline, followed by apples-to-apples before-vs-after
45-piece ``suite31_hw`` evaluation (14 HW + 31 Printed, clean + photo).

Launch detached from repo root:
    .\\.venv\\Scripts\\modal.exe run --detach experiments/2026-11-26T00-54-ablation-unlabeled-rism-rlvr/run_modal.py
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import modal

EXP_ID = "2026-11-26T00-54-ablation-unlabeled-rism-rlvr"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"

app = modal.App("exp5-ablation-unlabeled-rism-rlvr")
vol = modal.Volume.from_name("omr-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("tar")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "huggingface_hub>=0.26",
        "pillow",
        "numpy",
        "tqdm",
    )
    # Keep repo code LAST so edits rebuild in ~2s instead of reinstalling PyTorch.
    .add_local_dir("tmp-train/mvd", remote_path="/workspace/mvd")
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_tar_once(tar_path: Path, dest_dir: Path, sentinel_name: str = ".extracted") -> None:
    sentinel = dest_dir / sentinel_name
    if sentinel.exists():
        print(f"[cache] Already extracted: {dest_dir}", flush=True)
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cache] Extracting {tar_path} -> {dest_dir} ...", flush=True)
    t0 = time.perf_counter()
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(dest_dir)
    sentinel.write_text(_utc_now() + "\n", encoding="utf-8")
    print(f"[cache] Extracted {dest_dir} in {time.perf_counter() - t0:.1f}s", flush=True)


def _ensure_assets_on_volume() -> Tuple[Path, Path, Path]:
    """Download baseline ckpt + RISM 5,707 + Suite31_hw eval tars into /vol (idempotent)."""
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable not found (attach hf-secret).")

    ckpt_local = Path("/vol/checkpoints/iter10_soup_baseline/uni_omr_soup.pt")
    if not ckpt_local.exists():
        print("[hf] Downloading checkpoints/iter10_soup_baseline/uni_omr_soup.pt ...", flush=True)
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename="checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
            local_dir="/vol",
            token=token,
        )
        vol.commit()

    rism_dir = Path("/vol/datasets/rism_hw")
    if not (rism_dir / ".extracted").exists():
        print("[hf] Downloading datasets/rism_hw_5707.tar ...", flush=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename="datasets/rism_hw_5707.tar",
                local_dir="/tmp/hf_tars",
                token=token,
            )
        )
        _extract_tar_once(tar_p, rism_dir)
        try:
            tar_p.unlink(missing_ok=True)
        except OSError:
            pass
        vol.commit()

    s31_dir = Path("/vol/datasets/suite31_hw_eval")
    if not (s31_dir / ".extracted").exists():
        print("[hf] Downloading datasets/suite31_hw_eval.tar ...", flush=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename="datasets/suite31_hw_eval.tar",
                local_dir="/tmp/hf_tars",
                token=token,
            )
        )
        _extract_tar_once(tar_p, s31_dir)
        try:
            tar_p.unlink(missing_ok=True)
        except OSError:
            pass
        vol.commit()

    manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Could not find suite31_hw_manifest.json under {s31_dir}")

    return ckpt_local, rism_dir, manifests[0]


def _extract_musicxml_notes(xml_path: Path) -> List[Tuple[str, int, str, str, str]]:
    """Extract ordered (dur, n_dots, step, octave, alter) pitched notes from MusicXML.

    Pure-Python reference extractor matching ``rlvr_grpo.extract_pitch_dur_notes``
    so Modal Linux containers can score Note F1 without Windows .exe binaries.
    """
    notes: List[Tuple[str, int, str, str, str]] = []
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception:
        return notes

    def _strip_ns(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    for elem in root.iter():
        if _strip_ns(elem.tag) != "note":
            continue
        is_rest = False
        is_grace = False
        step = ""
        octave = ""
        alter = ""
        dur = ""
        n_dots = 0
        for child in elem:
            ctag = _strip_ns(child.tag)
            if ctag == "rest":
                is_rest = True
            elif ctag == "grace":
                is_grace = True
            elif ctag == "pitch":
                for pc in child:
                    ptag = _strip_ns(pc.tag)
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
        if is_rest or is_grace or not step or not octave or not dur:
            continue
        notes.append((dur, n_dots, step, octave, alter))
    return notes


def _upload_hf_file(local_path: Path, path_in_repo: str, commit_msg: str) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message=commit_msg,
    )
    print(f"[hf] Uploaded {path_in_repo}", flush=True)


def _upload_run_ledger(payload: Dict[str, Any], commit_msg: str) -> None:
    tmp = Path("/tmp/run_ledger.json")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        _upload_hf_file(tmp, f"runs/{EXP_ID}.json", commit_msg)
    except Exception as exc:
        print(f"[hf] Warning: failed to upload run ledger: {exc}", flush=True)


def _evaluate_suite31_python(
    model: Any,
    cfg: Any,
    manifest_path: Path,
    device: Any,
    *,
    tag: str,
    max_dim: int = 1024,
    max_len: int = 1536,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Apples-to-apples 45-piece suite31_hw eval (14 HW + 31 Printed, clean + photo).

    Computes exact pitch+duration Note F1 (via MusicXML ref vs decoded bekern),
    unlabeled structural validity (r_parse, r_rhythm, r_completeness, r_noloop),
    H/R note ratio, and saves decoded bekern per piece for optional Rust re-scoring.
    """
    from PIL import Image
    import torch

    import eval_uni_suite31 as eval_s31
    import rlvr_grpo
    import uni_omr

    _, pieces = eval_s31.resolve_manifest_pieces(manifest_path)
    _, _, eos_id, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))

    model.eval()
    piece_rows: List[Dict[str, Any]] = []
    t_all = time.perf_counter()

    for idx, piece in enumerate(pieces, start=1):
        slug = piece["slug"]
        is_hw = bool(piece["handwritten"])
        ref_notes = _extract_musicxml_notes(piece["gt"])
        prow: Dict[str, Any] = {
            "piece": slug,
            "handwritten": is_hw,
            "stratum": piece["stratum"],
            "ref_notes": len(ref_notes),
        }

        for domain in ("clean", "photo"):
            img_path = piece[domain]
            with Image.open(img_path) as raw_im:
                pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
            img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                if use_bf16 and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        seq, mtp_stats = eval_s31.generate_mtp_spec(
                            model,
                            img_t,
                            max_len=max_len,
                            use_grammar=True,
                            strict_grammar=False,
                            return_stats=True,
                        )
                else:
                    seq, mtp_stats = eval_s31.generate_mtp_spec(
                        model,
                        img_t,
                        max_len=max_len,
                        use_grammar=True,
                        strict_grammar=False,
                        return_stats=True,
                    )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = max(1e-6, time.perf_counter() - t0)

            seq_ids = seq[0].detach().cpu().tolist()
            has_eos = eos_id in seq_ids[1:]
            pred_raw_toks = eval_s31.decode_uni_compat(
                seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
            )
            if not has_eos:
                pred_raw_toks = eval_s31.trim_truncated_tokens(pred_raw_toks)
            hyp_page_toks, n_sys = eval_s31.stitch_uni_tokens(pred_raw_toks)

            struct = rlvr_grpo.compute_unlabeled_structural_reward(
                pred_raw_toks, vocab=id_to_tok, terminated_eos=has_eos
            )
            hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_page_toks)
            note_f1 = rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes)
            hr_ratio = (len(hyp_notes) / len(ref_notes)) if ref_notes else None

            pfx = domain + "_"
            prow[pfx + "note_f1_py"] = round(float(note_f1), 6)
            prow[pfx + "hyp_notes"] = len(hyp_notes)
            prow[pfx + "note_hr_ratio"] = round(float(hr_ratio), 4) if hr_ratio is not None else None
            prow[pfx + "struct_reward"] = round(float(struct["reward"]), 6)
            prow[pfx + "r_parse"] = round(float(struct["r_parse"]), 4)
            prow[pfx + "r_rhythm"] = round(float(struct["r_rhythm"]), 4)
            prow[pfx + "r_completeness"] = round(float(struct["r_completeness"]), 4)
            prow[pfx + "r_noloop"] = round(float(struct["r_noloop"]), 4)
            prow[pfx + "has_eos"] = bool(has_eos)
            prow[pfx + "n_sys"] = int(n_sys)
            prow[pfx + "n_pred_toks"] = len(hyp_page_toks)
            prow[pfx + "elapsed_secs"] = round(elapsed, 3)
            prow[pfx + "mtp_acc"] = round(float(mtp_stats["acceptance_rate"]), 4)
            prow[pfx + "hyp_bekern"] = " ".join(hyp_page_toks)

        if idx % 5 == 0 or is_hw or idx == len(pieces):
            kind = "HW" if is_hw else "PR"
            print(
                f"[eval:{tag}] [{idx:02d}/{len(pieces):02d}] {slug:30s} [{kind}] "
                f"clean_F1={prow['clean_note_f1_py']:.3f} (struct={prow['clean_struct_reward']:.2f}) | "
                f"photo_F1={prow['photo_note_f1_py']:.3f} (struct={prow['photo_struct_reward']:.2f})",
                flush=True,
            )
        piece_rows.append(prow)

    def _agg(rows: Sequence[Dict[str, Any]], domain: str) -> Dict[str, Any]:
        p = domain + "_"
        n = len(rows)
        if n == 0:
            return {"n": 0}

        def _m(key: str) -> Optional[float]:
            vals = [float(r[p + key]) for r in rows if r.get(p + key) is not None]
            return round(sum(vals) / len(vals), 6) if vals else None

        return {
            "n": n,
            "note_f1_py_mean": _m("note_f1_py"),
            "note_hr_ratio_mean": _m("note_hr_ratio"),
            "struct_reward_mean": _m("struct_reward"),
            "r_parse_mean": _m("r_parse"),
            "r_rhythm_mean": _m("r_rhythm"),
            "r_completeness_mean": _m("r_completeness"),
            "r_noloop_mean": _m("r_noloop"),
            "eos_rate": round(sum(1 for r in rows if r.get(p + "has_eos")) / n, 4),
        }

    pr_rows = [r for r in piece_rows if not r["handwritten"]]
    hw_rows = [r for r in piece_rows if r["handwritten"]]
    summary = {
        "printed_clean": _agg(pr_rows, "clean"),
        "printed_photo": _agg(pr_rows, "photo"),
        "hw_clean": _agg(hw_rows, "clean"),
        "hw_photo": _agg(hw_rows, "photo"),
        "all45_clean": _agg(piece_rows, "clean"),
        "all45_photo": _agg(piece_rows, "photo"),
    }

    print("=" * 96, flush=True)
    print(f"[eval:{tag}] Suite31_hw Summary ({time.perf_counter() - t_all:.1f}s)", flush=True)
    for k, s in summary.items():
        print(
            f"  {k:14s} (n={s['n']:2d}) | NoteF1_py={s['note_f1_py_mean']:.4f} | "
            f"StructR={s['struct_reward_mean']:.4f} | Parse={s['r_parse_mean']:.3f} | "
            f"Rhythm={s['r_rhythm_mean']:.3f} | NoLoop={s['r_noloop_mean']:.3f} | "
            f"EOS={s['eos_rate']:.2f}",
            flush=True,
        )
    print("=" * 96, flush=True)
    return {"summary": summary, "pieces": piece_rows, "elapsed_secs": round(time.perf_counter() - t_all, 2)}


@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def run_experiment(
    updates: int = 120,
    group_size: int = 4,
    max_len: int = 384,
    max_dim: int = 1024,
    lr: float = 1.0e-5,
    kl_beta: float = 0.04,
    clip_eps: float = 0.2,
    temperature: float = 0.8,
    top_p: float = 0.95,
    unlabeled_frac: float = 1.0,
    save_every_minutes: float = 5.0,
    eval_max_len: int = 1536,
    seed: int = 42,
) -> Dict[str, Any]:
    sys.path.insert(0, "/workspace/mvd")

    import copy
    from PIL import Image
    import torch

    import eval_uni_suite31 as eval_s31
    import rlvr_grpo
    import uni_omr

    t_start = time.perf_counter()
    started_at = _utc_now()
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
    use_bf16 = device.type == "cuda" and cc[0] >= 8
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(
        f"[exp5] Device={gpu_name} cc={cc} bf16={use_bf16} | updates={updates} "
        f"G={group_size} max_len={max_len} max_dim={max_dim} lr={lr} kl_beta={kl_beta}",
        flush=True,
    )

    ckpt_path, rism_dir, manifest_path = _ensure_assets_on_volume()
    rism_images = rlvr_grpo.collect_unlabeled_image_paths(rism_dir)
    if not rism_images:
        raise FileNotFoundError(f"No unlabeled RISM images found under {rism_dir}")
    print(f"[exp5] Found {len(rism_images)} unlabeled RISM pages under {rism_dir}", flush=True)

    ckpt_dir = Path("/vol/ckpts/exp5_rism_rlvr")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "uni_omr_rlvr_latest.pt"
    final_ckpt = ckpt_dir / "uni_omr_rlvr_final.pt"

    _upload_run_ledger(
        {
            "exp_id": EXP_ID,
            "status": "RUNNING",
            "started_at": started_at,
            "updated_at": _utc_now(),
            "platform": f"Modal L4 ({gpu_name}, cc={cc[0]}.{cc[1]})",
            "rism_pages": len(rism_images),
            "config": {
                "updates": updates,
                "group_size": group_size,
                "max_len": max_len,
                "max_dim": max_dim,
                "lr": lr,
                "kl_beta": kl_beta,
                "clip_eps": clip_eps,
                "temperature": temperature,
                "top_p": top_p,
                "unlabeled_frac": unlabeled_frac,
            },
        },
        commit_msg=f"exp5: mark {EXP_ID} RUNNING",
    )

    # Load baseline model and frozen reference policy
    model, cfg, ckpt_meta = eval_s31.load_uni_checkpoint(ckpt_path, device=device)
    ref_model = copy.deepcopy(model).to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    _, _, _, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))

    # 1. Baseline eval BEFORE RLVR (cached on volume if resuming)
    baseline_cache = ckpt_dir / "baseline_eval.json"
    if baseline_cache.exists():
        print(f"[exp5] Loading cached baseline eval from {baseline_cache}", flush=True)
        baseline_eval = json.loads(baseline_cache.read_text(encoding="utf-8"))
    else:
        print("[exp5] Running BEFORE-RLVR 45-piece suite31_hw eval on Iter-10 soup ...", flush=True)
        baseline_eval = _evaluate_suite31_python(
            model,
            cfg,
            manifest_path,
            device,
            tag="before_rlvr",
            max_dim=max_dim,
            max_len=eval_max_len,
            use_bf16=use_bf16,
        )
        baseline_cache.write_text(json.dumps(baseline_eval, indent=2), encoding="utf-8")
        vol.commit()

    # 2. Setup optimizer & optional resume from 5-min checkpoint
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    start_step = 0
    history: List[Dict[str, Any]] = []

    if latest_ckpt.exists():
        print(f"[exp5] Resuming from {latest_ckpt} ...", flush=True)
        resume_ck = torch.load(str(latest_ckpt), map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ck["model"], strict=False)
        if "optimizer" in resume_ck:
            optimizer.load_state_dict(resume_ck["optimizer"])
        start_step = int(resume_ck.get("step", 0))
        history = list(resume_ck.get("rlvr_history", []))
        print(f"[exp5] Resumed at step {start_step}/{updates}", flush=True)

    rng = random.Random(seed)
    order = list(range(len(rism_images)))
    rng.shuffle(order)

    grpo_cfg = rlvr_grpo.GRPOConfig(
        group_size=group_size,
        max_len=max_len,
        temperature=temperature,
        top_p=top_p,
        clip_eps=clip_eps,
        kl_beta=kl_beta,
        use_grammar=True,
        strict_grammar=False,
        unlabeled_dir=str(rism_dir),
        unlabeled_frac=unlabeled_frac,
    )

    def _save_atomic(path: Path, step_num: int) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": asdict(cfg),
                "step": step_num,
                "optimizer": optimizer.state_dict(),
                "rlvr_history": history,
                "base_ckpt": "checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
            },
            str(tmp),
        )
        os.replace(str(tmp), str(path))
        vol.commit()
        print(f"[ckpt] Atomically saved {path.name} @ step={step_num}", flush=True)

    last_ckpt_t = time.perf_counter()
    last_hf_t = time.perf_counter()
    warmup_steps = min(10, max(1, updates // 10))
    min_lr = lr * 0.2

    print(f"[exp5] Starting {updates - start_step} unlabeled-RISM GRPO updates ...", flush=True)
    for step in range(start_step + 1, updates + 1):
        # Warmup + cosine LR schedule
        if step <= warmup_steps:
            cur_lr = lr * (step / warmup_steps)
        else:
            prog = (step - warmup_steps) / max(1, updates - warmup_steps)
            cur_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * prog))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        img_path = rism_images[order[(step - 1) % len(order)]]
        try:
            with Image.open(img_path) as raw_im:
                pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
            img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
        except Exception as exc:
            print(f"[step {step:03d}] Skipping corrupt image {img_path.name}: {exc}", flush=True)
            continue

        model.train()
        optimizer.zero_grad(set_to_none=True)
        t_step = time.perf_counter()

        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, stats = rlvr_grpo.grpo_step(
                    model,
                    ref_model,
                    img_t,
                    [None],  # 100% label-free unlabeled structural-validity RLVR
                    cfg=grpo_cfg,
                    unlabeled_vocab=id_to_tok,
                )
        else:
            loss, stats = rlvr_grpo.grpo_step(
                model,
                ref_model,
                img_t,
                [None],
                cfg=grpo_cfg,
                unlabeled_vocab=id_to_tok,
            )

        updated = False
        grad_norm = 0.0
        if torch.isfinite(loss) and loss.requires_grad:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm = float(gn.detach().item()) if torch.isfinite(gn) else float("nan")
            if math.isfinite(grad_norm):
                optimizer.step()
                updated = True
            else:
                optimizer.zero_grad(set_to_none=True)

        dt = time.perf_counter() - t_step
        row = {
            "step": step,
            "image": img_path.name,
            "lr": round(cur_lr, 8),
            "loss": round(float(stats["loss"]), 5),
            "pg_loss": round(float(stats["pg_loss"]), 5),
            "kl_loss": round(float(stats["kl_loss"]), 5),
            "mean_reward": round(float(stats["mean_reward"]), 4),
            "std_reward": round(float(stats["std_reward"]), 4),
            "r_parse": round(float(stats["mean_r_parse"]), 4),
            "r_rhythm": round(float(stats["mean_r_rhythm"]), 4),
            "r_completeness": round(float(stats["mean_r_completeness"]), 4),
            "r_noloop": round(float(stats["mean_r_noloop"]), 4),
            "seq_len": int(stats["seq_len"]),
            "grad_norm": round(grad_norm, 4) if math.isfinite(grad_norm) else None,
            "updated": updated,
            "step_secs": round(dt, 2),
        }
        history.append(row)

        if step == 1 or step % 5 == 0 or step == updates:
            print(
                f"[grpo {step:03d}/{updates:03d}] loss={row['loss']:+.4f} "
                f"(pg={row['pg_loss']:+.4f} kl={row['kl_loss']:.4f}) | "
                f"R={row['mean_reward']:.3f}±{row['std_reward']:.3f} "
                f"(parse={row['r_parse']:.2f} rhythm={row['r_rhythm']:.2f} "
                f"comp={row['r_completeness']:.2f} noloop={row['r_noloop']:.2f}) | "
                f"L={row['seq_len']:3d} lr={cur_lr:.2e} {dt:.1f}s",
                flush=True,
            )

        now = time.perf_counter()
        if (now - last_ckpt_t) >= save_every_minutes * 60.0:
            _save_atomic(latest_ckpt, step)
            last_ckpt_t = now
        if (now - last_hf_t) >= 15.0 * 60.0:
            _upload_run_ledger(
                {
                    "exp_id": EXP_ID,
                    "status": "RUNNING",
                    "started_at": started_at,
                    "updated_at": _utc_now(),
                    "step": step,
                    "updates": updates,
                    "recent_mean_reward": round(
                        sum(h["mean_reward"] for h in history[-10:]) / max(1, len(history[-10:])), 4
                    ),
                    "recent_r_rhythm": round(
                        sum(h["r_rhythm"] for h in history[-10:]) / max(1, len(history[-10:])), 4
                    ),
                },
                commit_msg=f"exp5: progress step {step}/{updates}",
            )
            last_hf_t = now

    # Save final checkpoint
    _save_atomic(latest_ckpt, updates)
    _save_atomic(final_ckpt, updates)

    # 3. Post-RLVR 45-piece suite31_hw eval
    print("[exp5] Running AFTER-RLVR 45-piece suite31_hw eval ...", flush=True)
    post_eval = _evaluate_suite31_python(
        model,
        cfg,
        manifest_path,
        device,
        tag="after_rlvr",
        max_dim=max_dim,
        max_len=eval_max_len,
        use_bf16=use_bf16,
    )

    # Compute before-vs-after deltas
    deltas: Dict[str, Dict[str, Optional[float]]] = {}
    for sl in ("printed_clean", "printed_photo", "hw_clean", "hw_photo", "all45_clean", "all45_photo"):
        b = baseline_eval["summary"][sl]
        a = post_eval["summary"][sl]
        deltas[sl] = {
            "note_f1_py_delta": round(a["note_f1_py_mean"] - b["note_f1_py_mean"], 6)
            if (a.get("note_f1_py_mean") is not None and b.get("note_f1_py_mean") is not None)
            else None,
            "struct_reward_delta": round(a["struct_reward_mean"] - b["struct_reward_mean"], 6)
            if (a.get("struct_reward_mean") is not None and b.get("struct_reward_mean") is not None)
            else None,
            "r_rhythm_delta": round(a["r_rhythm_mean"] - b["r_rhythm_mean"], 6)
            if (a.get("r_rhythm_mean") is not None and b.get("r_rhythm_mean") is not None)
            else None,
            "r_noloop_delta": round(a["r_noloop_mean"] - b["r_noloop_mean"], 6)
            if (a.get("r_noloop_mean") is not None and b.get("r_noloop_mean") is not None)
            else None,
        }

    total_secs = round(time.perf_counter() - t_start, 2)
    report = {
        "exp_id": EXP_ID,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "total_elapsed_secs": total_secs,
        "platform": f"Modal L4 ({gpu_name}, cc={cc[0]}.{cc[1]})",
        "base_ckpt": "checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
        "base_ckpt_meta": ckpt_meta,
        "rism_pages_total": len(rism_images),
        "grpo_config": {
            "updates": updates,
            "group_size": group_size,
            "max_len": max_len,
            "max_dim": max_dim,
            "lr": lr,
            "kl_beta": kl_beta,
            "clip_eps": clip_eps,
            "temperature": temperature,
            "top_p": top_p,
            "unlabeled_frac": unlabeled_frac,
            "seed": seed,
        },
        "baseline_summary": baseline_eval["summary"],
        "post_rlvr_summary": post_eval["summary"],
        "deltas": deltas,
        "training_history": history,
        "baseline_pieces": baseline_eval["pieces"],
        "post_rlvr_pieces": post_eval["pieces"],
    }

    eval_out = ckpt_dir / f"{EXP_ID}.json"
    eval_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    vol.commit()

    # Upload checkpoint, eval JSON, and final run ledger to Hugging Face
    _upload_hf_file(
        final_ckpt,
        f"checkpoints/{EXP_ID}/uni_omr_rlvr_final.pt",
        f"exp5: upload final RLVR checkpoint ({updates} updates)",
    )
    _upload_hf_file(
        eval_out,
        f"evals/{EXP_ID}.json",
        f"exp5: upload before-vs-after 45-piece suite31_hw eval",
    )
    _upload_run_ledger(
        {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_at": started_at,
            "completed_at": report["completed_at"],
            "total_elapsed_secs": total_secs,
            "platform": report["platform"],
            "baseline_summary": baseline_eval["summary"],
            "post_rlvr_summary": post_eval["summary"],
            "deltas": deltas,
            "hf_checkpoint": f"checkpoints/{EXP_ID}/uni_omr_rlvr_final.pt",
            "hf_eval": f"evals/{EXP_ID}.json",
        },
        commit_msg=f"exp5: COMPLETED {EXP_ID}",
    )

    print("[exp5] DONE! Deltas:", json.dumps(deltas, indent=2), flush=True)
    return {
        "exp_id": EXP_ID,
        "total_elapsed_secs": total_secs,
        "deltas": deltas,
        "baseline_summary": baseline_eval["summary"],
        "post_rlvr_summary": post_eval["summary"],
    }


@app.local_entrypoint()
def main(
    updates: int = 120,
    group_size: int = 4,
    max_len: int = 384,
    lr: float = 1.0e-5,
    kl_beta: float = 0.04,
) -> None:
    res = run_experiment.remote(
        updates=updates,
        group_size=group_size,
        max_len=max_len,
        lr=lr,
        kl_beta=kl_beta,
    )
    print(json.dumps(res, indent=2))

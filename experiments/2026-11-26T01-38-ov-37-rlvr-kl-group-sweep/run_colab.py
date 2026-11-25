#!/usr/bin/env python3
"""Overnight Exp 37 (Batch 09, Wave 7) on Google Colab L4:
Beat-Conservation + <sys>-Loop RLVR kl_beta ({0.02, 0.06, 0.15}) x Group Size (G in {4, 8}) Sweep.

Sweeps 6 label-free GRPO arms on unlabeled RISM manuscripts (`datasets/rism_hw_5707.tar`)
starting from `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`, evaluating `suite31_hw`
HW gain vs printed retention on every arm and running a full 45-piece confirmation eval
on the winning arm.

Expects:
  - /content/bundle.tar.gz (or pre-extracted /content/workspace/sheet-music-converter/tmp-train/mvd)
  - /tmp/.hf_token (Hugging Face token; chmod 600, deleted immediately and in finally)
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime, timezone
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

EXP_ID = "2026-11-26T01-38-ov-37-rlvr-kl-group-sweep"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_PATH = "checkpoints/iter10_soup_baseline/uni_omr_soup.pt"
HF_RISM_TAR = "datasets/rism_hw_5707.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_BEST_CKPT_PATH = f"checkpoints/{EXP_ID}/best_rlvr_arm.pt"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

KL_BETAS = [0.02, 0.06, 0.15]
GROUP_SIZES = [4, 8]


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
        print(f"[{EXP_ID}] Downloading {HF_CKPT_PATH}...", flush=True)
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=HF_CKPT_PATH,
            local_dir=str(CACHE_ROOT),
            token=hf_token,
        )

    rism_dir = CACHE_ROOT / "datasets" / "rism_hw"
    if not (rism_dir / ".extracted").exists():
        print(f"[{EXP_ID}] Downloading {HF_RISM_TAR}...", flush=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=HF_RISM_TAR,
                local_dir=str(CACHE_ROOT / "tars"),
                token=hf_token,
            )
        )
        _extract_tar_once(tar_p, rism_dir)
        tar_p.unlink(missing_ok=True)

    s31_dir = CACHE_ROOT / "datasets" / "suite31_hw_eval"
    if not (s31_dir / ".extracted").exists():
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
        is_rest = False
        is_grace = False
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


def _evaluate_pieces_python(
    model: Any,
    cfg: Any,
    pieces: Sequence[Dict[str, Any]],
    device: Any,
    *,
    tag: str,
    max_dim: int = 1024,
    max_len: int = 1536,
    use_bf16: bool = True,
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
        slug = piece["slug"]
        is_hw = bool(piece["handwritten"])
        ref_notes = _extract_musicxml_notes(piece["gt"])
        prow: Dict[str, Any] = {
            "piece": slug,
            "handwritten": is_hw,
            "stratum": piece.get("stratum", ""),
            "ref_notes": len(ref_notes),
        }
        for domain in ("clean", "photo"):
            with Image.open(piece[domain]) as raw_im:
                pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
            img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
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

            pfx = domain + "_"
            prow[pfx + "note_f1_py"] = round(float(note_f1), 6)
            prow[pfx + "struct_reward"] = round(float(struct["reward"]), 6)
            prow[pfx + "r_parse"] = round(float(struct["r_parse"]), 4)
            prow[pfx + "r_rhythm"] = round(float(struct["r_rhythm"]), 4)
            prow[pfx + "r_noloop"] = round(float(struct["r_noloop"]), 4)
            prow[pfx + "has_eos"] = bool(has_eos)
            prow[pfx + "n_sys"] = int(n_sys)
            prow[pfx + "elapsed_secs"] = round(elapsed, 3)
            prow[pfx + "mtp_acc"] = round(float(mtp_stats["acceptance_rate"]), 4)

        rows.append(prow)
        if idx % 10 == 0 or idx == len(pieces):
            print(f"[eval:{tag}] {idx:02d}/{len(pieces):02d} done ({time.perf_counter() - t_all:.1f}s)", flush=True)

    def _agg(sub: Sequence[Dict[str, Any]], dom: str) -> Dict[str, Any]:
        p = dom + "_"
        n = len(sub)
        if n == 0:
            return {"n": 0}
        def _m(k: str) -> Optional[float]:
            vals = [float(r[p + k]) for r in sub if r.get(p + k) is not None]
            return round(sum(vals) / len(vals), 6) if vals else None
        return {
            "n": n,
            "note_f1_py_mean": _m("note_f1_py"),
            "struct_reward_mean": _m("struct_reward"),
            "r_parse_mean": _m("r_parse"),
            "r_rhythm_mean": _m("r_rhythm"),
            "r_noloop_mean": _m("r_noloop"),
            "eos_rate": round(sum(1 for r in sub if r.get(p + "has_eos")) / n, 4),
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


def _compute_deltas(base_sum: Dict[str, Any], arm_sum: Dict[str, Any]) -> Dict[str, Any]:
    deltas: Dict[str, Any] = {}
    for sl in ("printed_clean", "printed_photo", "hw_clean", "hw_photo"):
        b = base_sum.get(sl, {})
        a = arm_sum.get(sl, {})
        deltas[sl] = {
            "note_f1_py_delta": round(a["note_f1_py_mean"] - b["note_f1_py_mean"], 6)
            if (a.get("note_f1_py_mean") is not None and b.get("note_f1_py_mean") is not None)
            else None,
            "r_rhythm_delta": round(a["r_rhythm_mean"] - b["r_rhythm_mean"], 6)
            if (a.get("r_rhythm_mean") is not None and b.get("r_rhythm_mean") is not None)
            else None,
            "r_noloop_delta": round(a["r_noloop_mean"] - b["r_noloop_mean"], 6)
            if (a.get("r_noloop_mean") is not None and b.get("r_noloop_mean") is not None)
            else None,
        }
    return deltas


def run_sweep(
    updates: int = 40,
    max_len: int = 256,
    max_dim: int = 1024,
    lr: float = 1.0e-5,
    clip_eps: float = 0.2,
    temperature: float = 0.8,
    top_p: float = 0.95,
    eval_max_len: int = 1536,
    save_every_minutes: float = 5.0,
    seed: int = 42,
) -> Dict[str, Any]:
    try:
        hf_token = _read_and_purge_hf_token()
        mvd_dir = _setup_environment()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        from huggingface_hub import HfApi
        from PIL import Image
        import torch
        import eval_uni_suite31 as eval_s31
        import rlvr_grpo
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
        ckpt_path, rism_dir, manifest_path = _ensure_assets(hf_token)
        rism_images = rlvr_grpo.collect_unlabeled_image_paths(rism_dir)
        _, all_pieces = eval_s31.resolve_manifest_pieces(manifest_path)
        sweep_pieces = _select_sweep_pieces(all_pieces, n_printed=12)

        rng = random.Random(seed)
        page_order = list(range(len(rism_images)))
        rng.shuffle(page_order)

        # Upload RUNNING ledger
        ledger_tmp = CACHE_ROOT / "run_ledger.json"
        _atomic_write_json(
            ledger_tmp,
            {
                "exp_id": EXP_ID,
                "status": "RUNNING",
                "started_utc": started_utc,
                "platform": platform_str,
                "grid": {"kl_betas": KL_BETAS, "group_sizes": GROUP_SIZES, "updates": updates},
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

        model, cfg, ckpt_meta = eval_s31.load_uni_checkpoint(ckpt_path, device=device)
        ref_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ref_model = copy.deepcopy(model).to(device).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        _, _, _, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))

        partial_path = CACHE_ROOT / "evals" / f"{EXP_ID}_partial.json"
        arms_done: Dict[str, Any] = {}
        baseline_eval: Optional[Dict[str, Any]] = None
        if partial_path.exists():
            try:
                prev = json.loads(partial_path.read_text(encoding="utf-8"))
                baseline_eval = prev.get("baseline_eval")
                arms_done = dict(prev.get("arms", {}))
                print(f"[{EXP_ID}] Resuming: {len(arms_done)}/6 arms cached", flush=True)
            except Exception:
                pass

        if baseline_eval is None:
            print(f"[{EXP_ID}] Running baseline eval on {len(sweep_pieces)} pieces...", flush=True)
            baseline_eval = _evaluate_pieces_python(
                model, cfg, sweep_pieces, device,
                tag="baseline", max_dim=max_dim, max_len=eval_max_len, use_bf16=use_bf16,
            )

        best_score = -1e9
        best_arm_id: Optional[str] = None
        best_arm_state: Optional[Dict[str, Any]] = None
        last_save_t = time.perf_counter()

        for kl_beta in KL_BETAS:
            for g_size in GROUP_SIZES:
                arm_id = f"kl{kl_beta:.2f}_G{g_size}"
                if arm_id in arms_done:
                    print(f"[{EXP_ID}] Skipping cached arm {arm_id}", flush=True)
                    continue

                print(f"[{EXP_ID}] === Arm {arm_id} (updates={updates}, lr={lr}) ===", flush=True)
                torch.manual_seed(seed)
                random.seed(seed)
                model.load_state_dict({k: v.to(device) for k, v in ref_state_cpu.items()})
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
                grpo_cfg = rlvr_grpo.GRPOConfig(
                    group_size=g_size,
                    max_len=max_len,
                    temperature=temperature,
                    top_p=top_p,
                    clip_eps=clip_eps,
                    kl_beta=kl_beta,
                    use_grammar=True,
                    strict_grammar=False,
                    unlabeled_dir=str(rism_dir),
                    unlabeled_frac=1.0,
                )

                warmup = max(1, updates // 10)
                min_lr = lr * 0.2
                history: List[Dict[str, Any]] = []
                t_arm = time.perf_counter()

                for step in range(1, updates + 1):
                    if step <= warmup:
                        cur_lr = lr * (step / warmup)
                    else:
                        prog = (step - warmup) / max(1, updates - warmup)
                        cur_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * prog))
                    for pg in optimizer.param_groups:
                        pg["lr"] = cur_lr

                    img_path = rism_images[page_order[(step - 1) % len(page_order)]]
                    try:
                        with Image.open(img_path) as raw_im:
                            pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
                        img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
                    except Exception:
                        continue

                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    if use_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss, stats = rlvr_grpo.grpo_step(
                                model, ref_model, img_t, [None], cfg=grpo_cfg, unlabeled_vocab=id_to_tok
                            )
                    else:
                        loss, stats = rlvr_grpo.grpo_step(
                            model, ref_model, img_t, [None], cfg=grpo_cfg, unlabeled_vocab=id_to_tok
                        )

                    if torch.isfinite(loss) and loss.requires_grad:
                        loss.backward()
                        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if math.isfinite(float(gn.detach().item())):
                            optimizer.step()
                        else:
                            optimizer.zero_grad(set_to_none=True)

                    history.append({
                        "step": step,
                        "loss": round(float(stats["loss"]), 5),
                        "kl_loss": round(float(stats["kl_loss"]), 5),
                        "mean_reward": round(float(stats["mean_reward"]), 4),
                        "r_rhythm": round(float(stats["mean_r_rhythm"]), 4),
                        "r_noloop": round(float(stats["mean_r_noloop"]), 4),
                        "clip_frac": round(float(stats["clip_frac"]), 4),
                    })
                    if step == 1 or step % 10 == 0 or step == updates:
                        h = history[-1]
                        print(
                            f"[{arm_id} {step:02d}/{updates:02d}] loss={h['loss']:+.4f} kl={h['kl_loss']:.4f} "
                            f"R={h['mean_reward']:.3f} rhythm={h['r_rhythm']:.2f} noloop={h['r_noloop']:.2f}",
                            flush=True,
                        )

                arm_eval = _evaluate_pieces_python(
                    model, cfg, sweep_pieces, device,
                    tag=arm_id, max_dim=max_dim, max_len=eval_max_len, use_bf16=use_bf16,
                )
                deltas = _compute_deltas(baseline_eval["summary"], arm_eval["summary"])
                hw_gain = 0.5 * (
                    (deltas["hw_clean"]["note_f1_py_delta"] or 0.0)
                    + (deltas["hw_photo"]["note_f1_py_delta"] or 0.0)
                )
                pr_drop = min(0.0, deltas["printed_clean"]["note_f1_py_delta"] or 0.0)
                composite = hw_gain + 0.5 * pr_drop

                arms_done[arm_id] = {
                    "kl_beta": kl_beta,
                    "group_size": g_size,
                    "train_elapsed_secs": round(time.perf_counter() - t_arm, 2),
                    "summary": arm_eval["summary"],
                    "deltas": deltas,
                    "composite_score": round(composite, 6),
                    "mean_final_kl": round(sum(x["kl_loss"] for x in history[-10:]) / max(1, len(history[-10:])), 5),
                    "mean_final_reward": round(sum(x["mean_reward"] for x in history[-10:]) / max(1, len(history[-10:])), 4),
                    "history": history,
                }
                if composite > best_score:
                    best_score = composite
                    best_arm_id = arm_id
                    best_arm_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if (time.perf_counter() - last_save_t) >= save_every_minutes * 60.0 or True:
                    _atomic_write_json(
                        partial_path,
                        {
                            "exp_id": EXP_ID,
                            "status": "RUNNING",
                            "baseline_eval": baseline_eval,
                            "arms": arms_done,
                        },
                    )
                    last_save_t = time.perf_counter()

        # Full 45-piece confirmation eval on best arm
        winner_full45: Optional[Dict[str, Any]] = None
        uploaded_ckpt: Optional[str] = None
        if best_arm_id is not None and best_arm_state is not None:
            print(f"[{EXP_ID}] Running full 45-piece suite31_hw eval on winner {best_arm_id}...", flush=True)
            model.load_state_dict({k: v.to(device) for k, v in best_arm_state.items()})
            winner_full45 = _evaluate_pieces_python(
                model, cfg, all_pieces, device,
                tag=f"winner_{best_arm_id}_all45", max_dim=max_dim, max_len=eval_max_len, use_bf16=use_bf16,
            )
            win_d = arms_done[best_arm_id]["deltas"]
            hw_c_d = win_d["hw_clean"]["note_f1_py_delta"] or 0.0
            hw_p_d = win_d["hw_photo"]["note_f1_py_delta"] or 0.0
            pr_c_d = win_d["printed_clean"]["note_f1_py_delta"] or 0.0
            if (hw_c_d >= 0.015 or hw_p_d >= 0.015) and pr_c_d >= -0.010:
                best_ckpt_local = CACHE_ROOT / "best_rlvr_arm.pt"
                torch.save(
                    {"model": best_arm_state, "config": asdict(cfg), "arm_id": best_arm_id, "exp_id": EXP_ID},
                    str(best_ckpt_local),
                )
                api.upload_file(
                    path_or_fileobj=str(best_ckpt_local),
                    path_in_repo=HF_BEST_CKPT_PATH,
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    commit_message=f"{EXP_ID}: upload promoted RLVR checkpoint ({best_arm_id})",
                )
                uploaded_ckpt = HF_BEST_CKPT_PATH

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
            "config": {
                "kl_betas": KL_BETAS,
                "group_sizes": GROUP_SIZES,
                "updates": updates,
                "lr": lr,
                "max_len": max_len,
                "max_dim": max_dim,
                "eval_max_len": eval_max_len,
                "seed": seed,
            },
            "baseline_summary": baseline_eval["summary"],
            "arms": arms_done,
            "best_arm_id": best_arm_id,
            "winner_full45_summary": winner_full45["summary"] if winner_full45 else None,
            "uploaded_checkpoint": uploaded_ckpt,
        }
        final_local = CACHE_ROOT / "evals" / f"{EXP_ID}.json"
        _atomic_write_json(final_local, final_report)

        api.upload_file(
            path_or_fileobj=str(final_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload RLVR kl_beta x G sweep eval JSON",
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
                "best_arm_id": best_arm_id,
                "hf_eval_path": HF_EVAL_PATH,
                "uploaded_checkpoint": uploaded_ckpt,
            },
        )
        api.upload_file(
            path_or_fileobj=str(ledger_tmp),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark COMPLETED",
        )
        print(f"[{EXP_ID}] COMPLETED in {total_secs}s | best_arm={best_arm_id}", flush=True)
        return final_report
    finally:
        if TOKEN_PATH.exists():
            try:
                TOKEN_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    run_sweep()

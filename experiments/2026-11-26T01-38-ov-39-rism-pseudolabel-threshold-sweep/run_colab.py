#!/usr/bin/env python3
"""Overnight Exp 39 (Batch 09, Wave 7) on Google Colab L4:
Confidence- & Beat-Gated RISM Pseudo-Labeling Threshold Sweep:
  1. Generate FSA-constrained pseudo-labels on 300 unlabeled RISM pages from `uni_omr_soup.pt`
     with exact per-token `mean_logprob` and `beat_valid_frac` (`r_rhythm`).
  2. Sweep 3x3 filtering thresholds:
       mean_logprob >= {-0.35, -0.20, -0.10} x beat_valid_frac >= {0.70, 0.85, 0.95}
     and record full yield & quality statistics for all 9 cells.
  3. Run downstream noisy-student fine-tuning (60 optimizer steps) on up to 5 distinct
     qualifying threshold arms and measure `suite31_hw` HW lift vs printed retention.

Expects:
  - /content/bundle.tar.gz (or pre-extracted /content/workspace/sheet-music-converter/tmp-train/mvd)
  - /tmp/.hf_token (Hugging Face token; chmod 600, deleted immediately and in finally)
"""

from __future__ import annotations

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

EXP_ID = "2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_PATH = "checkpoints/iter10_soup_baseline/uni_omr_soup.pt"
HF_RISM_TAR = "datasets/rism_hw_5707.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

LOGPROB_THRESHOLDS = [-0.35, -0.20, -0.10]
BEAT_THRESHOLDS = [0.70, 0.85, 0.95]


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


def _select_eval_pieces(pieces: Sequence[Dict[str, Any]], n_printed: int = 12) -> List[Dict[str, Any]]:
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
        prow: Dict[str, Any] = {"piece": slug, "handwritten": is_hw}
        for domain in ("clean", "photo"):
            with Image.open(piece[domain]) as raw_im:
                pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
            img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
            with torch.no_grad():
                if use_bf16 and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        seq, _ = eval_s31.generate_mtp_spec(
                            model, img_t, max_len=max_len, use_grammar=True,
                            strict_grammar=False, return_stats=True,
                        )
                else:
                    seq, _ = eval_s31.generate_mtp_spec(
                        model, img_t, max_len=max_len, use_grammar=True,
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
        rows.append(prow)
        if idx % 10 == 0 or idx == len(pieces):
            print(f"[eval:{tag}] {idx:02d}/{len(pieces):02d} ({time.perf_counter() - t_all:.1f}s)", flush=True)

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
            "r_rhythm_mean": _m("r_rhythm"),
            "r_noloop_mean": _m("r_noloop"),
        }

    pr_rows = [r for r in rows if not r["handwritten"]]
    hw_rows = [r for r in rows if r["handwritten"]]
    return {
        "summary": {
            "printed_clean": _agg(pr_rows, "clean"),
            "printed_photo": _agg(pr_rows, "photo"),
            "hw_clean": _agg(hw_rows, "clean"),
            "hw_photo": _agg(hw_rows, "photo"),
        },
        "elapsed_secs": round(time.perf_counter() - t_all, 2),
    }


def _filter_records(
    records: Sequence[Dict[str, Any]],
    min_logprob: float,
    min_beat: float,
    min_tokens: int = 24,
) -> List[Dict[str, Any]]:
    return [
        r for r in records
        if (
            r.get("mean_logprob", -99.0) >= min_logprob
            and r.get("beat_valid_frac", 0.0) >= min_beat
            and r.get("r_parse", 0.0) >= 0.5
            and r.get("r_noloop", 0.0) >= 0.99
            and r.get("n_tokens", 0) >= min_tokens
        )
    ]


def run_sweep(
    n_rism_pages: int = 300,
    max_dim: int = 1024,
    gen_max_len: int = 1024,
    eval_max_len: int = 1536,
    ft_steps: int = 60,
    grad_accum: int = 4,
    ft_lr: float = 1.5e-5,
    max_ft_arms: int = 5,
    min_kept_for_ft: int = 6,
    save_every_minutes: float = 5.0,
    seed: int = 42,
) -> Dict[str, Any]:
    try:
        hf_token = _read_and_purge_hf_token()
        mvd_dir = _setup_environment()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        from huggingface_hub import HfApi
        from PIL import Image, ImageEnhance
        import torch
        import torch.nn.functional as F
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
        all_rism = rlvr_grpo.collect_unlabeled_image_paths(rism_dir)
        rng = random.Random(seed)
        rism_sample = list(all_rism)
        rng.shuffle(rism_sample)
        rism_sample = rism_sample[: min(int(n_rism_pages), len(rism_sample))]

        _, all_pieces = eval_s31.resolve_manifest_pieces(manifest_path)
        eval_pieces = _select_eval_pieces(all_pieces, n_printed=12)

        ledger_tmp = CACHE_ROOT / "run_ledger.json"
        _atomic_write_json(
            ledger_tmp,
            {
                "exp_id": EXP_ID,
                "status": "RUNNING",
                "started_utc": started_utc,
                "platform": platform_str,
                "n_rism_pages": len(rism_sample),
                "grid": {"logprob": LOGPROB_THRESHOLDS, "beat_valid_frac": BEAT_THRESHOLDS},
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
        base_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        pad_id, bos_id, eos_id, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))

        partial_path = CACHE_ROOT / "evals" / f"{EXP_ID}_partial.json"
        pseudo_records: List[Dict[str, Any]] = []
        decoded_paths: set[str] = set()
        baseline_eval: Optional[Dict[str, Any]] = None
        ft_arms_done: Dict[str, Any] = {}

        if partial_path.exists():
            try:
                prev = json.loads(partial_path.read_text(encoding="utf-8"))
                pseudo_records = list(prev.get("pseudo_records", []))
                decoded_paths = {r["image"] for r in pseudo_records}
                baseline_eval = prev.get("baseline_eval")
                ft_arms_done = dict(prev.get("ft_arms", {}))
                print(
                    f"[{EXP_ID}] Resuming: {len(pseudo_records)}/{len(rism_sample)} pseudo-labels, "
                    f"{len(ft_arms_done)} FT arms cached",
                    flush=True,
                )
            except Exception:
                pass

        # Stage 1: Generate FSA-constrained pseudo-labels on 300 RISM pages
        print(f"[{EXP_ID}] Stage 1: Decoding {len(rism_sample)} unlabeled RISM pages...", flush=True)
        model.eval()
        t_gen = time.perf_counter()
        last_save_t = time.perf_counter()

        for idx, img_path in enumerate(rism_sample, start=1):
            p_str = str(img_path)
            if p_str in decoded_paths:
                continue
            try:
                with Image.open(img_path) as raw_im:
                    pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
                img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
            except Exception as exc:
                print(f"[stage1] Skipping corrupt {img_path.name}: {exc}", flush=True)
                continue

            with torch.no_grad():
                if use_bf16 and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        seq, mtp_stats = eval_s31.generate_mtp_spec(
                            model, img_t, max_len=gen_max_len, use_grammar=True,
                            strict_grammar=False, return_stats=True,
                        )
                        # Compute exact mean per-token log-prob of the generated sequence
                        if seq.shape[1] >= 2:
                            out = model(img_t, seq[:, :-1], return_aux=False, return_mtp=False)
                            lp = F.log_softmax(out["logits"].float(), dim=-1)
                            tgt = seq[:, 1:]
                            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                            mean_lp = float(tok_lp.mean().item())
                        else:
                            mean_lp = -10.0
                else:
                    seq, mtp_stats = eval_s31.generate_mtp_spec(
                        model, img_t, max_len=gen_max_len, use_grammar=True,
                        strict_grammar=False, return_stats=True,
                    )
                    if seq.shape[1] >= 2:
                        out = model(img_t, seq[:, :-1], return_aux=False, return_mtp=False)
                        lp = F.log_softmax(out["logits"].float(), dim=-1)
                        tgt = seq[:, 1:]
                        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                        mean_lp = float(tok_lp.mean().item())
                    else:
                        mean_lp = -10.0

            seq_ids = seq[0].detach().cpu().tolist()
            has_eos = eos_id in seq_ids[1:]
            pred_raw = eval_s31.decode_uni_compat(
                seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
            )
            if not has_eos:
                pred_raw = eval_s31.trim_truncated_tokens(pred_raw)
            struct = rlvr_grpo.compute_unlabeled_structural_reward(
                pred_raw, vocab=id_to_tok, terminated_eos=has_eos
            )
            pseudo_records.append({
                "image": p_str,
                "name": img_path.name,
                "seq_ids": seq_ids,
                "n_tokens": len(pred_raw),
                "mean_logprob": round(mean_lp, 5),
                "beat_valid_frac": round(float(struct["r_rhythm"]), 5),
                "r_parse": round(float(struct["r_parse"]), 4),
                "r_noloop": round(float(struct["r_noloop"]), 4),
                "has_eos": bool(has_eos),
                "mtp_acc": round(float(mtp_stats["acceptance_rate"]), 4),
            })
            decoded_paths.add(p_str)

            if idx % 25 == 0 or idx == len(rism_sample):
                print(
                    f"[stage1] {idx:03d}/{len(rism_sample):03d} ({time.perf_counter() - t_gen:.1f}s) | "
                    f"last lp={mean_lp:.3f} beat={struct['r_rhythm']:.2f} n={len(pred_raw)}",
                    flush=True,
                )

            if (time.perf_counter() - last_save_t) >= save_every_minutes * 60.0:
                _atomic_write_json(
                    partial_path,
                    {
                        "exp_id": EXP_ID,
                        "status": "RUNNING",
                        "pseudo_records": pseudo_records,
                        "baseline_eval": baseline_eval,
                        "ft_arms": ft_arms_done,
                    },
                )
                last_save_t = time.perf_counter()

        # Stage 2: 3x3 Threshold Sweep Table
        threshold_grid: Dict[str, Any] = {}
        candidate_arms: List[Tuple[str, float, float, List[Dict[str, Any]]]] = []
        for lp_th in LOGPROB_THRESHOLDS:
            for beat_th in BEAT_THRESHOLDS:
                cell_id = f"lp{lp_th:.2f}_beat{beat_th:.2f}"
                kept = _filter_records(pseudo_records, min_logprob=lp_th, min_beat=beat_th)
                n_k = len(kept)
                threshold_grid[cell_id] = {
                    "min_logprob": lp_th,
                    "min_beat_valid_frac": beat_th,
                    "n_kept": n_k,
                    "yield_pct": round(100.0 * n_k / max(1, len(pseudo_records)), 2),
                    "mean_tokens": round(sum(r["n_tokens"] for r in kept) / n_k, 1) if n_k else 0.0,
                    "mean_logprob": round(sum(r["mean_logprob"] for r in kept) / n_k, 4) if n_k else None,
                    "mean_beat_valid_frac": round(sum(r["beat_valid_frac"] for r in kept) / n_k, 4) if n_k else None,
                    "eos_rate": round(sum(1 for r in kept if r["has_eos"]) / n_k, 4) if n_k else 0.0,
                }
                if n_k >= min_kept_for_ft:
                    candidate_arms.append((cell_id, lp_th, beat_th, kept))

        print(f"[{EXP_ID}] Stage 2: 3x3 Yield Grid:", json.dumps(threshold_grid, indent=2), flush=True)

        # Deduplicate arms with identical kept-image sets and pick up to max_ft_arms spanning yield range
        unique_arms: List[Tuple[str, float, float, List[Dict[str, Any]]]] = []
        seen_sets: set[Tuple[str, ...]] = set()
        for arm in candidate_arms:
            sig = tuple(sorted(r["image"] for r in arm[3]))
            if sig not in seen_sets:
                seen_sets.add(sig)
                unique_arms.append(arm)
        if len(unique_arms) > max_ft_arms:
            unique_arms.sort(key=lambda x: len(x[3]))
            idxs = [
                int(round(i * (len(unique_arms) - 1) / max(1, max_ft_arms - 1)))
                for i in range(max_ft_arms)
            ]
            unique_arms = [unique_arms[i] for i in sorted(set(idxs))]

        # Stage 3: Baseline eval + Downstream Fine-Tuning Lift
        if baseline_eval is None:
            print(f"[{EXP_ID}] Stage 3: Baseline eval on {len(eval_pieces)} pieces...", flush=True)
            baseline_eval = _evaluate_pieces_python(
                model, cfg, eval_pieces, device,
                tag="baseline", max_dim=max_dim, max_len=eval_max_len, use_bf16=use_bf16,
            )

        best_arm_id: Optional[str] = None
        best_hw_delta = -1e9

        for cell_id, lp_th, beat_th, kept in unique_arms:
            if cell_id in ft_arms_done:
                print(f"[{EXP_ID}] Skipping cached FT arm {cell_id}", flush=True)
                continue

            print(
                f"[{EXP_ID}] === FT Arm {cell_id} (n_kept={len(kept)}, steps={ft_steps}, lr={ft_lr}) ===",
                flush=True,
            )
            torch.manual_seed(seed)
            random.seed(seed)
            model.load_state_dict({k: v.to(device) for k, v in base_state_cpu.items()})
            optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=0.01)

            ft_rng = random.Random(seed)
            t_ft = time.perf_counter()
            losses: List[float] = []
            warmup = max(1, ft_steps // 10)
            min_lr = ft_lr * 0.2

            for step in range(1, ft_steps + 1):
                if step <= warmup:
                    cur_lr = ft_lr * (step / warmup)
                else:
                    prog = (step - warmup) / max(1, ft_steps - warmup)
                    cur_lr = min_lr + 0.5 * (ft_lr - min_lr) * (1.0 + math.cos(math.pi * prog))
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr

                model.train()
                optimizer.zero_grad(set_to_none=True)
                acc_loss = 0.0
                valid_micro = 0

                for _ in range(grad_accum):
                    rec = kept[ft_rng.randrange(len(kept))]
                    seq_ids = rec["seq_ids"]
                    if len(seq_ids) < 2:
                        continue
                    with Image.open(rec["image"]) as raw_im:
                        pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
                    # Mild noisy-student contrast/brightness jitter
                    pil_im = ImageEnhance.Contrast(pil_im).enhance(ft_rng.uniform(0.88, 1.12))
                    pil_im = ImageEnhance.Brightness(pil_im).enhance(ft_rng.uniform(0.94, 1.06))
                    img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
                    seq_t = torch.tensor([seq_ids], dtype=torch.long, device=device)

                    if use_bf16 and device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            out = model(img_t, seq_t[:, :-1], return_aux=False, return_mtp=False)
                            logits = out["logits"].float()
                            loss = F.cross_entropy(
                                logits.reshape(-1, logits.shape[-1]),
                                seq_t[:, 1:].reshape(-1),
                                ignore_index=pad_id,
                                label_smoothing=0.02,
                            ) / float(grad_accum)
                    else:
                        out = model(img_t, seq_t[:, :-1], return_aux=False, return_mtp=False)
                        logits = out["logits"].float()
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            seq_t[:, 1:].reshape(-1),
                            ignore_index=pad_id,
                            label_smoothing=0.02,
                        ) / float(grad_accum)

                    if torch.isfinite(loss):
                        loss.backward()
                        acc_loss += float(loss.detach().item())
                        valid_micro += 1

                if valid_micro > 0:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if math.isfinite(float(gn.detach().item())):
                        optimizer.step()
                    losses.append(round(acc_loss, 5))

                if step == 1 or step % 20 == 0 or step == ft_steps:
                    print(
                        f"[{cell_id} {step:02d}/{ft_steps:02d}] loss={losses[-1] if losses else 0.0:.4f} lr={cur_lr:.2e}",
                        flush=True,
                    )

            arm_eval = _evaluate_pieces_python(
                model, cfg, eval_pieces, device,
                tag=cell_id, max_dim=max_dim, max_len=eval_max_len, use_bf16=use_bf16,
            )
            deltas = {}
            for sl in ("printed_clean", "printed_photo", "hw_clean", "hw_photo"):
                b = baseline_eval["summary"][sl]["note_f1_py_mean"]
                a = arm_eval["summary"][sl]["note_f1_py_mean"]
                deltas[sl] = round(a - b, 6) if (a is not None and b is not None) else None

            hw_avg_delta = 0.5 * ((deltas["hw_clean"] or 0.0) + (deltas["hw_photo"] or 0.0))
            ft_arms_done[cell_id] = {
                "min_logprob": lp_th,
                "min_beat_valid_frac": beat_th,
                "n_kept": len(kept),
                "ft_elapsed_secs": round(time.perf_counter() - t_ft, 2),
                "final_loss": losses[-1] if losses else None,
                "summary": arm_eval["summary"],
                "deltas_note_f1_py": deltas,
                "hw_avg_delta": round(hw_avg_delta, 6),
            }
            if hw_avg_delta > best_hw_delta:
                best_hw_delta = hw_avg_delta
                best_arm_id = cell_id

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            _atomic_write_json(
                partial_path,
                {
                    "exp_id": EXP_ID,
                    "status": "RUNNING",
                    "pseudo_records": pseudo_records,
                    "baseline_eval": baseline_eval,
                    "ft_arms": ft_arms_done,
                },
            )

        total_secs = round(time.perf_counter() - t_start, 2)
        # Compact final report (omit raw seq_ids arrays to keep eval JSON lean)
        compact_records = [
            {k: v for k, v in r.items() if k != "seq_ids"} for r in pseudo_records
        ]
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
                "n_rism_pages": len(rism_sample),
                "logprob_thresholds": LOGPROB_THRESHOLDS,
                "beat_thresholds": BEAT_THRESHOLDS,
                "ft_steps": ft_steps,
                "grad_accum": grad_accum,
                "ft_lr": ft_lr,
                "max_dim": max_dim,
                "gen_max_len": gen_max_len,
                "eval_max_len": eval_max_len,
                "seed": seed,
            },
            "threshold_grid": threshold_grid,
            "baseline_summary": baseline_eval["summary"] if baseline_eval else None,
            "ft_arms": ft_arms_done,
            "best_arm_id": best_arm_id,
            "best_hw_avg_delta": round(best_hw_delta, 6) if best_arm_id else None,
            "pseudo_summary_records": compact_records,
        }
        final_local = CACHE_ROOT / "evals" / f"{EXP_ID}.json"
        _atomic_write_json(final_local, final_report)

        api.upload_file(
            path_or_fileobj=str(final_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload RISM pseudo-label threshold sweep eval JSON",
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
                "best_hw_avg_delta": final_report["best_hw_avg_delta"],
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

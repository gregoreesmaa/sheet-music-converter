#!/usr/bin/env python3
"""Google Colab L4 runner for i13-15: 2026-11-26T07-40-i13-15-confidence-selective-hires-redecode.

Commercial OMR Insight (Soundslice, PhotoScore, ScanScore, Newzik):
  80-90% of a scanned page is high-confidence and should stay on the fast 1x path,
  while errors concentrate in low-confidence / rhythm-inconsistent regions.
  Instead of paying 3x compute on every page (full multi-scale TTA), we use exact
  teacher-forced log-prob + FSA beat violation triage (~30 ms/page) to selectively
  re-decode ONLY flagged low-confidence pages or systems at higher resolution (1536px).

Also logs Soundslice/PhotoScore Suspect-Page Calibration (AUROC of confidence
flagger for predicting Strict Note F1 < 0.85).

Expects on Colab VM:
  - `/content/bundle.tar.gz` (contains `sheet-music-converter/...` and `mscz-rs/`)
  - `/tmp/.hf_token` (Hugging Face token; chmod 600, deleted immediately and in `finally:`)
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T07-40-i13-15-confidence-selective-hires-redecode"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_CKPT_PATH = os.environ.get("HF_CKPT_PATH", "checkpoints/iter10_soup_baseline/uni_omr_soup.pt")
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

BASE_DIM = 1280
HIRES_DIM = 1536
LORES_DIM = 1152
MAX_LEN = 3840
SAVE_EVERY_MINUTES = 5.0


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str] | str, *, shell: bool = False, env: Optional[Dict[str, str]] = None) -> None:
    print(f"[{EXP_ID}] $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd, shell=shell, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {cmd}")


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and HF_TOKEN_FILE.exists():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        token = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var")
    return token


def _find_repo_root() -> Path:
    for cand in (
        WORK_ROOT / "sheet-music-converter",
        Path("/workspace/sheet-music-converter"),
        Path("/content/sheet-music-converter"),
    ):
        if (cand / "tmp-train" / "mvd" / "eval_uni_suite31.py").exists():
            return cand
    if BUNDLE_PATH.exists():
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(BUNDLE_PATH, "r:*") as tf:
            tf.extractall(path=WORK_ROOT)
        repo = WORK_ROOT / "sheet-music-converter"
        if (repo / "tmp-train" / "mvd" / "eval_uni_suite31.py").exists():
            return repo
    raise FileNotFoundError(f"Could not locate sheet-music-converter workspace or bundle at {BUNDLE_PATH}")


def _setup_environment() -> Tuple[Path, Path, Path]:
    repo_root = _find_repo_root()
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    needed_pkgs = []
    for mod, pkg in [("huggingface_hub", "huggingface_hub>=0.26"), ("PIL", "pillow"), ("numpy", "numpy")]:
        try:
            __import__(mod)
        except ImportError:
            needed_pkgs.append(pkg)
    if needed_pkgs:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed_pkgs])

    cargo_bin_dir = Path.home() / ".cargo" / "bin"
    env = os.environ.copy()
    env["PATH"] = f"{cargo_bin_dir}:{env.get('PATH', '')}"

    aser_manifest = repo_root / "converter" / "assembly-serialize" / "Cargo.toml"
    eharness_manifest = repo_root / "converter" / "eval-harness" / "Cargo.toml"
    aser_bin = aser_manifest.parent / "target" / "release" / "assembly-serialize"
    eharness_bin = eharness_manifest.parent / "target" / "release" / "eval-harness"

    if not aser_bin.exists() or not eharness_bin.exists():
        if shutil.which("cargo", path=env["PATH"]) is None:
            _run_cmd(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
                "| sh -s -- -y --default-toolchain stable --profile minimal",
                shell=True,
                env=env,
            )
        if not aser_bin.exists():
            _run_cmd(["cargo", "build", "--release", "--manifest-path", str(aser_manifest)], env=env)
        if not eharness_bin.exists():
            _run_cmd(["cargo", "build", "--release", "--manifest-path", str(eharness_manifest)], env=env)

    assert aser_bin.exists(), f"Missing Linux binary: {aser_bin}"
    assert eharness_bin.exists(), f"Missing Linux binary: {eharness_bin}"
    return repo_root, aser_bin, eharness_bin


def _compute_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Exact rank-based AUROC (labels=1 for low-F1 suspect pages)."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _teacher_force_confidence(model: Any, img_t: Any, seq_ids: List[int]) -> Tuple[float, float, List[float]]:
    """Compute exact per-token log-probs & mean entropy (~30 ms/page)."""
    import torch

    if len(seq_ids) < 2:
        return -10.0, 10.0, []
    seq_t = torch.tensor([seq_ids], dtype=torch.long, device=img_t.device)
    tf_out = model(img_t, seq_t[:, :-1], return_aux=False, return_mtp=False)
    logits_f = tf_out["logits"].float()
    log_probs = torch.log_softmax(logits_f, dim=-1)
    tok_lp = log_probs.gather(-1, seq_t[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
    probs = log_probs.exp()
    mean_ent = float(-(probs * log_probs).sum(dim=-1).mean().item())
    lp_list = tok_lp.detach().cpu().tolist()
    return float(sum(lp_list) / max(1, len(lp_list))), mean_ent, lp_list


def _surgical_system_hires_redecode(
    model: Any,
    img_hires_t: Any,
    base_seq_ids: List[int],
    tok_lp_list: List[float],
    *,
    sys_lp_thresh: float,
    max_systems: int,
    ev_mod: Any,
    grammar_mod: Any,
) -> Tuple[List[int], int]:
    """Re-decode ONLY flagged low-confidence systems at 1536px via KV prefix prefill."""
    import torch

    device = img_hires_t.device
    vocab_size = int(model.cfg.vocab_size)
    _, bos_id, eos_id, sys_id, tok_to_id, id_to_tok = ev_mod._vocab_spec(vocab_size)

    end_limit = base_seq_ids.index(eos_id, 1) if eos_id in base_seq_ids[1:] else len(base_seq_ids)
    # Partition base_seq_ids into system segments
    sys_spans: List[Tuple[int, int]] = []
    cur = 1
    while cur <= end_limit:
        try:
            nxt = base_seq_ids.index(sys_id, cur, end_limit)
        except ValueError:
            nxt = end_limit
        sys_spans.append((cur, nxt))
        if nxt >= end_limit:
            break
        cur = nxt + 1

    # Flag low-confidence systems
    flagged: List[Tuple[float, int]] = []
    for s_idx, (st_p, en_p) in enumerate(sys_spans):
        if en_p <= st_p:
            continue
        slp = tok_lp_list[max(0, st_p - 1) : min(len(tok_lp_list), en_p - 1)]
        mean_slp = sum(slp) / max(1, len(slp))
        seg_toks = [id_to_tok.get(t, "") for t in base_seq_ids[st_p:en_p]]
        fsa_m = ev_mod.compute_fsa_beat_metrics(seg_toks, True, True)
        if mean_slp < sys_lp_thresh or fsa_m["mean_meter_err"] > 0.01:
            flagged.append((mean_slp - 0.5 * fsa_m["mean_meter_err"], s_idx))

    if not flagged:
        return base_seq_ids, 0

    flagged.sort(key=lambda x: x[0])
    chosen_indices = sorted(idx for _, idx in flagged[:max_systems])

    mem_hi, _, _ = model.encode_image(img_hires_t, return_aux=False)
    # Build rebuilt sequence splicing re-decoded systems in order
    rebuilt: List[int] = [bos_id]
    n_improved = 0
    for s_idx, (st_p, en_p) in enumerate(sys_spans):
        orig_seg = base_seq_ids[st_p:en_p]
        if s_idx > 0:
            rebuilt.append(sys_id)
        if s_idx not in chosen_indices:
            rebuilt.extend(orig_seg)
            continue

        # Prefill 1536px KV-cache with current prefix up to this system boundary
        pref_t = torch.tensor([rebuilt], dtype=torch.long, device=device)
        _, _, past_kvs = model.decode_tokens(pref_t, memory=mem_hi, return_mtp=False, past_kvs=None, use_cache=True)
        state = grammar_mod.PrefixState(strict=False, vocab_map=tok_to_id)
        max_sys_toks = min(640, max(128, int(len(orig_seg) * 1.5) + 32))
        new_seg: List[int] = []
        new_lps: List[float] = []
        cur_tok = rebuilt[-1]
        for _ in range(max_sys_toks):
            cur_t = torch.tensor([[cur_tok]], dtype=torch.long, device=device)
            logits, _, past_kvs = model.decode_tokens(
                cur_t, memory=None, return_mtp=False, past_kvs=past_kvs, use_cache=True
            )
            step_l = logits[0, -1, :].float()
            tok_id = ev_mod._select_constrained_token(step_l, state)
            if tok_id == sys_id or tok_id == eos_id:
                break
            lp = float(torch.log_softmax(step_l, dim=-1)[tok_id].item())
            new_seg.append(tok_id)
            new_lps.append(lp)
            state = ev_mod._advance_grammar_state(state, tok_id, strict=False, vocab_size=vocab_size)
            cur_tok = tok_id

        # Compare old vs new system via joint FSA + mean_logprob
        orig_toks = [id_to_tok.get(t, "") for t in orig_seg]
        new_toks = [id_to_tok.get(t, "") for t in new_seg]
        orig_fsa = ev_mod.compute_fsa_beat_metrics(orig_toks, True, True)["fsa_beat_score"]
        new_fsa = ev_mod.compute_fsa_beat_metrics(new_toks, True, True)["fsa_beat_score"]
        orig_slp = tok_lp_list[max(0, st_p - 1) : min(len(tok_lp_list), en_p - 1)]
        orig_score = orig_fsa + 0.5 * (sum(orig_slp) / max(1, len(orig_slp)))
        new_score = new_fsa + 0.5 * (sum(new_lps) / max(1, len(new_lps))) if new_seg else -99.0

        if new_seg and new_score > orig_score:
            rebuilt.extend(new_seg)
            n_improved += 1
        else:
            rebuilt.extend(orig_seg)

    rebuilt.append(eos_id)
    return rebuilt, n_improved


def _summarize_arm(piece_rows: List[Dict[str, Any]], base_total_secs: Optional[float]) -> Dict[str, Any]:
    slices = {
        "printed_clean": [r for r in piece_rows if not r["handwritten"]],
        "printed_photo": [r for r in piece_rows if not r["handwritten"]],
        "hw_clean": [r for r in piece_rows if r["handwritten"]],
        "hw_photo": [r for r in piece_rows if r["handwritten"]],
        "all45_clean": piece_rows,
        "all45_photo": piece_rows,
    }
    out: Dict[str, Any] = {}
    for sname, rows in slices.items():
        dom = "clean" if sname.endswith("clean") else "photo"
        nf1 = [float(r[f"{dom}_note_f1"]) for r in rows]
        dec = [1.0 if r[f"{dom}_decode_ok"] else 0.0 for r in rows]
        trig = [1.0 if r.get(f"{dom}_triggered", False) else 0.0 for r in rows]
        out[sname] = {
            "n": len(rows),
            "note_f1_mean": round(statistics.fmean(nf1), 6) if nf1 else 0.0,
            "decode_ok_rate": round(statistics.fmean(dec), 6) if dec else 0.0,
            "trigger_rate": round(statistics.fmean(trig), 4) if trig else 0.0,
        }
    total_secs = sum(float(r["clean_elapsed_secs"]) + float(r["photo_elapsed_secs"]) for r in piece_rows)
    overhead = ((total_secs - base_total_secs) / max(1e-6, base_total_secs) * 100.0) if base_total_secs else 0.0
    out["runtime"] = {
        "total_decode_secs_90pages": round(total_secs, 2),
        "overhead_vs_baseline_pct": round(overhead, 2),
    }
    return out


def run_experiment() -> Dict[str, Any]:
    started_ts = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    try:
        token = _read_and_purge_hf_token()
        repo_root, aser_bin, eharness_bin = _setup_environment()

        mvd_dir = repo_root / "tmp-train" / "mvd"
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        import torch
        from huggingface_hub import HfApi, hf_hub_download
        import eval_sys_gate as esg
        import eval_uni_suite31 as ev
        import grammar
        import uni_omr

        esg.ASER = str(aser_bin)
        esg.EHARNESS = str(eharness_bin)
        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda"
        platform_str = f"colab-{torch.cuda.get_device_name(0)}" if device.type == "cuda" else "cpu"

        api = HfApi(token=token)
        ckpt_local = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID, filename=HF_CKPT_PATH, repo_type="dataset",
                token=token, local_dir=str(CACHE_ROOT / "hf"),
            )
        )
        suite_tar = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID, filename=HF_SUITE31_TAR, repo_type="dataset",
                token=token, local_dir=str(CACHE_ROOT / "hf"),
            )
        )
        suite_dir = CACHE_ROOT / "suite31_eval"
        if not (suite_dir / "suite31" / "suite31_hw_manifest.json").exists():
            suite_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(suite_tar, "r:*") as tf:
                tf.extractall(path=suite_dir)

        manifest_path = suite_dir / "suite31" / "suite31_hw_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pieces = manifest["pieces"]
        suite_root = manifest_path.parent

        model, cfg, ckpt_meta = ev.load_uni_checkpoint(ckpt_local, device)
        _, _, eos_id, _, _, _ = ev._vocab_spec(int(cfg.vocab_size))

        workdir = CACHE_ROOT / EXP_ID
        workdir.mkdir(parents=True, exist_ok=True)
        partial_path = workdir / f"{EXP_ID}_partial.json"
        final_local_path = workdir / f"{EXP_ID}.json"

        refs_cache: Dict[str, Tuple[List[str], Path]] = {}
        for piece in pieces:
            slug = piece["slug"]
            ref_bek = suite_root / piece["bekern"]
            ref_toks = ref_bek.read_text(encoding="utf-8").strip().split()
            ref_notes = workdir / f"ref-{slug}.json"
            if not ref_notes.exists():
                esg.run_cli([esg.ASER, "bekern-decode", "--input", str(ref_bek), "--out", str(ref_notes)])
            refs_cache[slug] = (ref_toks, ref_notes)

        def _decode_and_score(slug: str, domain: str, img_p: Path, dim: int, tag: str) -> Dict[str, Any]:
            img_t = uni_omr.load_fullpage_image(img_p, max_dim=dim).unsqueeze(0).to(device, non_blocking=True)
            t0 = time.perf_counter()
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
            with torch.no_grad(), ctx:
                seq, _ = ev.generate_mtp_spec(
                    model, img_t, max_len=MAX_LEN, use_grammar=True, strict_grammar=False, return_stats=True
                )
                seq_ids = seq[0].detach().cpu().tolist()
                mean_lp, mean_ent, tok_lps = _teacher_force_confidence(model, img_t, seq_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = max(1e-6, time.perf_counter() - t0)

            hit_eos = eos_id in seq_ids[1:]
            pred_raw = ev.decode_uni_compat(seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True)
            if not hit_eos:
                pred_raw = ev.trim_truncated_tokens(pred_raw)
            hyp_toks, _ = ev.stitch_uni_tokens(pred_raw)

            ref_toks, ref_notes = refs_cache[slug]
            hyp_txt = workdir / f"{tag}-{slug}-{domain}-hyp.txt"
            hyp_txt.write_text(" ".join(hyp_toks) + "\n", encoding="utf-8")
            hyp_notes = workdir / f"{tag}-{slug}-{domain}-notes.json"
            p_dec = esg.run_cli([esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes)])
            bekern_ok = (p_dec.returncode == 0) and hyp_notes.exists()
            fsa_m = ev.compute_fsa_beat_metrics(hyp_toks, hit_eos, bekern_ok)

            if not bekern_ok:
                note_f1, decode_ok = 0.0, False
            else:
                sc = esg.score_notes_pair(ref_notes, hyp_notes, workdir, f"{tag}-{slug}-{domain}", esg.EHARNESS, esg.ASER)
                note_f1 = float(sc.get("f1", 0.0))
                decode_ok = not bool(sc.get("zeroed"))

            return {
                "seq_ids": seq_ids,
                "tok_lps": tok_lps,
                "mean_logprob": round(mean_lp, 6),
                "mean_entropy": round(mean_ent, 6),
                "note_f1": round(note_f1, 6),
                "decode_ok": decode_ok,
                "bekern_ok": bekern_ok,
                "elapsed_secs": round(elapsed, 4),
                "fsa_beat_score": float(fsa_m["fsa_beat_score"]),
                "beat_ok_rate": float(fsa_m["beat_conservation_rate"]),
                "has_loop": bool(fsa_m["has_loop"]),
            }

        # Stage 1: Fast canonical 1280px pass on all 90 pages
        base_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        alt_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        base_rows: List[Dict[str, Any]] = []
        last_save_ts = time.time()

        print(f"[{EXP_ID}] Pass 1: baseline_1280 + exact confidence across 90 pages...", flush=True)
        for p_idx, piece in enumerate(pieces, start=1):
            slug = piece["slug"]
            prow: Dict[str, Any] = {"piece": slug, "handwritten": bool(piece.get("handwritten", False))}
            for domain in ("clean", "photo"):
                res = _decode_and_score(slug, domain, suite_root / piece[domain], BASE_DIM, "base1280")
                base_cache[(slug, domain)] = res
                prow[f"{domain}_note_f1"] = res["note_f1"]
                prow[f"{domain}_decode_ok"] = res["decode_ok"]
                prow[f"{domain}_elapsed_secs"] = res["elapsed_secs"]
                prow[f"{domain}_mean_logprob"] = res["mean_logprob"]
                prow[f"{domain}_beat_ok_rate"] = res["beat_ok_rate"]
                prow[f"{domain}_triggered"] = False
            base_rows.append(prow)
            if p_idx % 15 == 0 or p_idx == len(pieces):
                print(f"[{EXP_ID}] Pass 1 progress {p_idx}/{len(pieces)}", flush=True)

        base_summary = _summarize_arm(base_rows, None)
        base_total_secs = base_summary["runtime"]["total_decode_secs_90pages"]

        # Suspect-Page Calibration Audit (Soundslice/PhotoScore UX metric)
        cal_labels: List[int] = []
        cal_neg_lp: List[float] = []
        cal_beat_risk: List[float] = []
        cal_joint: List[float] = []
        for (_, _), res in base_cache.items():
            cal_labels.append(1 if res["note_f1"] < 0.85 else 0)
            cal_neg_lp.append(-res["mean_logprob"])
            cal_beat_risk.append(1.0 - res["beat_ok_rate"])
            cal_joint.append(-res["mean_logprob"] + 0.5 * (1.0 - res["beat_ok_rate"]))

        calibration = {
            "target": "note_f1 < 0.85",
            "n_pages": len(cal_labels),
            "n_suspect": sum(cal_labels),
            "auroc_neg_mean_logprob": round(_compute_auroc(cal_labels, cal_neg_lp), 4),
            "auroc_beat_violation": round(_compute_auroc(cal_labels, cal_beat_risk), 4),
            "auroc_joint_risk": round(_compute_auroc(cal_labels, cal_joint), 4),
        }
        print(f"[{EXP_ID}] Calibration Audit: {calibration}", flush=True)

        def _get_alt_view(slug: str, domain: str, img_p: Path, dim: int) -> Dict[str, Any]:
            key = (slug, domain, dim)
            if key not in alt_cache:
                alt_cache[key] = _decode_and_score(slug, domain, img_p, dim, f"alt{dim}")
            return alt_cache[key]

        results_by_arm: Dict[str, Any] = {
            "baseline_1280": {"summary": base_summary, "pieces": base_rows}
        }

        # Stage 2: Page-level selective 3-scale triage arms
        page_arms = [
            ("conf_triage_page_p25", -0.18, 0.85),
            ("conf_triage_page_p40", -0.12, 0.92),
        ]
        for akey, lp_th, beat_th in page_arms:
            rows: List[Dict[str, Any]] = []
            for piece in pieces:
                slug = piece["slug"]
                prow = {"piece": slug, "handwritten": bool(piece.get("handwritten", False))}
                for domain in ("clean", "photo"):
                    b = base_cache[(slug, domain)]
                    flagged = (
                        b["mean_logprob"] < lp_th
                        or b["beat_ok_rate"] < beat_th
                        or b["has_loop"]
                        or (not b["bekern_ok"])
                    )
                    if not flagged:
                        prow[f"{domain}_note_f1"] = b["note_f1"]
                        prow[f"{domain}_decode_ok"] = b["decode_ok"]
                        prow[f"{domain}_elapsed_secs"] = b["elapsed_secs"]
                        prow[f"{domain}_triggered"] = False
                        continue
                    v1152 = _get_alt_view(slug, domain, suite_root / piece[domain], LORES_DIM)
                    v1536 = _get_alt_view(slug, domain, suite_root / piece[domain], HIRES_DIM)
                    cands = [b, v1152, v1536]
                    best = max(cands, key=lambda c: c["fsa_beat_score"] + 0.5 * c["mean_logprob"])
                    prow[f"{domain}_note_f1"] = best["note_f1"]
                    prow[f"{domain}_decode_ok"] = best["decode_ok"]
                    prow[f"{domain}_elapsed_secs"] = round(
                        b["elapsed_secs"] + v1152["elapsed_secs"] + v1536["elapsed_secs"], 4
                    )
                    prow[f"{domain}_triggered"] = True
                rows.append(prow)
            results_by_arm[akey] = {"summary": _summarize_arm(rows, base_total_secs), "pieces": rows}
            _atomic_write_json(partial_path, {"exp_id": EXP_ID, "status": "RUNNING", "calibration": calibration, "arms": results_by_arm})

        # Stage 3: Surgical system-level 1536px re-decode
        sys_rows: List[Dict[str, Any]] = []
        for piece in pieces:
            slug = piece["slug"]
            ref_toks, ref_notes = refs_cache[slug]
            prow = {"piece": slug, "handwritten": bool(piece.get("handwritten", False))}
            for domain in ("clean", "photo"):
                b = base_cache[(slug, domain)]
                img_hi = uni_omr.load_fullpage_image(suite_root / piece[domain], max_dim=HIRES_DIM).unsqueeze(0).to(device)
                t0 = time.perf_counter()
                ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
                with torch.no_grad(), ctx:
                    new_ids, n_imp = _surgical_system_hires_redecode(
                        model, img_hi, b["seq_ids"], b["tok_lps"],
                        sys_lp_thresh=-0.20, max_systems=4, ev_mod=ev, grammar_mod=grammar,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                extra_t = time.perf_counter() - t0 if n_imp > 0 or new_ids != b["seq_ids"] else 0.0
                if new_ids == b["seq_ids"]:
                    prow[f"{domain}_note_f1"] = b["note_f1"]
                    prow[f"{domain}_decode_ok"] = b["decode_ok"]
                    prow[f"{domain}_elapsed_secs"] = b["elapsed_secs"]
                    prow[f"{domain}_triggered"] = False
                else:
                    pred_raw = ev.decode_uni_compat(new_ids, vocab_size=int(cfg.vocab_size), skip_special=True)
                    if eos_id not in new_ids[1:]:
                        pred_raw = ev.trim_truncated_tokens(pred_raw)
                    hyp_toks, _ = ev.stitch_uni_tokens(pred_raw)
                    hyp_txt = workdir / f"syshi-{slug}-{domain}-hyp.txt"
                    hyp_txt.write_text(" ".join(hyp_toks) + "\n", encoding="utf-8")
                    hyp_notes = workdir / f"syshi-{slug}-{domain}-notes.json"
                    p_dec = esg.run_cli([esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes)])
                    bekern_ok = (p_dec.returncode == 0) and hyp_notes.exists()
                    if not bekern_ok:
                        nf1, dok = 0.0, False
                    else:
                        sc = esg.score_notes_pair(ref_notes, hyp_notes, workdir, f"syshi-{slug}-{domain}", esg.EHARNESS, esg.ASER)
                        nf1, dok = float(sc.get("f1", 0.0)), not bool(sc.get("zeroed"))
                    prow[f"{domain}_note_f1"] = round(nf1, 6)
                    prow[f"{domain}_decode_ok"] = dok
                    prow[f"{domain}_elapsed_secs"] = round(b["elapsed_secs"] + extra_t, 4)
                    prow[f"{domain}_triggered"] = True
            sys_rows.append(prow)
            if (time.time() - last_save_ts) >= SAVE_EVERY_MINUTES * 60.0:
                _atomic_write_json(partial_path, {"exp_id": EXP_ID, "status": "RUNNING", "calibration": calibration, "arms": results_by_arm})
                last_save_ts = time.time()

        results_by_arm["conf_triage_system_hires"] = {
            "summary": _summarize_arm(sys_rows, base_total_secs),
            "pieces": sys_rows,
        }

        comp = {
            k: {
                s: v["summary"][s]["note_f1_mean"]
                for s in ("printed_clean", "printed_photo", "hw_clean", "hw_photo", "all45_clean", "all45_photo")
            } | {"overhead_pct": v["summary"]["runtime"]["overhead_vs_baseline_pct"]}
            for k, v in results_by_arm.items()
        }
        final_report = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_iso,
            "elapsed_secs": round(time.time() - started_ts, 2),
            "platform": platform_str,
            "ckpt_hf_path": HF_CKPT_PATH,
            "ckpt_meta": ckpt_meta,
            "calibration_audit": calibration,
            "comparison_table": comp,
            "arms": results_by_arm,
        }
        _atomic_write_json(final_local_path, final_report)
        _atomic_write_json(partial_path, final_report)

        api.upload_file(
            path_or_fileobj=str(final_local_path),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload confidence-triggered selective hires redecode JSON (Colab L4)",
        )
        ledger = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_iso,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_secs": final_report["elapsed_secs"],
            "platform": platform_str,
            "hf_eval_path": HF_EVAL_PATH,
            "ckpt_hf_path": HF_CKPT_PATH,
            "calibration_audit": calibration,
            "comparison_table": comp,
        }
        tmp_ledger = workdir / "run_ledger_final.json"
        _atomic_write_json(tmp_ledger, ledger)
        api.upload_file(
            path_or_fileobj=str(tmp_ledger),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark COMPLETED (Colab L4)",
        )
        print(f"[{EXP_ID}] Uploaded HF artifacts: {HF_EVAL_PATH} & {HF_RUN_PATH}", flush=True)
        return final_report
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    run_experiment()

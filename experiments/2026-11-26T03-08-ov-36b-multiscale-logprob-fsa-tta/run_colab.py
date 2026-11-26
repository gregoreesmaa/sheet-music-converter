#!/usr/bin/env python3
"""Google Colab L4 runner for Exp 36b: 2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta.

Follow-up to ov-36 (`2026-11-26T01-38-ov-36-hw-multiview-tta-voting`):
- Drops destructive `clahe`, `deskew`, and OOD `1792px` views completely.
- Evaluates 5 clean in-distribution scale views (`1024, 1152, 1280, 1408, 1536`)
  at `max_len=3840` (per `ov-35`) across all 14 real historical HW pieces
  (`clean` + `photo` = 28 pages).
- Adds exact teacher-forced length-normalized `mean_logprob` and `mean_entropy`
  (~30 ms/view) alongside `fsa_beat_score` and `mbr_vote_score`, and reports
  full 5-view plus fast 2-view `{1280, 1536}` and 3-view `{1152, 1280, 1536}`
  Pareto subsets.

Expects on Colab VM:
  - `/content/bundle.tar.gz` (contains `sheet-music-converter/...` and `mscz-rs/`)
  - `/tmp/.hf_token` (Hugging Face token; chmod 600, deleted immediately and in `finally:`)
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_CKPT_PATH = "checkpoints/iter10_soup_baseline/uni_omr_soup.pt"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

MAX_LEN = 3840
SAVE_EVERY_MINUTES = 5.0
VIEW_SPECS: List[Tuple[str, int]] = [
    ("scale_1280", 1280),
    ("base_1536", 1536),
    ("scale_1024", 1024),
    ("scale_1152", 1152),
    ("scale_1408", 1408),
]
VIEW_NAMES: List[str] = [name for name, _ in VIEW_SPECS]
VIEW_SUBSETS: Dict[str, List[str]] = {
    "all5": VIEW_NAMES,
    "fast2_1280_1536": ["scale_1280", "base_1536"],
    "fast3_1152_1280_1536": ["scale_1280", "base_1536", "scale_1152"],
}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str] | str, *, shell: bool = False, env: Optional[Dict[str, str]] = None) -> None:
    print(f"[{EXP_ID}] $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        shell=shell,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
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
        print(f"[{EXP_ID}] Extracting {BUNDLE_PATH} -> {WORK_ROOT}...", flush=True)
        with tarfile.open(BUNDLE_PATH, "r:*") as tf:
            tf.extractall(path=WORK_ROOT)
        repo = WORK_ROOT / "sheet-music-converter"
        if (repo / "tmp-train" / "mvd" / "eval_uni_suite31.py").exists():
            return repo
    raise FileNotFoundError(
        f"Could not locate sheet-music-converter workspace or bundle at {BUNDLE_PATH}"
    )


def _setup_environment() -> Tuple[Path, Path, Path]:
    repo_root = _find_repo_root()
    env = os.environ.copy()
    cargo_bin = Path.home() / ".cargo" / "bin"
    env["PATH"] = f"{cargo_bin}:{env.get('PATH', '')}"

    if not shutil.which("cargo", path=env["PATH"]):
        print(f"[{EXP_ID}] Installing Rust toolchain (rustup minimal)...", flush=True)
        _run_cmd(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
            shell=True,
            env=env,
        )

    for pkg in ("huggingface_hub", "pillow"):
        try:
            __import__("PIL" if pkg == "pillow" else pkg)
        except ImportError:
            _run_cmd([sys.executable, "-m", "pip", "install", "-q", pkg], env=env)

    aser_manifest = repo_root / "converter" / "assembly-serialize" / "Cargo.toml"
    eharness_manifest = repo_root / "converter" / "eval-harness" / "Cargo.toml"
    aser_bin = repo_root / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize"
    eharness_bin = repo_root / "converter" / "eval-harness" / "target" / "release" / "eval-harness"

    if not aser_bin.exists():
        _run_cmd(["cargo", "build", "--release", "--manifest-path", str(aser_manifest)], env=env)
    if not eharness_bin.exists():
        _run_cmd(["cargo", "build", "--release", "--manifest-path", str(eharness_manifest)], env=env)

    if not aser_bin.exists() or not eharness_bin.exists():
        raise FileNotFoundError(f"Missing Rust scoring binaries: {aser_bin}, {eharness_bin}")
    return repo_root, aser_bin, eharness_bin


def _compute_fsa_beat_metrics(
    hyp_page_toks: Sequence[str],
    hit_eos: bool,
    bekern_decode_ok: bool,
    grammar_mod: Any,
    ev_mod: Any,
    uni_omr_mod: Any,
) -> Dict[str, Any]:
    st = grammar_mod.PrefixState(strict=False)
    n_bars = 0
    n_both_ok = 0
    meter_errs: List[float] = []
    voice_errs: List[float] = []
    broken_steps = 0

    for tok in hyp_page_toks:
        if tok == uni_omr_mod.SYS_TOKEN:
            st = grammar_mod.PrefixState(strict=False)
            continue
        if tok in grammar_mod.BAR_LIKES:
            m_err, v_err = ev_mod._measure_violation_errors(st)
            n_bars += 1
            meter_errs.append(min(4.0, float(m_err)))
            voice_errs.append(min(4.0, float(v_err)))
            if m_err <= 1e-3 and v_err <= 1e-3:
                n_both_ok += 1
        try:
            st.step(tok)
        except ValueError:
            broken_steps += 1

    beat_ok_rate = (float(n_both_ok) / float(n_bars)) if n_bars > 0 else 0.0
    mean_m_err = (sum(meter_errs) / len(meter_errs)) if meter_errs else 1.0
    mean_v_err = (sum(voice_errs) / len(voice_errs)) if voice_errs else 1.0
    broken_rate = float(broken_steps) / float(max(1, len(hyp_page_toks)))

    score = (
        (1.0 if bekern_decode_ok else -2.0)
        + (0.25 if hit_eos else -0.5)
        + 1.5 * beat_ok_rate
        - 0.5 * mean_m_err
        - 0.5 * mean_v_err
        - 0.25 * broken_rate
    )
    return {
        "n_bars": n_bars,
        "beat_conservation_rate": round(beat_ok_rate, 6),
        "mean_meter_err": round(mean_m_err, 6),
        "mean_voice_err": round(mean_v_err, 6),
        "broken_steps": broken_steps,
        "fsa_beat_score": round(float(score), 6),
    }


def _score_subset_selectors(
    view_results: Dict[str, Dict[str, Any]],
    view_toks: Dict[str, List[str]],
    subset: List[str],
    ev_mod: Any,
) -> Dict[str, Dict[str, Any]]:
    # Recompute MBR within the subset so fast2/fast3 don't leak info from other views
    mbr_map: Dict[str, float] = {}
    for vi in subset:
        w_sum = 0.0
        s_sum = 0.0
        for vj in subset:
            if vi == vj:
                continue
            w = 1.0 if view_results[vj]["bekern_decode_ok"] else 0.25
            sim = ev_mod.compute_token_lcs_f1(view_toks[vi], view_toks[vj])
            s_sum += w * sim
            w_sum += w
        mbr_map[vi] = s_sum / max(1e-6, w_sum)

    def _valid_bonus(vr: Dict[str, Any]) -> float:
        return (1.0 if vr["bekern_decode_ok"] else -2.0) + (0.25 if vr["hit_eos"] else -0.5)

    def _pick(score_fn) -> Dict[str, Any]:
        best_v = max(subset, key=lambda v: (score_fn(v), -subset.index(v)))
        vr = view_results[best_v]
        return {
            "selected_view": best_v,
            "note_f1": vr["note_f1"],
            "tok_lcs": vr["tok_lcs"],
            "decode_ok": vr["decode_ok"],
        }

    return {
        "best_fsa_beat": _pick(lambda v: float(view_results[v]["fsa_beat_score"])),
        "best_logprob": _pick(
            lambda v: _valid_bonus(view_results[v]) + 2.0 * float(view_results[v]["mean_logprob"])
        ),
        "best_neg_entropy": _pick(
            lambda v: _valid_bonus(view_results[v]) - 1.0 * float(view_results[v]["mean_entropy"])
        ),
        "best_fsa_logprob": _pick(
            lambda v: float(view_results[v]["fsa_beat_score"]) + 2.0 * float(view_results[v]["mean_logprob"])
        ),
        "best_mbr_vote": _pick(lambda v: mbr_map[v]),
        "best_fsa_logprob_mbr": _pick(
            lambda v: 0.5 * float(view_results[v]["fsa_beat_score"])
            + 1.5 * float(view_results[v]["mean_logprob"])
            + 1.0 * mbr_map[v]
        ),
        "oracle_best_view": _pick(lambda v: float(view_results[v]["note_f1"])),
    }


def _summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
        xs = [float(v) for v in vals if v is not None]
        return round(sum(xs) / len(xs), 6) if xs else None

    summary: Dict[str, Any] = {}
    for domain in ("clean", "photo"):
        drows = [r for r in rows if r["domain"] == domain]
        base_nf1 = _mean([r["views"]["base_1536"]["note_f1"] for r in drows])
        s1280_nf1 = _mean([r["views"]["scale_1280"]["note_f1"] for r in drows])
        per_view: Dict[str, Any] = {}
        for vname in VIEW_NAMES:
            vnf1 = _mean([r["views"][vname]["note_f1"] for r in drows])
            vtf1 = _mean([r["views"][vname]["tok_lcs"] for r in drows])
            per_view[vname] = {
                "note_f1": vnf1,
                "delta_vs_base": round(vnf1 - base_nf1, 6) if (vnf1 is not None and base_nf1 is not None) else None,
                "delta_vs_1280": round(vnf1 - s1280_nf1, 6) if (vnf1 is not None and s1280_nf1 is not None) else None,
                "tok_lcs": vtf1,
                "mean_logprob": _mean([r["views"][vname]["mean_logprob"] for r in drows]),
                "mean_entropy": _mean([r["views"][vname]["mean_entropy"] for r in drows]),
                "decode_ok_rate": round(
                    sum(1 for r in drows if r["views"][vname]["decode_ok"]) / max(1, len(drows)), 4
                ),
            }

        subsets_summary: Dict[str, Any] = {}
        for sub_name in VIEW_SUBSETS:
            sel_summary: Dict[str, Any] = {}
            for sel_key in (
                "best_fsa_beat",
                "best_logprob",
                "best_neg_entropy",
                "best_fsa_logprob",
                "best_mbr_vote",
                "best_fsa_logprob_mbr",
                "oracle_best_view",
            ):
                snf1 = _mean([r["subsets"][sub_name][sel_key]["note_f1"] for r in drows])
                stf1 = _mean([r["subsets"][sub_name][sel_key]["tok_lcs"] for r in drows])
                w_base = sum(
                    1
                    for r in drows
                    if float(r["subsets"][sub_name][sel_key]["note_f1"])
                    > float(r["views"]["base_1536"]["note_f1"]) + 1e-6
                )
                l_base = sum(
                    1
                    for r in drows
                    if float(r["subsets"][sub_name][sel_key]["note_f1"])
                    < float(r["views"]["base_1536"]["note_f1"]) - 1e-6
                )
                w_1280 = sum(
                    1
                    for r in drows
                    if float(r["subsets"][sub_name][sel_key]["note_f1"])
                    > float(r["views"]["scale_1280"]["note_f1"]) + 1e-6
                )
                l_1280 = sum(
                    1
                    for r in drows
                    if float(r["subsets"][sub_name][sel_key]["note_f1"])
                    < float(r["views"]["scale_1280"]["note_f1"]) - 1e-6
                )
                sel_summary[sel_key] = {
                    "note_f1": snf1,
                    "delta_vs_base": round(snf1 - base_nf1, 6) if (snf1 is not None and base_nf1 is not None) else None,
                    "delta_vs_1280": round(snf1 - s1280_nf1, 6) if (snf1 is not None and s1280_nf1 is not None) else None,
                    "tok_lcs": stf1,
                    "vs_base_wtl": [w_base, len(drows) - w_base - l_base, l_base],
                    "vs_1280_wtl": [w_1280, len(drows) - w_1280 - l_1280, l_1280],
                }
            subsets_summary[sub_name] = sel_summary

        tot_secs = round(
            sum(sum(float(r["views"][v]["elapsed_secs"]) for v in VIEW_NAMES) for r in drows),
            2,
        )
        summary[f"hw_{domain}"] = {
            "n_pages": len(drows),
            "base_1536_note_f1": base_nf1,
            "scale_1280_note_f1": s1280_nf1,
            "total_tta_decode_secs": tot_secs,
            "per_view": per_view,
            "subsets": subsets_summary,
        }
    return summary


def run_experiment() -> Dict[str, Any]:
    try:
        hf_token = _read_and_purge_hf_token()
        repo_root, aser_bin, eharness_bin = _setup_environment()

        import torch
        from PIL import Image
        from huggingface_hub import HfApi, hf_hub_download

        t_start = time.time()
        started_iso = datetime.now(timezone.utc).isoformat()

        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        use_bf16 = bool(torch.cuda.is_available() and cc[0] >= 8)
        platform_str = f"colab-{gpu_name}"
        print(
            f"[{EXP_ID}] Starting on {platform_str} (cc={cc[0]}.{cc[1]}, bf16={use_bf16}, "
            f"torch={torch.__version__}) | views={VIEW_NAMES} max_len={MAX_LEN}",
            flush=True,
        )

        api = HfApi(token=hf_token)

        suite31_dir = CACHE_ROOT / "datasets" / "suite31_hw_eval"
        manifest_path = suite31_dir / "suite31" / "suite31_hw_manifest.json"
        if not manifest_path.exists():
            print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR} from HF...", flush=True)
            tar_local = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_SUITE31_TAR,
                repo_type="dataset",
                token=hf_token,
                local_dir=str(CACHE_ROOT),
            )
            suite31_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_local, "r:*") as tf:
                tf.extractall(path=suite31_dir)

        ckpt_path = CACHE_ROOT / HF_CKPT_PATH
        if not ckpt_path.exists():
            print(f"[{EXP_ID}] Downloading {HF_CKPT_PATH} from HF...", flush=True)
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_CKPT_PATH,
                repo_type="dataset",
                token=hf_token,
                local_dir=str(CACHE_ROOT),
            )

        mvd_dir = str(repo_root / "tmp-train" / "mvd")
        if mvd_dir not in sys.path:
            sys.path.insert(0, mvd_dir)
        import eval_sys_gate as esg
        import eval_uni_suite31 as ev
        import grammar
        import uni_omr

        esg.ASER = aser_bin
        esg.EHARNESS = eharness_bin

        model, cfg, ckpt_meta = ev.load_uni_checkpoint(ckpt_path, device=device)
        _, all_pieces = ev.resolve_manifest_pieces(manifest_path)
        hw_pieces = [p for p in all_pieces if p["handwritten"]]
        _, _, eos_id, _, _, _ = ev._vocab_spec(int(cfg.vocab_size))
        print(
            f"[{EXP_ID}] Loaded ckpt step={ckpt_meta.get('step')} | hw_pieces={len(hw_pieces)}",
            flush=True,
        )

        workdir = Path(tempfile.mkdtemp(prefix="ov36b_scale_tta_"))
        try:
            init_ledger = {
                "exp_id": EXP_ID,
                "status": "RUNNING",
                "started_utc": started_iso,
                "platform": platform_str,
                "hf_eval_path": HF_EVAL_PATH,
                "ckpt_hf_path": HF_CKPT_PATH,
                "views": VIEW_NAMES,
                "max_len": MAX_LEN,
                "n_hw_pieces": len(hw_pieces),
            }
            tmp_ledger = workdir / "run_ledger_init.json"
            _atomic_write_json(tmp_ledger, init_ledger)
            api.upload_file(
                path_or_fileobj=str(tmp_ledger),
                path_in_repo=HF_RUN_PATH,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"{EXP_ID}: mark RUNNING on Colab {gpu_name}",
            )
        except Exception as exc:
            print(f"[{EXP_ID}] Warning: initial HF ledger upload failed: {exc}", flush=True)

        partial_path = CACHE_ROOT / "evals" / f"{EXP_ID}_partial.json"
        final_local_path = CACHE_ROOT / "evals" / f"{EXP_ID}.json"
        rows: List[Dict[str, Any]] = []
        done_keys: set[Tuple[str, str]] = set()
        if partial_path.exists():
            try:
                prev = json.loads(partial_path.read_text(encoding="utf-8"))
                if isinstance(prev.get("rows"), list):
                    rows = prev["rows"]
                    done_keys = {(r["piece"], r["domain"]) for r in rows}
                    print(
                        f"[{EXP_ID}] Resuming: {len(done_keys)}/{len(hw_pieces) * 2} HW pages done",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[{EXP_ID}] Ignoring unreadable partial file: {exc}", flush=True)

        def _build_report(status: str) -> Dict[str, Any]:
            return {
                "exp_id": EXP_ID,
                "status": status,
                "started_utc": started_iso,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_secs": round(time.time() - t_start, 2),
                "platform": platform_str,
                "bf16": use_bf16,
                "ckpt_hf_path": HF_CKPT_PATH,
                "ckpt_meta": ckpt_meta,
                "model_config": asdict(cfg),
                "eval_config": {
                    "views": VIEW_NAMES,
                    "view_dims": dict(VIEW_SPECS),
                    "subsets": VIEW_SUBSETS,
                    "max_len": MAX_LEN,
                    "use_grammar": True,
                    "strict_grammar": False,
                    "use_mtp_spec": True,
                    "n_hw_pieces": len(hw_pieces),
                },
                "summary": _summarize_rows(rows),
                "rows": rows,
            }

        last_save_ts = time.time()
        save_interval_s = max(30.0, SAVE_EVERY_MINUTES * 60.0)

        for idx, piece in enumerate(hw_pieces, start=1):
            slug = piece["slug"]
            ref_page_toks, page_ref_notes = esg.suite31_page_refs(
                piece["gt"], "ov36b-" + slug, workdir
            )

            for domain in ("clean", "photo"):
                if (slug, domain) in done_keys:
                    continue

                with Image.open(piece[domain]) as raw_im:
                    rgb = raw_im.convert("RGB")
                    pil_views = {
                        vname: uni_omr.resize_preserve_aspect(rgb, max_dim=vdim)
                        for vname, vdim in VIEW_SPECS
                    }

                view_results: Dict[str, Dict[str, Any]] = {}
                view_toks: Dict[str, List[str]] = {}

                for vname, _ in VIEW_SPECS:
                    img_t = uni_omr._pil_to_chw_tensor(pil_views[vname]).unsqueeze(0).to(device)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        ctx = (
                            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                            if use_bf16
                            else contextlib.nullcontext()
                        )
                        with ctx:
                            seq, _ = ev.generate_mtp_spec(
                                model,
                                img_t,
                                max_len=MAX_LEN,
                                use_grammar=True,
                                strict_grammar=False,
                                return_stats=True,
                                measure_beam_size=0,
                            )
                            # Exact teacher-forced length-normalized log-prob & entropy
                            if seq.shape[1] >= 2:
                                tf_out = model(
                                    img_t,
                                    seq[:, :-1],
                                    return_aux=False,
                                    return_mtp=False,
                                )
                                logits_f = tf_out["logits"].float()
                                log_probs = torch.log_softmax(logits_f, dim=-1)
                                tok_lp = log_probs.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
                                mean_lp = float(tok_lp.mean().item())
                                probs = log_probs.exp()
                                mean_ent = float(-(probs * log_probs).sum(dim=-1).mean().item())
                            else:
                                mean_lp = -10.0
                                mean_ent = 10.0
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = max(1e-6, time.perf_counter() - t0)

                    seq_ids = seq[0].detach().cpu().tolist()
                    hit_eos = eos_id in seq_ids[1:]
                    pred_raw = ev.decode_uni_compat(
                        seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
                    )
                    if not hit_eos:
                        pred_raw = ev.trim_truncated_tokens(pred_raw)
                    hyp_page_toks, n_sys = ev.stitch_uni_tokens(pred_raw)
                    view_toks[vname] = hyp_page_toks

                    hyp_txt = workdir / f"ov36b-{slug}-{domain}-{vname}-hyp.txt"
                    hyp_txt.write_text(" ".join(hyp_page_toks) + "\n", encoding="utf-8")
                    hyp_notes = workdir / f"ov36b-{slug}-{domain}-{vname}-notes.json"
                    p_dec = esg.run_cli([
                        esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes),
                    ])
                    bekern_ok = (p_dec.returncode == 0) and hyp_notes.exists()

                    fsa_metrics = _compute_fsa_beat_metrics(
                        hyp_page_toks, hit_eos, bekern_ok, grammar, ev, uni_omr
                    )

                    _, _, tok_f1 = uni_omr.compute_token_f1(ref_page_toks, hyp_page_toks)
                    tok_lcs = ev.compute_token_lcs_f1(ref_page_toks, hyp_page_toks)
                    if not bekern_ok:
                        note_f1, note_ser, decode_ok = 0.0, 1.0, False
                    else:
                        sc = esg.score_notes_pair(
                            page_ref_notes,
                            hyp_notes,
                            workdir,
                            f"ov36b-{slug}-{domain}-{vname}",
                            esg.EHARNESS,
                            esg.ASER,
                        )
                        note_f1 = float(sc.get("f1", 0.0))
                        note_ser = float(sc.get("ser", 1.0))
                        decode_ok = not bool(sc.get("zeroed"))

                    view_results[vname] = {
                        "tok_f1": round(tok_f1, 6),
                        "tok_lcs": round(tok_lcs, 6),
                        "note_f1": round(note_f1, 6),
                        "note_ser": round(note_ser, 6),
                        "decode_ok": decode_ok,
                        "bekern_decode_ok": bekern_ok,
                        "hit_eos": hit_eos,
                        "n_sys": n_sys,
                        "n_pred_toks": len(hyp_page_toks),
                        "elapsed_secs": round(elapsed, 4),
                        "mean_logprob": round(mean_lp, 6),
                        "mean_entropy": round(mean_ent, 6),
                        **fsa_metrics,
                    }

                subsets_res = {
                    sub_name: _score_subset_selectors(view_results, view_toks, sub_views, ev)
                    for sub_name, sub_views in VIEW_SUBSETS.items()
                }

                row = {
                    "piece": slug,
                    "handwritten": True,
                    "domain": domain,
                    "views": view_results,
                    "subsets": subsets_res,
                }
                rows.append(row)
                done_keys.add((slug, domain))

                all5 = subsets_res["all5"]
                print(
                    f"[{idx:02d}/{len(hw_pieces):02d}] {slug:34s} [{domain:5s}] | "
                    f"1280={view_results['scale_1280']['note_f1']:.3f} "
                    f"1536={view_results['base_1536']['note_f1']:.3f} "
                    f"fsa={all5['best_fsa_beat']['note_f1']:.3f}({all5['best_fsa_beat']['selected_view']}) "
                    f"lp={all5['best_logprob']['note_f1']:.3f}({all5['best_logprob']['selected_view']}) "
                    f"fsa+lp={all5['best_fsa_logprob']['note_f1']:.3f}({all5['best_fsa_logprob']['selected_view']}) "
                    f"orc={all5['oracle_best_view']['note_f1']:.3f}({all5['oracle_best_view']['selected_view']})",
                    flush=True,
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if (time.time() - last_save_ts) >= save_interval_s or domain == "photo":
                    _atomic_write_json(partial_path, _build_report("RUNNING"))
                    last_save_ts = time.time()

        final_report = _build_report("COMPLETED")
        _atomic_write_json(final_local_path, final_report)
        _atomic_write_json(partial_path, final_report)

        print("=" * 96, flush=True)
        print(f"[{EXP_ID}] FINAL SUMMARY (elapsed={final_report['elapsed_secs']}s)", flush=True)
        print(json.dumps(final_report["summary"], indent=2), flush=True)
        print("=" * 96, flush=True)

        api.upload_file(
            path_or_fileobj=str(final_local_path),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload clean multi-scale TTA + logprob/FSA reranking JSON (Colab L4)",
        )
        final_ledger = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_iso,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_secs": final_report["elapsed_secs"],
            "platform": platform_str,
            "hf_eval_path": HF_EVAL_PATH,
            "ckpt_hf_path": HF_CKPT_PATH,
            "summary": final_report["summary"],
        }
        tmp_final_ledger = workdir / "run_ledger_final.json"
        _atomic_write_json(tmp_final_ledger, final_ledger)
        api.upload_file(
            path_or_fileobj=str(tmp_final_ledger),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark COMPLETED with multi-scale logprob+FSA summary (Colab L4)",
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

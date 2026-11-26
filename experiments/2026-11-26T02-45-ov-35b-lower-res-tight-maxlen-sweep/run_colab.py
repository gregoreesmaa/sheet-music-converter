#!/usr/bin/env python3
"""Google Colab L4 runner for Exp 35b: 2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep.

Sweeps inference `--max-len` in {2560, 3072, 3584, 3840} x input resolution `--max-dim`
in {1024, 1152, 1280, 1408} across all 45 `suite31_hw` pieces (`clean` + `photo` = 90 pages),
isolating truncation gains vs fine-detail resolution gains.

Uses exact prefix-equivalence of deterministic greedy+MTP-4+FSA decoding: each
resolution is decoded once at `max_len=3840` and scored cleanly at all four
cutoffs (2560, 3072, 3584, 3840) with `trim_truncated_tokens` applied whenever `eos_id`
is not reached within the cutoff.

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
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_CKPT_PATH = "checkpoints/iter10_soup_baseline/uni_omr_soup.pt"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

RESOLUTIONS: List[int] = [1024, 1152, 1280, 1408]
MAX_LENS: List[int] = [2560, 3072, 3584, 3840]
DECODE_HORIZON = max(MAX_LENS)
SAVE_EVERY_MINUTES = 5.0


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
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    needed_pkgs = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
    ]:
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
            print(f"[{EXP_ID}] Installing minimal Rust toolchain via rustup...", flush=True)
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


def _summarize_slice(rows: Sequence[Dict[str, Any]], domain: str, max_len: int) -> Dict[str, Any]:
    def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
        xs = [float(v) for v in vals if v is not None]
        return round(sum(xs) / len(xs), 6) if xs else None

    def _med(vals: Sequence[Optional[float]]) -> Optional[float]:
        xs = [float(v) for v in vals if v is not None]
        return round(float(statistics.median(xs)), 4) if xs else None

    ckey = f"ml{max_len}"
    n = len(rows)
    cells = [r[domain]["cutoffs"][ckey] for r in rows]
    dec_ok = sum(1 for c in cells if c.get("decode_ok"))
    trunc_cnt = sum(1 for c in cells if c.get("truncated"))
    return {
        "n_pieces": n,
        "tok_f1_mean": _mean([c.get("tok_f1") for c in cells]),
        "tok_lcs_mean": _mean([c.get("tok_lcs") for c in cells]),
        "note_f1_mean": _mean([c.get("note_f1") for c in cells]),
        "note_ser_mean": _mean([c.get("note_ser") for c in cells]),
        "decode_ok_rate": round(dec_ok / max(1, n), 4),
        "truncated_pages": trunc_cnt,
        "truncated_rate": round(trunc_cnt / max(1, n), 4),
        "elapsed_secs_median_horizon": _med([r[domain].get("elapsed_secs") for r in rows]),
        "tok_per_sec_mean_horizon": _mean([r[domain].get("tok_per_sec") for r in rows]),
    }


def _summarize_resolution(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pr_rows = [r for r in rows if not r["handwritten"]]
    hw_rows = [r for r in rows if r["handwritten"]]
    by_max_len: Dict[str, Any] = {}
    for ml in MAX_LENS:
        by_max_len[str(ml)] = {
            "printed_clean": _summarize_slice(pr_rows, "clean", ml),
            "printed_photo": _summarize_slice(pr_rows, "photo", ml),
            "hw_clean": _summarize_slice(hw_rows, "clean", ml),
            "hw_photo": _summarize_slice(hw_rows, "photo", ml),
            "all45_clean": _summarize_slice(rows, "clean", ml),
            "all45_photo": _summarize_slice(rows, "photo", ml),
        }
    total_decode_secs = round(
        sum(
            float(r["clean"]["elapsed_secs"]) + float(r["photo"]["elapsed_secs"])
            for r in rows
        ),
        3,
    )
    return {
        "total_decode_secs_90pages_horizon": total_decode_secs,
        "by_max_len": by_max_len,
    }


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
            f"torch={torch.__version__}) | resolutions={RESOLUTIONS} max_lens={MAX_LENS}",
            flush=True,
        )

        api = HfApi(token=hf_token)

        # 1. Download & unpack suite31_hw_eval dataset
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

        # 2. Download iter10_soup_baseline checkpoint
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

        # 3. Wire Linux Rust binaries & import UniOMR evaluator
        mvd_dir = str(repo_root / "tmp-train" / "mvd")
        if mvd_dir not in sys.path:
            sys.path.insert(0, mvd_dir)
        import eval_sys_gate as esg
        import eval_uni_suite31 as ev
        import uni_omr

        esg.ASER = aser_bin
        esg.EHARNESS = eharness_bin

        model, cfg, ckpt_meta = ev.load_uni_checkpoint(ckpt_path, device=device)
        _, pieces = ev.resolve_manifest_pieces(manifest_path)
        _, _, eos_id, _, _, _ = ev._vocab_spec(int(cfg.vocab_size))
        print(
            f"[{EXP_ID}] Loaded ckpt step={ckpt_meta.get('step')} | pieces={len(pieces)}",
            flush=True,
        )

        # 4. Upload initial RUNNING ledger to HF
        workdir = Path(tempfile.mkdtemp(prefix="ov35b_lower_res_"))
        try:
            init_ledger = {
                "exp_id": EXP_ID,
                "status": "RUNNING",
                "started_utc": started_iso,
                "platform": platform_str,
                "hf_eval_path": HF_EVAL_PATH,
                "ckpt_hf_path": HF_CKPT_PATH,
                "resolutions": RESOLUTIONS,
                "max_lens": MAX_LENS,
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

        # 5. Pre-compute GT reference notes once
        refs_cache: Dict[str, Tuple[List[str], Path]] = {}
        for piece in pieces:
            slug = piece["slug"]
            refs_cache[slug] = esg.suite31_page_refs(piece["gt"], "ov35b-" + slug, workdir)

        # 6. Load resume state if present
        partial_path = CACHE_ROOT / "evals" / f"{EXP_ID}_partial.json"
        final_local_path = CACHE_ROOT / "evals" / f"{EXP_ID}.json"
        results_by_res: Dict[str, Dict[str, Any]] = {}
        if partial_path.exists():
            try:
                prev = json.loads(partial_path.read_text(encoding="utf-8"))
                if isinstance(prev.get("resolutions"), dict):
                    results_by_res = prev["resolutions"]
                    print(
                        f"[{EXP_ID}] Resuming: {len(results_by_res)}/{len(RESOLUTIONS)} resolutions done",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[{EXP_ID}] Ignoring unreadable partial file: {exc}", flush=True)

        def _score_cutoff(
            slug: str,
            domain: str,
            res_px: int,
            ml: int,
            seq_ids: List[int],
            ref_page_toks: List[str],
            page_ref_notes: Path,
        ) -> Tuple[List[str], Dict[str, Any]]:
            # seq_ids[0] is BOS; generated tokens are seq_ids[1 : 1 + ml]
            sliced_ids = seq_ids[: 1 + ml]
            hit_eos = eos_id in sliced_ids[1:]
            pred_raw = ev.decode_uni_compat(
                sliced_ids, vocab_size=int(cfg.vocab_size), skip_special=True
            )
            if not hit_eos:
                pred_raw = ev.trim_truncated_tokens(pred_raw)
            hyp_page_toks, n_sys = ev.stitch_uni_tokens(pred_raw)

            _, _, tok_f1 = uni_omr.compute_token_f1(ref_page_toks, hyp_page_toks)
            tok_lcs = ev.compute_token_lcs_f1(ref_page_toks, hyp_page_toks)
            tag = f"r{res_px}-ml{ml}"
            hyp_txt = workdir / f"{tag}-{slug}-{domain}-hyp.txt"
            hyp_txt.write_text(" ".join(hyp_page_toks) + "\n", encoding="utf-8")
            hyp_notes = workdir / f"{tag}-{slug}-{domain}-notes.json"
            p_dec = esg.run_cli([
                esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes),
            ])
            if p_dec.returncode != 0 or not hyp_notes.exists():
                return hyp_page_toks, {
                    "tok_f1": round(tok_f1, 6),
                    "tok_lcs": round(tok_lcs, 6),
                    "note_f1": 0.0,
                    "note_ser": 1.0,
                    "decode_ok": False,
                    "zero_reason": f"bekern-decode-rc-{p_dec.returncode}",
                    "truncated": not hit_eos,
                    "n_sys": n_sys,
                    "n_pred_toks": len(hyp_page_toks),
                }
            sc = esg.score_notes_pair(
                page_ref_notes,
                hyp_notes,
                workdir,
                f"{tag}-{slug}-{domain}",
                esg.EHARNESS,
                esg.ASER,
            )
            return hyp_page_toks, {
                "tok_f1": round(tok_f1, 6),
                "tok_lcs": round(tok_lcs, 6),
                "note_f1": round(float(sc.get("f1", 0.0)), 6),
                "note_ser": round(float(sc.get("ser", 1.0)), 6),
                "decode_ok": not bool(sc.get("zeroed")),
                "zero_reason": sc.get("zero_reason"),
                "truncated": not hit_eos,
                "n_sys": n_sys,
                "n_pred_toks": len(hyp_page_toks),
            }

        def _build_report(status: str) -> Dict[str, Any]:
            matrix: List[Dict[str, Any]] = []
            for res_px in RESOLUTIONS:
                rkey = str(res_px)
                if rkey not in results_by_res:
                    continue
                rsum = results_by_res[rkey]["summary"]
                for ml in MAX_LENS:
                    s = rsum["by_max_len"][str(ml)]
                    matrix.append({
                        "max_dim": res_px,
                        "max_len": ml,
                        "printed_clean_note_f1": s["printed_clean"]["note_f1_mean"],
                        "printed_photo_note_f1": s["printed_photo"]["note_f1_mean"],
                        "hw_clean_note_f1": s["hw_clean"]["note_f1_mean"],
                        "hw_photo_note_f1": s["hw_photo"]["note_f1_mean"],
                        "all45_clean_note_f1": s["all45_clean"]["note_f1_mean"],
                        "all45_photo_note_f1": s["all45_photo"]["note_f1_mean"],
                        "truncated_pages_90": (
                            s["all45_clean"]["truncated_pages"]
                            + s["all45_photo"]["truncated_pages"]
                        ),
                        "total_decode_secs_horizon": rsum["total_decode_secs_90pages_horizon"],
                    })
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
                    "resolutions": RESOLUTIONS,
                    "max_lens": MAX_LENS,
                    "decode_horizon": DECODE_HORIZON,
                    "use_grammar": True,
                    "strict_grammar": False,
                    "use_mtp_spec": True,
                    "n_pieces": len(pieces),
                },
                "matrix_table": matrix,
                "resolutions": results_by_res,
            }

        last_save_ts = time.time()
        save_interval_s = max(30.0, SAVE_EVERY_MINUTES * 60.0)

        for r_idx, res_px in enumerate(RESOLUTIONS, start=1):
            rkey = str(res_px)
            if rkey in results_by_res:
                print(f"[{EXP_ID}] [{r_idx}/{len(RESOLUTIONS)}] Skipping completed {res_px}px", flush=True)
                continue

            print(
                f"[{EXP_ID}] [{r_idx}/{len(RESOLUTIONS)}] Running max_dim={res_px}px "
                f"(horizon max_len={DECODE_HORIZON})",
                flush=True,
            )
            piece_rows: List[Dict[str, Any]] = []
            for p_idx, piece in enumerate(pieces, start=1):
                slug = piece["slug"]
                ref_page_toks, page_ref_notes = refs_cache[slug]
                prow: Dict[str, Any] = {
                    "piece": slug,
                    "handwritten": piece["handwritten"],
                    "stratum": piece["stratum"],
                    "ref_tokens": len(ref_page_toks),
                }

                for domain in ("clean", "photo"):
                    with Image.open(piece[domain]) as raw_im:
                        pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), res_px)
                    img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)

                    t0 = time.perf_counter()
                    with torch.no_grad():
                        ctx = (
                            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                            if use_bf16
                            else contextlib.nullcontext()
                        )
                        with ctx:
                            seq, mtp_stats = ev.generate_mtp_spec(
                                model,
                                img_t,
                                max_len=DECODE_HORIZON,
                                use_grammar=True,
                                strict_grammar=False,
                                return_stats=True,
                                measure_beam_size=0,
                            )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = max(1e-6, time.perf_counter() - t0)

                    seq_ids = seq[0].detach().cpu().tolist()
                    raw_gen_len = max(0, len(seq_ids) - 1)
                    hit_eos_horizon = eos_id in seq_ids[1:]

                    cutoffs: Dict[str, Any] = {}
                    prev_toks: Optional[List[str]] = None
                    prev_score: Optional[Dict[str, Any]] = None
                    for ml in MAX_LENS:
                        sliced = seq_ids[: 1 + ml]
                        # If previous cutoff already hit EOS, identical sequence -> reuse score!
                        if prev_score is not None and not prev_score["truncated"] and prev_toks is not None:
                            cutoffs[f"ml{ml}"] = dict(prev_score)
                        else:
                            hyp_toks, sc_dict = _score_cutoff(
                                slug, domain, res_px, ml, sliced, ref_page_toks, page_ref_notes
                            )
                            cutoffs[f"ml{ml}"] = sc_dict
                            prev_toks = hyp_toks
                            prev_score = sc_dict

                    prow[domain] = {
                        "raw_gen_len": raw_gen_len,
                        "hit_eos_horizon": hit_eos_horizon,
                        "elapsed_secs": round(elapsed, 4),
                        "tok_per_sec": round(max(1, raw_gen_len) / elapsed, 2),
                        "mtp_acceptance_rate": mtp_stats.get("acceptance_rate"),
                        "cutoffs": cutoffs,
                    }

                piece_rows.append(prow)
                if p_idx % 15 == 0 or p_idx == len(pieces):
                    c_max = prow["clean"]["cutoffs"][f"ml{DECODE_HORIZON}"]["note_f1"]
                    p_max = prow["photo"]["cutoffs"][f"ml{DECODE_HORIZON}"]["note_f1"]
                    print(
                        f"[{EXP_ID}] {res_px}px progress {p_idx}/{len(pieces)} | "
                        f"last={slug} clean_nf1={c_max:.3f} photo_nf1={p_max:.3f}",
                        flush=True,
                    )
                if (time.time() - last_save_ts) >= save_interval_s:
                    _atomic_write_json(partial_path, _build_report("RUNNING"))
                    last_save_ts = time.time()

            res_summary = _summarize_resolution(piece_rows)
            results_by_res[rkey] = {
                "max_dim": res_px,
                "summary": res_summary,
                "pieces": piece_rows,
            }
            for ml in MAX_LENS:
                s = res_summary["by_max_len"][str(ml)]
                print(
                    f"[{EXP_ID}] DONE {res_px}px / max_len={ml}: "
                    f"all45_clean={s['all45_clean']['note_f1_mean']:.4f} "
                    f"all45_photo={s['all45_photo']['note_f1_mean']:.4f} "
                    f"hw_clean={s['hw_clean']['note_f1_mean']:.4f} "
                    f"trunc={s['all45_clean']['truncated_pages'] + s['all45_photo']['truncated_pages']}/90",
                    flush=True,
                )
            _atomic_write_json(partial_path, _build_report("RUNNING"))
            last_save_ts = time.time()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_report = _build_report("COMPLETED")
        _atomic_write_json(final_local_path, final_report)
        _atomic_write_json(partial_path, final_report)

        print("=" * 96, flush=True)
        print(f"[{EXP_ID}] FINAL 4x4 MATRIX (elapsed={final_report['elapsed_secs']}s)", flush=True)
        print(json.dumps(final_report["matrix_table"], indent=2), flush=True)
        print("=" * 96, flush=True)

        api.upload_file(
            path_or_fileobj=str(final_local_path),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload final max_len x resolution sweep JSON (Colab L4)",
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
            "matrix_table": final_report["matrix_table"],
        }
        tmp_final_ledger = workdir / "run_ledger_final.json"
        _atomic_write_json(tmp_final_ledger, final_ledger)
        api.upload_file(
            path_or_fileobj=str(tmp_final_ledger),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark COMPLETED with 4x4 matrix (Colab L4)",
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

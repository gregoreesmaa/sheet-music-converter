#!/usr/bin/env python3
"""Google Colab L4 runner for Exp 34: 2026-11-26T01-38-ov-34-measure-beam-header-guard-sweep.

Sweeps header-guarded `--measure-beam-size` in {1, 2, 4, 8} x `const_offset_limit`
in {2, 3, 5} across all 45 `suite31_hw` pieces (`clean` + `photo` = 90 pages),
measuring Strict Note F1, Token LCS F1, header-guard trip counts, and runtime
overhead relative to `measure_beam_size=1` (MTP-4 + FSA grammar baseline).

Expects on Colab VM:
  - `/content/bundle.tar.gz` (contains `sheet-music-converter/...` and `mscz-rs/`)
  - `/tmp/.hf_token` (Hugging Face token; chmod 600, deleted immediately and in `finally:`)
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict
from datetime import datetime, timezone
import functools
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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T01-38-ov-34-measure-beam-header-guard-sweep"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_CKPT_PATH = "checkpoints/iter10_soup_baseline/uni_omr_soup.pt"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

MAX_DIM = 1536
MAX_LEN = 4096
MAX_BAR_TOKS = 120
SAVE_EVERY_MINUTES = 5.0

# (measure_beam_size, const_offset_limit)
# measure_beam_size=1 is the shared greedy+MTP-4+FSA baseline (use_measure_beam=False).
SWEEP_CONFIGS: List[Tuple[int, int]] = [
    (1, 0),
    (2, 2),
    (2, 3),
    (2, 5),
    (4, 2),
    (4, 3),
    (4, 5),
    (8, 2),
    (8, 3),
    (8, 5),
]


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


@contextlib.contextmanager
def _patch_header_guard_limit(ev_mod: Any, const_offset_limit: int) -> Iterator[None]:
    """Bind MeasureBeamHeaderGuard's default const_offset_limit for one sweep cell."""
    orig_cls = ev_mod.MeasureBeamHeaderGuard
    ev_mod.MeasureBeamHeaderGuard = functools.partial(
        orig_cls, const_offset_limit=int(const_offset_limit)
    )
    try:
        yield
    finally:
        ev_mod.MeasureBeamHeaderGuard = orig_cls


def _cfg_key(beam_size: int, limit: int) -> str:
    if beam_size <= 1:
        return "mbeam1_baseline"
    return f"mbeam{beam_size}_lim{limit}"


def _summarize_slice(rows: Sequence[Dict[str, Any]], domain: str) -> Dict[str, Any]:
    def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
        xs = [float(v) for v in vals if v is not None]
        return round(sum(xs) / len(xs), 6) if xs else None

    def _med(vals: Sequence[Optional[float]]) -> Optional[float]:
        xs = [float(v) for v in vals if v is not None]
        return round(float(statistics.median(xs)), 4) if xs else None

    p = domain + "_"
    n = len(rows)
    dec_ok = sum(1 for r in rows if r.get(p + "decode_ok"))
    triggers = sum(int(r.get(p + "measure_beam_triggers", 0) or 0) for r in rows)
    repaired = sum(int(r.get(p + "measure_beam_repaired", 0) or 0) for r in rows)
    tripped = sum(1 for r in rows if r.get(p + "header_guard_tripped"))
    total_secs = round(sum(float(r.get(p + "elapsed_secs", 0.0) or 0.0) for r in rows), 3)
    return {
        "n_pieces": n,
        "tok_f1_mean": _mean([r.get(p + "tok_f1") for r in rows]),
        "tok_lcs_mean": _mean([r.get(p + "tok_lcs") for r in rows]),
        "note_f1_mean": _mean([r.get(p + "note_f1") for r in rows]),
        "note_ser_mean": _mean([r.get(p + "note_ser") for r in rows]),
        "decode_ok_rate": round(dec_ok / max(1, n), 4),
        "tok_per_sec_mean": _mean([r.get(p + "tok_per_sec") for r in rows]),
        "elapsed_secs_total": total_secs,
        "elapsed_secs_median": _med([r.get(p + "elapsed_secs") for r in rows]),
        "measure_beam_triggers": triggers,
        "measure_beam_repaired": repaired,
        "header_guard_tripped_pages": tripped,
    }


def _summarize_config(rows: Sequence[Dict[str, Any]], base_total_secs: Optional[float] = None) -> Dict[str, Any]:
    pr_rows = [r for r in rows if not r["handwritten"]]
    hw_rows = [r for r in rows if r["handwritten"]]
    summary = {
        "printed_clean": _summarize_slice(pr_rows, "clean"),
        "printed_photo": _summarize_slice(pr_rows, "photo"),
        "hw_clean": _summarize_slice(hw_rows, "clean"),
        "hw_photo": _summarize_slice(hw_rows, "photo"),
        "all45_clean": _summarize_slice(rows, "clean"),
        "all45_photo": _summarize_slice(rows, "photo"),
    }
    tot_secs = round(
        float(summary["all45_clean"]["elapsed_secs_total"])
        + float(summary["all45_photo"]["elapsed_secs_total"]),
        3,
    )
    overhead_pct: Optional[float] = None
    if base_total_secs is not None and base_total_secs > 0:
        overhead_pct = round(((tot_secs - base_total_secs) / base_total_secs) * 100.0, 2)
    summary["runtime"] = {
        "total_decode_secs_90pages": tot_secs,
        "overhead_vs_mbeam1_pct": overhead_pct,
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
            f"torch={torch.__version__}) | configs={len(SWEEP_CONFIGS)} max_dim={MAX_DIM} max_len={MAX_LEN}",
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
        workdir = Path(tempfile.mkdtemp(prefix="ov34_mbeam_"))
        try:
            init_ledger = {
                "exp_id": EXP_ID,
                "status": "RUNNING",
                "started_utc": started_iso,
                "platform": platform_str,
                "hf_eval_path": HF_EVAL_PATH,
                "ckpt_hf_path": HF_CKPT_PATH,
                "sweep_configs": [
                    {"key": _cfg_key(b, l), "measure_beam_size": b, "const_offset_limit": l}
                    for b, l in SWEEP_CONFIGS
                ],
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

        # 5. Pre-compute GT reference notes & resized tensors once across all configs
        refs_cache: Dict[str, Tuple[List[str], Path]] = {}
        img_cache: Dict[Tuple[str, str], torch.Tensor] = {}
        for piece in pieces:
            slug = piece["slug"]
            refs_cache[slug] = esg.suite31_page_refs(piece["gt"], "ov34-" + slug, workdir)
            for domain in ("clean", "photo"):
                with Image.open(piece[domain]) as raw_im:
                    pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), MAX_DIM)
                img_cache[(slug, domain)] = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0)

        # 6. Load resume state if present
        partial_path = CACHE_ROOT / "evals" / f"{EXP_ID}_partial.json"
        final_local_path = CACHE_ROOT / "evals" / f"{EXP_ID}.json"
        results_by_cfg: Dict[str, Dict[str, Any]] = {}
        if partial_path.exists():
            try:
                prev = json.loads(partial_path.read_text(encoding="utf-8"))
                if isinstance(prev.get("configs"), dict):
                    results_by_cfg = prev["configs"]
                    print(
                        f"[{EXP_ID}] Resuming: {len(results_by_cfg)}/{len(SWEEP_CONFIGS)} configs done",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[{EXP_ID}] Ignoring unreadable partial file: {exc}", flush=True)

        # Cache baseline decoded tokens & scores per (slug, domain) so configs with
        # identical output (measure_beam_repaired == 0) skip redundant Rust CLI scoring.
        baseline_hyp_cache: Dict[Tuple[str, str], Tuple[List[str], Dict[str, Any]]] = {}

        def _score_hyp(
            slug: str,
            domain: str,
            tag_prefix: str,
            hyp_page_toks: List[str],
            ref_page_toks: List[str],
            page_ref_notes: Path,
        ) -> Dict[str, Any]:
            _, _, tok_f1 = uni_omr.compute_token_f1(ref_page_toks, hyp_page_toks)
            tok_lcs = ev.compute_token_lcs_f1(ref_page_toks, hyp_page_toks)
            hyp_txt = workdir / f"{tag_prefix}-{slug}-{domain}-hyp.txt"
            hyp_txt.write_text(" ".join(hyp_page_toks) + "\n", encoding="utf-8")
            hyp_notes = workdir / f"{tag_prefix}-{slug}-{domain}-notes.json"
            p_dec = esg.run_cli([
                esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes),
            ])
            if p_dec.returncode != 0 or not hyp_notes.exists():
                return {
                    "tok_f1": round(tok_f1, 6),
                    "tok_lcs": round(tok_lcs, 6),
                    "note_f1": 0.0,
                    "note_ser": 1.0,
                    "decode_ok": False,
                    "zero_reason": f"bekern-decode-rc-{p_dec.returncode}",
                }
            sc = esg.score_notes_pair(
                page_ref_notes,
                hyp_notes,
                workdir,
                f"{tag_prefix}-{slug}-{domain}",
                esg.EHARNESS,
                esg.ASER,
            )
            return {
                "tok_f1": round(tok_f1, 6),
                "tok_lcs": round(tok_lcs, 6),
                "note_f1": round(float(sc.get("f1", 0.0)), 6),
                "note_ser": round(float(sc.get("ser", 1.0)), 6),
                "decode_ok": not bool(sc.get("zeroed")),
                "zero_reason": sc.get("zero_reason"),
            }

        def _build_report(status: str) -> Dict[str, Any]:
            base_secs = None
            base_entry = results_by_cfg.get(_cfg_key(1, 0))
            if isinstance(base_entry, dict):
                base_secs = base_entry.get("summary", {}).get("runtime", {}).get("total_decode_secs_90pages")
            comparison: List[Dict[str, Any]] = []
            for b, l in SWEEP_CONFIGS:
                ckey = _cfg_key(b, l)
                if ckey not in results_by_cfg:
                    continue
                s = results_by_cfg[ckey]["summary"]
                comparison.append({
                    "key": ckey,
                    "measure_beam_size": b,
                    "const_offset_limit": l,
                    "printed_clean_note_f1": s["printed_clean"]["note_f1_mean"],
                    "printed_photo_note_f1": s["printed_photo"]["note_f1_mean"],
                    "hw_clean_note_f1": s["hw_clean"]["note_f1_mean"],
                    "hw_photo_note_f1": s["hw_photo"]["note_f1_mean"],
                    "all45_clean_note_f1": s["all45_clean"]["note_f1_mean"],
                    "all45_photo_note_f1": s["all45_photo"]["note_f1_mean"],
                    "total_decode_secs": s["runtime"]["total_decode_secs_90pages"],
                    "overhead_vs_mbeam1_pct": s["runtime"]["overhead_vs_mbeam1_pct"],
                    "triggers_total": (
                        s["all45_clean"]["measure_beam_triggers"]
                        + s["all45_photo"]["measure_beam_triggers"]
                    ),
                    "repaired_total": (
                        s["all45_clean"]["measure_beam_repaired"]
                        + s["all45_photo"]["measure_beam_repaired"]
                    ),
                    "guard_tripped_pages": (
                        s["all45_clean"]["header_guard_tripped_pages"]
                        + s["all45_photo"]["header_guard_tripped_pages"]
                    ),
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
                    "max_dim": MAX_DIM,
                    "max_len": MAX_LEN,
                    "max_bar_toks": MAX_BAR_TOKS,
                    "use_grammar": True,
                    "strict_grammar": False,
                    "use_mtp_spec": True,
                    "n_pieces": len(pieces),
                },
                "base_total_decode_secs": base_secs,
                "comparison_table": comparison,
                "configs": results_by_cfg,
            }

        last_save_ts = time.time()
        save_interval_s = max(30.0, SAVE_EVERY_MINUTES * 60.0)
        base_total_secs: Optional[float] = None
        if _cfg_key(1, 0) in results_by_cfg:
            base_total_secs = (
                results_by_cfg[_cfg_key(1, 0)]
                .get("summary", {})
                .get("runtime", {})
                .get("total_decode_secs_90pages")
            )

        for cfg_idx, (beam_size, offset_limit) in enumerate(SWEEP_CONFIGS, start=1):
            ckey = _cfg_key(beam_size, offset_limit)
            if ckey in results_by_cfg:
                print(f"[{EXP_ID}] [{cfg_idx}/{len(SWEEP_CONFIGS)}] Skipping completed {ckey}", flush=True)
                continue

            print(
                f"[{EXP_ID}] [{cfg_idx}/{len(SWEEP_CONFIGS)}] Running {ckey} "
                f"(measure_beam_size={beam_size}, const_offset_limit={offset_limit})",
                flush=True,
            )
            piece_rows: List[Dict[str, Any]] = []
            mbeam_arg = 0 if beam_size <= 1 else int(beam_size)
            guard_ctx = (
                _patch_header_guard_limit(ev, offset_limit)
                if mbeam_arg > 1
                else contextlib.nullcontext()
            )

            with guard_ctx:
                for p_idx, piece in enumerate(pieces, start=1):
                    slug = piece["slug"]
                    ref_page_toks, page_ref_notes = refs_cache[slug]
                    prow: Dict[str, Any] = {
                        "piece": slug,
                        "handwritten": piece["handwritten"],
                        "stratum": piece["stratum"],
                    }

                    for domain in ("clean", "photo"):
                        img_t = img_cache[(slug, domain)].to(device, non_blocking=True)
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
                                    max_len=MAX_LEN,
                                    use_grammar=True,
                                    strict_grammar=False,
                                    return_stats=True,
                                    measure_beam_size=mbeam_arg,
                                    max_bar_toks=MAX_BAR_TOKS,
                                )
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        elapsed = max(1e-6, time.perf_counter() - t0)

                        seq_ids = seq[0].detach().cpu().tolist()
                        pred_raw_toks = ev.decode_uni_compat(
                            seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
                        )
                        if eos_id not in seq_ids[1:]:
                            pred_raw_toks = ev.trim_truncated_tokens(pred_raw_toks)
                        hyp_page_toks, n_sys = ev.stitch_uni_tokens(pred_raw_toks)

                        cached = baseline_hyp_cache.get((slug, domain))
                        if cached is not None and hyp_page_toks == cached[0]:
                            scores = dict(cached[1])
                        else:
                            scores = _score_hyp(
                                slug, domain, ckey, hyp_page_toks, ref_page_toks, page_ref_notes
                            )
                            if beam_size <= 1:
                                baseline_hyp_cache[(slug, domain)] = (hyp_page_toks, dict(scores))

                        n_gen = max(1, len(pred_raw_toks) + 1)
                        pfx = domain + "_"
                        prow[pfx + "tok_f1"] = scores["tok_f1"]
                        prow[pfx + "tok_lcs"] = scores["tok_lcs"]
                        prow[pfx + "note_f1"] = scores["note_f1"]
                        prow[pfx + "note_ser"] = scores["note_ser"]
                        prow[pfx + "decode_ok"] = scores["decode_ok"]
                        prow[pfx + "zero_reason"] = scores["zero_reason"]
                        prow[pfx + "n_sys"] = n_sys
                        prow[pfx + "n_pred_toks"] = len(hyp_page_toks)
                        prow[pfx + "elapsed_secs"] = round(elapsed, 4)
                        prow[pfx + "tok_per_sec"] = round(n_gen / elapsed, 2)
                        prow[pfx + "measure_beam_triggers"] = int(mtp_stats.get("measure_beam_triggers", 0))
                        prow[pfx + "measure_beam_repaired"] = int(mtp_stats.get("measure_beam_repaired", 0))
                        prow[pfx + "header_guard_tripped"] = bool(
                            mtp_stats.get("measure_beam_header_guard_tripped", False)
                        )

                    piece_rows.append(prow)
                    if p_idx % 15 == 0 or p_idx == len(pieces):
                        print(
                            f"[{EXP_ID}] {ckey} progress {p_idx}/{len(pieces)} | "
                            f"last={slug} clean_nf1={prow['clean_note_f1']:.3f} photo_nf1={prow['photo_note_f1']:.3f}",
                            flush=True,
                        )
                    if (time.time() - last_save_ts) >= save_interval_s:
                        _atomic_write_json(partial_path, _build_report("RUNNING"))
                        last_save_ts = time.time()

            cfg_summary = _summarize_config(piece_rows, base_total_secs=base_total_secs)
            if beam_size <= 1:
                base_total_secs = cfg_summary["runtime"]["total_decode_secs_90pages"]
            results_by_cfg[ckey] = {
                "measure_beam_size": beam_size,
                "const_offset_limit": offset_limit,
                "summary": cfg_summary,
                "pieces": piece_rows,
            }
            print(
                f"[{EXP_ID}] DONE {ckey}: all45_clean={cfg_summary['all45_clean']['note_f1_mean']:.4f} "
                f"all45_photo={cfg_summary['all45_photo']['note_f1_mean']:.4f} "
                f"hw_clean={cfg_summary['hw_clean']['note_f1_mean']:.4f} "
                f"secs={cfg_summary['runtime']['total_decode_secs_90pages']} "
                f"overhead={cfg_summary['runtime']['overhead_vs_mbeam1_pct']}%",
                flush=True,
            )
            _atomic_write_json(partial_path, _build_report("RUNNING"))
            last_save_ts = time.time()

        final_report = _build_report("COMPLETED")
        _atomic_write_json(final_local_path, final_report)
        _atomic_write_json(partial_path, final_report)

        print("=" * 96, flush=True)
        print(f"[{EXP_ID}] FINAL COMPARISON TABLE (elapsed={final_report['elapsed_secs']}s)", flush=True)
        print(json.dumps(final_report["comparison_table"], indent=2), flush=True)
        print("=" * 96, flush=True)

        api.upload_file(
            path_or_fileobj=str(final_local_path),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload final measure-beam header-guard sweep JSON (Colab L4)",
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
            "comparison_table": final_report["comparison_table"],
        }
        tmp_final_ledger = workdir / "run_ledger_final.json"
        _atomic_write_json(tmp_final_ledger, final_ledger)
        api.upload_file(
            path_or_fileobj=str(tmp_final_ledger),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark COMPLETED with comparison table (Colab L4)",
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

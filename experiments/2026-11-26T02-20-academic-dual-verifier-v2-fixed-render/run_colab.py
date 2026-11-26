#!/usr/bin/env python3
"""Academic Experiment A v2 on Google Colab L4:
Dual-Verified Render-Cycle Verifier Gate & TTC Scaling Curve (K in {1, 4, 16})
with FIXED visual rendering (importlib.invalidate_caches + fail-fast startup
render smoke test + thread-safe _VEROVIO_LOCK) and MATCHED rollout_max_len=2048.

Predecessor: 2026-11-26T01-30-academic-dual-verifier-rlvr-ttc (ITERATE)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T02-20-academic-dual-verifier-v2-fixed-render"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

ROLLOUT_MAX_LEN = 2048
GREEDY_MAX_LEN = 2048


def _run(cmd: Sequence[str], *, env: Optional[Dict[str, str]] = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"[cmd] {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(list(cmd), env=env, check=check)


def _setup_env() -> Tuple[Path, Path]:
    """Unpack workspace bundle, install deps (+ invalidate_caches), build Rust eval binaries."""
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    mvd_dir = WORK_ROOT / "sheet-music-converter" / "tmp-train" / "mvd"
    if not mvd_dir.exists():
        assert BUNDLE_PATH.exists(), f"Missing bundle at {BUNDLE_PATH}"
        print(f"[setup] extracting {BUNDLE_PATH} -> {WORK_ROOT}", flush=True)
        with tarfile.open(str(BUNDLE_PATH), "r:*") as tf:
            tf.extractall(path=str(WORK_ROOT))

    # Install lightweight Python packages (torch is pre-installed on Colab)
    missing_pkgs = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("verovio", "verovio"),
        ("resvg_py", "resvg-py"),
        ("scipy", "scipy"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing_pkgs.append(pkg)
    if missing_pkgs:
        _run([sys.executable, "-m", "pip", "install", "-q", *missing_pkgs])
        # CRITICAL: clear Python finder caches so newly installed packages import cleanly
        importlib.invalidate_caches()
        for mod in ("huggingface_hub", "verovio", "resvg_py", "scipy"):
            sys.modules.pop(mod, None)

    # Build Rust assembly-serialize & eval-harness if not already built
    aser_bin = WORK_ROOT / "sheet-music-converter" / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize"
    eharness_bin = WORK_ROOT / "sheet-music-converter" / "converter" / "eval-harness" / "target" / "release" / "eval-harness"

    cargo_env = os.environ.copy()
    cargo_bin_dir = str(Path.home() / ".cargo" / "bin")
    cargo_env["PATH"] = cargo_bin_dir + ":" + cargo_env.get("PATH", "")

    if not aser_bin.exists() or not eharness_bin.exists():
        if shutil.which("cargo", path=cargo_env["PATH"]) is None:
            print("[setup] installing minimal rustup toolchain...", flush=True)
            _run(
                [
                    "bash",
                    "-lc",
                    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal",
                ],
                env=cargo_env,
            )
        t_build = time.perf_counter()
        if not aser_bin.exists():
            _run(
                [
                    "cargo",
                    "build",
                    "--release",
                    "--manifest-path",
                    str(WORK_ROOT / "sheet-music-converter" / "converter" / "assembly-serialize" / "Cargo.toml"),
                ],
                env=cargo_env,
            )
        if not eharness_bin.exists():
            _run(
                [
                    "cargo",
                    "build",
                    "--release",
                    "--manifest-path",
                    str(WORK_ROOT / "sheet-music-converter" / "converter" / "eval-harness" / "Cargo.toml"),
                ],
                env=cargo_env,
            )
        print(f"[setup] Rust release builds finished in {time.perf_counter() - t_build:.1f}s", flush=True)

    for b in (aser_bin, eharness_bin):
        exe_link = b.with_name(b.name + ".exe")
        if not exe_link.exists():
            exe_link.symlink_to(b)

    return aser_bin, eharness_bin


def _select_pieces(pieces: Sequence[Dict[str, Any]], n_printed: int = 10) -> List[Dict[str, Any]]:
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


def _safe_spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    from scipy.stats import spearmanr

    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    res = spearmanr(xs, ys)
    val = float(res.statistic)
    if math.isnan(val) or math.isinf(val):
        return None
    return round(val, 6)


def _mean_opt(vals: Sequence[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return round(sum(xs) / len(xs), 6) if xs else None


def _summarize_group(pages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not pages:
        return {"n_pages": 0}

    all_f1: List[float] = []
    all_rsym: List[float] = []
    all_rvis: List[float] = []
    all_rdual: List[float] = []

    within_rho_sym: List[Optional[float]] = []
    within_rho_vis: List[Optional[float]] = []
    within_rho_dual: List[Optional[float]] = []

    greedy_f1s: List[float] = []
    k1_samp_f1s: List[float] = []

    bo4_dual_f1s: List[float] = []
    bo4_sym_f1s: List[float] = []
    bo4_vis_f1s: List[float] = []
    oracle4_f1s: List[float] = []

    bo16_dual_f1s: List[float] = []
    bo16_sym_f1s: List[float] = []
    bo16_vis_f1s: List[float] = []
    oracle16_f1s: List[float] = []
    oracle16_plus_greedy_f1s: List[float] = []
    bo16_plus_greedy_dual_f1s: List[float] = []
    bo16_plus_greedy_sym_f1s: List[float] = []
    bo16_plus_greedy_vis_f1s: List[float] = []

    render_ok_cnt = 0
    decode_ok_cnt = 0
    total_rollouts = 0

    for pg in pages:
        g = pg["greedy"]
        rolls = pg["rollouts"]
        greedy_f1s.append(float(g["note_f1"]))
        k1_samp_f1s.append(sum(float(r["note_f1"]) for r in rolls) / max(1, len(rolls)))

        f1s = [float(r["note_f1"]) for r in rolls]
        rsyms = [float(r["r_sym"]) for r in rolls]
        rviss = [float(r["r_vis"]) for r in rolls]
        rduals = [float(r["r_dual"]) for r in rolls]

        all_f1.extend(f1s)
        all_rsym.extend(rsyms)
        all_rvis.extend(rviss)
        all_rdual.extend(rduals)

        within_rho_sym.append(_safe_spearman(rsyms, f1s))
        within_rho_vis.append(_safe_spearman(rviss, f1s))
        within_rho_dual.append(_safe_spearman(rduals, f1s))

        for r in rolls:
            total_rollouts += 1
            if r.get("render_ok"):
                render_ok_cnt += 1
            if r.get("decode_ok"):
                decode_ok_cnt += 1

        c_dual, c_sym, c_vis, c_orc = [], [], [], []
        for start in range(0, min(16, len(rolls)), 4):
            chunk = rolls[start : start + 4]
            if not chunk:
                continue
            c_dual.append(float(max(chunk, key=lambda x: (x["r_dual"], x["r_sym"]))["note_f1"]))
            c_sym.append(float(max(chunk, key=lambda x: x["r_sym"])["note_f1"]))
            c_vis.append(float(max(chunk, key=lambda x: x["r_vis"])["note_f1"]))
            c_orc.append(float(max(chunk, key=lambda x: x["note_f1"])["note_f1"]))
        bo4_dual_f1s.append(sum(c_dual) / len(c_dual))
        bo4_sym_f1s.append(sum(c_sym) / len(c_sym))
        bo4_vis_f1s.append(sum(c_vis) / len(c_vis))
        oracle4_f1s.append(sum(c_orc) / len(c_orc))

        bo16_dual_f1s.append(float(max(rolls, key=lambda x: (x["r_dual"], x["r_sym"]))["note_f1"]))
        bo16_sym_f1s.append(float(max(rolls, key=lambda x: x["r_sym"])["note_f1"]))
        bo16_vis_f1s.append(float(max(rolls, key=lambda x: x["r_vis"])["note_f1"]))
        oracle16_f1s.append(float(max(rolls, key=lambda x: x["note_f1"])["note_f1"]))

        all17 = [g] + list(rolls)
        oracle16_plus_greedy_f1s.append(float(max(all17, key=lambda x: x["note_f1"])["note_f1"]))
        bo16_plus_greedy_dual_f1s.append(
            float(max(all17, key=lambda x: (x["r_dual"], x["r_sym"]))["note_f1"])
        )
        bo16_plus_greedy_sym_f1s.append(
            float(max(all17, key=lambda x: x["r_sym"])["note_f1"])
        )
        bo16_plus_greedy_vis_f1s.append(
            float(max(all17, key=lambda x: x["r_vis"])["note_f1"])
        )

    k1_g = _mean_opt(greedy_f1s) or 0.0
    k1_s = _mean_opt(k1_samp_f1s) or 0.0
    b4_d = _mean_opt(bo4_dual_f1s) or 0.0
    b16_d = _mean_opt(bo16_dual_f1s) or 0.0
    b16_pg_d = _mean_opt(bo16_plus_greedy_dual_f1s) or 0.0
    orc16 = _mean_opt(oracle16_f1s) or 0.0
    orc17 = _mean_opt(oracle16_plus_greedy_f1s) or 0.0

    # Zero-cost offline calibration sweep over fusion weights
    fusion_grid: Dict[str, Dict[str, Any]] = {}
    formulas = {
        "mult_f0.10": lambda c: c["r_sym"] * (0.10 + 0.90 * c["r_vis"]),
        "mult_f0.20": lambda c: c["r_sym"] * (0.20 + 0.80 * c["r_vis"]),
        "mult_f0.40": lambda c: c["r_sym"] * (0.40 + 0.60 * c["r_vis"]),
        "add_w0.30": lambda c: 0.70 * c["r_sym"] + 0.30 * c["r_vis"],
        "add_w0.50": lambda c: 0.50 * c["r_sym"] + 0.50 * c["r_vis"],
        "add_w0.70": lambda c: 0.30 * c["r_sym"] + 0.70 * c["r_vis"],
    }
    for fname, ffn in formulas.items():
        all_sc: List[float] = []
        w_rhos: List[Optional[float]] = []
        b16_pg_vals: List[float] = []
        for pg in pages:
            rolls = pg["rollouts"]
            scs = [float(ffn(r)) for r in rolls]
            f1s = [float(r["note_f1"]) for r in rolls]
            all_sc.extend(scs)
            w_rhos.append(_safe_spearman(scs, f1s))
            all17 = [pg["greedy"]] + list(rolls)
            b16_pg_vals.append(float(max(all17, key=lambda c: (ffn(c), c["r_sym"]))["note_f1"]))
        b16_pg_m = _mean_opt(b16_pg_vals) or 0.0
        fusion_grid[fname] = {
            "spearman_pooled": _safe_spearman(all_sc, all_f1),
            "spearman_within_page_mean": _mean_opt(w_rhos),
            "bo16_plus_greedy": b16_pg_m,
            "delta_vs_greedy": round(b16_pg_m - k1_g, 6),
        }

    return {
        "n_pages": len(pages),
        "n_rollouts_total": total_rollouts,
        "render_ok_rate": round(render_ok_cnt / max(1, total_rollouts), 4),
        "decode_ok_rate": round(decode_ok_cnt / max(1, total_rollouts), 4),
        "spearman_pooled": {
            "r_sym": _safe_spearman(all_rsym, all_f1),
            "r_vis": _safe_spearman(all_rvis, all_f1),
            "r_dual": _safe_spearman(all_rdual, all_f1),
        },
        "spearman_within_page_mean": {
            "r_sym": _mean_opt(within_rho_sym),
            "r_vis": _mean_opt(within_rho_vis),
            "r_dual": _mean_opt(within_rho_dual),
        },
        "ttc_note_f1": {
            "k1_greedy": k1_g,
            "k1_sampled_mean": k1_s,
            "bo4_r_dual": b4_d,
            "bo4_r_sym": _mean_opt(bo4_sym_f1s),
            "bo4_r_vis": _mean_opt(bo4_vis_f1s),
            "oracle4": _mean_opt(oracle4_f1s),
            "bo16_r_dual": b16_d,
            "bo16_r_sym": _mean_opt(bo16_sym_f1s),
            "bo16_r_vis": _mean_opt(bo16_vis_f1s),
            "bo16_plus_greedy_r_dual": b16_pg_d,
            "bo16_plus_greedy_r_sym": _mean_opt(bo16_plus_greedy_sym_f1s),
            "bo16_plus_greedy_r_vis": _mean_opt(bo16_plus_greedy_vis_f1s),
            "oracle16": orc16,
            "oracle16_plus_greedy": orc17,
            "delta_bo16_vs_greedy": round(b16_d - k1_g, 6),
            "delta_bo16_pg_vs_greedy": round(b16_pg_d - k1_g, 6),
            "delta_bo16_vs_k1_sampled": round(b16_d - k1_s, 6),
            "oracle_headroom_vs_greedy": round(orc17 - k1_g, 6),
            "ttc_slope_per_log2K_vs_sampled": round((b16_d - k1_s) / 4.0, 6),
            "ttc_slope_per_log2K_vs_greedy": round((b16_d - k1_g) / 4.0, 6),
            "ttc_slope_per_log2K_pg_vs_greedy": round((b16_pg_d - k1_g) / 4.0, 6),
        },
        "fusion_calibration_grid": fusion_grid,
    }


def main() -> None:
    t_all0 = time.perf_counter()
    try:
        assert TOKEN_PATH.exists(), f"Missing HF token file at {TOKEN_PATH}"
        os.chmod(str(TOKEN_PATH), 0o600)
        hf_token = TOKEN_PATH.read_text(encoding="utf-8").strip()
        assert hf_token, "Empty HF token"

        aser_bin, eharness_bin = _setup_env()

        mvd_dir = WORK_ROOT / "sheet-music-converter" / "tmp-train" / "mvd"
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        import numpy as np
        from PIL import Image
        import torch
        from huggingface_hub import HfApi, hf_hub_download

        import dual_verifier as dv
        import eval_sys_gate as esg
        import eval_uni_suite31 as eval_s31
        import rlvr_grpo
        import uni_omr

        # FAIL-FAST STARTUP SMOKE TEST: never waste GPU if verovio/resvg_py failed to load!
        assert dv.verovio is not None, "FATAL: dual_verifier.verovio is None after pip install!"
        assert dv.resvg_py is not None or dv.cairosvg is not None, "FATAL: no SVG rasterizer loaded!"
        _smoke_bekern = (
            "clef-G-2 key+0 time-4-4 quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar "
            "<sys> clef-G-2 key+0 time-4-4 half G 4 half A 4 bar-light-heavy"
        )
        with ThreadPoolExecutor(max_workers=4) as _smoke_pool:
            _smoke_imgs = list(
                _smoke_pool.map(
                    lambda _: dv.render_bekern_to_gray(_smoke_bekern, aser_bin=aser_bin, use_fallback=True),
                    range(4),
                )
            )
        assert all(im is not None and im.ndim == 2 and im.shape[0] > 40 for im in _smoke_imgs), (
            f"FATAL: startup render smoke test failed: {[None if x is None else x.shape for x in _smoke_imgs]}"
        )
        print(f"[acad-a-v2] startup render smoke test PASSED (4/4 threads ok, shape={_smoke_imgs[0].shape})", flush=True)

        esg.ASER = aser_bin
        esg.EHARNESS = eharness_bin

        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
        use_bf16 = device.type == "cuda" and cc[0] >= 8
        print(f"[acad-a-v2] device={device} ({gpu_name}, cc={cc}) bf16={use_bf16}", flush=True)

        api = HfApi(token=hf_token)

        eval_unpack = CACHE_ROOT / "suite31_hw_eval"
        if not (eval_unpack / ".ok").exists():
            tar_p = Path(
                hf_hub_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    filename="datasets/suite31_hw_eval.tar",
                    token=hf_token,
                    local_dir=str(CACHE_ROOT / "hf"),
                )
            )
            eval_unpack.mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(tar_p), "r:*") as tf:
                tf.extractall(path=str(eval_unpack))
            (eval_unpack / ".ok").write_text("ok\n", encoding="utf-8")

        ckpt_path = Path(
            hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename="checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
                token=hf_token,
                local_dir=str(CACHE_ROOT / "hf"),
            )
        )
        manifests = list(eval_unpack.rglob("suite31_hw_manifest.json"))
        assert manifests, f"suite31_hw_manifest.json not found in {eval_unpack}"
        manifest_path = manifests[0]

        _, all_pieces = eval_s31.resolve_manifest_pieces(manifest_path)
        selected_pieces = _select_pieces(all_pieces, n_printed=10)
        n_hw = sum(1 for p in selected_pieces if p["handwritten"])
        n_pr = sum(1 for p in selected_pieces if not p["handwritten"])
        print(f"[acad-a-v2] selected {len(selected_pieces)} pieces ({n_hw} HW + {n_pr} PR) = {len(selected_pieces) * 2} pages", flush=True)

        model, cfg, ckpt_meta = eval_s31.load_uni_checkpoint(ckpt_path, device=device)
        vocab_size = int(cfg.vocab_size)
        _, _, eos_id, _, _, _ = eval_s31._vocab_spec(vocab_size)
        print(f"[acad-a-v2] loaded uni_omr_soup.pt (step={ckpt_meta.get('step')}, vocab={vocab_size})", flush=True)

        verifier = dv.DualVerifier(
            dv.DualVerifierConfig(aser_bin=aser_bin, use_python_musicxml_fallback=True)
        )
        workdir = Path(tempfile.mkdtemp(prefix="acad_a_v2_s31_"))
        partial_path = Path("/content/exp_a_v2_partial.jsonl")

        def _score_candidate(
            cand_idx: int,
            pred_raw_toks: List[str],
            slug: str,
            domain: str,
            raw_gray: np.ndarray,
            real_prof: Any,
            ref_page_toks: List[str],
            page_ref_notes: Path,
        ) -> Dict[str, Any]:
            hyp_page_toks, n_sys = eval_s31.stitch_uni_tokens(pred_raw_toks)
            tok_lcs = eval_s31.compute_token_lcs_f1(ref_page_toks, hyp_page_toks)

            dv_res = verifier.score(
                raw_gray,
                pred_raw_toks,
                precomputed_real_profiles=real_prof,
            )

            tag_id = f"acadv2-{slug}-{domain}-k{cand_idx}"
            hyp_txt = workdir / f"{tag_id}-hyp.txt"
            hyp_notes = workdir / f"{tag_id}-notes.json"
            hyp_txt.write_text(" ".join(hyp_page_toks) + "\n", encoding="utf-8")
            p_dec = esg.run_cli([
                esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes),
            ])
            if p_dec.returncode != 0 or not hyp_notes.exists():
                note_f1 = 0.0
                note_ser = 1.0
                decode_ok = False
            else:
                sc = esg.score_notes_pair(
                    page_ref_notes,
                    hyp_notes,
                    workdir,
                    tag_id,
                    esg.EHARNESS,
                    esg.ASER,
                )
                note_f1 = float(sc.get("f1", 0.0))
                note_ser = float(sc.get("ser", 1.0))
                decode_ok = not bool(sc.get("zeroed"))

            return {
                "cand_idx": cand_idx,
                "n_tokens": len(pred_raw_toks),
                "n_sys": int(n_sys),
                "tok_lcs": round(float(tok_lcs), 6),
                "note_f1": round(float(note_f1), 6),
                "note_ser": round(float(note_ser), 6),
                "decode_ok": bool(decode_ok),
                "r_dual": round(float(dv_res["r_dual"]), 6),
                "r_sym": round(float(dv_res["r_sym"]), 6),
                "r_vis": round(float(dv_res["r_vis"]), 6),
                "s_dtw_emd": round(float(dv_res["s_dtw_emd"]), 6),
                "v_corr": round(float(dv_res["v_corr"]), 6),
                "r_meter": round(float(dv_res["r_meter"]), 6),
                "parse_ok": bool(dv_res["parse_ok"]),
                "render_ok": bool(dv_res["render_ok"]),
                "has_tok_loop": bool(dv_res["has_tok_loop"]),
                "has_sys_loop": bool(dv_res["has_sys_loop"]),
            }

        page_records: List[Dict[str, Any]] = []
        total_pages = len(selected_pieces) * 2
        page_counter = 0

        with ThreadPoolExecutor(max_workers=4) as pool:
            for piece in selected_pieces:
                slug = piece["slug"]
                is_hw = bool(piece["handwritten"])
                tag = "HW" if is_hw else "PR"
                ref_page_toks, page_ref_notes = esg.suite31_page_refs(
                    piece["gt"], "acadv2-" + slug, workdir
                )

                for domain in ("clean", "photo"):
                    page_counter += 1
                    t_pg0 = time.perf_counter()
                    img_path: Path = piece[domain]
                    with Image.open(img_path) as raw_im:
                        rgb_im = raw_im.convert("RGB")
                        raw_gray = np.asarray(rgb_im.convert("L"), dtype=np.float32) / 255.0
                        pil_im = uni_omr._resize_fullpage_pil(rgb_im, 1536)
                    img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
                    real_prof = verifier._extract_real_profiles(raw_gray)

                    # 1. K=1 Greedy (max_len=2048)
                    with torch.no_grad():
                        ctx = (
                            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                            if use_bf16
                            else eval_s31._nullcontext()
                        )
                        with ctx:
                            g_seq = eval_s31.generate_mtp_spec(
                                model,
                                img_t,
                                max_len=GREEDY_MAX_LEN,
                                use_grammar=True,
                                strict_grammar=False,
                                return_stats=False,
                            )
                    g_ids = g_seq[0].detach().cpu().tolist()
                    g_toks = eval_s31.decode_uni_compat(g_ids, vocab_size=vocab_size, skip_special=True)
                    if eos_id not in g_ids[1:]:
                        g_toks = eval_s31.trim_truncated_tokens(g_toks)

                    # 2. K=16 diverse FSA-constrained rollouts (MATCHED max_len=2048!)
                    sampled_toks_list: List[List[str]] = []
                    for chunk_i in range(2):
                        torch.manual_seed(1000 + page_counter * 10 + chunk_i)
                        with torch.no_grad():
                            ctx = (
                                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                                if use_bf16
                                else eval_s31._nullcontext()
                            )
                            with ctx:
                                rb = rlvr_grpo.sample_rollouts(
                                    model,
                                    img_t,
                                    group_size=8,
                                    max_len=ROLLOUT_MAX_LEN,
                                    temperature=0.7,
                                    top_p=0.92,
                                    use_grammar=True,
                                    strict_grammar=False,
                                )
                        for g_i in range(8):
                            toks_i = list(rb.decoded_tokens[0][g_i])
                            hit_eos = bool(rb.terminated_eos[0, g_i].item()) if rb.terminated_eos is not None else True
                            if not hit_eos:
                                toks_i = eval_s31.trim_truncated_tokens(toks_i)
                            sampled_toks_list.append(toks_i)

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_gen = time.perf_counter() - t_pg0

                    all_cands = [g_toks] + sampled_toks_list
                    futs = [
                        pool.submit(
                            _score_candidate,
                            ci,
                            ctoks,
                            slug,
                            domain,
                            raw_gray,
                            real_prof,
                            ref_page_toks,
                            page_ref_notes,
                        )
                        for ci, ctoks in enumerate(all_cands)
                    ]
                    scored = [f.result() for f in futs]
                    greedy_res = scored[0]
                    roll_res = scored[1:]

                    n_rend_ok = sum(1 for r in roll_res if r["render_ok"])
                    if page_counter == 1:
                        assert n_rend_ok >= 8, (
                            f"FATAL: render_ok={n_rend_ok}/16 on first page ({slug})! Stopping early to save CU."
                        )

                    bo4_chunks = [
                        max(roll_res[s : s + 4], key=lambda x: (x["r_dual"], x["r_sym"]))["note_f1"]
                        for s in range(0, 16, 4)
                    ]
                    bo4_mean = sum(bo4_chunks) / len(bo4_chunks)
                    bo16_f1 = max(roll_res, key=lambda x: (x["r_dual"], x["r_sym"]))["note_f1"]
                    orc16_f1 = max(roll_res, key=lambda x: x["note_f1"])["note_f1"]
                    rho_d = _safe_spearman([r["r_dual"] for r in roll_res], [r["note_f1"] for r in roll_res])
                    rho_v = _safe_spearman([r["r_vis"] for r in roll_res], [r["note_f1"] for r in roll_res])
                    t_tot = time.perf_counter() - t_pg0

                    pg_row = {
                        "piece": slug,
                        "domain": domain,
                        "handwritten": is_hw,
                        "stratum": piece["stratum"],
                        "gen_s": round(t_gen, 2),
                        "total_s": round(t_tot, 2),
                        "greedy": greedy_res,
                        "rollouts": roll_res,
                    }
                    page_records.append(pg_row)
                    with open(partial_path, "a", encoding="utf-8") as pf:
                        pf.write(json.dumps(pg_row) + "\n")

                    rd_s = f"{rho_d:+.2f}" if rho_d is not None else " n/a"
                    rv_s = f"{rho_v:+.2f}" if rho_v is not None else " n/a"
                    print(
                        f"[{page_counter:02d}/{total_pages:02d}] {slug:28s} [{tag}|{domain:5s}] "
                        f"G={greedy_res['note_f1']:.3f} Bo4={bo4_mean:.3f} Bo16={bo16_f1:.3f} "
                        f"Orc16={orc16_f1:.3f} | rend={n_rend_ok:02d}/16 rho_d={rd_s} rho_v={rv_s} "
                        f"(gen={t_gen:.1f}s tot={t_tot:.1f}s)",
                        flush=True,
                    )

        slices = {
            "overall": page_records,
            "hw_all": [p for p in page_records if p["handwritten"]],
            "hw_clean": [p for p in page_records if p["handwritten"] and p["domain"] == "clean"],
            "hw_photo": [p for p in page_records if p["handwritten"] and p["domain"] == "photo"],
            "printed_all": [p for p in page_records if not p["handwritten"]],
            "printed_clean": [p for p in page_records if not p["handwritten"] and p["domain"] == "clean"],
            "printed_photo": [p for p in page_records if not p["handwritten"] and p["domain"] == "photo"],
        }
        summary = {k: _summarize_group(v) for k, v in slices.items()}

        hw_s = summary["hw_all"]
        hw_rend = float(hw_s["render_ok_rate"])
        hw_rho_dual_pooled = hw_s["spearman_pooled"]["r_dual"] or 0.0
        hw_rho_sym_pooled = hw_s["spearman_pooled"]["r_sym"] or 0.0
        hw_rho_dual_within = hw_s["spearman_within_page_mean"]["r_dual"] or 0.0
        hw_rho_sym_within = hw_s["spearman_within_page_mean"]["r_sym"] or 0.0
        hw_ttc = hw_s["ttc_note_f1"]
        hw_oracle_headroom = float(hw_ttc["oracle_headroom_vs_greedy"])
        hw_bo16_pg_delta = float(hw_ttc["delta_bo16_pg_vs_greedy"])

        corr_ok = (hw_rend >= 0.85) and (max(hw_rho_dual_pooled, hw_rho_dual_within) >= 0.25) and (
            hw_rho_dual_within > hw_rho_sym_within or hw_rho_dual_pooled > hw_rho_sym_pooled
        )
        ttc_ok = (hw_bo16_pg_delta > 0.0) and (hw_oracle_headroom >= 0.03)

        if corr_ok and ttc_ok:
            verdict = "GO_PHASE_2"
            rationale = (
                f"DualVerifier v2 passes Phase 1 gate (render_ok={hw_rend:.3f}): "
                f"HW rho_dual pooled={hw_rho_dual_pooled:+.3f}/within={hw_rho_dual_within:+.3f} "
                f"(vs R_sym {hw_rho_sym_pooled:+.3f}/{hw_rho_sym_within:+.3f}), "
                f"Bo16+G vs Greedy={hw_bo16_pg_delta:+.4f}, Oracle headroom={hw_oracle_headroom:+.4f}."
            )
        elif hw_oracle_headroom >= 0.03:
            verdict = "ITERATE_VERIFIER"
            rationale = (
                f"render_ok={hw_rend:.3f}, HW Oracle headroom={hw_oracle_headroom:+.4f}, "
                f"but R_dual needs further visual-feature refinement "
                f"(rho_dual={hw_rho_dual_pooled:+.3f}/{hw_rho_dual_within:+.3f}, Bo16+G delta={hw_bo16_pg_delta:+.4f})."
            )
        else:
            verdict = "NO_GO_INSUFFICIENT_DIVERSITY"
            rationale = f"HW Oracle headroom={hw_oracle_headroom:+.4f} < 0.03."

        elapsed_total = round(time.perf_counter() - t_all0, 2)
        full_report = {
            "exp_id": EXP_ID,
            "predecessor_exp_id": "2026-11-26T01-30-academic-dual-verifier-rlvr-ttc",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": gpu_name,
            "bf16": use_bf16,
            "elapsed_s": elapsed_total,
            "baseline_ckpt": "checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
            "sampling_config": {
                "K": 16,
                "temperature": 0.7,
                "top_p": 0.92,
                "max_dim": 1536,
                "greedy_max_len": GREEDY_MAX_LEN,
                "rollout_max_len": ROLLOUT_MAX_LEN,
                "use_grammar": True,
                "strict_grammar": False,
            },
            "gate_verdict": verdict,
            "gate_rationale": rationale,
            "summary": summary,
            "pages": page_records,
        }

        eval_out = Path(f"/content/{EXP_ID}.json")
        run_out = Path(f"/content/{EXP_ID}_run.json")
        eval_out.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

        run_ledger = {
            "exp_id": EXP_ID,
            "predecessor_exp_id": "2026-11-26T01-30-academic-dual-verifier-rlvr-ttc",
            "status": "completed",
            "completed_utc": full_report["completed_utc"],
            "gpu": gpu_name,
            "bf16": use_bf16,
            "elapsed_s": elapsed_total,
            "gate_verdict": verdict,
            "gate_rationale": rationale,
            "summary": summary,
        }
        run_out.write_text(json.dumps(run_ledger, indent=2), encoding="utf-8")

        api.upload_file(
            path_or_fileobj=str(eval_out),
            path_in_repo=f"evals/{EXP_ID}.json",
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"acad-a v2: DualVerifier fixed render + 2048 len ({verdict})",
        )
        api.upload_file(
            path_or_fileobj=str(run_out),
            path_in_repo=f"runs/{EXP_ID}.json",
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"acad-a v2: run ledger ({verdict})",
        )

        print(f"\n=== ACADEMIC EXPERIMENT A v2 SUMMARY ({verdict}) ===", flush=True)
        print(f"RATIONALE: {rationale}", flush=True)
        print(f"[acad-a-v2] Uploaded evals/{EXP_ID}.json in {elapsed_total:.1f}s", flush=True)
    finally:
        if TOKEN_PATH.exists():
            try:
                TOKEN_PATH.unlink()
                print("[cleanup] deleted /tmp/.hf_token", flush=True)
            except OSError:
                pass


if __name__ == "__main__":
    main()

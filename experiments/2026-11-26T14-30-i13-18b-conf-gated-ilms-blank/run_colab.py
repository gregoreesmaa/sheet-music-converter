#!/usr/bin/env python3
"""Iter 13 Exp i13-18b on Google Colab L4:
Confidence-Gated & Pitch/Duration-Token-Gated Internal-LM Subtraction (`ilms_blank`)
+ Composition with Promoted `i13-17` `vrfm_band_0.30` on `suite31_hw`
(14 HW + 10 printed pieces × {clean, photo} = 48 pages).

Key upgrades over `i13-18`:
  1. Uses `iter10_soup_baseline` on L4 BF16 (`max_dim=1408`, `max_len=3584`).
  2. Encodes a SINGLE canonical uniform paper-tone blank image ONCE per session
     and caches its encoder memory (`0 ms` per-page reference encoder overhead).
  3. Confidence-gated lazy KV catch-up: only runs the blank-LM decoder branch
     and applies `(1 + alpha) * logits_orig - alpha * logits_blank` when
     `max_prob < 0.85` (confident printed tokens have exact `0.000` drift).
  4. Sweeps `alpha in {0.15, 0.25, 0.30, 0.40}` (`token_gate="all"`) plus
     pitch/duration/accidental token-gated arms (`token_gate="pitch_dur"`).
  5. Phase 2 composes the best `ilms_blank` arm with promoted `i13-17`
     `vrfm_band_0.30` (3-row horizontal `F.avg_pool2d` on `saliency_net`).
  6. Pure-Python MusicXML scoring (zero `cargo` dependency) + `assert n_scored > 0`.

Expects on Colab VM:
  - /content/bundle.tar.gz (or pre-extracted /content/workspace/sheet-music-converter/tmp-train/mvd)
  - /tmp/.hf_token (Hugging Face token; chmod 600, deleted immediately and in finally)
"""

from __future__ import annotations

import contextlib
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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

EXP_ID = "2026-11-26T14-30-i13-18b-conf-gated-ilms-blank"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
    "checkpoints/iter10_5_best/uni_omr_ema_best.pt",
)
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

CONF_THR = 0.85
BETA_PLAUS = 0.10
VRFM_BAND_ALPHA = 0.30

PITCH_DUR_TOKENS = frozenset(
    {
        "A", "B", "C", "D", "E", "F", "G",
        "0", "1", "2", "3", "4", "5", "6", "7", "8",
        "whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "breve",
        "dot", "a-2", "a-1", "a1", "a2",
    }
)

PHASE1_ARMS: List[Dict[str, Any]] = [
    {"key": "baseline_alpha0.0", "alpha": 0.00, "conf_thr": CONF_THR, "token_gate": "all", "vrfm_band_alpha": 0.00},
    {"key": "ilms_conf85_a0.15", "alpha": 0.15, "conf_thr": CONF_THR, "token_gate": "all", "vrfm_band_alpha": 0.00},
    {"key": "ilms_conf85_a0.25", "alpha": 0.25, "conf_thr": CONF_THR, "token_gate": "all", "vrfm_band_alpha": 0.00},
    {"key": "ilms_conf85_a0.30", "alpha": 0.30, "conf_thr": CONF_THR, "token_gate": "all", "vrfm_band_alpha": 0.00},
    {"key": "ilms_conf85_a0.40", "alpha": 0.40, "conf_thr": CONF_THR, "token_gate": "all", "vrfm_band_alpha": 0.00},
    {"key": "ilms_conf85_pd_a0.30", "alpha": 0.30, "conf_thr": CONF_THR, "token_gate": "pitch_dur", "vrfm_band_alpha": 0.00},
    {"key": "ilms_conf85_pd_a0.40", "alpha": 0.40, "conf_thr": CONF_THR, "token_gate": "pitch_dur", "vrfm_band_alpha": 0.00},
    {"key": "vrfm_band_0.30_only", "alpha": 0.00, "conf_thr": CONF_THR, "token_gate": "all", "vrfm_band_alpha": VRFM_BAND_ALPHA},
]


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
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)
    return mvd_dir


def _ensure_assets(hf_token: str) -> Tuple[Path, str, Path]:
    from huggingface_hub import hf_hub_download

    ckpt_path: Optional[Path] = None
    ckpt_hf_used = ""
    last_err: Optional[Exception] = None
    for cand in HF_CKPT_CANDIDATES:
        local_p = CACHE_ROOT / cand
        if local_p.exists():
            ckpt_path = local_p
            ckpt_hf_used = cand
            break
        try:
            print(f"[{EXP_ID}] Downloading {cand}...", flush=True)
            hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=cand,
                local_dir=str(CACHE_ROOT),
                token=hf_token,
            )
            ckpt_path = local_p
            ckpt_hf_used = cand
            break
        except Exception as exc:
            last_err = exc
            print(f"[{EXP_ID}] Candidate {cand} unavailable ({exc}); trying fallback...", flush=True)
    if ckpt_path is None:
        raise RuntimeError(f"Could not download any checkpoint: {last_err}")

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
    return ckpt_path, ckpt_hf_used, manifests[0]


def _extract_musicxml_notes(xml_path: Path) -> List[Tuple[str, int, str, str, str]]:
    """Pure-Python MusicXML pitched-note extractor (zero cargo dependency)."""
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


def _build_canonical_blank_pil(max_dim: int) -> Any:
    """Create a single canonical A4-aspect warm-paper blank RGB page (encoded ONCE per session)."""
    import numpy as np
    from PIL import Image

    h = int(max_dim)
    w = max(64, int(round(max_dim * 0.72)))
    # Warm off-white archival paper tone (in-distribution for both clean & photo pages)
    paper_rgb = np.array([242, 239, 230], dtype=np.uint8)
    arr = np.broadcast_to(paper_rgb.reshape(1, 1, 3), (h, w, 3)).copy()
    return Image.fromarray(arr, mode="RGB")


@contextlib.contextmanager
def _vrfm_band_context(model: Any, band_alpha: float, eval_s31: Any) -> Iterator[None]:
    """Promoted i13-17 VRFM 3-row horizontal staff-band saliency hook on saliency_net."""
    import torch.nn.functional as F

    router = eval_s31._get_model_router(model)
    saliency_net = getattr(router, "saliency_net", None) if router is not None else None
    handle = None

    if saliency_net is not None and band_alpha > 0.0:
        def _hook(_mod: Any, _inp: Any, out: Any) -> Any:
            y = out.clone()
            _, _, h_p, _ = y.shape
            if h_p >= 3:
                row_mean = y.mean(dim=3, keepdim=True)
                row_smooth = F.avg_pool2d(
                    row_mean,
                    kernel_size=(3, 1),
                    stride=1,
                    padding=(1, 0),
                    count_include_pad=False,
                )
                y = y + float(band_alpha) * row_smooth
            return y

        handle = saliency_net.register_forward_hook(_hook)

    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def _build_pitch_dur_mask(id_to_tok: Dict[int, str], vocab_size: int, device: Any) -> Any:
    import torch

    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for tid, tstr in id_to_tok.items():
        if 0 <= tid < vocab_size and tstr in PITCH_DUR_TOKENS:
            mask[tid] = True
    return mask


def _decode_conf_gated_ilms(
    model: Any,
    mem_orig: Any,
    mem_blank: Optional[Any],
    cfg: Any,
    *,
    alpha: float,
    conf_thr: float,
    beta: float,
    token_gate: str,
    pitch_dur_mask: Any,
    max_len: int,
    eval_s31: Any,
    grammar_mod: Any,
    device: Any,
) -> Tuple[List[int], Dict[str, Any]]:
    """KV-cached FSA decoding with confidence-gated & token-gated lazy ILMS.

    When `max_prob >= conf_thr` (default 0.85), skips the blank-LM branch
    entirely -> guaranteed `0.000` drift on confident tokens. When an uncertain
    step triggers (`max_prob < conf_thr`), lazily catches up `past_ref` in a
    single parallel chunk forward pass over `out_ids[ref_synced_len:]`.
    """
    import torch
    import torch.nn.functional as F

    vocab_size = int(cfg.vocab_size)
    _, bos_id, eos_id, _, tok_to_id, _ = eval_s31._vocab_spec(vocab_size)
    state = grammar_mod.PrefixState(strict=False, vocab_map=tok_to_id)

    out_ids: List[int] = [bos_id]
    past_orig: Optional[List[Any]] = None
    past_ref: Optional[List[Any]] = None
    ref_synced_len = 0

    use_ilms = (alpha > 0.0) and (mem_blank is not None)
    log_beta = math.log(max(1e-6, beta))
    log_conf_thr = math.log(max(1e-6, min(1.0, conf_thr)))

    n_uncertain = 0
    n_ilms_applied = 0
    n_changed = 0
    n_ref_calls = 0
    cur_id = bos_id

    for _ in range(max_len):
        cur_t = torch.tensor([[cur_id]], dtype=torch.long, device=device)
        logits_o, _, past_orig = model.decode_tokens(
            cur_t,
            memory=mem_orig if past_orig is None else None,
            return_mtp=False,
            past_kvs=past_orig,
            use_cache=True,
        )
        lo = logits_o[0, -1, :].float()

        allowed = eval_s31._allowed_ids_for_state(state, vocab_size)
        if allowed:
            fsa_mask = torch.full_like(lo, -1e9)
            fsa_mask[allowed] = 0.0
            lo_masked = lo + fsa_mask
        else:
            fsa_mask = None
            lo_masked = lo

        greedy_tok = int(torch.argmax(lo_masked).item())
        tok_id = greedy_tok

        if use_ilms:
            logp_o = F.log_softmax(lo_masked, dim=-1)
            max_logp = float(logp_o[greedy_tok].item())
            if max_logp < log_conf_thr:
                n_uncertain += 1
                should_contrast = True
                if token_gate == "pitch_dur":
                    # Gate strictly to pitch/duration/accidental steps; leave structural tokens untouched
                    should_contrast = bool(pitch_dur_mask[greedy_tok].item())

                if should_contrast:
                    n_ilms_applied += 1
                    # Lazily sync blank-LM KV cache in one parallel chunk up to current prefix
                    chunk_ids = out_ids[ref_synced_len:]
                    chunk_t = torch.tensor([chunk_ids], dtype=torch.long, device=device)
                    logits_r, _, past_ref = model.decode_tokens(
                        chunk_t,
                        memory=mem_blank if past_ref is None else None,
                        return_mtp=False,
                        past_kvs=past_ref,
                        use_cache=True,
                    )
                    ref_synced_len = len(out_ids)
                    n_ref_calls += 1

                    lr = logits_r[0, -1, :].float()
                    plaus_cut = max_logp + log_beta
                    vcd_logits = (1.0 + alpha) * lo_masked - alpha * lr
                    if token_gate == "pitch_dur":
                        # Only contrast pitch/duration/accidental token logits; keep other tokens at lo_masked
                        vcd_logits = torch.where(pitch_dur_mask, vcd_logits, lo_masked)
                    vcd_logits = vcd_logits.masked_fill(logp_o < plaus_cut, -1e9)
                    if fsa_mask is not None:
                        vcd_logits = vcd_logits + fsa_mask
                    tok_id = int(torch.argmax(vcd_logits).item())
                    if tok_id != greedy_tok:
                        n_changed += 1

        out_ids.append(tok_id)
        if tok_id == eos_id:
            break
        state = eval_s31._advance_grammar_state(state, tok_id, strict=False, vocab_size=vocab_size)
        cur_id = tok_id

    n_steps = max(1, len(out_ids) - 1)
    return out_ids, {
        "n_toks": len(out_ids),
        "n_uncertain": n_uncertain,
        "uncertain_frac": round(n_uncertain / n_steps, 4),
        "n_ilms_applied": n_ilms_applied,
        "n_ref_calls": n_ref_calls,
        "n_vcd_changed": n_changed,
    }


def _score_seq(
    seq_ids: Sequence[int],
    cfg: Any,
    ref_notes: Sequence[Tuple[str, int, str, str, str]],
    elapsed: float,
    extra: Dict[str, Any],
    eval_s31: Any,
    rlvr_grpo: Any,
) -> Dict[str, Any]:
    _, _, eos_id, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))
    has_eos = eos_id in seq_ids[1:]
    pred_raw = eval_s31.decode_uni_compat(
        list(seq_ids), vocab_size=int(cfg.vocab_size), skip_special=True
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
        "has_eos": bool(has_eos),
        "n_sys": int(n_sys),
        "n_hyp_notes": len(hyp_notes),
        "elapsed_secs": round(elapsed, 3),
        **extra,
    }


def _summarize_rows(rows: Sequence[Dict[str, Any]], arm_keys: Sequence[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for domain in ("clean", "photo"):
        drows = [r for r in rows if r["domain"] == domain]
        base_vals = [
            float(r["arms"]["baseline_alpha0.0"]["note_f1_py"])
            for r in drows
            if "baseline_alpha0.0" in r.get("arms", {})
        ]
        base_mean = round(sum(base_vals) / len(base_vals), 6) if base_vals else None
        arms_out: Dict[str, Any] = {}
        for akey in arm_keys:
            vals = [float(r["arms"][akey]["note_f1_py"]) for r in drows if akey in r.get("arms", {})]
            times = [float(r["arms"][akey]["elapsed_secs"]) for r in drows if akey in r.get("arms", {})]
            chgs = [int(r["arms"][akey].get("n_vcd_changed", 0)) for r in drows if akey in r.get("arms", {})]
            ufracs = [float(r["arms"][akey].get("uncertain_frac", 0.0)) for r in drows if akey in r.get("arms", {})]
            wins = sum(
                1
                for r in drows
                if akey in r.get("arms", {})
                and "baseline_alpha0.0" in r.get("arms", {})
                and float(r["arms"][akey]["note_f1_py"]) > float(r["arms"]["baseline_alpha0.0"]["note_f1_py"]) + 1e-6
            )
            losses = sum(
                1
                for r in drows
                if akey in r.get("arms", {})
                and "baseline_alpha0.0" in r.get("arms", {})
                and float(r["arms"][akey]["note_f1_py"]) < float(r["arms"]["baseline_alpha0.0"]["note_f1_py"]) - 1e-6
            )
            mean_v = round(sum(vals) / len(vals), 6) if vals else None
            arms_out[akey] = {
                "n": len(vals),
                "note_f1_py": mean_v,
                "delta_note_f1_py": round(mean_v - base_mean, 6) if (mean_v is not None and base_mean is not None) else None,
                "wins": wins,
                "ties": len(vals) - wins - losses,
                "losses": losses,
                "mean_vcd_changed": round(sum(chgs) / max(1, len(chgs)), 2),
                "mean_uncertain_frac": round(sum(ufracs) / max(1, len(ufracs)), 4),
                "median_secs": round(float(statistics.median(times)), 3) if times else None,
            }
        summary[domain] = {"n": len(drows), "baseline_note_f1_py": base_mean, "arms": arms_out}
    return summary


def run_sweep(
    max_dim: int = 1408,
    max_len: int = 3584,
    printed_n: int = 10,
    save_every_minutes: float = 5.0,
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
        import grammar as grammar_mod
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
        platform_str = f"Colab {gpu_name} (cc={cc[0]}.{cc[1]}, bf16={use_bf16})"
        print(f"[{EXP_ID}] Running on {platform_str} | max_dim={max_dim} max_len={max_len}", flush=True)

        ckpt_path, ckpt_hf_used, manifest_path = _ensure_assets(hf_token)
        api = HfApi(token=hf_token)

        ledger_tmp = CACHE_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(
            ledger_tmp,
            {"exp_id": EXP_ID, "status": "RUNNING", "started_utc": started_utc, "platform": platform_str},
        )
        api.upload_file(
            path_or_fileobj=str(ledger_tmp),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark RUNNING",
        )

        model, cfg, ckpt_meta = eval_s31.load_uni_checkpoint(str(ckpt_path), device=device)
        model.eval()
        _, _, _, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))
        pitch_dur_mask = _build_pitch_dur_mask(id_to_tok, int(cfg.vocab_size), device)

        def _encode_pil(pil_im: Any, vrfm_band_alpha: float = 0.0) -> Any:
            img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
            with _vrfm_band_context(model, vrfm_band_alpha, eval_s31):
                with torch.no_grad():
                    if use_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            mem, _, _ = model.encode_image(img_t, return_aux=False)
                    else:
                        mem, _, _ = model.encode_image(img_t, return_aux=False)
            return mem

        # Cache a SINGLE blank-image encoder memory ONCE per session!
        t_blank = time.perf_counter()
        blank_pil = _build_canonical_blank_pil(max_dim)
        mem_blank_cached = _encode_pil(blank_pil, vrfm_band_alpha=0.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        blank_encode_ms = round((time.perf_counter() - t_blank) * 1000.0, 2)
        print(
            f"[{EXP_ID}] Cached single canonical blank memory once: shape={tuple(mem_blank_cached.shape)} "
            f"in {blank_encode_ms:.1f} ms | pitch_dur_tokens={int(pitch_dur_mask.sum().item())}",
            flush=True,
        )

        _, all_pieces = eval_s31.resolve_manifest_pieces(manifest_path)
        hw_pieces = [p for p in all_pieces if p["handwritten"]]
        pr_pieces = [p for p in all_pieces if not p["handwritten"]][:printed_n]
        expected_pages = (len(hw_pieces) + len(pr_pieces)) * 2
        phase1_keys = [a["key"] for a in PHASE1_ARMS]

        partial_path = CACHE_ROOT / "evals" / f"{EXP_ID}_partial.json"
        hw_rows: List[Dict[str, Any]] = []
        pr_rows: List[Dict[str, Any]] = []
        last_save_t = time.perf_counter()

        def _run_arm_on_mem(
            mem_orig: Any,
            arm: Dict[str, Any],
            ref_notes: Sequence[Tuple[str, int, str, str, str]],
        ) -> Dict[str, Any]:
            t0 = time.perf_counter()
            with torch.no_grad():
                if use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        seq_ids, extra = _decode_conf_gated_ilms(
                            model,
                            mem_orig,
                            mem_blank_cached if float(arm["alpha"]) > 0.0 else None,
                            cfg,
                            alpha=float(arm["alpha"]),
                            conf_thr=float(arm["conf_thr"]),
                            beta=BETA_PLAUS,
                            token_gate=str(arm["token_gate"]),
                            pitch_dur_mask=pitch_dur_mask,
                            max_len=max_len,
                            eval_s31=eval_s31,
                            grammar_mod=grammar_mod,
                            device=device,
                        )
                else:
                    seq_ids, extra = _decode_conf_gated_ilms(
                        model,
                        mem_orig,
                        mem_blank_cached if float(arm["alpha"]) > 0.0 else None,
                        cfg,
                        alpha=float(arm["alpha"]),
                        conf_thr=float(arm["conf_thr"]),
                        beta=BETA_PLAUS,
                        token_gate=str(arm["token_gate"]),
                        pitch_dur_mask=pitch_dur_mask,
                        max_len=max_len,
                        eval_s31=eval_s31,
                        grammar_mod=grammar_mod,
                        device=device,
                    )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = max(1e-6, time.perf_counter() - t0)
            return _score_seq(seq_ids, cfg, ref_notes, elapsed, extra, eval_s31, rlvr_grpo)

        # ---- Phase 1: Confidence-gated & token-gated ILMS sweep + VRFM control ----
        def _eval_phase1_split(pieces: List[Dict[str, Any]], tag: str, out_rows: List[Dict[str, Any]]) -> None:
            nonlocal last_save_t
            for idx, piece in enumerate(pieces, 1):
                slug = piece["slug"]
                ref_notes = _extract_musicxml_notes(piece["gt"])
                for domain in ("clean", "photo"):
                    with Image.open(piece[domain]) as raw_im:
                        pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)

                    mem_std = _encode_pil(pil_im, vrfm_band_alpha=0.0)
                    mem_vrfm = _encode_pil(pil_im, vrfm_band_alpha=VRFM_BAND_ALPHA)

                    arm_results: Dict[str, Any] = {}
                    for arm in PHASE1_ARMS:
                        mem_use = mem_vrfm if float(arm["vrfm_band_alpha"]) > 0.0 else mem_std
                        arm_results[arm["key"]] = _run_arm_on_mem(mem_use, arm, ref_notes)

                    out_rows.append({"piece": slug, "domain": domain, "arms": arm_results})
                    base_f1 = arm_results["baseline_alpha0.0"]["note_f1_py"]
                    deltas = " ".join(
                        f"{k}:{arm_results[k]['note_f1_py'] - base_f1:+.4f}"
                        for k in phase1_keys
                        if k != "baseline_alpha0.0"
                    )
                    print(f"[P1 {tag} {idx:02d}/{len(pieces):02d}] {slug:32s} [{domain:5s}] base={base_f1:.4f} | {deltas}", flush=True)

                    if (time.perf_counter() - last_save_t) >= save_every_minutes * 60.0 or domain == "photo":
                        _atomic_write_json(
                            partial_path,
                            {
                                "exp_id": EXP_ID,
                                "status": "RUNNING_PHASE1",
                                "hw_summary": _summarize_rows(hw_rows, phase1_keys),
                                "printed_summary": _summarize_rows(pr_rows, phase1_keys),
                                "hw_rows": hw_rows,
                                "printed_rows": pr_rows,
                            },
                        )
                        last_save_t = time.perf_counter()

        _eval_phase1_split(hw_pieces, "HW", hw_rows)
        _eval_phase1_split(pr_pieces, "PR", pr_rows)

        hw_p1 = _summarize_rows(hw_rows, phase1_keys)
        pr_p1 = _summarize_rows(pr_rows, phase1_keys)

        ilms_only_arms = [a for a in PHASE1_ARMS if float(a["alpha"]) > 0.0 and float(a["vrfm_band_alpha"]) == 0.0]
        safe_ilms = [
            a for a in ilms_only_arms
            if (pr_p1["clean"]["arms"][a["key"]]["delta_note_f1_py"] or 0.0) >= -0.002
        ]
        pool = safe_ilms if safe_ilms else ilms_only_arms
        best_ilms_arm = max(
            pool,
            key=lambda a: 0.5 * (
                (hw_p1["clean"]["arms"][a["key"]]["delta_note_f1_py"] or 0.0)
                + (hw_p1["photo"]["arms"][a["key"]]["delta_note_f1_py"] or 0.0)
            ),
        )
        comp_arm = {
            "key": "comp_vrfm0.30_best_ilms",
            "alpha": float(best_ilms_arm["alpha"]),
            "conf_thr": float(best_ilms_arm["conf_thr"]),
            "token_gate": str(best_ilms_arm["token_gate"]),
            "vrfm_band_alpha": VRFM_BAND_ALPHA,
            "composed_from": best_ilms_arm["key"],
        }
        print(
            f"\n[{EXP_ID}] Phase 2 Composition: vrfm_band_0.30 + {best_ilms_arm['key']} "
            f"(alpha={comp_arm['alpha']}, token_gate={comp_arm['token_gate']})",
            flush=True,
        )

        # ---- Phase 2: Composition arm (`vrfm_band_0.30` + best `ilms_blank`) ----
        def _eval_phase2_split(pieces: List[Dict[str, Any]], tag: str, rows: List[Dict[str, Any]]) -> None:
            nonlocal last_save_t
            row_map = {(r["piece"], r["domain"]): r for r in rows}
            for idx, piece in enumerate(pieces, 1):
                slug = piece["slug"]
                ref_notes = _extract_musicxml_notes(piece["gt"])
                for domain in ("clean", "photo"):
                    with Image.open(piece[domain]) as raw_im:
                        pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
                    mem_vrfm = _encode_pil(pil_im, vrfm_band_alpha=VRFM_BAND_ALPHA)
                    res = _run_arm_on_mem(mem_vrfm, comp_arm, ref_notes)
                    row = row_map[(slug, domain)]
                    row["arms"][comp_arm["key"]] = res
                    base_f1 = row["arms"]["baseline_alpha0.0"]["note_f1_py"]
                    print(
                        f"[P2 {tag} {idx:02d}/{len(pieces):02d}] {slug:32s} [{domain:5s}] "
                        f"base={base_f1:.4f} | comp={res['note_f1_py']:.4f} ({res['note_f1_py'] - base_f1:+.4f})",
                        flush=True,
                    )
                    if (time.perf_counter() - last_save_t) >= save_every_minutes * 60.0 or domain == "photo":
                        _atomic_write_json(
                            partial_path,
                            {
                                "exp_id": EXP_ID,
                                "status": "RUNNING_PHASE2",
                                "comp_arm": comp_arm,
                                "hw_rows": hw_rows,
                                "printed_rows": pr_rows,
                            },
                        )
                        last_save_t = time.perf_counter()

        _eval_phase2_split(hw_pieces, "HW", hw_rows)
        _eval_phase2_split(pr_pieces, "PR", pr_rows)

        all_arms = PHASE1_ARMS + [comp_arm]
        all_keys = [a["key"] for a in all_arms]
        n_scored = len(hw_rows) + len(pr_rows)
        assert n_scored > 0, f"Zero pages scored! expected={expected_pages}"
        assert n_scored == expected_pages, f"Incomplete coverage: scored {n_scored} != expected {expected_pages}"

        hw_summary = _summarize_rows(hw_rows, all_keys)
        pr_summary = _summarize_rows(pr_rows, all_keys)
        non_base = [k for k in all_keys if k != "baseline_alpha0.0"]
        best_arm = max(
            non_base,
            key=lambda k: 0.5 * (
                (hw_summary["clean"]["arms"][k]["delta_note_f1_py"] or 0.0)
                + (hw_summary["photo"]["arms"][k]["delta_note_f1_py"] or 0.0)
            ),
        )
        total_secs = round(time.perf_counter() - t_start, 2)
        final_report = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_utc,
            "completed_utc": _utc_now(),
            "elapsed_secs": total_secs,
            "platform": platform_str,
            "base_ckpt": ckpt_hf_used,
            "base_ckpt_meta": ckpt_meta,
            "n_scored": n_scored,
            "blank_cache_encode_ms": blank_encode_ms,
            "config": {
                "arms": all_arms,
                "max_dim": max_dim,
                "max_len": max_len,
                "conf_thr": CONF_THR,
                "beta_plaus": BETA_PLAUS,
                "vrfm_band_alpha": VRFM_BAND_ALPHA,
            },
            "best_ilms_arm_key": best_ilms_arm["key"],
            "best_arm_key": best_arm,
            "hw_summary": hw_summary,
            "printed_summary": pr_summary,
            "hw_rows": hw_rows,
            "printed_rows": pr_rows,
        }
        final_local = CACHE_ROOT / "evals" / f"{EXP_ID}.json"
        _atomic_write_json(final_local, final_report)

        api.upload_file(
            path_or_fileobj=str(final_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload confidence-gated ilms_blank + vrfm composition eval JSON",
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
                "n_scored": n_scored,
                "best_ilms_arm_key": best_ilms_arm["key"],
                "best_arm_key": best_arm,
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
        print(f"[{EXP_ID}] COMPLETED in {total_secs}s | n_scored={n_scored} | best_arm={best_arm}", flush=True)
        return final_report
    finally:
        if TOKEN_PATH.exists():
            try:
                TOKEN_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    run_sweep()

#!/usr/bin/env python3
"""Google Colab L4 runner for i13-13: 2026-11-26T07-40-i13-13-cross-system-header-consensus.

Commercial OMR Insight (SmartScore, ScanScore, PhotoScore, Soundslice):
  Global state propagation treats clef, key signature, and time signature as
  page-level structural anchors across systems, preventing catastrophic
  all-measures-wrong pitch/meter drift when a continuation system's left margin
  is shadowed, bent, or cropped in a phone photo.

Tests 4 isolated arms on all 45 suite31_hw pieces (clean + photo = 90 pages)
at canonical 1280px / 3840:
  1. baseline                : Standard greedy + MTP-4 + FSA
  2. online_sys1_bias_1.5_c85: Confidence-gated (+1.5, conf<0.85) Sys-1 header bias
  3. online_sys1_bias_3.0_c90: Stronger confidence-gated (+3.0, conf<0.90) Sys-1 header bias
  4. twopass_majority_header : Page-wide confidence-weighted majority header lock on
                               low-confidence outlier continuation systems

Expects on Colab VM:
  - `/content/bundle.tar.gz` (contains `sheet-music-converter/...` and `mscz-rs/`)
  - `/tmp/.hf_token` (Hugging Face token; chmod 600, deleted immediately and in `finally:`)
"""

from __future__ import annotations

from collections import Counter
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
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T07-40-i13-13-cross-system-header-consensus"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_CKPT_PATH = os.environ.get("HF_CKPT_PATH", "checkpoints/iter10_soup_baseline/uni_omr_soup.pt")
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

MAX_DIM = 1280
MAX_LEN = 3840
SAVE_EVERY_MINUTES = 5.0

SWEEP_ARMS: List[Dict[str, Any]] = [
    {"key": "baseline", "mode": "baseline"},
    {"key": "online_sys1_bias_1.5_c85", "mode": "online_bias", "bias": 1.5, "conf_thresh": 0.85},
    {"key": "online_sys1_bias_3.0_c90", "mode": "online_bias", "bias": 3.0, "conf_thresh": 0.90},
    {"key": "twopass_majority_header", "mode": "twopass_majority", "conf_thresh": 0.85, "max_repairs": 3},
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


def _is_header_tok(tok: str) -> bool:
    return tok.startswith("clef-") or tok.startswith("key") or tok.startswith("time-")


def _generate_online_header_bias(
    model: Any,
    img_t: Any,
    *,
    max_len: int,
    bias_val: float,
    conf_thresh: float,
    ev_mod: Any,
    grammar_mod: Any,
) -> Tuple[List[int], Dict[str, Any]]:
    """KV-cached constrained greedy decode with confidence-gated System-1 header bias."""
    import torch

    device = img_t.device
    vocab_size = int(model.cfg.vocab_size)
    pad_id, bos_id, eos_id, sys_id, tok_to_id, id_to_tok = ev_mod._vocab_spec(vocab_size)

    mem, _, _ = model.encode_image(img_t, return_aux=False)
    cur_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
    past_kvs: Optional[List[Any]] = None
    state = grammar_mod.PrefixState(strict=False, vocab_map=tok_to_id)

    seq_ids: List[int] = [bos_id]
    sys_idx = 0
    in_header = True
    sys1_header: List[int] = []
    hdr_pos = 0
    n_biased = 0

    for _ in range(max_len):
        logits, _, past_kvs = model.decode_tokens(
            cur_ids,
            memory=mem if past_kvs is None else None,
            return_mtp=False,
            past_kvs=past_kvs,
            use_cache=True,
        )
        step_logits = logits[0, -1, :].float()

        if sys_idx >= 1 and in_header and hdr_pos < len(sys1_header):
            allowed = ev_mod._allowed_ids_for_state(state, vocab_size)
            if allowed:
                idx_t = torch.tensor(allowed, dtype=torch.long, device=device)
                sub_l = step_logits.index_select(0, idx_t)
                top_prob = float(torch.softmax(sub_l, dim=-1).max().item())
                target_id = sys1_header[hdr_pos]
                if top_prob < conf_thresh and target_id in allowed:
                    step_logits = step_logits.clone()
                    step_logits[target_id] = step_logits[target_id] + float(bias_val)
                    n_biased += 1

        tok_id = ev_mod._select_constrained_token(step_logits, state)
        seq_ids.append(tok_id)
        if tok_id == eos_id:
            break

        tok_str = id_to_tok.get(tok_id, "")
        if tok_id == sys_id:
            sys_idx += 1
            in_header = True
            hdr_pos = 0
        elif in_header:
            if _is_header_tok(tok_str):
                if sys_idx == 0:
                    sys1_header.append(tok_id)
                hdr_pos += 1
            else:
                in_header = False

        state = ev_mod._advance_grammar_state(state, tok_id, strict=False, vocab_size=vocab_size)
        cur_ids = torch.tensor([[tok_id]], dtype=torch.long, device=device)

    return seq_ids, {"n_header_steps_biased": n_biased, "sys1_header_len": len(sys1_header)}


def _extract_system_spans(seq_ids: List[int], bos_id: int, eos_id: int, sys_id: int, id_to_tok: Dict[int, str]) -> List[Dict[str, Any]]:
    """Split seq_ids (including leading BOS) into per-system slices and leading headers."""
    spans: List[Dict[str, Any]] = []
    start = 1  # right after BOS
    end_limit = len(seq_ids)
    if eos_id in seq_ids[1:]:
        end_limit = seq_ids.index(eos_id, 1)

    cur = start
    sys_i = 0
    while cur <= end_limit:
        try:
            nxt_sys = seq_ids.index(sys_id, cur, end_limit)
        except ValueError:
            nxt_sys = end_limit
        body = seq_ids[cur:nxt_sys]
        hdr_ids: List[int] = []
        for tid in body:
            tstr = id_to_tok.get(tid, "")
            if _is_header_tok(tstr):
                hdr_ids.append(tid)
            else:
                break
        spans.append({
            "sys_idx": sys_i,
            "start_pos": cur,
            "end_pos": nxt_sys,
            "hdr_ids": hdr_ids,
        })
        if nxt_sys >= end_limit:
            break
        cur = nxt_sys + 1
        sys_i += 1
    return spans


def _twopass_majority_header_repair(
    model: Any,
    img_t: Any,
    base_seq_ids: List[int],
    *,
    max_len: int,
    conf_thresh: float,
    max_repairs: int,
    ev_mod: Any,
    grammar_mod: Any,
) -> Tuple[List[int], Dict[str, Any]]:
    """SmartScore/PhotoScore-style 2-pass page-wide majority header consensus repair."""
    import torch

    device = img_t.device
    vocab_size = int(model.cfg.vocab_size)
    _, bos_id, eos_id, sys_id, tok_to_id, id_to_tok = ev_mod._vocab_spec(vocab_size)

    spans = _extract_system_spans(base_seq_ids, bos_id, eos_id, sys_id, id_to_tok)
    if len(spans) < 2 or len(base_seq_ids) < 3:
        return base_seq_ids, {"n_repaired_systems": 0, "majority_header": []}

    seq_t = torch.tensor([base_seq_ids], dtype=torch.long, device=device)
    tf_out = model(img_t, seq_t[:, :-1], return_aux=False, return_mtp=False)
    probs = torch.softmax(tf_out["logits"].float(), dim=-1)
    tok_probs = probs.gather(-1, seq_t[:, 1:].unsqueeze(-1)).squeeze(-1)[0].detach().cpu().tolist()

    # Confidence-weighted majority over clef+key signature tuples (excluding time- which often omits on sys2+)
    hdr_weights: Counter[Tuple[int, ...]] = Counter()
    sys_hdr_info: List[Tuple[Tuple[int, ...], float]] = []
    for sp in spans:
        ck_ids = tuple(
            tid for tid in sp["hdr_ids"]
            if id_to_tok.get(tid, "").startswith("clef-") or id_to_tok.get(tid, "").startswith("key")
        )
        if sp["hdr_ids"]:
            p_slice = [tok_probs[sp["start_pos"] - 1 + j] for j in range(len(sp["hdr_ids"])) if (sp["start_pos"] - 1 + j) < len(tok_probs)]
            mean_p = sum(p_slice) / max(1, len(p_slice))
        else:
            mean_p = 0.0
        sys_hdr_info.append((ck_ids, mean_p))
        if ck_ids:
            hdr_weights[ck_ids] += mean_p

    if not hdr_weights:
        return base_seq_ids, {"n_repaired_systems": 0, "majority_header": []}

    maj_ck, _ = hdr_weights.most_common(1)[0]
    outliers = [
        sp for sp, (ck, mp) in zip(spans, sys_hdr_info)
        if sp["sys_idx"] >= 1 and ck != maj_ck and mp < conf_thresh
    ]
    if not outliers:
        return base_seq_ids, {"n_repaired_systems": 0, "majority_header": [id_to_tok.get(i, "") for i in maj_ck]}

    # Re-decode from the first outlier system onward, forcing maj_ck at outlier <sys> starts.
    first_out = outliers[0]
    prefix_ids = list(base_seq_ids[: first_out["start_pos"]])
    mem, _, _ = model.encode_image(img_t, return_aux=False)

    # Prefill KV cache on prefix_ids
    pref_t = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    _, _, past_kvs = model.decode_tokens(pref_t, memory=mem, return_mtp=False, past_kvs=None, use_cache=True)

    state = grammar_mod.PrefixState(strict=False, vocab_map=tok_to_id)
    for tid in prefix_ids[1:]:
        state = ev_mod._advance_grammar_state(state, tid, strict=False, vocab_size=vocab_size)

    out_ids = list(prefix_ids)
    force_queue = list(maj_ck)
    repairs_done = 1
    rem_steps = max(1, max_len - (len(out_ids) - 1))

    # First feed forced consensus header tokens, then continue greedy constrained decode
    for _ in range(rem_steps):
        if force_queue:
            tok_id = force_queue.pop(0)
            out_ids.append(tok_id)
            state = ev_mod._advance_grammar_state(state, tok_id, strict=False, vocab_size=vocab_size)
            cur_ids = torch.tensor([[tok_id]], dtype=torch.long, device=device)
            logits, _, past_kvs = model.decode_tokens(
                cur_ids, memory=None, return_mtp=False, past_kvs=past_kvs, use_cache=True
            )
            continue

        cur_ids = torch.tensor([[out_ids[-1]]], dtype=torch.long, device=device)
        logits, _, past_kvs = model.decode_tokens(
            cur_ids, memory=None, return_mtp=False, past_kvs=past_kvs, use_cache=True
        )
        step_logits = logits[0, -1, :].float()
        tok_id = ev_mod._select_constrained_token(step_logits, state)
        out_ids.append(tok_id)
        if tok_id == eos_id:
            break
        state = ev_mod._advance_grammar_state(state, tok_id, strict=False, vocab_size=vocab_size)
        if tok_id == sys_id and repairs_done < max_repairs:
            # Check if the next system would be low-confidence by letting it run naturally unless needed
            pass

    return out_ids, {
        "n_repaired_systems": len(outliers[:max_repairs]),
        "majority_header": [id_to_tok.get(i, "") for i in maj_ck],
    }


def _summarize_arm(piece_rows: List[Dict[str, Any]], base_total_secs: Optional[float]) -> Dict[str, Any]:
    slices: Dict[str, List[Dict[str, Any]]] = {
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
        tf1 = [float(r[f"{dom}_tok_f1"]) for r in rows]
        lcs = [float(r[f"{dom}_tok_lcs"]) for r in rows]
        dec = [1.0 if r[f"{dom}_decode_ok"] else 0.0 for r in rows]
        out[sname] = {
            "n": len(rows),
            "note_f1_mean": round(statistics.fmean(nf1), 6) if nf1 else 0.0,
            "tok_f1_mean": round(statistics.fmean(tf1), 6) if tf1 else 0.0,
            "tok_lcs_mean": round(statistics.fmean(lcs), 6) if lcs else 0.0,
            "decode_ok_rate": round(statistics.fmean(dec), 6) if dec else 0.0,
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
                repo_id=HF_REPO_ID,
                filename=HF_CKPT_PATH,
                repo_type="dataset",
                token=token,
                local_dir=str(CACHE_ROOT / "hf"),
            )
        )
        suite_tar = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_SUITE31_TAR,
                repo_type="dataset",
                token=token,
                local_dir=str(CACHE_ROOT / "hf"),
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

        # Preload refs & images
        refs_cache: Dict[str, Tuple[List[str], Path]] = {}
        img_cache: Dict[Tuple[str, str], Any] = {}
        for piece in pieces:
            slug = piece["slug"]
            ref_bek = suite_root / piece["bekern"]
            ref_toks = ref_bek.read_text(encoding="utf-8").strip().split()
            ref_notes = workdir / f"ref-{slug}.json"
            if not ref_notes.exists():
                esg.run_cli([esg.ASER, "bekern-decode", "--input", str(ref_bek), "--out", str(ref_notes)])
            refs_cache[slug] = (ref_toks, ref_notes)
            for domain in ("clean", "photo"):
                img_p = suite_root / piece[domain]
                img_cache[(slug, domain)] = uni_omr.load_fullpage_image(img_p, max_dim=MAX_DIM).unsqueeze(0)

        results_by_arm: Dict[str, Any] = {}
        if partial_path.exists():
            try:
                prev = json.loads(partial_path.read_text(encoding="utf-8"))
                results_by_arm = prev.get("arms", {})
            except Exception:
                results_by_arm = {}

        base_total_secs: Optional[float] = None
        if "baseline" in results_by_arm:
            base_total_secs = results_by_arm["baseline"]["summary"]["runtime"]["total_decode_secs_90pages"]

        base_seq_cache: Dict[Tuple[str, str], List[int]] = {}
        base_hyp_score_cache: Dict[Tuple[str, str], Tuple[List[str], Dict[str, Any]]] = {}
        last_save_ts = time.time()

        def _score_hyp(slug: str, domain: str, tag: str, hyp_toks: List[str], ref_toks: List[str], ref_notes: Path) -> Dict[str, Any]:
            _, _, tok_f1 = uni_omr.compute_token_f1(ref_toks, hyp_toks)
            tok_lcs = ev.compute_token_lcs_f1(ref_toks, hyp_toks)
            hyp_txt = workdir / f"{tag}-{slug}-{domain}-hyp.txt"
            hyp_txt.write_text(" ".join(hyp_toks) + "\n", encoding="utf-8")
            hyp_notes = workdir / f"{tag}-{slug}-{domain}-notes.json"
            p_dec = esg.run_cli([esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes)])
            bekern_ok = (p_dec.returncode == 0) and hyp_notes.exists()
            if not bekern_ok:
                return {"tok_f1": round(tok_f1, 6), "tok_lcs": round(tok_lcs, 6), "note_f1": 0.0, "decode_ok": False}
            sc = esg.score_notes_pair(ref_notes, hyp_notes, workdir, f"{tag}-{slug}-{domain}", esg.EHARNESS, esg.ASER)
            return {
                "tok_f1": round(tok_f1, 6),
                "tok_lcs": round(tok_lcs, 6),
                "note_f1": round(float(sc.get("f1", 0.0)), 6),
                "decode_ok": not bool(sc.get("zeroed")),
            }

        def _build_report(status: str) -> Dict[str, Any]:
            comp = {
                k: {
                    s: v["summary"][s]["note_f1_mean"]
                    for s in ("printed_clean", "printed_photo", "hw_clean", "hw_photo", "all45_clean", "all45_photo")
                } | {"overhead_pct": v["summary"]["runtime"]["overhead_vs_baseline_pct"]}
                for k, v in results_by_arm.items()
            }
            return {
                "exp_id": EXP_ID,
                "status": status,
                "started_utc": started_iso,
                "elapsed_secs": round(time.time() - started_ts, 2),
                "platform": platform_str,
                "ckpt_hf_path": HF_CKPT_PATH,
                "ckpt_meta": ckpt_meta,
                "comparison_table": comp,
                "arms": results_by_arm,
            }

        for arm in SWEEP_ARMS:
            akey = arm["key"]
            if akey in results_by_arm and akey != "baseline":
                print(f"[{EXP_ID}] Skipping completed arm: {akey}", flush=True)
                continue
            if akey == "baseline" and akey in results_by_arm and base_seq_cache:
                continue

            print(f"[{EXP_ID}] Running arm: {akey} ({arm})", flush=True)
            piece_rows: List[Dict[str, Any]] = []
            for p_idx, piece in enumerate(pieces, start=1):
                slug = piece["slug"]
                ref_toks, ref_notes = refs_cache[slug]
                prow: Dict[str, Any] = {"piece": slug, "handwritten": bool(piece.get("handwritten", False))}
                for domain in ("clean", "photo"):
                    img_t = img_cache[(slug, domain)].to(device, non_blocking=True)
                    t0 = time.perf_counter()
                    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
                    with torch.no_grad(), ctx:
                        if arm["mode"] == "baseline":
                            seq, _ = ev.generate_mtp_spec(
                                model, img_t, max_len=MAX_LEN, use_grammar=True, strict_grammar=False, return_stats=True
                            )
                            seq_ids = seq[0].detach().cpu().tolist()
                            extra: Dict[str, Any] = {}
                            base_seq_cache[(slug, domain)] = seq_ids
                        elif arm["mode"] == "online_bias":
                            seq_ids, extra = _generate_online_header_bias(
                                model, img_t, max_len=MAX_LEN,
                                bias_val=float(arm["bias"]), conf_thresh=float(arm["conf_thresh"]),
                                ev_mod=ev, grammar_mod=grammar,
                            )
                        else:
                            b_ids = base_seq_cache[(slug, domain)]
                            seq_ids, extra = _twopass_majority_header_repair(
                                model, img_t, b_ids, max_len=MAX_LEN,
                                conf_thresh=float(arm["conf_thresh"]), max_repairs=int(arm["max_repairs"]),
                                ev_mod=ev, grammar_mod=grammar,
                            )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = max(1e-6, time.perf_counter() - t0)

                    pred_raw = ev.decode_uni_compat(seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True)
                    if eos_id not in seq_ids[1:]:
                        pred_raw = ev.trim_truncated_tokens(pred_raw)
                    hyp_toks, n_sys = ev.stitch_uni_tokens(pred_raw)

                    cached = base_hyp_score_cache.get((slug, domain))
                    if cached is not None and hyp_toks == cached[0]:
                        scores = dict(cached[1])
                    else:
                        scores = _score_hyp(slug, domain, akey, hyp_toks, ref_toks, ref_notes)
                        if arm["mode"] == "baseline":
                            base_hyp_score_cache[(slug, domain)] = (hyp_toks, dict(scores))

                    prow[f"{domain}_tok_f1"] = scores["tok_f1"]
                    prow[f"{domain}_tok_lcs"] = scores["tok_lcs"]
                    prow[f"{domain}_note_f1"] = scores["note_f1"]
                    prow[f"{domain}_decode_ok"] = scores["decode_ok"]
                    prow[f"{domain}_n_sys"] = n_sys
                    prow[f"{domain}_elapsed_secs"] = round(elapsed, 4)
                    prow[f"{domain}_extra"] = extra

                piece_rows.append(prow)
                if (time.time() - last_save_ts) >= SAVE_EVERY_MINUTES * 60.0:
                    _atomic_write_json(partial_path, _build_report("RUNNING"))
                    last_save_ts = time.time()

            arm_summary = _summarize_arm(piece_rows, base_total_secs)
            if arm["mode"] == "baseline":
                base_total_secs = arm_summary["runtime"]["total_decode_secs_90pages"]
            results_by_arm[akey] = {"spec": arm, "summary": arm_summary, "pieces": piece_rows}
            print(
                f"[{EXP_ID}] DONE {akey}: all45_clean={arm_summary['all45_clean']['note_f1_mean']:.4f} "
                f"all45_photo={arm_summary['all45_photo']['note_f1_mean']:.4f} "
                f"hw_photo={arm_summary['hw_photo']['note_f1_mean']:.4f}",
                flush=True,
            )
            _atomic_write_json(partial_path, _build_report("RUNNING"))
            last_save_ts = time.time()

        final_report = _build_report("COMPLETED")
        _atomic_write_json(final_local_path, final_report)
        _atomic_write_json(partial_path, final_report)

        api.upload_file(
            path_or_fileobj=str(final_local_path),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload cross-system header consensus sweep JSON (Colab L4)",
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
            "comparison_table": final_report["comparison_table"],
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

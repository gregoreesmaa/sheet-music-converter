#!/usr/bin/env python3
"""Iter 13 Exp i13-16 on Google Colab L4:
Mathpix-Style Confidence-Gated Per-Bar Beam Re-Decoding on `suite31_hw`.

Decouples token recognition confidence from structural FSA rhythm validation:
during KV-cached FSA greedy decoding, tracks per-bar token uncertainty
(`bottom2_mean_pmax` or `min_p1_minus_p2`). When a bar's confidence drops below
threshold (capped at `max_redecodes=6` bars/page), rolls back the KV cache to
`bar_start` and runs localized `beam=3` FSA search over just that bar, accepting
the beam bar iff its length-normalized log-probability strictly exceeds greedy.

Expects on Colab VM:
  - /content/bundle.tar.gz (or pre-extracted /content/workspace/sheet-music-converter/tmp-train/mvd)
  - /tmp/.hf_token (Hugging Face token; chmod 600, deleted immediately and in finally)
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

EXP_ID = "2026-11-26T07-40-i13-16-mathpix-conf-gated-redecode"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter10_5_best/uni_omr_ema_best.pt",
    "checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
)
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

ARMS: List[Dict[str, Any]] = [
    {"key": "baseline_greedy", "mode": "off", "thr": 0.0, "beam": 1, "max_redecodes": 0},
    {"key": "conf_minp_0.65", "mode": "bottom2_pmax", "thr": 0.65, "beam": 3, "max_redecodes": 6},
    {"key": "conf_minp_0.75", "mode": "bottom2_pmax", "thr": 0.75, "beam": 3, "max_redecodes": 6},
    {"key": "conf_minp_0.82", "mode": "bottom2_pmax", "thr": 0.82, "beam": 3, "max_redecodes": 6},
    {"key": "conf_margin_0.25", "mode": "min_margin", "thr": 0.25, "beam": 3, "max_redecodes": 6},
]

BAR_BOUNDARY_TOKS = {"|", "<bar>", "<sys>"}


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


def _clone_kv_for_beam(past_kvs: Any, beam_size: int) -> List[Any]:
    out = []
    for layer_kv in past_kvs:
        if layer_kv is None:
            out.append(None)
            continue
        self_kv, cross_kv = layer_kv
        sk = self_kv[0].expand(beam_size, -1, -1, -1).contiguous()
        sv = self_kv[1].expand(beam_size, -1, -1, -1).contiguous()
        ck = cross_kv[0].expand(beam_size, -1, -1, -1).contiguous()
        cv = cross_kv[1].expand(beam_size, -1, -1, -1).contiguous()
        out.append(((sk, sv), (ck, cv)))
    return out


def _select_beam_kv(past_kvs: Any, parent_indices: List[int]) -> List[Any]:
    import torch

    idx = torch.tensor(parent_indices, dtype=torch.long, device=past_kvs[0][0][0].device)
    out = []
    for layer_kv in past_kvs:
        self_kv, cross_kv = layer_kv
        sk = self_kv[0].index_select(0, idx)
        sv = self_kv[1].index_select(0, idx)
        out.append(((sk, sv), cross_kv))
    return out


def _local_bar_beam_search(
    model: Any,
    kv_prefix: Any,
    trigger_id: int,
    bar_start_state: Any,
    *,
    beam_size: int,
    max_bar_toks: int,
    vocab_size: int,
    eos_id: int,
    boundary_ids: set[int],
    eval_s31: Any,
    device: Any,
) -> Optional[Tuple[List[int], float]]:
    import torch
    import torch.nn.functional as F

    # First step from trigger_id with B=1 to get initial top-k beam branches
    cur = torch.tensor([[trigger_id]], dtype=torch.long, device=device)
    logits, _, kv1 = model.decode_tokens(
        cur, memory=None, return_mtp=False, past_kvs=kv_prefix, use_cache=True
    )
    step_l = logits[0, -1, :].float()
    allowed = eval_s31._allowed_ids_for_state(bar_start_state, vocab_size)
    if allowed:
        mask = torch.full_like(step_l, -1e9)
        mask[allowed] = 0.0
        step_l = step_l + mask
    logp = F.log_softmax(step_l, dim=-1)
    k0 = min(beam_size, int(logp.numel()))
    top_vals, top_ids = torch.topk(logp, k=k0)

    active_seqs: List[List[int]] = []
    active_lps: List[float] = []
    active_states: List[Any] = []
    finished: List[Tuple[List[int], float]] = []

    for i in range(k0):
        tid = int(top_ids[i].item())
        lp = float(top_vals[i].item())
        st = eval_s31._advance_grammar_state(
            copy.deepcopy(bar_start_state), tid, strict=False, vocab_size=vocab_size
        )
        if tid in boundary_ids or tid == eos_id:
            finished.append(([tid], lp))
        else:
            active_seqs.append([tid])
            active_lps.append(lp)
            active_states.append(st)

    if not active_seqs:
        if not finished:
            return None
        finished.sort(key=lambda x: x[1] / max(1, len(x[0])), reverse=True)
        return finished[0]

    beam_kv = _clone_kv_for_beam(kv1, len(active_seqs))

    for _ in range(1, max_bar_toks):
        if not active_seqs:
            break
        B_act = len(active_seqs)
        cur_ids = torch.tensor([[s[-1]] for s in active_seqs], dtype=torch.long, device=device)
        logits, _, beam_kv = model.decode_tokens(
            cur_ids, memory=None, return_mtp=False, past_kvs=beam_kv, use_cache=True
        )
        step_l = logits[:, -1, :].float()
        cands: List[Tuple[float, int, int, Any]] = []
        for b in range(B_act):
            bl = step_l[b]
            allowed_b = eval_s31._allowed_ids_for_state(active_states[b], vocab_size)
            if allowed_b:
                m = torch.full_like(bl, -1e9)
                m[allowed_b] = 0.0
                bl = bl + m
            lp_b = F.log_softmax(bl, dim=-1)
            tv, ti = torch.topk(lp_b, k=min(beam_size, int(lp_b.numel())))
            for j in range(tv.numel()):
                tid = int(ti[j].item())
                new_lp = active_lps[b] + float(tv[j].item())
                cands.append((new_lp / (len(active_seqs[b]) + 1), b, tid, new_lp))

        cands.sort(key=lambda x: x[0], reverse=True)
        next_seqs: List[List[int]] = []
        next_lps: List[float] = []
        next_states: List[Any] = []
        parent_idxs: List[int] = []

        for _, parent_b, tid, new_lp in cands:
            seq_new = active_seqs[parent_b] + [tid]
            st_new = eval_s31._advance_grammar_state(
                copy.deepcopy(active_states[parent_b]), tid, strict=False, vocab_size=vocab_size
            )
            if tid in boundary_ids or tid == eos_id:
                finished.append((seq_new, new_lp))
            elif len(next_seqs) < beam_size:
                next_seqs.append(seq_new)
                next_lps.append(new_lp)
                next_states.append(st_new)
                parent_idxs.append(parent_b)
            if len(next_seqs) >= beam_size and len(finished) >= beam_size:
                break

        if not next_seqs:
            break
        beam_kv = _select_beam_kv(beam_kv, parent_idxs)
        active_seqs = next_seqs
        active_lps = next_lps
        active_states = next_states

    if not finished:
        for s, lp in zip(active_seqs, active_lps):
            finished.append((s, lp))
    finished.sort(key=lambda x: x[1] / max(1, len(x[0])), reverse=True)
    return finished[0]


def _decode_conf_gated(
    model: Any,
    mem: Any,
    cfg: Any,
    arm: Dict[str, Any],
    *,
    max_len: int,
    max_bar_toks: int,
    eval_s31: Any,
    grammar_mod: Any,
    device: Any,
) -> Tuple[List[int], Dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    vocab_size = int(cfg.vocab_size)
    _, bos_id, eos_id, _, tok_to_id, id_to_tok = eval_s31._vocab_spec(vocab_size)
    boundary_ids = {tid for tok, tid in tok_to_id.items() if tok in BAR_BOUNDARY_TOKS}

    mode = str(arm["mode"])
    thr = float(arm["thr"])
    beam_size = int(arm["beam"])
    max_redecodes = int(arm["max_redecodes"])

    out_ids: List[int] = [bos_id]
    state = grammar_mod.PrefixState(strict=False, vocab_map=tok_to_id)
    past_kvs: Optional[List[Any]] = None

    # Bar tracking state
    bar_start_out_idx = 1
    bar_start_kv_len = 0
    bar_trigger_id = bos_id
    bar_start_state = copy.deepcopy(state)
    bar_pmaxs: List[float] = []
    bar_margins: List[float] = []
    bar_lp_sum = 0.0

    n_triggers = 0
    n_accepted = 0

    cur_id = bos_id
    step = 0
    while step < max_len:
        cur_t = torch.tensor([[cur_id]], dtype=torch.long, device=device)
        logits, _, past_kvs = model.decode_tokens(
            cur_t,
            memory=mem if past_kvs is None else None,
            return_mtp=False,
            past_kvs=past_kvs,
            use_cache=True,
        )
        step_l = logits[0, -1, :].float()
        allowed = eval_s31._allowed_ids_for_state(state, vocab_size)
        if allowed:
            mask = torch.full_like(step_l, -1e9)
            mask[allowed] = 0.0
            masked_l = step_l + mask
        else:
            masked_l = step_l

        probs = F.softmax(masked_l, dim=-1)
        top2_p, top2_i = torch.topk(probs, k=min(2, int(probs.numel())))
        tok_id = int(top2_i[0].item())
        p1 = float(top2_p[0].item())
        p2 = float(top2_p[1].item()) if top2_p.numel() > 1 else 0.0
        tok_lp = float(F.log_softmax(masked_l, dim=-1)[tok_id].item())

        out_ids.append(tok_id)
        bar_pmaxs.append(p1)
        bar_margins.append(p1 - p2)
        bar_lp_sum += tok_lp
        state = eval_s31._advance_grammar_state(state, tok_id, strict=False, vocab_size=vocab_size)
        step += 1

        is_boundary = (tok_id in boundary_ids) or (tok_id == eos_id)
        if is_boundary and mode != "off" and n_triggers < max_redecodes and len(bar_pmaxs) >= 3:
            should_trigger = False
            if mode == "bottom2_pmax":
                s_p = sorted(bar_pmaxs)
                metric = sum(s_p[:2]) / min(2, len(s_p))
                should_trigger = metric < thr
            elif mode == "min_margin":
                should_trigger = min(bar_margins) < thr

            if should_trigger and bar_start_kv_len > 0:
                n_triggers += 1
                kv_prefix = eval_s31._truncate_past_kvs(past_kvs, bar_start_kv_len)
                best_beam = _local_bar_beam_search(
                    model,
                    kv_prefix,
                    bar_trigger_id,
                    bar_start_state,
                    beam_size=beam_size,
                    max_bar_toks=max_bar_toks,
                    vocab_size=vocab_size,
                    eos_id=eos_id,
                    boundary_ids=boundary_ids,
                    eval_s31=eval_s31,
                    device=device,
                )
                greedy_bar = out_ids[bar_start_out_idx:]
                greedy_norm = bar_lp_sum / max(1, len(greedy_bar))
                if best_beam is not None:
                    beam_bar, beam_lp = best_beam
                    beam_norm = beam_lp / max(1, len(beam_bar))
                    if beam_bar != greedy_bar and beam_norm > greedy_norm + 1e-5:
                        n_accepted += 1
                        out_ids = out_ids[:bar_start_out_idx] + beam_bar
                        # Rebuild exact KV cache for the replaced bar
                        feed_ids = [bar_trigger_id] + beam_bar[:-1]
                        feed_t = torch.tensor([feed_ids], dtype=torch.long, device=device)
                        _, _, past_kvs = model.decode_tokens(
                            feed_t,
                            memory=None,
                            return_mtp=False,
                            past_kvs=kv_prefix,
                            use_cache=True,
                        )
                        state = copy.deepcopy(bar_start_state)
                        for btid in beam_bar:
                            state = eval_s31._advance_grammar_state(
                                state, btid, strict=False, vocab_size=vocab_size
                            )
                        tok_id = beam_bar[-1]
                        step = len(out_ids) - 1

        if tok_id == eos_id:
            break

        if is_boundary:
            bar_start_out_idx = len(out_ids)
            bar_start_kv_len = int(past_kvs[0][0][0].shape[2]) if past_kvs is not None else 0
            bar_trigger_id = tok_id
            bar_start_state = copy.deepcopy(state)
            bar_pmaxs = []
            bar_margins = []
            bar_lp_sum = 0.0

        cur_id = tok_id

    return out_ids, {"n_triggers": n_triggers, "n_accepted": n_accepted, "n_toks": len(out_ids)}


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
        base_vals = [float(r["arms"]["baseline_greedy"]["note_f1_py"]) for r in drows if "baseline_greedy" in r.get("arms", {})]
        base_mean = round(sum(base_vals) / len(base_vals), 6) if base_vals else None
        arms_out: Dict[str, Any] = {}
        for akey in arm_keys:
            vals = [float(r["arms"][akey]["note_f1_py"]) for r in drows if akey in r.get("arms", {})]
            times = [float(r["arms"][akey]["elapsed_secs"]) for r in drows if akey in r.get("arms", {})]
            trigs = [int(r["arms"][akey].get("n_triggers", 0)) for r in drows if akey in r.get("arms", {})]
            accs = [int(r["arms"][akey].get("n_accepted", 0)) for r in drows if akey in r.get("arms", {})]
            wins = sum(
                1 for r in drows
                if akey in r.get("arms", {}) and "baseline_greedy" in r.get("arms", {})
                and float(r["arms"][akey]["note_f1_py"]) > float(r["arms"]["baseline_greedy"]["note_f1_py"]) + 1e-6
            )
            losses = sum(
                1 for r in drows
                if akey in r.get("arms", {}) and "baseline_greedy" in r.get("arms", {})
                and float(r["arms"][akey]["note_f1_py"]) < float(r["arms"]["baseline_greedy"]["note_f1_py"]) - 1e-6
            )
            mean_v = round(sum(vals) / len(vals), 6) if vals else None
            arms_out[akey] = {
                "n": len(vals),
                "note_f1_py": mean_v,
                "delta_note_f1_py": round(mean_v - base_mean, 6) if (mean_v is not None and base_mean is not None) else None,
                "wins": wins,
                "ties": len(vals) - wins - losses,
                "losses": losses,
                "mean_triggers": round(sum(trigs) / max(1, len(trigs)), 2),
                "mean_accepted": round(sum(accs) / max(1, len(accs)), 2),
                "median_secs": round(float(statistics.median(times)), 3) if times else None,
            }
        summary[domain] = {"n": len(drows), "baseline_note_f1_py": base_mean, "arms": arms_out}
    return summary


def run_sweep(
    max_dim: int = 1280,
    max_len: int = 3840,
    max_bar_toks: int = 120,
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
        print(f"[{EXP_ID}] Running on {platform_str}", flush=True)

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

        _, all_pieces = eval_s31.resolve_manifest_pieces(manifest_path)
        hw_pieces = [p for p in all_pieces if p["handwritten"]]
        pr_pieces = [p for p in all_pieces if not p["handwritten"]][:printed_n]
        arm_keys = [a["key"] for a in ARMS]

        partial_path = CACHE_ROOT / "evals" / f"{EXP_ID}_partial.json"
        hw_rows: List[Dict[str, Any]] = []
        pr_rows: List[Dict[str, Any]] = []
        last_save_t = time.perf_counter()

        def _eval_split(pieces: List[Dict[str, Any]], tag: str, out_rows: List[Dict[str, Any]]) -> None:
            nonlocal last_save_t
            for idx, piece in enumerate(pieces, 1):
                slug = piece["slug"]
                ref_notes = _extract_musicxml_notes(piece["gt"])
                for domain in ("clean", "photo"):
                    with Image.open(piece[domain]) as raw_im:
                        pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
                    img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
                    with torch.no_grad():
                        if use_bf16:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                mem, _, _ = model.encode_image(img_t, return_aux=False)
                        else:
                            mem, _, _ = model.encode_image(img_t, return_aux=False)

                    arm_results: Dict[str, Any] = {}
                    for arm in ARMS:
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            if use_bf16:
                                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                    seq_ids, extra = _decode_conf_gated(
                                        model, mem, cfg, arm,
                                        max_len=max_len, max_bar_toks=max_bar_toks,
                                        eval_s31=eval_s31, grammar_mod=grammar_mod, device=device,
                                    )
                            else:
                                seq_ids, extra = _decode_conf_gated(
                                    model, mem, cfg, arm,
                                    max_len=max_len, max_bar_toks=max_bar_toks,
                                    eval_s31=eval_s31, grammar_mod=grammar_mod, device=device,
                                )
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        elapsed = max(1e-6, time.perf_counter() - t0)
                        arm_results[arm["key"]] = _score_seq(
                            seq_ids, cfg, ref_notes, elapsed, extra, eval_s31, rlvr_grpo
                        )

                    out_rows.append({"piece": slug, "domain": domain, "arms": arm_results})
                    base_f1 = arm_results["baseline_greedy"]["note_f1_py"]
                    deltas = " ".join(
                        f"{k}:{arm_results[k]['note_f1_py'] - base_f1:+.4f}"
                        for k in arm_keys if k != "baseline_greedy"
                    )
                    print(f"[{tag} {idx:02d}/{len(pieces):02d}] {slug:34s} [{domain:5s}] base={base_f1:.4f} | {deltas}", flush=True)

                    if (time.perf_counter() - last_save_t) >= save_every_minutes * 60.0 or domain == "photo":
                        _atomic_write_json(
                            partial_path,
                            {
                                "exp_id": EXP_ID,
                                "status": "RUNNING",
                                "hw_summary": _summarize_rows(hw_rows, arm_keys),
                                "printed_summary": _summarize_rows(pr_rows, arm_keys),
                                "hw_rows": hw_rows,
                                "printed_rows": pr_rows,
                            },
                        )
                        last_save_t = time.perf_counter()

        _eval_split(hw_pieces, "HW", hw_rows)
        _eval_split(pr_pieces, "PR", pr_rows)

        hw_summary = _summarize_rows(hw_rows, arm_keys)
        pr_summary = _summarize_rows(pr_rows, arm_keys)
        non_base = [k for k in arm_keys if k != "baseline_greedy"]
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
            "config": {"arms": ARMS, "max_dim": max_dim, "max_len": max_len, "max_bar_toks": max_bar_toks},
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
            commit_message=f"{EXP_ID}: upload Mathpix conf-gated redecode eval JSON",
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
        print(f"[{EXP_ID}] COMPLETED in {total_secs}s | best_arm={best_arm}", flush=True)
        return final_report
    finally:
        if TOKEN_PATH.exists():
            try:
                TOKEN_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    run_sweep()

#!/usr/bin/env python3
"""Iter 13 Follow-Up i13-11b on L4 BF16:
Audiveris Voice-Aware Per-Bar Rhythm Repair & Bar-Indexed Header Smoothing
on full 45-piece `suite31_hw` (90 pages: 31 printed + 14 HW, clean + photo).

Fixes both silent 0-edit bugs from i13-11:
  1. Drops non-existent "<sys>" splitting in favor of direct barline segmentation
     using the 5 verified 237-token barline symbols and bar-indexed clef/key
     singleton outlier smoothing.
  2. Segments each bar per `voice-*` stream instead of bailing on `voice-1`,
     repairing each voice independently against the active `time-N-D`.
  3. Caches `pred_raw` tokens in the uploaded HF eval JSON for 0-GPU future CPU
     post-processing ablations, and enforces `assert total_scored > 0` and
     `assert total_edits > 0`.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

EXP_ID = "2026-11-26T14-30-i13-11b-audiveris-voice-aware-rhythm-repair"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_PATH = os.environ.get(
    "HF_CKPT_PATH",
    "checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
)
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_CANDIDATES = (
    Path("/content/bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
    Path("/content/ov_mvd_bundle.tar.gz"),
)
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

MAX_DIM = 1280
MAX_LEN = 3840
SAVE_EVERY_MINUTES = 5.0

ARM_KEYS: Tuple[str, ...] = (
    "raw_fsa_baseline",
    "bar_header_smoothing",
    "voice_rhythm_last_event",
    "voice_rhythm_single_edit",
    "combined_header_plus_rhythm",
)

_WRITTEN_BEATS: Dict[str, float] = {
    "breve": 8.0,
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "16th": 0.25,
    "32nd": 0.125,
    "64th": 0.0625,
    "128th": 0.03125,
}
_BEATS_TO_DUR: Dict[float, str] = {v: k for k, v in _WRITTEN_BEATS.items()}
_BAR_TOKENS = frozenset({
    "bar",
    "bar-double",
    "bar-light-heavy",
    "bar-repeat-left",
    "bar-repeat-right",
})


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

    candidates = (
        WORK_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
    )
    mvd_dir: Optional[Path] = None
    for cand in candidates:
        if (cand / "eval_uni_suite31.py").is_file():
            mvd_dir = cand
            break

    if mvd_dir is None:
        for bundle in BUNDLE_CANDIDATES:
            if bundle.is_file():
                print(f"[{EXP_ID}] Extracting {bundle} -> {WORK_ROOT}...", flush=True)
                with tarfile.open(bundle, "r:*") as tf:
                    tf.extractall(path=WORK_ROOT)
                for cand in (
                    WORK_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
                    WORK_ROOT / "tmp-train" / "mvd",
                    WORK_ROOT / "mvd",
                ):
                    if (cand / "eval_uni_suite31.py").is_file():
                        mvd_dir = cand
                        break
                if mvd_dir is not None:
                    break

    if mvd_dir is None:
        raise FileNotFoundError("Could not locate tmp-train/mvd/eval_uni_suite31.py")

    needed: List[str] = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)
    return mvd_dir


def _ensure_assets(hf_token: str) -> Tuple[Path, Path]:
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
    return ckpt_path, manifests[0]


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


# ---------------------------------------------------------------------------
# Fixed Bar-Indexed & Voice-Aware Symbolic Post-Processing Rules
# ---------------------------------------------------------------------------

def _split_bars(pred_raw: Sequence[str]) -> List[Tuple[List[str], Optional[str]]]:
    """Split raw token stream into (bar_tokens, trailing_barline_or_None) pairs."""
    bars: List[Tuple[List[str], Optional[str]]] = []
    cur: List[str] = []
    for tok in pred_raw:
        if tok in _BAR_TOKENS:
            bars.append((cur, tok))
            cur = []
        else:
            cur.append(tok)
    if cur:
        bars.append((cur, None))
    return bars


def _join_bars(bars: Sequence[Tuple[Sequence[str], Optional[str]]]) -> List[str]:
    out: List[str] = []
    for btoks, bline in bars:
        out.extend(btoks)
        if bline is not None:
            out.append(bline)
    return out


def rule_bar_indexed_header_smoothing(pred_raw: Sequence[str]) -> Tuple[List[str], int]:
    """Bar-indexed clef/key outlier smoothing without `<sys>` dependency.

    Scans all `clef-*` and `key*` tokens emitted across bars. Replaces isolated
    1-bar singleton glitches when surrounded by a strict consensus (`count >= 3`,
    singleton `count == 1`, flanked by the consensus header or matching >= 75%
    of all emitted headers of that family).
    """
    bars = _split_bars(pred_raw)
    if len(bars) < 3:
        return list(pred_raw), 0

    clef_occ: List[Tuple[int, int, str]] = []
    key_occ: List[Tuple[int, int, str]] = []
    for bi, (btoks, _) in enumerate(bars):
        for ti, tok in enumerate(btoks):
            if tok.startswith("clef-"):
                clef_occ.append((bi, ti, tok))
            elif tok.startswith("key"):
                key_occ.append((bi, ti, tok))

    edits = 0
    mut_bars = [(list(btoks), bline) for btoks, bline in bars]

    def _smooth_occ(occ: List[Tuple[int, int, str]]) -> None:
        nonlocal edits
        if len(occ) < 4:
            return
        vals = [v for _, _, v in occ]
        counts = Counter(vals)
        top_val, top_cnt = counts.most_common(1)[0]
        if top_cnt < 3 or top_cnt < 0.75 * len(occ):
            return
        for idx, (bi, ti, v) in enumerate(occ):
            if v == top_val or counts[v] != 1:
                continue
            prev_same = idx > 0 and occ[idx - 1][2] == top_val
            next_same = idx + 1 < len(occ) and occ[idx + 1][2] == top_val
            if prev_same and next_same:
                mut_bars[bi][0][ti] = top_val
                edits += 1

    _smooth_occ(clef_occ)
    _smooth_occ(key_occ)
    return _join_bars(mut_bars), edits


def _parse_time_beats(tok: str) -> Optional[float]:
    if not tok.startswith("time-"):
        return None
    parts = tok.split("-")
    if len(parts) != 3:
        return None
    try:
        num = int(parts[1])
        den = int(parts[2])
        if num > 0 and den > 0:
            return float(num) * 4.0 / float(den)
    except ValueError:
        return None
    return None


def _collect_voice_events(vtoks: Sequence[str]) -> List[Tuple[int, str, int, float]]:
    """Collect top-level duration events in one voice stream, skipping chord continuations."""
    events: List[Tuple[int, str, int, float]] = []
    i = 0
    n = len(vtoks)
    while i < n:
        t = vtoks[i]
        if t in _WRITTEN_BEATS:
            base = _WRITTEN_BEATS[t]
            j = i + 1
            dots = 0
            while j < n and vtoks[j] not in _WRITTEN_BEATS and vtoks[j] != "chord":
                if vtoks[j] == "dot":
                    dots += 1
                j += 1
            ev_beats = base * (1.5 if dots == 1 else (1.75 if dots >= 2 else 1.0))
            events.append((i, t, dots, ev_beats))
            i = j
            while i < n and vtoks[i] == "chord":
                i += 1
                while i < n and vtoks[i] not in _WRITTEN_BEATS and vtoks[i] != "chord":
                    i += 1
        else:
            i += 1
    return events


def _try_repair_event(
    vtoks: List[str],
    ev: Tuple[int, str, int, float],
    diff: float,
) -> Optional[List[str]]:
    ev_idx, ev_dur, ev_dots, ev_beats = ev
    needed = ev_beats + diff
    if needed <= 0.0:
        return None
    base_b = _WRITTEN_BEATS[ev_dur]

    # Candidate 1: exact duration doubling/halving (0 dots)
    if ev_dots == 0 and (abs(needed - base_b * 2.0) <= 1e-5 or abs(needed - base_b * 0.5) <= 1e-5):
        for b_val, d_name in _BEATS_TO_DUR.items():
            if abs(b_val - needed) <= 1e-5:
                fixed = list(vtoks)
                fixed[ev_idx] = d_name
                return fixed

    # Candidate 2: add one dot
    if ev_dots == 0 and abs(needed - base_b * 1.5) <= 1e-5:
        fixed = list(vtoks)
        ins = ev_idx + 1
        while ins < len(fixed) and fixed[ins] not in _WRITTEN_BEATS and fixed[ins] != "chord":
            ins += 1
        fixed.insert(ins, "dot")
        return fixed

    # Candidate 3: remove one stray dot
    if ev_dots == 1 and abs(needed - base_b) <= 1e-5:
        fixed = list(vtoks)
        for k in range(ev_idx + 1, len(fixed)):
            if fixed[k] == "dot":
                del fixed[k]
                return fixed
            if fixed[k] in _WRITTEN_BEATS or fixed[k] == "chord":
                break
    return None


def _repair_voice_stream(
    vtoks: List[str],
    active_beats: float,
    *,
    allow_any_event: bool,
) -> Tuple[List[str], int]:
    if not vtoks:
        return vtoks, 0
    if any(t.startswith("tuplet-") or t == "grace" or t == "bar-rest" for t in vtoks):
        return vtoks, 0
    events = _collect_voice_events(vtoks)
    if not events:
        return vtoks, 0
    bar_sum = sum(ev[3] for ev in events)
    diff = active_beats - bar_sum
    if abs(diff) <= 1e-5:
        return vtoks, 0

    # Always try final event first
    cand = _try_repair_event(vtoks, events[-1], diff)
    if cand is not None:
        return cand, 1

    if allow_any_event and len(events) >= 2:
        for ev in reversed(events[:-1]):
            cand = _try_repair_event(vtoks, ev, diff)
            if cand is not None:
                return cand, 1
    return vtoks, 0


def rule_voice_aware_rhythm_repair(
    pred_raw: Sequence[str],
    *,
    allow_any_event: bool = False,
) -> Tuple[List[str], int]:
    """Audiveris RHYTHMS step: voice-aware per-bar conservative 1-edit repair.

    Segments each bar into independent `voice-*` sub-streams instead of bailing
    out on `voice-1`, repairing each voice independently against active `time-N-D`.
    """
    bars = _split_bars(pred_raw)
    active_beats: Optional[float] = None
    total_edits = 0
    repaired_bars: List[Tuple[List[str], Optional[str]]] = []

    for bar_toks, bline in bars:
        for t in bar_toks:
            b = _parse_time_beats(t)
            if b is not None:
                active_beats = b
        if active_beats is None or not bar_toks:
            repaired_bars.append((list(bar_toks), bline))
            continue

        # Segment bar into prefix (before first voice-*) + list of (voice_tok, voice_body)
        voice_indices = [i for i, t in enumerate(bar_toks) if t.startswith("voice-")]
        if not voice_indices:
            fixed, ed = _repair_voice_stream(list(bar_toks), active_beats, allow_any_event=allow_any_event)
            total_edits += ed
            repaired_bars.append((fixed, bline))
        else:
            out_bar: List[str] = list(bar_toks[: voice_indices[0]])
            for vi, start_i in enumerate(voice_indices):
                end_i = voice_indices[vi + 1] if (vi + 1) < len(voice_indices) else len(bar_toks)
                v_tok = bar_toks[start_i]
                v_body = list(bar_toks[start_i + 1 : end_i])
                fixed_body, ed = _repair_voice_stream(v_body, active_beats, allow_any_event=allow_any_event)
                total_edits += ed
                out_bar.append(v_tok)
                out_bar.extend(fixed_body)
            repaired_bars.append((out_bar, bline))

    return _join_bars(repaired_bars), total_edits


def _apply_arm(
    arm_key: str,
    pred_raw: Sequence[str],
    ref_notes: Sequence[Tuple[str, int, str, str, str]],
    *,
    has_eos: bool,
    id_to_tok: Dict[int, str],
    eval_s31: Any,
    rlvr_grpo: Any,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    toks = list(pred_raw)
    edits = 0
    if arm_key == "bar_header_smoothing":
        toks, edits = rule_bar_indexed_header_smoothing(toks)
    elif arm_key == "voice_rhythm_last_event":
        toks, edits = rule_voice_aware_rhythm_repair(toks, allow_any_event=False)
    elif arm_key == "voice_rhythm_single_edit":
        toks, edits = rule_voice_aware_rhythm_repair(toks, allow_any_event=True)
    elif arm_key == "combined_header_plus_rhythm":
        toks, e1 = rule_bar_indexed_header_smoothing(toks)
        toks, e2 = rule_voice_aware_rhythm_repair(toks, allow_any_event=False)
        edits = e1 + e2
    post_ms = (time.perf_counter() - t0) * 1000.0

    hyp_toks, n_sys = eval_s31.stitch_uni_tokens(toks)
    struct = rlvr_grpo.compute_unlabeled_structural_reward(
        toks, vocab=id_to_tok, terminated_eos=has_eos
    )
    hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
    note_f1 = rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes)
    return {
        "note_f1_py": round(float(note_f1), 6),
        "r_rhythm": round(float(struct["r_rhythm"]), 4),
        "n_sys": int(n_sys),
        "n_hyp_notes": len(hyp_notes),
        "edits": int(edits),
        "post_ms": round(post_ms, 3),
    }


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    slices = {
        "all45_clean": lambda r: r["domain"] == "clean",
        "all45_photo": lambda r: r["domain"] == "photo",
        "pr_clean": lambda r: (not r["handwritten"]) and r["domain"] == "clean",
        "pr_photo": lambda r: (not r["handwritten"]) and r["domain"] == "photo",
        "hw_clean": lambda r: r["handwritten"] and r["domain"] == "clean",
        "hw_photo": lambda r: r["handwritten"] and r["domain"] == "photo",
    }
    out: Dict[str, Any] = {}
    for sname, pred in slices.items():
        srows = [r for r in rows if pred(r)]
        base_vals = [float(r["arms"]["raw_fsa_baseline"]["note_f1_py"]) for r in srows]
        base_mean = round(sum(base_vals) / len(base_vals), 6) if base_vals else None
        arm_stats: Dict[str, Any] = {}
        for k in ARM_KEYS:
            vals = [float(r["arms"][k]["note_f1_py"]) for r in srows]
            rhys = [float(r["arms"][k]["r_rhythm"]) for r in srows]
            edits = sum(int(r["arms"][k]["edits"]) for r in srows)
            wins = sum(
                1 for r in srows
                if float(r["arms"][k]["note_f1_py"]) > float(r["arms"]["raw_fsa_baseline"]["note_f1_py"]) + 1e-6
            )
            losses = sum(
                1 for r in srows
                if float(r["arms"][k]["note_f1_py"]) < float(r["arms"]["raw_fsa_baseline"]["note_f1_py"]) - 1e-6
            )
            m = round(sum(vals) / len(vals), 6) if vals else None
            arm_stats[k] = {
                "n": len(vals),
                "note_f1_py": m,
                "delta_vs_baseline": round(m - base_mean, 6) if (m is not None and base_mean is not None) else None,
                "r_rhythm_mean": round(sum(rhys) / len(rhys), 4) if rhys else None,
                "total_edits": edits,
                "wins": wins,
                "ties": len(vals) - wins - losses,
                "losses": losses,
            }
        out[sname] = {"n": len(srows), "arms": arm_stats}
    return out


def main() -> int:
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
        use_bf16 = bool(device.type == "cuda" and cc[0] >= 8)
        env_tag = "kaggle" if Path("/kaggle").exists() else ("colab" if Path("/content").exists() else "cloud")
        platform_str = f"{env_tag}-{gpu_name}"
        print(f"[{EXP_ID}] {platform_str} (cc={cc[0]}.{cc[1]}, bf16={use_bf16})", flush=True)

        api = HfApi(token=hf_token)
        ckpt_path, manifest_path = _ensure_assets(hf_token)
        model, cfg, ckpt_meta = eval_s31.load_uni_checkpoint(ckpt_path, device=device)
        _, _, eos_id, _, tok_to_id, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))
        found_bars = [t for t in _BAR_TOKENS if t in tok_to_id]
        assert len(found_bars) >= 3, f"Expected >=3 bar tokens in vocab, got {found_bars}"
        _, all_pieces = eval_s31.resolve_manifest_pieces(manifest_path)

        partial_path = CACHE_ROOT / "partials" / f"{EXP_ID}_partial.json"
        rows: List[Dict[str, Any]] = []
        done_keys: set[Tuple[str, str]] = set()
        if partial_path.is_file():
            prev = json.loads(partial_path.read_text(encoding="utf-8"))
            rows = list(prev.get("rows", []))
            for r in rows:
                done_keys.add((str(r["piece"]), str(r["domain"])))
            print(f"[{EXP_ID}] Resumed {len(done_keys)}/90 pages from {partial_path}", flush=True)

        last_save = time.perf_counter()
        model.eval()

        for idx, piece in enumerate(all_pieces, start=1):
            slug = str(piece["slug"])
            is_hw = bool(piece["handwritten"])
            ref_notes = _extract_musicxml_notes(Path(piece["gt"]))
            for domain in ("clean", "photo"):
                if (slug, domain) in done_keys:
                    continue
                img_path = Path(piece["clean"] if domain == "clean" else piece["photo"])
                img_pil = Image.open(img_path).convert("RGB")
                pil_resized = uni_omr._resize_fullpage_pil(img_pil, MAX_DIM)
                img_t = uni_omr._pil_to_chw_tensor(pil_resized).unsqueeze(0).to(device)

                t0 = time.perf_counter()
                with torch.no_grad():
                    if use_bf16 and device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            seq, mtp_stats = eval_s31.generate_mtp_spec(
                                model, img_t, max_len=MAX_LEN, use_grammar=True,
                                strict_grammar=False, return_stats=True,
                            )
                    else:
                        seq, mtp_stats = eval_s31.generate_mtp_spec(
                            model, img_t, max_len=MAX_LEN, use_grammar=True,
                            strict_grammar=False, return_stats=True,
                        )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                decode_secs = round(time.perf_counter() - t0, 3)

                seq_ids = seq[0].detach().cpu().tolist()
                has_eos = eos_id in seq_ids[1:]
                pred_raw = eval_s31.decode_uni_compat(
                    seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
                )
                if not has_eos:
                    pred_raw = eval_s31.trim_truncated_tokens(pred_raw)

                arm_res: Dict[str, Any] = {}
                for k in ARM_KEYS:
                    arm_res[k] = _apply_arm(
                        k, pred_raw, ref_notes, has_eos=has_eos,
                        id_to_tok=id_to_tok, eval_s31=eval_s31, rlvr_grpo=rlvr_grpo,
                    )

                rows.append({
                    "piece": slug,
                    "handwritten": is_hw,
                    "domain": domain,
                    "has_eos": bool(has_eos),
                    "decode_secs": decode_secs,
                    "mtp_acc": round(float(mtp_stats["acceptance_rate"]), 4),
                    "n_ref_notes": len(ref_notes),
                    "pred_raw": pred_raw,
                    "arms": arm_res,
                })
                done_keys.add((slug, domain))
                b_f1 = arm_res["raw_fsa_baseline"]["note_f1_py"]
                print(
                    f"[{idx:02d}/{len(all_pieces):02d}] {slug:32s} [{domain:5s}] base={b_f1:.4f} | "
                    + " ".join(
                        f"{k[:8]}:{arm_res[k]['note_f1_py'] - b_f1:+.4f}(e={arm_res[k]['edits']})"
                        for k in ARM_KEYS[1:]
                    ),
                    flush=True,
                )
                if (time.perf_counter() - last_save) >= SAVE_EVERY_MINUTES * 60.0 or domain == "photo":
                    _atomic_write_json(
                        partial_path,
                        {
                            "exp_id": EXP_ID,
                            "status": "RUNNING",
                            "summary": _summarize(rows),
                            "rows": rows,
                        },
                    )
                    last_save = time.perf_counter()

        total_scored = len(rows)
        assert total_scored > 0, f"{EXP_ID}: total_scored == 0!"
        total_edits = sum(
            int(r["arms"][k]["edits"])
            for r in rows
            for k in ARM_KEYS[1:]
        )
        assert total_edits > 0, f"{EXP_ID}: silent 0-edit bug — total_edits == 0 across {total_scored} pages!"

        summary = _summarize(rows)
        elapsed_secs = round(time.perf_counter() - t_start, 2)
        final_report = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_utc,
            "completed_utc": _utc_now(),
            "elapsed_secs": elapsed_secs,
            "platform": platform_str,
            "base_ckpt": HF_CKPT_PATH,
            "base_ckpt_meta": ckpt_meta,
            "total_scored": total_scored,
            "total_edits": total_edits,
            "config": {
                "arm_keys": list(ARM_KEYS),
                "max_dim": MAX_DIM,
                "max_len": MAX_LEN,
            },
            "summary": summary,
            "rows": rows,
        }
        final_local = CACHE_ROOT / "evals" / f"{EXP_ID}.json"
        _atomic_write_json(final_local, final_report)
        api.upload_file(
            path_or_fileobj=str(final_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload voice-aware rhythm repair eval JSON",
        )
        ledger_local = CACHE_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(
            ledger_local,
            {
                "exp_id": EXP_ID,
                "status": "COMPLETED",
                "started_utc": started_utc,
                "completed_utc": final_report["completed_utc"],
                "elapsed_secs": elapsed_secs,
                "platform": platform_str,
                "total_scored": total_scored,
                "total_edits": total_edits,
                "hf_eval_path": HF_EVAL_PATH,
                "summary": summary,
            },
        )
        api.upload_file(
            path_or_fileobj=str(ledger_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark COMPLETED",
        )
        print(f"[{EXP_ID}] COMPLETED in {elapsed_secs}s (scored={total_scored}, edits={total_edits})", flush=True)
        return 0
    finally:
        if TOKEN_PATH.exists():
            try:
                TOKEN_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

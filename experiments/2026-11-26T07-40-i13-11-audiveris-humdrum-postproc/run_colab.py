#!/usr/bin/env python3
"""Iter 13 (i13-11) on Google Colab L4:
Audiveris & Humdrum/Verovio Deterministic Symbolic Post-Processing Ablation
(Clef/Key Continuity, Accidental-Scope Normalization, Measure-Rhythm Repair)
on full 45-piece `suite31_hw` (90 pages: 31 printed + 14 HW, clean + photo).

Runs 1 GPU decode pass and evaluates all 5 deterministic CPU symbolic arms on
identical raw decoded token streams (0 ms GPU overhead if promoted).

Expects:
  - /content/bundle.tar.gz (or pre-extracted sheet-music-converter/tmp-train/mvd)
  - /tmp/.hf_token (Hugging Face token; chmod 600, deleted immediately and in finally)
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

EXP_ID = "2026-11-26T07-40-i13-11-audiveris-humdrum-postproc"
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
    Path("/content/ov_mvd_bundle.tar.gz"),
)
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

MAX_DIM = 1280
MAX_LEN = 3840
SAVE_EVERY_MINUTES = 5.0

ARM_KEYS: Tuple[str, ...] = (
    "raw_fsa_baseline",
    "audiveris_header_continuity",
    "humdrum_accidental_norm",
    "audiveris_rhythm_repair",
    "combined_all_rules",
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
_BAR_TOKENS = frozenset({"bar", "bar-light-heavy", "bar-repeat-right", "bar-double"})
_ALTER_TOKENS = frozenset({"a-2", "a-1", "a1", "a2"})
_STEP_TOKENS = frozenset({"A", "B", "C", "D", "E", "F", "G"})


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
# Competitor-Inspired Deterministic Symbolic Post-Processing Rules
# ---------------------------------------------------------------------------

def _split_systems(pred_raw: Sequence[str], sys_tok: str) -> List[List[str]]:
    systems: List[List[str]] = [[]]
    for tok in pred_raw:
        if tok == sys_tok:
            systems.append([])
        else:
            systems[-1].append(tok)
    return [s for s in systems if s]


def _join_systems(systems: Sequence[Sequence[str]], sys_tok: str) -> List[str]:
    out: List[str] = []
    for i, s in enumerate(systems):
        if i > 0:
            out.append(sys_tok)
        out.extend(s)
    return out


def rule_audiveris_header_continuity(pred_raw: Sequence[str], sys_tok: str = "<sys>") -> Tuple[List[str], int]:
    """Audiveris HEADERS rule: propagate consensus clef & key across systems.

    Recovers missing continuation headers and corrects 1-system singleton clef
    outliers when surrounded by identical consensus clefs on >= 3-system pages.
    """
    systems = _split_systems(pred_raw, sys_tok)
    if len(systems) < 2:
        return list(pred_raw), 0

    def _lead_headers(sys_toks: Sequence[str]) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str]]:
        clef_i = key_i = None
        clef_v = key_v = None
        for idx, t in enumerate(sys_toks[:8]):
            if t in _BAR_TOKENS or t in _WRITTEN_BEATS:
                break
            if t.startswith("clef-") and clef_i is None:
                clef_i, clef_v = idx, t
            elif t.startswith("key") and key_i is None:
                key_i, key_v = idx, t
        return clef_i, clef_v, key_i, key_v

    headers = [_lead_headers(s) for s in systems]
    clefs = [h[1] for h in headers if h[1] is not None]
    keys = [h[3] for h in headers if h[3] is not None]
    cons_clef = Counter(clefs).most_common(1)[0][0] if clefs else None
    cons_key = Counter(keys).most_common(1)[0][0] if keys else None

    edits = 0
    new_systems: List[List[str]] = []
    for i, sys_toks in enumerate(systems):
        s = list(sys_toks)
        ci, cv, ki, kv = headers[i]
        # Replace isolated 1-system singleton clef outlier on >= 3-system page
        if (
            len(systems) >= 3
            and ci is not None
            and cons_clef is not None
            and cv != cons_clef
            and clefs.count(cv) == 1
            and clefs.count(cons_clef) >= len(systems) - 1
        ):
            s[ci] = cons_clef
            edits += 1
        # Restore dropped continuation clef
        elif i > 0 and ci is None and cons_clef is not None:
            s.insert(0, cons_clef)
            edits += 1
            if ki is not None:
                ki += 1
        # Restore dropped continuation key signature
        if i > 0 and ki is None and cons_key is not None and keys.count(cons_key) >= max(2, len(systems) - 1):
            insert_pos = 1 if (len(s) > 0 and s[0].startswith("clef-")) else 0
            s.insert(insert_pos, cons_key)
            edits += 1
        new_systems.append(s)
    return _join_systems(new_systems, sys_tok), edits


def rule_humdrum_accidental_norm(pred_raw: Sequence[str], sys_tok: str = "<sys>") -> Tuple[List[str], int]:
    """Humdrum **kern accidental-scope cleanup.

    Strips malformed duplicate consecutive alterations on a single notehead and
    removes orphan alter tokens not immediately preceded by a valid pitch (step + octave).
    """
    out: List[str] = []
    edits = 0
    for tok in pred_raw:
        if tok in _ALTER_TOKENS:
            # Must follow a digit octave which follows a step token
            if len(out) >= 2 and out[-2] in _STEP_TOKENS and out[-1].lstrip("-").isdigit():
                out.append(tok)
            else:
                edits += 1
                continue
        else:
            out.append(tok)
    return out, edits


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


def rule_audiveris_rhythm_repair(pred_raw: Sequence[str], sys_tok: str = "<sys>") -> Tuple[List[str], int]:
    """Audiveris RHYTHMS step: conservative 1-edit measure duration repair.

    Tracks active time signature `time-B-T` across the page. Within each single-voice
    measure segment, sums written beats (including dots). If the measure duration
    deviates from the active time signature and can be fixed by a single conservative
    edit on the final note/rest (adding/removing one `dot`, or halving/doubling the
    final duration token), applies that exact repair.
    """
    systems = _split_systems(pred_raw, sys_tok)
    active_beats: Optional[float] = None
    total_edits = 0
    repaired_systems: List[List[str]] = []

    for sys_toks in systems:
        # Split into measures preserving barline tokens
        out_sys: List[str] = []
        cur_bar: List[str] = []

        def _flush_bar(bar_toks: List[str]) -> List[str]:
            nonlocal active_beats, total_edits
            for t in bar_toks:
                b = _parse_time_beats(t)
                if b is not None:
                    active_beats = b
            if active_beats is None or not bar_toks:
                return bar_toks
            # Only repair clean single-voice, non-tuplet, non-grace measures
            if any(t.startswith("voice-") or t.startswith("tuplet-") or t == "grace" or t == "bar-rest" for t in bar_toks):
                return bar_toks

            # Collect top-level duration events (skipping chord continuations)
            events: List[Tuple[int, str, int, float]] = []  # (dur_idx, dur_tok, n_dots, beats)
            i = 0
            while i < len(bar_toks):
                t = bar_toks[i]
                if t in _WRITTEN_BEATS:
                    base = _WRITTEN_BEATS[t]
                    j = i + 1
                    dots = 0
                    while j < len(bar_toks) and bar_toks[j] not in _WRITTEN_BEATS and bar_toks[j] != "chord":
                        if bar_toks[j] == "dot":
                            dots += 1
                        j += 1
                    ev_beats = base * (1.5 if dots == 1 else (1.75 if dots >= 2 else 1.0))
                    events.append((i, t, dots, ev_beats))
                    i = j
                    # Skip chord-stacked noteheads on the same onset
                    while i < len(bar_toks) and bar_toks[i] == "chord":
                        i += 1
                        while i < len(bar_toks) and bar_toks[i] not in _WRITTEN_BEATS and bar_toks[i] != "chord":
                            i += 1
                else:
                    i += 1

            if not events:
                return bar_toks

            bar_sum = sum(ev[3] for ev in events)
            diff = active_beats - bar_sum
            if abs(diff) <= 1e-5:
                return bar_toks

            last_idx, last_dur, last_dots, last_beats = events[-1]
            needed = last_beats + diff
            if needed <= 0.0:
                return bar_toks

            base_b = _WRITTEN_BEATS[last_dur]
            # Candidate 1: exact duration doubling/halving (0 dots)
            if last_dots == 0 and (abs(needed - base_b * 2.0) <= 1e-5 or abs(needed - base_b * 0.5) <= 1e-5):
                for b_val, d_name in _BEATS_TO_DUR.items():
                    if abs(b_val - needed) <= 1e-5:
                        fixed = list(bar_toks)
                        fixed[last_idx] = d_name
                        total_edits += 1
                        return fixed

            # Candidate 2: add one dot to final event
            if last_dots == 0 and abs(needed - base_b * 1.5) <= 1e-5:
                fixed = list(bar_toks)
                # Insert dot right after final event's pitch/rest tokens
                ins = last_idx + 1
                while ins < len(fixed) and fixed[ins] not in _WRITTEN_BEATS and fixed[ins] != "chord":
                    ins += 1
                fixed.insert(ins, "dot")
                total_edits += 1
                return fixed

            # Candidate 3: remove one stray dot from final event
            if last_dots == 1 and abs(needed - base_b) <= 1e-5:
                fixed = list(bar_toks)
                for k in range(last_idx + 1, len(fixed)):
                    if fixed[k] == "dot":
                        del fixed[k]
                        total_edits += 1
                        return fixed
                    if fixed[k] in _WRITTEN_BEATS or fixed[k] == "chord":
                        break

            return bar_toks

        for tok in sys_toks:
            if tok in _BAR_TOKENS:
                out_sys.extend(_flush_bar(cur_bar))
                out_sys.append(tok)
                cur_bar = []
            else:
                cur_bar.append(tok)
        if cur_bar:
            out_sys.extend(_flush_bar(cur_bar))
        repaired_systems.append(out_sys)

    return _join_systems(repaired_systems, sys_tok), total_edits


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
    if arm_key == "audiveris_header_continuity":
        toks, edits = rule_audiveris_header_continuity(toks)
    elif arm_key == "humdrum_accidental_norm":
        toks, edits = rule_humdrum_accidental_norm(toks)
    elif arm_key == "audiveris_rhythm_repair":
        toks, edits = rule_audiveris_rhythm_repair(toks)
    elif arm_key == "combined_all_rules":
        toks, e1 = rule_audiveris_header_continuity(toks)
        toks, e2 = rule_humdrum_accidental_norm(toks)
        toks, e3 = rule_audiveris_rhythm_repair(toks)
        edits = e1 + e2 + e3
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
        platform_str = f"colab-{gpu_name}"
        print(f"[{EXP_ID}] {platform_str} (cc={cc[0]}.{cc[1]}, bf16={use_bf16})", flush=True)

        api = HfApi(token=hf_token)
        ckpt_path, manifest_path = _ensure_assets(hf_token)
        model, cfg, ckpt_meta = eval_s31.load_uni_checkpoint(ckpt_path, device=device)
        _, _, eos_id, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))
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
                                model,
                                img_t,
                                max_len=MAX_LEN,
                                use_grammar=True,
                                strict_grammar=False,
                                return_stats=True,
                            )
                    else:
                        seq, mtp_stats = eval_s31.generate_mtp_spec(
                            model,
                            img_t,
                            max_len=MAX_LEN,
                            use_grammar=True,
                            strict_grammar=False,
                            return_stats=True,
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
                        k,
                        pred_raw,
                        ref_notes,
                        has_eos=has_eos,
                        id_to_tok=id_to_tok,
                        eval_s31=eval_s31,
                        rlvr_grpo=rlvr_grpo,
                    )

                rows.append({
                    "piece": slug,
                    "handwritten": is_hw,
                    "domain": domain,
                    "has_eos": bool(has_eos),
                    "decode_secs": decode_secs,
                    "mtp_acc": round(float(mtp_stats["acceptance_rate"]), 4),
                    "n_ref_notes": len(ref_notes),
                    "arms": arm_res,
                })
                done_keys.add((slug, domain))
                b_f1 = arm_res["raw_fsa_baseline"]["note_f1_py"]
                print(
                    f"[{idx:02d}/{len(all_pieces):02d}] {slug:32s} [{domain:5s}] base={b_f1:.4f} | "
                    + " ".join(f"{k[:6]}:{arm_res[k]['note_f1_py'] - b_f1:+.4f}" for k in ARM_KEYS[1:]),
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
            commit_message=f"{EXP_ID}: upload Audiveris/Humdrum symbolic postproc eval JSON",
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
        print(f"[{EXP_ID}] COMPLETED in {elapsed_secs}s", flush=True)
        return 0
    finally:
        if TOKEN_PATH.exists():
            try:
                TOKEN_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

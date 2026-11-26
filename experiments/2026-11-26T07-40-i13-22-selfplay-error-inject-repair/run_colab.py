#!/usr/bin/env python3
"""Iter 13 Moonshot i13-22 on Google Colab L4 / Kaggle T4:
Self-Play Synthetic-Error-Injection Critique & Repair (SCoRe-OMR).

Tests whether a lightweight zero-initialized Draft Memory Projector (~0.6M params)
trained with synthetic OMR error-injection (duration/dot/octave/barline/accidental
corruptions) + on-policy model drafts enables a Pass-2 self-repair decode that breaks
autoregressive exposure-bias cascades on suite31_hw.

Uploads to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T07-40-i13-22-selfplay-error-inject-repair.json
  - runs/2026-11-26T07-40-i13-22-selfplay-error-inject-repair.json
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

EXP_ID = "2026-11-26T07-40-i13-22-selfplay-error-inject-repair"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_PATH = "checkpoints/iter10_soup_baseline/uni_omr_soup.pt"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_CANDIDATES = (
    Path("/content/ov_bundle.tar.gz"),
    Path("/content/bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
)
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

SEED = 42
ARM_STEPS = 300
GRAD_ACCUM = 4
LR_BACKBONE = 2.0e-5
LR_PROJECTOR = 2.0e-4
TRAIN_MAX_DIM = 1280
TRAIN_MAX_SEQ = 1536
DRAFT_MAX_LEN = 512
EVAL_MAX_DIM = 1280
EVAL_MAX_LEN = 3840

# (arm_id, corrupt_prob, onpolicy_prob, corrupt_rate)
ARMS: List[Tuple[str, float, float, float]] = [
    ("arm0_sft_control", 0.0, 0.0, 0.0),
    ("arm1_selfplay_corrupt_p05", 0.50, 0.0, 0.12),
    ("arm2_selfplay_onpolicy_mix", 0.30, 0.30, 0.12),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
        alt = Path("/workspace/sheet-music-converter/tmp-train/mvd")
        if alt.exists():
            mvd_dir = alt
        else:
            bundle = next((b for b in BUNDLE_CANDIDATES if b.exists()), None)
            if bundle is None:
                raise FileNotFoundError(f"Missing bundle in {BUNDLE_CANDIDATES}")
            print(f"[{EXP_ID}] Extracting {bundle} -> {WORK_ROOT}...", flush=True)
            with tarfile.open(bundle, "r:*") as tf:
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
        importlib.invalidate_caches()
    return mvd_dir


def _extract_tar_once(tar_path: Path, dest_dir: Path) -> None:
    sentinel = dest_dir / ".extracted"
    if sentinel.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=dest_dir)
    sentinel.write_text(_utc_now() + "\n", encoding="utf-8")


def _ensure_assets(hf_token: str) -> Tuple[Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    ckpt_path = CACHE_ROOT / HF_CKPT_PATH
    if not ckpt_path.exists():
        hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", filename=HF_CKPT_PATH,
            local_dir=str(CACHE_ROOT), token=hf_token,
        )

    proxy_dir = CACHE_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    if not train_abs.exists():
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO, repo_type="dataset", filename=HF_PROXY_TAR,
                local_dir=str(CACHE_ROOT / "tars"), token=hf_token,
            )
        )
        _extract_tar_once(tar_p, proxy_dir)
        tar_p.unlink(missing_ok=True)
        src_path = proxy_dir / "proxy_train.jsonl"
        out_lines: List[str] = []
        for raw in src_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            rec["image"] = str((proxy_dir / rec["image"]).resolve())
            rec["bekerns"] = [str((proxy_dir / b).resolve()) for b in rec.get("bekerns", [])]
            out_lines.append(json.dumps(rec))
        train_abs.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    s31_dir = CACHE_ROOT / "datasets" / "suite31_hw_eval"
    if not (s31_dir / ".extracted").exists():
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO, repo_type="dataset", filename=HF_SUITE31_TAR,
                local_dir=str(CACHE_ROOT / "tars"), token=hf_token,
            )
        )
        _extract_tar_once(tar_p, s31_dir)
        tar_p.unlink(missing_ok=True)

    manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json under {s31_dir}")
    return ckpt_path, train_abs, manifests[0]


def _load_manifest_pairs(manifest_path: Path) -> List[Tuple[Path, List[Path], str]]:
    pairs: List[Tuple[Path, List[Path], str]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        img_p = Path(rec["image"])
        beks = [Path(b) for b in rec.get("bekerns", [])]
        domain = str(rec.get("domain", "")).strip().lower()
        if img_p.is_file() and beks:
            pairs.append((img_p, beks, domain))
    return pairs


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


def _select_sweep_pieces(pieces: Sequence[Dict[str, Any]], n_printed: int = 12) -> List[Dict[str, Any]]:
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


def _build_corruption_tables(id_to_tok: Sequence[str]) -> Dict[str, List[int]]:
    tables: Dict[str, List[int]] = {
        "dur": [],
        "oct": [],
        "step": [],
        "bar": [],
        "acc": [],
        "dot": [],
        "voice": [],
    }
    for tid, tok in enumerate(id_to_tok):
        if tok.startswith("dur-"):
            tables["dur"].append(tid)
        elif tok.startswith("oct-"):
            tables["oct"].append(tid)
        elif tok.startswith("step-"):
            tables["step"].append(tid)
        elif tok == "<bar>" or tok.startswith("bar-"):
            tables["bar"].append(tid)
        elif tok in ("a1", "a-1", "a2", "a-2", "a0"):
            tables["acc"].append(tid)
        elif tok == "dot":
            tables["dot"].append(tid)
        elif tok.startswith("voice-"):
            tables["voice"].append(tid)
    return tables


def _corrupt_bekern_ids(
    seq_ids: Sequence[int],
    tables: Dict[str, List[int]],
    special_ids: set[int],
    rng: random.Random,
    *,
    corrupt_rate: float = 0.12,
    max_len: int = DRAFT_MAX_LEN,
) -> List[int]:
    """Inject realistic OMR errors (duration drift, dot drop/insert, octave slip,
    barline drop/dup, accidental drop) into a ground-truth token sequence."""
    out: List[int] = []
    dur_set = set(tables["dur"])
    oct_list = tables["oct"]
    oct_pos = {tid: i for i, tid in enumerate(oct_list)}
    step_list = tables["step"]
    step_pos = {tid: i for i, tid in enumerate(step_list)}
    bar_set = set(tables["bar"])
    acc_set = set(tables["acc"])
    dot_ids = tables["dot"]
    voice_list = tables["voice"]

    for tid in seq_ids[:max_len]:
        if tid in special_ids:
            out.append(int(tid))
            continue
        if rng.random() >= corrupt_rate:
            out.append(int(tid))
            continue

        if tid in dur_set and len(tables["dur"]) >= 2:
            # Duration drift or stray dot injection
            if dot_ids and rng.random() < 0.35:
                out.append(int(tid))
                out.append(int(dot_ids[0]))
            else:
                out.append(int(rng.choice(tables["dur"])))
        elif tid in oct_pos and len(oct_list) >= 2:
            idx = oct_pos[tid]
            delta = rng.choice([-1, 1])
            nidx = max(0, min(len(oct_list) - 1, idx + delta))
            out.append(int(oct_list[nidx]))
        elif tid in step_pos and len(step_list) >= 2:
            idx = step_pos[tid]
            delta = rng.choice([-1, 1])
            nidx = max(0, min(len(step_list) - 1, idx + delta))
            out.append(int(step_list[nidx]))
        elif tid in bar_set:
            # 50% drop barline, 50% duplicate barline
            if rng.random() < 0.50:
                continue
            out.append(int(tid))
            out.append(int(tid))
        elif tid in acc_set or (dot_ids and tid == dot_ids[0]):
            # Drop accidental or augmentation dot
            continue
        elif voice_list and tid in voice_list and len(voice_list) >= 2:
            out.append(int(rng.choice(voice_list)))
        else:
            out.append(int(tid))

    return out[:max_len]


def _make_draft_projector(d_model: int, vocab_size: int, pad_id: int, device: Any) -> Any:
    import torch
    import torch.nn as nn

    class DraftMemoryProjector(nn.Module):
        """Lightweight (~0.6M) zero-gated draft-sequence encoder appended to visual memory."""

        def __init__(self) -> None:
            super().__init__()
            self.pad_id = int(pad_id)
            self.pos_emb = nn.Embedding(DRAFT_MAX_LEN, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=6,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
            self.norm = nn.LayerNorm(d_model)
            # Zero-initialized gate so Pass-2 starts bit-identical to Pass-1 at step 0
            self.alpha_draft = nn.Parameter(torch.zeros(1))

        def forward(self, draft_ids: Any, tok_emb: nn.Embedding) -> Any:
            bsz, seq_len = draft_ids.shape
            seq_len = min(seq_len, DRAFT_MAX_LEN)
            ids = draft_ids[:, :seq_len]
            pos = torch.arange(seq_len, device=ids.device).unsqueeze(0).expand(bsz, seq_len)
            x = tok_emb(ids) + self.pos_emb(pos)
            pad_mask = ids.eq(self.pad_id)
            h = self.encoder(x, src_key_padding_mask=pad_mask)
            h = self.norm(h)
            return torch.tanh(self.alpha_draft) * h

    return DraftMemoryProjector().to(device)


def _greedy_decode_with_memory(
    model: Any,
    mem: Any,
    *,
    vocab_size: int,
    max_len: int,
    device: Any,
    amp_dtype: Any,
) -> Tuple[List[int], bool]:
    import torch
    import eval_uni_suite31 as eval_s31
    import grammar

    _, bos_id, eos_id, _, _, id_to_tok = eval_s31._vocab_spec(int(vocab_size))
    fsa = grammar.PrefixState(id_to_tok, strict=False)
    generated: List[int] = [bos_id]
    past_kvs = None
    cur_tok = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        with torch.no_grad():
            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    dec_out = model.decode_tokens(
                        cur_tok, mem, return_mtp=False, past_kvs=past_kvs,
                        use_cache=True, apply_tok_dropout=False,
                    )
            else:
                dec_out = model.decode_tokens(
                    cur_tok, mem, return_mtp=False, past_kvs=past_kvs,
                    use_cache=True, apply_tok_dropout=False,
                )
        logits = dec_out["logits"][:, -1, :].float()
        past_kvs = dec_out.get("past_kvs")
        mask = fsa.valid_mask(device=device)
        if mask is not None and mask.shape[-1] == logits.shape[-1]:
            logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)
        next_id = int(torch.argmax(logits, dim=-1).item())
        generated.append(next_id)
        if next_id == eos_id:
            return generated, True
        fsa.step(next_id)
        cur_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return generated, False


def _evaluate_two_pass(
    model: Any,
    projector: Optional[Any],
    cfg: Any,
    pieces: Sequence[Dict[str, Any]],
    device: Any,
    *,
    tag: str,
    run_pass2: bool,
    amp_dtype: Any,
) -> Dict[str, Any]:
    from PIL import Image
    import torch
    import eval_uni_suite31 as eval_s31
    import rlvr_grpo
    import uni_omr

    pad_id, _, eos_id, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))
    model.eval()
    if projector is not None:
        projector.eval()

    rows: List[Dict[str, Any]] = []
    t_all = time.perf_counter()

    def _score_seq(seq_ids: List[int], has_eos: bool, ref_notes: Sequence[Any]) -> Dict[str, Any]:
        pred_raw = eval_s31.decode_uni_compat(
            seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
        )
        if not has_eos:
            pred_raw = eval_s31.trim_truncated_tokens(pred_raw)
        hyp_toks, _ = eval_s31.stitch_uni_tokens(pred_raw)
        struct = rlvr_grpo.compute_unlabeled_structural_reward(
            pred_raw, vocab=id_to_tok, terminated_eos=has_eos
        )
        hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
        note_f1 = rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes)
        return {
            "note_f1": float(note_f1),
            "r_sym": float(struct["reward"]),
            "r_rhythm": float(struct["r_rhythm"]),
            "r_noloop": float(struct["r_noloop"]),
            "has_eos": bool(has_eos),
        }

    for idx, piece in enumerate(pieces, start=1):
        ref_notes = _extract_musicxml_notes(piece["gt"])
        prow: Dict[str, Any] = {
            "piece": piece["slug"],
            "handwritten": bool(piece["handwritten"]),
            "stratum": piece.get("stratum", ""),
        }
        for domain in ("clean", "photo"):
            with Image.open(piece[domain]) as raw_im:
                pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), EVAL_MAX_DIM)
            img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)

            with torch.no_grad():
                if amp_dtype is not None and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        seq1, _ = eval_s31.generate_mtp_spec(
                            model, img_t, max_len=EVAL_MAX_LEN, use_grammar=True,
                            strict_grammar=False, return_stats=True,
                        )
                else:
                    seq1, _ = eval_s31.generate_mtp_spec(
                        model, img_t, max_len=EVAL_MAX_LEN, use_grammar=True,
                        strict_grammar=False, return_stats=True,
                    )
            seq1_ids = seq1[0].detach().cpu().tolist()
            has_eos1 = eos_id in seq1_ids[1:]
            s1 = _score_seq(seq1_ids, has_eos1, ref_notes)

            s2_raw = dict(s1)
            s2_gated = dict(s1)
            edited = False
            edit_improved = False

            if run_pass2 and projector is not None:
                with torch.no_grad():
                    if amp_dtype is not None and device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            vis_mem, _, _ = model.encode_image(img_t)
                            draft_t = torch.tensor(
                                [seq1_ids[:DRAFT_MAX_LEN]], dtype=torch.long, device=device
                            )
                            draft_mem = projector(draft_t, model.decoder.tok_emb)
                            mem_aug = torch.cat([vis_mem, draft_mem], dim=1)
                    else:
                        vis_mem, _, _ = model.encode_image(img_t)
                        draft_t = torch.tensor(
                            [seq1_ids[:DRAFT_MAX_LEN]], dtype=torch.long, device=device
                        )
                        draft_mem = projector(draft_t, model.decoder.tok_emb)
                        mem_aug = torch.cat([vis_mem, draft_mem], dim=1)
                seq2_ids, has_eos2 = _greedy_decode_with_memory(
                    model, mem_aug, vocab_size=int(cfg.vocab_size),
                    max_len=EVAL_MAX_LEN, device=device, amp_dtype=amp_dtype,
                )
                s2_raw = _score_seq(seq2_ids, has_eos2, ref_notes)
                edited = seq2_ids != seq1_ids
                edit_improved = edited and (s2_raw["note_f1"] > s1["note_f1"] + 1e-6)
                if s2_raw["r_sym"] >= s1["r_sym"] and s2_raw["r_noloop"] >= 1.0:
                    s2_gated = s2_raw

            pfx = domain + "_"
            prow[pfx + "p1_f1"] = round(s1["note_f1"], 6)
            prow[pfx + "p2_raw_f1"] = round(s2_raw["note_f1"], 6)
            prow[pfx + "p2_gated_f1"] = round(s2_gated["note_f1"], 6)
            prow[pfx + "p1_r_rhythm"] = round(s1["r_rhythm"], 4)
            prow[pfx + "p2_r_rhythm"] = round(s2_gated["r_rhythm"], 4)
            prow[pfx + "edited"] = bool(edited)
            prow[pfx + "edit_improved"] = bool(edit_improved)

        rows.append(prow)
        if idx % 10 == 0 or idx == len(pieces):
            print(f"[eval:{tag}] {idx:02d}/{len(pieces):02d} ({time.perf_counter() - t_all:.1f}s)", flush=True)

    def _agg(sub: Sequence[Dict[str, Any]], dom: str) -> Dict[str, Any]:
        p = dom + "_"
        n = max(1, len(sub))
        edits = sum(1 for r in sub if r[p + "edited"])
        edit_wins = sum(1 for r in sub if r[p + "edit_improved"])
        return {
            "n": len(sub),
            "p1_f1_mean": round(sum(r[p + "p1_f1"] for r in sub) / n, 6),
            "p2_raw_f1_mean": round(sum(r[p + "p2_raw_f1"] for r in sub) / n, 6),
            "p2_gated_f1_mean": round(sum(r[p + "p2_gated_f1"] for r in sub) / n, 6),
            "edit_rate": round(edits / n, 4),
            "edit_precision": round(edit_wins / max(1, edits), 4) if edits else 0.0,
        }

    pr_rows = [r for r in rows if not r["handwritten"]]
    hw_rows = [r for r in rows if r["handwritten"]]
    summary = {
        "printed_clean": _agg(pr_rows, "clean"),
        "printed_photo": _agg(pr_rows, "photo"),
        "hw_clean": _agg(hw_rows, "clean"),
        "hw_photo": _agg(hw_rows, "photo"),
    }
    return {"summary": summary, "pieces": rows, "elapsed_secs": round(time.perf_counter() - t_all, 2)}


def main() -> None:
    started_utc = _utc_now()
    t_start = time.perf_counter()
    hf_token = _read_and_purge_hf_token()
    mvd_dir = _setup_environment()
    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))

    import torch
    import torch.nn.functional as F
    from huggingface_hub import HfApi
    import eval_uni_suite31 as eval_s31
    import uni_omr

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
        amp_dtype = (
            torch.bfloat16 if (device.type == "cuda" and cc[0] >= 8)
            else (torch.float16 if device.type == "cuda" else None)
        )
        host_pfx = "kaggle" if Path("/kaggle/working").exists() else "colab"
        platform_str = f"{host_pfx}-{gpu_name}"

        ckpt_path, train_manifest, s31_manifest = _ensure_assets(hf_token)
        train_pairs = _load_manifest_pairs(train_manifest)
        all_pieces = eval_s31.load_suite31_manifest(s31_manifest)
        sweep_pieces = _select_sweep_pieces(all_pieces, n_printed=12)

        api = HfApi(token=hf_token)
        ledger_tmp = CACHE_ROOT / f"{EXP_ID}_run.json"
        _atomic_write_json(
            ledger_tmp,
            {
                "exp_id": EXP_ID,
                "status": "RUNNING",
                "started_utc": started_utc,
                "platform": platform_str,
                "arms": [a[0] for a in ARMS],
            },
        )
        try:
            api.upload_file(
                path_or_fileobj=str(ledger_tmp), path_in_repo=HF_RUN_PATH,
                repo_id=HF_REPO, repo_type="dataset",
                commit_message=f"{EXP_ID}: mark RUNNING on {platform_str}",
            )
        except Exception as exc:
            print(f"[{EXP_ID}] Warning: initial HF ledger upload failed: {exc}", flush=True)

        model, cfg, _ = eval_s31.load_uni_checkpoint(ckpt_path, device=device)
        ref_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        pad_id, bos_id, eos_id, sys_id, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))
        special_ids = {pad_id, bos_id, eos_id, sys_id}
        corrupt_tables = _build_corruption_tables(id_to_tok)

        train_ds = uni_omr.UniFullPageDataset(
            train_pairs,
            fullpage_max_dim=TRAIN_MAX_DIM,
            max_seq_len=TRAIN_MAX_SEQ,
            augment=True,
            return_uint8=True,
        )

        baseline_eval = _evaluate_two_pass(
            model, None, cfg, sweep_pieces, device,
            tag="baseline", run_pass2=False, amp_dtype=amp_dtype,
        )

        arms_out: Dict[str, Any] = {}
        for arm_id, corrupt_prob, onpolicy_prob, corrupt_rate in ARMS:
            print(
                f"[{EXP_ID}] === {arm_id} (corrupt_p={corrupt_prob}, onpolicy_p={onpolicy_prob}) ===",
                flush=True,
            )
            rng = random.Random(SEED)
            torch.manual_seed(SEED)
            model.load_state_dict({k: v.to(device) for k, v in ref_state_cpu.items()})

            use_proj = (corrupt_prob > 0.0) or (onpolicy_prob > 0.0)
            projector = (
                _make_draft_projector(int(cfg.d_model), int(cfg.vocab_size), pad_id, device)
                if use_proj else None
            )
            param_groups: List[Dict[str, Any]] = [
                {"params": list(model.parameters()), "lr": LR_BACKBONE}
            ]
            if projector is not None:
                param_groups.append({"params": list(projector.parameters()), "lr": LR_PROJECTOR})
            optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
            scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

            t_arm = time.perf_counter()
            loss_hist: List[float] = []
            mode_counts = {"sft": 0, "corrupt": 0, "onpolicy": 0}

            for step in range(1, ARM_STEPS + 1):
                prog = (step - 1) / max(1, ARM_STEPS - 1)
                lr_mult = 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * prog))
                optimizer.param_groups[0]["lr"] = LR_BACKBONE * lr_mult
                if len(optimizer.param_groups) > 1:
                    optimizer.param_groups[1]["lr"] = LR_PROJECTOR * lr_mult

                model.train()
                if projector is not None:
                    projector.train()
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0

                for _ in range(GRAD_ACCUM):
                    sample = train_ds[rng.randrange(len(train_ds))]
                    img_u8, tgt_ids = sample[0], sample[1]
                    img_t = uni_omr._images_to_device_float(img_u8.unsqueeze(0), device)
                    tgt_t = tgt_ids.unsqueeze(0).to(device)
                    inp_t = tgt_t[:, :-1]
                    lbl_t = tgt_t[:, 1:]

                    u = rng.random()
                    if use_proj and u < onpolicy_prob:
                        mode = "onpolicy"
                    elif use_proj and u < (onpolicy_prob + corrupt_prob):
                        mode = "corrupt"
                    else:
                        mode = "sft"
                    mode_counts[mode] += 1

                    draft_t: Optional[Any] = None
                    if mode == "corrupt":
                        c_ids = _corrupt_bekern_ids(
                            tgt_ids.tolist(), corrupt_tables, special_ids, rng,
                            corrupt_rate=corrupt_rate, max_len=DRAFT_MAX_LEN,
                        )
                        draft_t = torch.tensor([c_ids], dtype=torch.long, device=device)
                    elif mode == "onpolicy":
                        model.eval()
                        with torch.no_grad():
                            vis_m, _, _ = model.encode_image(img_t)
                            op_ids, _ = _greedy_decode_with_memory(
                                model, vis_m, vocab_size=int(cfg.vocab_size),
                                max_len=min(256, DRAFT_MAX_LEN), device=device, amp_dtype=amp_dtype,
                            )
                        model.train()
                        draft_t = torch.tensor([op_ids], dtype=torch.long, device=device)

                    ctx = (
                        torch.autocast(device_type="cuda", dtype=amp_dtype)
                        if (amp_dtype is not None and device.type == "cuda")
                        else torch.no_grad() if False else None
                    )
                    if ctx is not None:
                        with ctx:
                            vis_mem, _, _ = model.encode_image(img_t)
                            if draft_t is not None and projector is not None:
                                d_mem = projector(draft_t, model.decoder.tok_emb)
                                mem_in = torch.cat([vis_mem, d_mem], dim=1)
                            else:
                                mem_in = vis_mem
                            dec_out = model.decode_tokens(inp_t, mem_in, return_mtp=False)
                            logits = dec_out["logits"]
                            loss = F.cross_entropy(
                                logits.reshape(-1, logits.size(-1)),
                                lbl_t.reshape(-1),
                                ignore_index=pad_id,
                                label_smoothing=0.02,
                            ) / GRAD_ACCUM
                    else:
                        vis_mem, _, _ = model.encode_image(img_t)
                        if draft_t is not None and projector is not None:
                            d_mem = projector(draft_t, model.decoder.tok_emb)
                            mem_in = torch.cat([vis_mem, d_mem], dim=1)
                        else:
                            mem_in = vis_mem
                        dec_out = model.decode_tokens(inp_t, mem_in, return_mtp=False)
                        logits = dec_out["logits"]
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            lbl_t.reshape(-1),
                            ignore_index=pad_id,
                            label_smoothing=0.02,
                        ) / GRAD_ACCUM

                    if torch.isfinite(loss):
                        if scaler.is_enabled():
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        step_loss += float(loss.detach().item())

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if projector is not None:
                        torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if projector is not None:
                        torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
                    optimizer.step()

                loss_hist.append(step_loss)
                if step == 1 or step % 50 == 0 or step == ARM_STEPS:
                    alpha_val = (
                        float(torch.tanh(projector.alpha_draft).detach().item())
                        if projector is not None else 0.0
                    )
                    print(
                        f"[{arm_id}] step {step:03d}/{ARM_STEPS} loss={step_loss:.4f} "
                        f"alpha={alpha_val:+.4f} ({time.perf_counter() - t_arm:.1f}s)",
                        flush=True,
                    )

            arm_eval = _evaluate_two_pass(
                model, projector, cfg, sweep_pieces, device,
                tag=arm_id, run_pass2=use_proj, amp_dtype=amp_dtype,
            )
            arms_out[arm_id] = {
                "corrupt_prob": corrupt_prob,
                "onpolicy_prob": onpolicy_prob,
                "mode_counts": mode_counts,
                "final_alpha_draft": (
                    round(float(torch.tanh(projector.alpha_draft).detach().item()), 6)
                    if projector is not None else 0.0
                ),
                "mean_last50_loss": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5),
                "train_secs": round(time.perf_counter() - t_arm, 2),
                "eval": arm_eval,
            }

        payload = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_utc,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.perf_counter() - t_start, 2),
            "platform": platform_str,
            "baseline_ckpt": HF_CKPT_PATH,
            "baseline_eval": baseline_eval,
            "arms": arms_out,
        }
        eval_local = CACHE_ROOT / "evals" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, payload)
        api.upload_file(
            path_or_fileobj=str(eval_local), path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO, repo_type="dataset",
            commit_message=f"{EXP_ID}: upload completed eval JSON",
        )
        _atomic_write_json(
            ledger_tmp,
            {
                "exp_id": EXP_ID,
                "status": "COMPLETED",
                "started_utc": started_utc,
                "completed_utc": payload["completed_utc"],
                "elapsed_secs": payload["elapsed_secs"],
                "platform": platform_str,
                "hf_eval_path": HF_EVAL_PATH,
            },
        )
        api.upload_file(
            path_or_fileobj=str(ledger_tmp), path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO, repo_type="dataset",
            commit_message=f"{EXP_ID}: mark COMPLETED",
        )
        print(f"[{EXP_ID}] COMPLETED and uploaded to {HF_EVAL_PATH}", flush=True)
    finally:
        if TOKEN_PATH.exists():
            try:
                TOKEN_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()

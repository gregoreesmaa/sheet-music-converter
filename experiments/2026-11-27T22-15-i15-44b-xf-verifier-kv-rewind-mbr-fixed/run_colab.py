#!/usr/bin/env python3
"""Colab A100 / Local RTX 4080 SUPER runner — Iter 15 Exp i15-44b: Fixed Verifier-Triggered KV-Cache Rewind + Structure-Conditional MBR.

Experiment ID: 2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed
Cross-field origin: Compilers (VerMCTS/FIM), Drift/RL (Verifier-Triggered KV-Cache Rewind),
Connectomics (NEURD), HITL (Structure-Conditional MBR).

Clean ITERATE successor to i15-44 fixing the two bugs found in i15-44/OUTCOME.md:
  1. Proper `suite31_hw` MusicXML Note F1 & wFPP scoring via `eval_uni_suite31` /
     `eval_sys_gate` Rust note-matching pipeline (with pure-Python MusicXML
     ElementTree fallback on Colab Linux; NEVER `piece["gt"].read_text().split()`).
  2. Hard-ban `<eos>` inside `_rollout_bar_candidate` + heavy penalty (-100.0) in
     `_select_mbr_bar_candidate` for premature `<eos>` or unclosed rollouts, plus
     violation-only rewind trigger (no noisy `peak_ent > 1.60` trigger) and `K=2`
     candidates (`MAX_RETRIES_PER_BAR=1`, `MAX_REWINDS_PER_PAGE=8`) targeting `<= 1.35x`
     latency overhead.

Measures per arm (overall, printed, hw):
  - Note F1 (real MusicXML note-matching), TokLCS F1, BSVR (Bar-Sum Violation Rate),
    wFPP (weighted Fixes-Per-Page proxy), wall-clock ms/page, rewinds/page.

Includes 1-arm HF partial checkpoint sync + startup auto-resume and Windows cuDNN 9.10 safety.
Security: reads HF_TOKEN from /tmp/.hf_token (chmod 600), HF_TOKEN env, or ~/.cache/huggingface/token;
never prints it, always unlinks /tmp/.hf_token in finally.
"""

from __future__ import annotations

import argparse
import difflib
import gc
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed"
EXP_CODE = "i15-44b"
ARMS = ("greedy_fsa", "kv_rewind_mbr")
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter14-v2a-final/uni_omr_v2_ema_latest.pt",
    "checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt",
)
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_CKPT_PREFIX = f"checkpoints/{EXP_ID}"

HF_TOKEN_FILE = Path("/tmp/.hf_token")


def _default_work_root() -> Path:
    env_root = os.environ.get("OMR_WORK_ROOT", "").strip()
    if env_root:
        return Path(env_root) / "i15_44b"
    if Path("/vol").is_dir():
        return Path("/vol/i15_44b")
    if Path("/content").is_dir():
        return Path("/content/i15_44b")
    if sys.platform == "win32":
        return Path("D:/tmp/omr_work/i15_44b")
    return Path("/tmp/i15_44b")


WORK_ROOT = _default_work_root()

SEED = 42
MAX_DIM = 1408
MAX_LEN = 2048
MAX_MEM_TOKENS = 4096
N_PRINTED_PAGES = 6
N_HW_PAGES = 6

# Violation-only KV-Rewind + MBR hyperparameters (low-latency <= 1.35x target)
MAX_RETRIES_PER_BAR = 1
MAX_REWINDS_PER_PAGE = 8
MBR_K_CANDIDATES = 2
MBR_TEMP = 0.70
MBR_TOP_P = 0.90
MAX_BAR_RETRY_TOKENS = 96
BEAT_TOL = 1e-3
PREMATURE_EOS_PENALTY = -100.0
UNCLOSED_BAR_PENALTY = -50.0

PREFERRED_HARD_PRINTED = (
    "beethoven-op13",
    "chopin-op28",
    "bach-wtc",
    "schumann",
    "schubert",
    "mozart",
    "brahms",
    "liszt",
)


# ---------------------------------------------------------------------------
# Infrastructure Helpers
# ---------------------------------------------------------------------------

def _configure_backend_safety() -> None:
    import torch

    if sys.platform == "win32":
        torch.backends.cudnn.enabled = False
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_hf_token() -> str:
    if HF_TOKEN_FILE.is_file():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        tok = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    cache_tok = Path.home() / ".cache" / "huggingface" / "token"
    if cache_tok.is_file():
        tok = cache_tok.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    raise RuntimeError("HF token missing: provide /tmp/.hf_token, HF_TOKEN env var, or ~/.cache/huggingface/token")


def _purge_hf_token() -> None:
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass


def _locate_mvd() -> Path:
    candidates = [
        Path.cwd() / "tmp-train" / "mvd",
        Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path("/workspace/mvd"),
        Path("/content/mvd"),
        WORK_ROOT / "mvd",
    ]
    for cand in candidates:
        if (cand / "uni_omr_v2.py").is_file() and (cand / "eval_uni_suite31.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    for bundle in (Path("/content/bundle.tar.gz"), Path("/content/mvd_bundle.tar.gz")):
        if bundle.is_file():
            WORK_ROOT.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORK_ROOT)
            for p in WORK_ROOT.rglob("uni_omr_v2.py"):
                if str(p.parent) not in sys.path:
                    sys.path.insert(0, str(p.parent))
                return p.parent
    raise FileNotFoundError("Could not locate tmp-train/mvd/uni_omr_v2.py")


def _ensure_deps() -> None:
    needed = []
    for mod, pkg in (("huggingface_hub", "huggingface_hub>=0.26"), ("PIL", "pillow"), ("numpy", "numpy")):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _upload_json_to_hf(hf_token: str, local_path: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"{EXP_CODE}({EXP_ID}): upload {path_in_repo}",
    )
    print(f"[{EXP_ID}] Uploaded {path_in_repo} -> {HF_REPO_ID}", flush=True)


def _try_resume_from_hf(hf_token: str) -> Dict[str, Dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    recovered: Dict[str, Dict[str, Any]] = {}
    dl_dir = WORK_ROOT / "hf_dl" / "resume"
    dl_dir.mkdir(parents=True, exist_ok=True)

    try:
        ev_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_EVAL_PATH,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(dl_dir),
        )
        ev_data = json.loads(Path(ev_path).read_text(encoding="utf-8"))
        for arm_name, arm_res in (ev_data.get("arms") or {}).items():
            if arm_name in ARMS and isinstance(arm_res, dict) and "note_f1" in arm_res:
                recovered[arm_name] = arm_res
    except Exception:
        pass

    for arm_name in ARMS:
        if arm_name in recovered:
            continue
        try:
            p = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"{HF_CKPT_PREFIX}/{arm_name}_arm_result.json",
                repo_type="dataset",
                token=hf_token,
                local_dir=str(dl_dir),
            )
            arm_res = json.loads(Path(p).read_text(encoding="utf-8"))
            if isinstance(arm_res, dict) and "note_f1" in arm_res:
                recovered[arm_name] = arm_res
        except Exception:
            pass

    for arm_name, arm_res in recovered.items():
        cached = WORK_ROOT / "ckpts" / arm_name / "arm_result.json"
        if not cached.is_file():
            _atomic_write_json(cached, arm_res)
        print(f"[{EXP_ID}] Auto-resumed completed arm '{arm_name}' from HF Private Hub", flush=True)

    return recovered


def _ensure_suite31(hf_token: str) -> Path:
    from huggingface_hub import hf_hub_download

    # Check local repo manifest first if available and images exist
    local_manifest = Path(__file__).resolve().parents[2] / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"
    if local_manifest.is_file():
        import eval_uni_suite31
        _, pieces = eval_uni_suite31.resolve_manifest_pieces(local_manifest)
        if pieces and pieces[0]["photo"].is_file():
            return local_manifest

    s31_dir = WORK_ROOT / "datasets" / "suite31_hw_eval"
    sentinel = s31_dir / ".extracted"
    if not sentinel.is_file():
        dl_dir = WORK_ROOT / "hf_dl"
        dl_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR} from {HF_REPO_ID}...", flush=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_SUITE31_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(dl_dir),
        )
        s31_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=s31_dir)
        sentinel.write_text(_utc_now() + "\n", encoding="utf-8")
        try:
            os.remove(tar_p)
        except OSError:
            pass

    manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json under {s31_dir}")
    return manifests[0]


def _ensure_checkpoint(hf_token: str, device: "torch.device") -> Tuple[Any, str]:
    from huggingface_hub import hf_hub_download
    import eval_uni_suite31

    local_override = os.environ.get("UNIOMR_CKPT", "").strip()
    if local_override and Path(local_override).is_file():
        model, _, meta = eval_uni_suite31.load_uni_checkpoint(local_override, device)
        return model, f"local:{local_override} (step={meta.get('step')})"

    dl_dir = WORK_ROOT / "hf_dl" / "ckpt"
    dl_dir.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for cand in HF_CKPT_CANDIDATES:
        try:
            print(f"[{EXP_ID}] Fetching {cand} from {HF_REPO_ID}...", flush=True)
            p = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=cand,
                repo_type="dataset",
                token=hf_token,
                local_dir=str(dl_dir),
            )
            model, _, meta = eval_uni_suite31.load_uni_checkpoint(p, device)
            return model, f"hf:{cand} (step={meta.get('step')})"
        except Exception as exc:
            last_err = exc
            print(f"[{EXP_ID}] Fallback: could not load {cand} ({exc})", flush=True)
    raise RuntimeError(f"Could not load any HF checkpoint from {HF_CKPT_CANDIDATES}: {last_err}")


def _select_hard_12_subset(pieces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministically pick 6 challenging printed + 6 real handwritten photo pages from suite31_hw."""
    printed = [p for p in pieces if not p["handwritten"] and p["photo"].is_file()]
    hw = [p for p in pieces if p["handwritten"] and p["photo"].is_file()]

    def _printed_rank(p: Dict[str, Any]) -> Tuple[int, str]:
        slug = p["slug"].lower()
        for idx, pref in enumerate(PREFERRED_HARD_PRINTED):
            if pref in slug:
                return (idx, slug)
        return (999, slug)

    printed_sorted = sorted(printed, key=_printed_rank)
    hw_sorted = sorted(hw, key=lambda p: p["slug"].lower())
    return printed_sorted[:N_PRINTED_PAGES] + hw_sorted[:N_HW_PAGES]


def _pil_to_tensor(img_pil: Any, max_dim: int) -> "torch.Tensor":
    import numpy as np
    import torch
    from PIL import Image

    rgb = img_pil.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, float(max_dim) / float(max(w, h)))
    nw = max(32, int(round(w * scale / 16.0)) * 16)
    nh = max(32, int(round(h * scale / 16.0)) * 16)
    if (nw, nh) != (w, h):
        rgb = rgb.resize((nw, nh), Image.BICUBIC)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return (t - 0.5) / 0.5


# ---------------------------------------------------------------------------
# Verifier-Triggered KV-Cache Rewind + Structure-Conditional MBR Decoder
# ---------------------------------------------------------------------------

def _slice_past_kvs(past_kvs: List[Any], keep_len: int) -> List[Any]:
    """Rewind self-attention KV caches back to `keep_len` in O(L) views;
    cross-attention KV caches (index 1) are immutable and reused for free."""
    rewound: List[Any] = []
    for layer_kv in past_kvs:
        if layer_kv is None:
            rewound.append(None)
            continue
        self_kv, cross_kv = layer_kv
        if self_kv is None or keep_len <= 0:
            new_self = None
        else:
            k_s, v_s = self_kv
            new_self = (k_s[:, :, :keep_len, :].contiguous(), v_s[:, :, :keep_len, :].contiguous())
        rewound.append((new_self, cross_kv))
    return rewound


def _kv_seq_len(past_kvs: Optional[List[Any]]) -> int:
    if not past_kvs or past_kvs[0] is None or past_kvs[0][0] is None:
        return 0
    return int(past_kvs[0][0][0].shape[2])


def _apply_fsa_mask(
    logits_1d: "torch.Tensor",
    fsa_state: Any,
    eos_id: int,
    sys_id: int,
    pad_id: int,
    bos_id: int,
    extra_ban_id: Optional[int] = None,
    ban_eos: bool = False,
) -> "torch.Tensor":
    import torch

    allow_ids, eos_ok = fsa_state.allowed()
    V = logits_1d.shape[0]
    mask = torch.full((V,), float("-inf"), device=logits_1d.device, dtype=logits_1d.dtype)
    valid = [i for i in allow_ids if 0 <= i < V]
    if eos_ok:
        if (not ban_eos) and 0 <= eos_id < V:
            valid.append(eos_id)
        if 0 <= sys_id < V:
            valid.append(sys_id)
    if valid:
        mask[valid] = 0.0
    else:
        mask.zero_()
    for sp in (pad_id, bos_id):
        if 0 <= sp < V:
            mask[sp] = float("-inf")
    if ban_eos and 0 <= eos_id < V:
        mask[eos_id] = float("-inf")
    if extra_ban_id is not None and 0 <= extra_ban_id < V and len(valid) > 1:
        mask[extra_ban_id] = float("-inf")
    masked = logits_1d + mask
    if not bool(torch.isfinite(masked).any()):
        # Safe fallback: still forbid pad/bos/(eos if ban_eos) if possible
        fallback = logits_1d.clone()
        for sp in (pad_id, bos_id):
            if 0 <= sp < V:
                fallback[sp] = float("-inf")
        if ban_eos and 0 <= eos_id < V:
            fallback[eos_id] = float("-inf")
        if bool(torch.isfinite(fallback).any()):
            return fallback
        return logits_1d
    return masked


def _sample_top_p(logits_1d: "torch.Tensor", temperature: float, top_p: float) -> Tuple[int, float]:
    import torch
    import torch.nn.functional as F

    scaled = logits_1d.float() / max(1e-3, float(temperature))
    log_probs = F.log_softmax(scaled, dim=-1)
    probs = torch.exp(log_probs)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    remove = cum > top_p
    remove[0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    total = sorted_probs.sum()
    if total <= 0.0 or not torch.isfinite(total):
        tok_id = int(torch.argmax(logits_1d).item())
        return tok_id, float(log_probs[tok_id].item())
    sorted_probs = sorted_probs / total
    choice = int(torch.multinomial(sorted_probs, 1).item())
    tok_id = int(sorted_idx[choice].item())
    return tok_id, float(log_probs[tok_id].item())


def _is_structural_tok(tok: str) -> bool:
    return (
        tok == "<sys>"
        or tok.startswith(("clef-", "key", "time-", "bar", "staff-", "voice-", "tuplet-"))
        or tok in ("breve", "whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "128th")
    )


def _asymmetric_bar_edit_cost(toks_a: Sequence[str], toks_b: Sequence[str]) -> float:
    """Asymmetric edit cost between two candidate bars: structural/header/duration diffs cost 2.0x, others 1.0x."""
    sm = difflib.SequenceMatcher(None, toks_a, toks_b, autojunk=False)
    cost = 0.0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for t in toks_a[i1:i2]:
            cost += 2.0 if _is_structural_tok(t) else 1.0
        for t in toks_b[j1:j2]:
            cost += 2.0 if _is_structural_tok(t) else 1.0
    denom = max(1.0, float(len(toks_a) + len(toks_b)))
    return cost / denom


def _select_mbr_bar_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Structure-Conditional MBR selection over K+1 bar candidates.

    Maximizes: conservation_bonus + 0.5 * mean_logprob - 1.5 * mean_asymmetric_edit_cost_to_peers.

    Bug #2 fix: NEVER award `cons_term = +2.0` to a premature-`<eos>` or unclosed bar
    rollout; heavily penalize premature `<eos>` (`-100.0`) and unclosed bars (`-50.0`) so
    bar-rewind candidates cannot truncate pages.
    """
    n = len(candidates)
    if n == 1:
        return candidates[0]

    best_cand = candidates[0]
    best_score = -1e9
    for i, ci in enumerate(candidates):
        is_orig = bool(ci.get("is_orig", False))
        hit_eos = bool(ci.get("hit_eos", False))
        closed_bar = bool(ci.get("closed_bar", True))
        m_err = float(ci["meter_err"])
        v_err = float(ci["voice_err"])

        if hit_eos and not is_orig:
            cons_term = PREMATURE_EOS_PENALTY
        elif (not closed_bar) and (not is_orig):
            cons_term = UNCLOSED_BAR_PENALTY
        elif m_err <= BEAT_TOL and v_err <= BEAT_TOL:
            cons_term = 2.0
        else:
            cons_term = -1.0 * min(4.0, m_err + v_err)

        peer_costs = [
            _asymmetric_bar_edit_cost(ci["toks"], cj["toks"])
            for j, cj in enumerate(candidates)
            if j != i
        ]
        mbr_risk = sum(peer_costs) / max(1, len(peer_costs))
        score = cons_term + 0.5 * float(ci["mean_lp"]) - 1.5 * mbr_risk
        if score > best_score:
            best_score = score
            best_cand = ci
    return best_cand


def decode_page(
    model: Any,
    img_t: "torch.Tensor",
    *,
    mode: str,
    max_len: int,
    use_bf16: bool,
    force_smoke_rewind: bool = False,
) -> Dict[str, Any]:
    """Decode one page with either `greedy_fsa` or violation-only `kv_rewind_mbr`."""
    import torch
    import torch.nn.functional as F
    import eval_uni_suite31
    import grammar

    device = img_t.device
    v = int(model.cfg.vocab_size)
    pad_id, bos_id, eos_id, sys_id, _, id2tok = eval_uni_suite31._vocab_spec(v)

    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            mem, _, _ = model.encode_image(img_t, return_aux=False)
            past_kvs = model.precompute_cross_kv(mem)

        # Feed BOS once to establish initial KV + logits
        cur_id_t = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits, _, past_kvs = model.decode_tokens(
                cur_id_t, memory=None, return_mtp=False, return_aux=False,
                past_kvs=past_kvs, use_cache=True, apply_tok_dropout=False,
            )
        next_logits = logits[0, -1, :].float()

        fsa = grammar.PrefixState(strict=False)
        out_ids: List[int] = [bos_id]
        hit_eos = False

        # Bar-boundary snapshot for KV-cache rewind
        bar_kv_len = _kv_seq_len(past_kvs)
        bar_next_logits = next_logits.clone()
        bar_fsa_snap = fsa.copy()
        bar_out_start = len(out_ids)
        cur_bar_ids: List[int] = []
        cur_bar_entropies: List[float] = []
        cur_bar_lps: List[float] = []
        retries_this_bar = 0
        total_rewinds = 0
        rewinds_accepted = 0
        smoke_triggered = False

        def _rollout_bar_candidate(
            start_kvs: List[Any],
            start_logits: torch.Tensor,
            start_fsa: Any,
            *,
            greedy: bool,
            ban_step_idx: Optional[int] = None,
            ban_tok_id: Optional[int] = None,
            step_budget: int = MAX_BAR_RETRY_TOKENS,
        ) -> Dict[str, Any]:
            c_kvs = _slice_past_kvs(start_kvs, bar_kv_len)
            c_logits = start_logits
            c_fsa = start_fsa.copy()
            c_ids: List[int] = []
            c_toks: List[str] = []
            c_lps: List[float] = []
            c_m_err, c_v_err = 0.0, 0.0
            c_hit_eos = False
            c_closed_bar = False

            for s_idx in range(step_budget):
                ban_id = ban_tok_id if (ban_step_idx is not None and s_idx == ban_step_idx) else None
                # Hard-ban <eos> inside bar rollouts so a bar retry can never truncate the page
                masked = _apply_fsa_mask(
                    c_logits, c_fsa, eos_id, sys_id, pad_id, bos_id,
                    extra_ban_id=ban_id, ban_eos=True,
                )
                if greedy:
                    lp_all = F.log_softmax(masked.float(), dim=-1)
                    tid = int(torch.argmax(masked).item())
                    lp = float(lp_all[tid].item())
                else:
                    tid, lp = _sample_top_p(masked, MBR_TEMP, MBR_TOP_P)

                c_ids.append(tid)
                c_lps.append(lp)
                tok_str = id2tok.get(tid, "")
                c_toks.append(tok_str)

                if tid == eos_id:
                    c_hit_eos = True
                    c_m_err, c_v_err = eval_uni_suite31._measure_violation_errors(c_fsa)
                    break

                # Advance KV cache with this token
                t_in = torch.tensor([[tid]], dtype=torch.long, device=device)
                with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    lg_out, _, c_kvs = model.decode_tokens(
                        t_in, memory=None, return_mtp=False, return_aux=False,
                        past_kvs=c_kvs, use_cache=True, apply_tok_dropout=False,
                    )
                c_logits = lg_out[0, -1, :].float()

                if tid == sys_id:
                    c_m_err, c_v_err = eval_uni_suite31._measure_violation_errors(c_fsa)
                    c_fsa = grammar.PrefixState(strict=False)
                    c_closed_bar = True
                    break
                if tok_str in grammar.BAR_LIKES:
                    c_m_err, c_v_err = eval_uni_suite31._measure_violation_errors(c_fsa)
                    try:
                        c_fsa.step(tok_str)
                    except ValueError:
                        pass
                    c_closed_bar = True
                    break
                try:
                    c_fsa.step(tok_str)
                except ValueError:
                    pass

            if (not c_closed_bar) and (not c_hit_eos):
                c_m_err, c_v_err = eval_uni_suite31._measure_violation_errors(c_fsa)

            return {
                "ids": c_ids,
                "toks": c_toks,
                "mean_lp": sum(c_lps) / max(1, len(c_lps)),
                "meter_err": float(c_m_err),
                "voice_err": float(c_v_err),
                "kvs": c_kvs,
                "next_logits": c_logits,
                "fsa": c_fsa,
                "hit_eos": c_hit_eos,
                "closed_bar": c_closed_bar,
            }

        step = 0
        while step < max_len:
            masked = _apply_fsa_mask(next_logits, fsa, eos_id, sys_id, pad_id, bos_id, ban_eos=False)
            log_probs = F.log_softmax(masked.float(), dim=-1)
            probs = torch.exp(log_probs)
            ent = float(-(probs * log_probs.clamp(min=-1e4)).sum().item())
            tid = int(torch.argmax(masked).item())
            lp = float(log_probs[tid].item())
            tok_str = id2tok.get(tid, "")

            if tid == eos_id:
                out_ids.append(tid)
                hit_eos = True
                break

            # Advance KV cache for chosen token
            t_in = torch.tensor([[tid]], dtype=torch.long, device=device)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                lg_out, _, past_kvs = model.decode_tokens(
                    t_in, memory=None, return_mtp=False, return_aux=False,
                    past_kvs=past_kvs, use_cache=True, apply_tok_dropout=False,
                )
            next_logits = lg_out[0, -1, :].float()

            out_ids.append(tid)
            cur_bar_ids.append(tid)
            cur_bar_entropies.append(ent)
            cur_bar_lps.append(lp)
            step += 1

            is_bar_close = (tok_str in grammar.BAR_LIKES) or (tid == sys_id)
            if force_smoke_rewind and not smoke_triggered and len(cur_bar_ids) >= 4:
                is_bar_close = True

            if is_bar_close:
                m_err, v_err = eval_uni_suite31._measure_violation_errors(fsa)
                # Violation-only trigger (removed noisy peak_ent > 1.60 trigger from i15-44)
                trigger_rewind = (
                    mode == "kv_rewind_mbr"
                    and retries_this_bar < MAX_RETRIES_PER_BAR
                    and total_rewinds < MAX_REWINDS_PER_PAGE
                    and (
                        m_err > BEAT_TOL
                        or v_err > BEAT_TOL
                        or (force_smoke_rewind and not smoke_triggered)
                    )
                )

                if trigger_rewind:
                    smoke_triggered = True
                    total_rewinds += 1
                    retries_this_bar += 1
                    peak_idx = (
                        max(range(len(cur_bar_entropies)), key=lambda idx: cur_bar_entropies[idx])
                        if cur_bar_entropies
                        else 0
                    )
                    peak_tok_id = cur_bar_ids[peak_idx] if cur_bar_ids else None

                    # Build original candidate record
                    orig_fsa_post = fsa.copy()
                    if tid == sys_id:
                        orig_fsa_post = grammar.PrefixState(strict=False)
                    else:
                        try:
                            orig_fsa_post.step(tok_str)
                        except ValueError:
                            pass
                    orig_cand = {
                        "ids": list(cur_bar_ids),
                        "toks": [id2tok.get(i, "") for i in cur_bar_ids],
                        "mean_lp": sum(cur_bar_lps) / max(1, len(cur_bar_lps)),
                        "meter_err": float(m_err),
                        "voice_err": float(v_err),
                        "kvs": past_kvs,
                        "next_logits": next_logits,
                        "fsa": orig_fsa_post,
                        "hit_eos": False,
                        "closed_bar": True,
                        "is_orig": True,
                    }

                    # Generate K=2 candidates from rewound bar-boundary KV cache
                    cands: List[Dict[str, Any]] = [orig_cand]
                    budget = min(MAX_BAR_RETRY_TOKENS, max(4, max_len - bar_out_start + 1))
                    # Candidate 1: greedy with peak-entropy top-1 token masked
                    c1 = _rollout_bar_candidate(
                        past_kvs, bar_next_logits, bar_fsa_snap,
                        greedy=True, ban_step_idx=peak_idx, ban_tok_id=peak_tok_id, step_budget=budget,
                    )
                    c1["is_orig"] = False
                    cands.append(c1)
                    # Candidates 2..K: temperature-sampled FSA continuations
                    for _ in range(MBR_K_CANDIDATES - 1):
                        ck = _rollout_bar_candidate(
                            past_kvs, bar_next_logits, bar_fsa_snap,
                            greedy=False, step_budget=budget,
                        )
                        ck["is_orig"] = False
                        cands.append(ck)

                    winner = _select_mbr_bar_candidate(cands)
                    if not winner["is_orig"]:
                        rewinds_accepted += 1
                    # Commit winning bar's tokens, KV cache, logits, and FSA state
                    out_ids = out_ids[:bar_out_start] + winner["ids"]
                    past_kvs = winner["kvs"]
                    next_logits = winner["next_logits"]
                    fsa = winner["fsa"]
                    step = len(out_ids) - 1
                    if winner["hit_eos"]:
                        hit_eos = True
                        break
                else:
                    if tid == sys_id:
                        fsa = grammar.PrefixState(strict=False)
                    else:
                        try:
                            fsa.step(tok_str)
                        except ValueError:
                            pass

                # Advance bar-boundary snapshot
                bar_kv_len = _kv_seq_len(past_kvs)
                bar_next_logits = next_logits.clone()
                bar_fsa_snap = fsa.copy()
                bar_out_start = len(out_ids)
                cur_bar_ids = []
                cur_bar_entropies = []
                cur_bar_lps = []
                retries_this_bar = 0
            else:
                try:
                    fsa.step(tok_str)
                except ValueError:
                    pass

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    pred_toks = eval_uni_suite31.decode_uni_compat(out_ids, vocab_size=v, skip_special=True)
    return {
        "pred_toks": pred_toks,
        "hit_eos": hit_eos,
        "elapsed_ms": round(elapsed_ms, 2),
        "n_tokens": len(pred_toks),
        "total_rewinds": total_rewinds,
        "rewinds_accepted": rewinds_accepted,
    }


# ---------------------------------------------------------------------------
# Proper suite31_hw MusicXML Note F1 & wFPP Scoring (Bug #1 Fix)
# ---------------------------------------------------------------------------

def _extract_musicxml_notes_headers_toks(
    xml_path: Path,
) -> Tuple[List[Tuple[str, int, str, str, str]], List[str], List[str]]:
    """Pure-Python MusicXML parser returning (ref_notes, ref_headers, approx_bekern_toks).

    Used on platforms without compiled Rust `assembly-serialize`/`eval-harness` binaries
    (e.g. Colab Linux) and as a cross-check alongside the Rust scorer on Windows.
    """
    notes: List[Tuple[str, int, str, str, str]] = []
    headers: List[str] = []
    toks: List[str] = []
    root = ET.parse(str(xml_path)).getroot()

    def _strip(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    for part in root:
        if _strip(part.tag) != "part":
            continue
        for meas in part:
            if _strip(meas.tag) != "measure":
                continue
            for child in meas:
                ctag = _strip(child.tag)
                if ctag == "attributes":
                    attr_clefs: List[str] = []
                    attr_keys: List[str] = []
                    attr_times: List[str] = []
                    for attr in child:
                        atag = _strip(attr.tag)
                        if atag == "clef":
                            sign = line = ""
                            for cc in attr:
                                cct = _strip(cc.tag)
                                txt = (cc.text or "").strip()
                                if cct == "sign":
                                    sign = txt.upper()
                                elif cct == "line":
                                    line = txt
                            if sign and line:
                                attr_clefs.append(f"clef-{sign}-{line}")
                        elif atag == "key":
                            for kc in attr:
                                if _strip(kc.tag) == "fifths":
                                    txt = (kc.text or "").strip()
                                    if txt:
                                        try:
                                            attr_keys.append(f"key{int(txt):+d}")
                                        except ValueError:
                                            pass
                        elif atag == "time":
                            beats = btype = ""
                            for tc in attr:
                                tct = _strip(tc.tag)
                                txt = (tc.text or "").strip()
                                if tct == "beats":
                                    beats = txt
                                elif tct == "beat-type":
                                    btype = txt
                            if beats and btype:
                                attr_times.append(f"time-{beats}-{btype}")
                    canon_hdrs = attr_clefs + attr_keys + attr_times
                    headers.extend(canon_hdrs)
                    toks.extend(canon_hdrs)
                elif ctag == "note":
                    is_rest = is_grace = False
                    step = octave = alter = dur = ""
                    n_dots = 0
                    for nc in child:
                        ntag = _strip(nc.tag)
                        if ntag == "rest":
                            is_rest = True
                        elif ntag == "grace":
                            is_grace = True
                        elif ntag == "pitch":
                            for pc in nc:
                                ptag = _strip(pc.tag)
                                txt = (pc.text or "").strip()
                                if ptag == "step":
                                    step = txt.upper()
                                elif ptag == "octave":
                                    octave = txt
                                elif ptag == "alter":
                                    try:
                                        av = int(round(float(txt)))
                                        if av != 0:
                                            alter = f"a{av}"
                                    except ValueError:
                                        pass
                        elif ntag == "type":
                            dur = (nc.text or "").strip().lower()
                        elif ntag == "dot":
                            n_dots += 1
                    if is_rest:
                        if dur:
                            toks.extend(["rest", dur] + ["dot"] * n_dots)
                    elif (not is_grace) and step and octave and dur:
                        notes.append((dur, n_dots, step, octave, alter))
                        toks.extend([dur, step, octave])
                        if alter:
                            toks.append(alter)
                        toks.extend(["dot"] * n_dots)
            toks.append("bar")
    return notes, headers, toks


def _extract_headers(toks: Sequence[str]) -> List[str]:
    return [t for t in toks if t.startswith(("clef-", "key-", "key", "time-"))]


def score_page_prediction(
    pred_toks: Sequence[str],
    gt_xml_path: Path,
    hit_eos: bool,
    *,
    slug: str = "page",
    workdir: Optional[Path] = None,
    prefer_rust: bool = True,
) -> Dict[str, Any]:
    """Compute real MusicXML Note F1, TokLCS F1, BSVR, and wFPP against a `.musicxml` GT file.

    Uses the exact `eval_uni_suite31` / `eval_sys_gate` Rust scorer (`assembly-serialize` +
    `eval-harness`) whenever compiled binaries are present, and cleanly falls back to
    pure-Python `xml.etree.ElementTree` MusicXML note/header matching on Colab Linux.
    """
    import eval_sys_gate as esg
    import eval_uni_suite31
    import rlvr_grpo

    cleaned_toks = list(pred_toks)
    if not hit_eos and cleaned_toks:
        cleaned_toks = eval_uni_suite31.trim_truncated_tokens(cleaned_toks)
    hyp_page_toks, _ = eval_uni_suite31.stitch_uni_tokens(cleaned_toks)

    use_rust = bool(prefer_rust and esg.ASER.is_file() and esg.EHARNESS.is_file())
    if use_rust:
        tmp_ctx: Optional[tempfile.TemporaryDirectory] = None
        if workdir is None:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="i15_44b_score_")
            wd = Path(tmp_ctx.name)
        else:
            wd = workdir
            wd.mkdir(parents=True, exist_ok=True)
        try:
            ref_page_toks, page_ref_notes = esg.suite31_page_refs(gt_xml_path, f"i1544b-{slug}", wd)
            tok_lcs_f1 = eval_uni_suite31.compute_token_lcs_f1(ref_page_toks, hyp_page_toks)

            hyp_txt = wd / f"i1544b-{slug}-hyp-bekern.txt"
            hyp_txt.write_text(" ".join(hyp_page_toks) + "\n", encoding="utf-8")
            hyp_notes = wd / f"i1544b-{slug}-hyp-notes.json"
            p_dec = esg.run_cli([
                esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes),
            ])
            bekern_ok = (p_dec.returncode == 0) and hyp_notes.exists()

            beat_m = eval_uni_suite31.compute_fsa_beat_metrics(
                hyp_page_toks, hit_eos=hit_eos, bekern_decode_ok=bekern_ok
            )
            n_bars = int(beat_m["n_bars"])
            bsvr = round(1.0 - float(beat_m["beat_conservation_rate"]), 6) if n_bars > 0 else 1.0
            n_bar_violations = int(round(bsvr * n_bars))

            ref_notes_py = rlvr_grpo.extract_pitch_dur_notes(ref_page_toks)
            pred_notes_py = rlvr_grpo.extract_pitch_dur_notes(hyp_page_toks)
            py_note_f1 = rlvr_grpo.compute_sequence_f1(ref_notes_py, pred_notes_py)

            if bekern_ok:
                sc = esg.score_notes_pair(
                    page_ref_notes, hyp_notes, wd, f"i1544b-{slug}", esg.EHARNESS, esg.ASER
                )
                note_f1 = float(sc.get("f1", 0.0))
                proxy_fixes = sc.get("proxy_fixes")
                ref_n_cnt = int(sc.get("ref_len") or len(ref_notes_py))
                hyp_n_cnt = int(sc.get("hyp_len") or len(pred_notes_py))
            else:
                note_f1 = 0.0
                proxy_fixes = None
                ref_n_cnt = len(ref_notes_py)
                hyp_n_cnt = len(pred_notes_py)

            sm_h = difflib.SequenceMatcher(
                None, _extract_headers(ref_page_toks), _extract_headers(hyp_page_toks), autojunk=False
            )
            header_edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm_h.get_opcodes() if tag != "equal")
            if proxy_fixes is not None:
                note_edits = int(proxy_fixes)
            else:
                sm_n = difflib.SequenceMatcher(None, ref_notes_py, pred_notes_py, autojunk=False)
                note_edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm_n.get_opcodes() if tag != "equal")
            wfpp = round(3.0 * header_edits + 2.0 * n_bar_violations + 1.0 * note_edits, 2)

            return {
                "note_f1": round(float(note_f1), 5),
                "py_note_f1": round(float(py_note_f1), 5),
                "tok_lcs_f1": round(float(tok_lcs_f1), 5),
                "bsvr": bsvr,
                "n_bars": n_bars,
                "ref_notes": ref_n_cnt,
                "hyp_notes": hyp_n_cnt,
                "wfpp": wfpp,
                "bekern_decode_ok": bekern_ok,
                "scorer": "eval_uni_suite31_rust",
            }
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    # Pure-Python MusicXML fallback (e.g. Colab Linux without Rust binaries)
    ref_notes, ref_headers, ref_toks = _extract_musicxml_notes_headers_toks(gt_xml_path)
    pred_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_page_toks)
    note_f1 = rlvr_grpo.compute_sequence_f1(ref_notes, pred_notes)
    tok_lcs_f1 = eval_uni_suite31.compute_token_lcs_f1(ref_toks, hyp_page_toks)

    beat_m = eval_uni_suite31.compute_fsa_beat_metrics(
        hyp_page_toks, hit_eos=hit_eos, bekern_decode_ok=True
    )
    n_bars = int(beat_m["n_bars"])
    bsvr = round(1.0 - float(beat_m["beat_conservation_rate"]), 6) if n_bars > 0 else 1.0
    n_bar_violations = int(round(bsvr * n_bars))

    sm_h = difflib.SequenceMatcher(
        None, ref_headers, _extract_headers(hyp_page_toks), autojunk=False
    )
    header_edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm_h.get_opcodes() if tag != "equal")
    sm_n = difflib.SequenceMatcher(None, ref_notes, pred_notes, autojunk=False)
    note_edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm_n.get_opcodes() if tag != "equal")
    wfpp = round(3.0 * header_edits + 2.0 * n_bar_violations + 1.0 * note_edits, 2)

    return {
        "note_f1": round(float(note_f1), 5),
        "py_note_f1": round(float(note_f1), 5),
        "tok_lcs_f1": round(float(tok_lcs_f1), 5),
        "bsvr": bsvr,
        "n_bars": n_bars,
        "ref_notes": len(ref_notes),
        "hyp_notes": len(pred_notes),
        "wfpp": wfpp,
        "bekern_decode_ok": True,
        "scorer": "python_musicxml_lcs",
    }


def _aggregate_page_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n_pages": 0, "note_f1": 0.0, "tok_lcs_f1": 0.0, "bsvr": 1.0, "wfpp": 0.0, "ms_per_page": 0.0}
    n = float(len(rows))
    tot_rew = sum(r["total_rewinds"] for r in rows)
    tot_acc = sum(r["rewinds_accepted"] for r in rows)
    return {
        "n_pages": len(rows),
        "note_f1": round(sum(r["note_f1"] for r in rows) / n, 5),
        "tok_lcs_f1": round(sum(r["tok_lcs_f1"] for r in rows) / n, 5),
        "bsvr": round(sum(r["bsvr"] for r in rows) / n, 5),
        "wfpp": round(sum(r["wfpp"] for r in rows) / n, 2),
        "ms_per_page": round(sum(r["elapsed_ms"] for r in rows) / n, 1),
        "rewinds_per_page": round(tot_rew / n, 2),
        "rewind_accept_rate": round(float(tot_acc) / float(max(1, tot_rew)), 4),
    }


# ---------------------------------------------------------------------------
# CPU Smoke Test
# ---------------------------------------------------------------------------

_MOCK_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Music</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><type>half</type></note>
    </measure>
  </part>
</score-partwise>
"""


def run_smoke_test() -> int:
    """CPU verification covering:
    1. Non-zero Note F1 (exact 1.0) on a mock MusicXML pair (both Python and Rust scorers if present).
    2. Premature `<eos>` rejection in `_select_mbr_bar_candidate` and `_apply_fsa_mask(ban_eos=True)`.
    3. 1-page CPU rewind decoding for `greedy_fsa` vs `kv_rewind_mbr` (`K=2`).
    """
    _locate_mvd()
    import torch
    import grammar
    import uni_omr_v2
    import eval_uni_suite31

    device = torch.device("cpu")
    _seed_everything(SEED)

    # --- Check 1: Proper MusicXML Note F1 & wFPP scoring ---
    mock_pred = ["clef-G-2", "key+0", "time-4-4", "quarter", "C", "4", "quarter", "D", "4", "half", "E", "4", "bar"]
    with tempfile.TemporaryDirectory(prefix="i15_44b_smoke_") as td:
        xml_p = Path(td) / "gt-p1-smoke.musicxml"
        xml_p.write_text(_MOCK_MUSICXML, encoding="utf-8")

        sc_py = score_page_prediction(mock_pred, xml_p, hit_eos=True, slug="smoke-py", prefer_rust=False)
        assert sc_py["note_f1"] == 1.0, f"Expected python MusicXML note_f1==1.0, got {sc_py}"
        assert sc_py["ref_notes"] == 3 and sc_py["hyp_notes"] == 3, f"Expected 3 notes, got {sc_py}"
        assert sc_py["wfpp"] == 0.0, f"Expected wfpp==0.0 on exact match, got {sc_py}"
        print(
            f"[{EXP_CODE}][SMOKE] MusicXML python scorer: note_f1={sc_py['note_f1']} "
            f"tok_lcs_f1={sc_py['tok_lcs_f1']} wfpp={sc_py['wfpp']} ref_notes={sc_py['ref_notes']}",
            flush=True,
        )

        sc_auto = score_page_prediction(mock_pred, xml_p, hit_eos=True, slug="smoke-auto", prefer_rust=True)
        assert sc_auto["note_f1"] > 0.99, f"Expected auto MusicXML note_f1>0.99, got {sc_auto}"
        print(
            f"[{EXP_CODE}][SMOKE] MusicXML auto scorer ({sc_auto['scorer']}): note_f1={sc_auto['note_f1']} "
            f"tok_lcs_f1={sc_auto['tok_lcs_f1']} wfpp={sc_auto['wfpp']} ref_notes={sc_auto['ref_notes']}",
            flush=True,
        )

        # --- Check 2: Premature <eos> hard-ban & MBR rejection ---
        orig_violating = {
            "ids": [1, 2, 3],
            "toks": ["quarter", "C", "4", "bar"],
            "mean_lp": -0.5,
            "meter_err": 1.0,
            "voice_err": 0.0,
            "hit_eos": False,
            "closed_bar": True,
            "is_orig": True,
        }
        premature_eos_cand = {
            "ids": [99],
            "toks": ["<eos>"],
            "mean_lp": -0.01,
            "meter_err": 0.0,
            "voice_err": 0.0,
            "hit_eos": True,
            "closed_bar": False,
            "is_orig": False,
        }
        winner_no_eos = _select_mbr_bar_candidate([orig_violating, premature_eos_cand])
        assert not winner_no_eos["hit_eos"], f"Premature <eos> must be rejected! Got {winner_no_eos}"
        assert winner_no_eos["is_orig"], "Expected orig_violating to beat premature <eos> candidate"

        # Verify _apply_fsa_mask(ban_eos=True) sets eos_id logit to -inf
        v = 237
        pad_id, bos_id, eos_id, sys_id, _, _ = eval_uni_suite31._vocab_spec(v)
        st = grammar.PrefixState(strict=False)
        for tok in ("clef-G-2", "key+0", "time-4-4", "quarter", "C", "4"):
            st.step(tok)
        raw_logits = torch.zeros(v)
        raw_logits[eos_id] = 100.0
        masked_no_eos = _apply_fsa_mask(raw_logits, st, eos_id, sys_id, pad_id, bos_id, ban_eos=True)
        assert int(torch.argmax(masked_no_eos).item()) != eos_id, "ban_eos=True must block eos_id!"
        assert not math.isfinite(float(masked_no_eos[eos_id].item())), "eos_id must be -inf when ban_eos=True"
        print(f"[{EXP_CODE}][SMOKE] Premature <eos> rejection: PASS (MBR + ban_eos mask)", flush=True)

        # --- Check 3: 1-page CPU rewind decode ---
        tiny_cfg = uni_omr_v2.UniOMRv2Config(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            n_axial_layers=0,
            n_global_layers=2,
            n_dec_layers=2,
            ffn_mult=2.0,
            axial_ffn_mult=2.0,
            backbone_channels=(16, 32, 48, 64),
            backbone_blocks=(1, 1, 1, 1),
            backbone_expand=2.0,
            router_hidden=32,
            max_mem_tokens=256,
            mtp_k=4,
            grad_checkpoint=False,
        )
        model = uni_omr_v2.UniOMRv2Net(tiny_cfg).to(device).eval()
        img_t = torch.randn(1, 3, 64, 96, device=device)

        r0 = decode_page(model, img_t, mode="greedy_fsa", max_len=16, use_bf16=False)
        s0 = score_page_prediction(r0["pred_toks"], xml_p, r0["hit_eos"], slug="smoke-r0")

        r1 = decode_page(
            model, img_t, mode="kv_rewind_mbr", max_len=16, use_bf16=False, force_smoke_rewind=True
        )
        s1 = score_page_prediction(r1["pred_toks"], xml_p, r1["hit_eos"], slug="smoke-r1")

        assert r0["total_rewinds"] == 0, f"greedy_fsa should have 0 rewinds, got {r0['total_rewinds']}"
        assert r1["total_rewinds"] >= 1, f"kv_rewind_mbr smoke test must trigger >=1 KV rewind, got {r1['total_rewinds']}"
        assert math.isfinite(s0["note_f1"]) and math.isfinite(s1["note_f1"])
        assert math.isfinite(s0["bsvr"]) and math.isfinite(s1["bsvr"])

        print(
            f"[{EXP_CODE}][SMOKE] greedy_fsa:    n_toks={r0['n_tokens']} rewinds={r0['total_rewinds']} "
            f"ms={r0['elapsed_ms']} note_f1={s0['note_f1']} bsvr={s0['bsvr']} wfpp={s0['wfpp']}",
            flush=True,
        )
        print(
            f"[{EXP_CODE}][SMOKE] kv_rewind_mbr: n_toks={r1['n_tokens']} rewinds={r1['total_rewinds']} "
            f"accepted={r1['rewinds_accepted']} ms={r1['elapsed_ms']} note_f1={s1['note_f1']} bsvr={s1['bsvr']} wfpp={s1['wfpp']}",
            flush=True,
        )
    print(f"[{EXP_CODE}][SMOKE] PASS!", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{EXP_ID} runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run CPU smoke test and exit")
    args = parser.parse_args()
    if args.smoke_test:
        return run_smoke_test()

    t_start = time.perf_counter()
    started_at = _utc_now()
    hf_token: Optional[str] = None
    status = "RUNNING"
    error_msg: Optional[str] = None
    arm_results: Dict[str, Dict[str, Any]] = {}

    try:
        hf_token = _read_hf_token()
        _ensure_deps()
        _locate_mvd()
        _configure_backend_safety()
        _seed_everything(SEED)

        import torch
        from PIL import Image
        import eval_uni_suite31

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        print(
            f"[{EXP_ID}] Device={gpu_name} | bf16={use_bf16} | cudnn={torch.backends.cudnn.enabled} | torch={torch.__version__}",
            flush=True,
        )

        arm_results.update(_try_resume_from_hf(hf_token))

        manifest_path = _ensure_suite31(hf_token)
        _, all_pieces = eval_uni_suite31.resolve_manifest_pieces(manifest_path)
        subset = _select_hard_12_subset(all_pieces)
        print(
            f"[{EXP_ID}] Selected {len(subset)} hard suite31_hw pages: {[p['slug'] for p in subset]}",
            flush=True,
        )

        model, ckpt_source = _ensure_checkpoint(hf_token, device)
        if hasattr(model, "cfg") and hasattr(model.cfg, "max_mem_tokens"):
            model.cfg.max_mem_tokens = max(int(model.cfg.max_mem_tokens), MAX_MEM_TOKENS)
        print(f"[{EXP_ID}] Loaded checkpoint: {ckpt_source}", flush=True)

        score_workdir = WORK_ROOT / "score_tmp"
        score_workdir.mkdir(parents=True, exist_ok=True)

        for arm_name in ARMS:
            arm_dir = WORK_ROOT / "ckpts" / arm_name
            arm_dir.mkdir(parents=True, exist_ok=True)
            cached = arm_dir / "arm_result.json"
            if arm_name in arm_results or cached.is_file():
                if arm_name not in arm_results:
                    arm_results[arm_name] = json.loads(cached.read_text(encoding="utf-8"))
                print(f"[{EXP_ID}] Reusing cached arm '{arm_name}'", flush=True)
                continue

            pages_cache = arm_dir / "pages.json"
            done_rows: Dict[str, Dict[str, Any]] = {}
            if pages_cache.is_file():
                done_rows = json.loads(pages_cache.read_text(encoding="utf-8"))

            for idx, piece in enumerate(subset, start=1):
                slug = piece["slug"]
                if slug in done_rows:
                    continue
                img = Image.open(piece["photo"]).convert("RGB")
                img_t = _pil_to_tensor(img, MAX_DIM).unsqueeze(0).to(device)

                dec = decode_page(model, img_t, mode=arm_name, max_len=MAX_LEN, use_bf16=use_bf16)
                sc = score_page_prediction(
                    dec["pred_toks"],
                    Path(piece["gt"]),
                    dec["hit_eos"],
                    slug=f"{arm_name}-{slug}",
                    workdir=score_workdir,
                )
                row = {
                    "slug": slug,
                    "handwritten": bool(piece["handwritten"]),
                    "elapsed_ms": dec["elapsed_ms"],
                    "n_tokens": dec["n_tokens"],
                    "hit_eos": dec["hit_eos"],
                    "total_rewinds": dec["total_rewinds"],
                    "rewinds_accepted": dec["rewinds_accepted"],
                    **sc,
                }
                done_rows[slug] = row
                _atomic_write_json(pages_cache, done_rows)
                print(
                    f"[{EXP_ID}][{arm_name}] ({idx}/{len(subset)}) {slug}: "
                    f"note_f1={row['note_f1']:.4f} tok_lcs={row['tok_lcs_f1']:.4f} "
                    f"bsvr={row['bsvr']:.4f} wfpp={row['wfpp']:.1f} "
                    f"ms={row['elapsed_ms']:.0f} rew={row['total_rewinds']}({row['rewinds_accepted']}) "
                    f"[{row['scorer']}]",
                    flush=True,
                )

            ordered_rows = [done_rows[p["slug"]] for p in subset if p["slug"] in done_rows]
            pr_rows = [r for r in ordered_rows if not r["handwritten"]]
            hw_rows = [r for r in ordered_rows if r["handwritten"]]
            res: Dict[str, Any] = {
                "arm": arm_name,
                "ckpt_source": ckpt_source,
                **_aggregate_page_metrics(ordered_rows),
                "printed": _aggregate_page_metrics(pr_rows),
                "hw": _aggregate_page_metrics(hw_rows),
                "per_page": ordered_rows,
            }
            _atomic_write_json(cached, res)
            arm_results[arm_name] = res
            _upload_json_to_hf(hf_token, cached, f"{HF_CKPT_PREFIX}/{arm_name}_arm_result.json")

            eval_payload: Dict[str, Any] = {
                "exp_id": EXP_ID,
                "exp_code": EXP_CODE,
                "status": "RUNNING" if len(arm_results) < len(ARMS) else "COMPLETED",
                "started_at": started_at,
                "updated_at": _utc_now(),
                "gpu": gpu_name,
                "bf16": use_bf16,
                "ckpt_source": ckpt_source,
                "seed": SEED,
                "hparams": {
                    "trigger": "violation_only",
                    "k_candidates": MBR_K_CANDIDATES,
                    "max_retries_per_bar": MAX_RETRIES_PER_BAR,
                    "max_rewinds_per_page": MAX_REWINDS_PER_PAGE,
                    "ban_eos_in_rollout": True,
                },
                "arms": arm_results,
            }
            if "greedy_fsa" in arm_results and "kv_rewind_mbr" in arm_results:
                g0 = arm_results["greedy_fsa"]
                g1 = arm_results["kv_rewind_mbr"]
                eval_payload["delta_kv_rewind_minus_greedy"] = {
                    "note_f1": round(g1["note_f1"] - g0["note_f1"], 5),
                    "tok_lcs_f1": round(g1["tok_lcs_f1"] - g0["tok_lcs_f1"], 5),
                    "bsvr": round(g1["bsvr"] - g0["bsvr"], 5),
                    "wfpp": round(g1["wfpp"] - g0["wfpp"], 2),
                    "ms_per_page": round(g1["ms_per_page"] - g0["ms_per_page"], 1),
                    "latency_ratio": round(g1["ms_per_page"] / max(1.0, g0["ms_per_page"]), 3),
                }
            eval_local = WORK_ROOT / "out" / "eval.json"
            _atomic_write_json(eval_local, eval_payload)
            _upload_json_to_hf(hf_token, eval_local, HF_EVAL_PATH)

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        status = "COMPLETED"
    except Exception as exc:
        status = "FAILED"
        error_msg = f"{type(exc).__name__}: {exc}"
        print(f"[{EXP_ID}] ERROR: {error_msg}", flush=True)
        raise
    finally:
        total_sec = time.perf_counter() - t_start
        run_payload = {
            "exp_id": EXP_ID,
            "exp_code": EXP_CODE,
            "status": status,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_sec": round(total_sec, 2),
            "n_arms_completed": len(arm_results),
            "eval_path": HF_EVAL_PATH,
            "error": error_msg,
        }
        run_local = WORK_ROOT / "out" / "run.json"
        _atomic_write_json(run_local, run_payload)
        if hf_token:
            try:
                _upload_json_to_hf(hf_token, run_local, HF_RUN_PATH)
            except Exception as up_exc:
                print(f"[{EXP_ID}] WARN: failed to upload run ledger: {up_exc}", flush=True)
        _purge_hf_token()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

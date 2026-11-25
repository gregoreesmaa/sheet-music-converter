#!/usr/bin/env python3
"""Apples-to-apples Suite31+HW full-page evaluator for UniOMR.

Includes:
  - FSA grammar-constrained greedy decoding (``generate_constrained``)
    reusing ``grammar.PrefixState`` with clean ``<sys>`` state resets.
  - MTP-4 self-speculative decoding (``generate_mtp_spec``) using
    ``UniOMRNet.mtp_heads`` with exact token-for-token parity against
    standard greedy decoding (and optional grammar constraints).
  - Full ``suite31_hw_manifest.json`` (45 pieces: 31 printed + 14 HW)
    evaluation scoring BOTH Token F1/SER and real Note-Level F1/SER
    via ``assembly-serialize.exe`` + ``eval-harness.exe``.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/eval_uni_suite31.py \\
        --ckpt tmp-train/mvd-runs/uni-omr/uni_omr_best.pt \\
        --use-grammar --use-mtp-spec --device cuda \\
        --out tmp-train/mvd-runs/uni-omr/suite31_hw_eval.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402
import eval_sys_gate as esg  # noqa: E402
import grammar  # noqa: E402
import uni_omr  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"


# ---------------------------------------------------------------------------
# Helpers: Backward-Compat Vocab IDs, Grammar Masking, Truncation & KV Slicing
# ---------------------------------------------------------------------------

def _vocab_spec(
    vocab_size: int,
) -> Tuple[int, int, int, int, Dict[str, int], Dict[int, str]]:
    """Derive special IDs and token maps for any append-only UniOMR ``vocab_size``.

    Keeps legacy 182-vocab checkpoints (178 base + PAD/BOS/EOS/SYS) 100%
    evaluable alongside expanded vocabs (e.g. Option D1: 233 base + 4 = 237).
    """
    n_base = int(vocab_size) - 4
    if n_base < 0 or n_base > len(data.VOCAB):
        raise ValueError(
            f"Unsupported vocab_size={vocab_size} (expected 4..{len(data.VOCAB) + 4})"
        )
    pad_id = vocab_size - 4
    bos_id = vocab_size - 3
    eos_id = vocab_size - 2
    sys_id = vocab_size - 1
    base_vocab = data.VOCAB[:n_base]
    tok_to_id = {t: i for i, t in enumerate(base_vocab)}
    id_to_tok = {i: t for i, t in enumerate(base_vocab)}
    id_to_tok[pad_id] = "<pad>"
    id_to_tok[bos_id] = "<bos>"
    id_to_tok[eos_id] = "<eos>"
    id_to_tok[sys_id] = uni_omr.SYS_TOKEN
    return pad_id, bos_id, eos_id, sys_id, tok_to_id, id_to_tok


def decode_uni_compat(
    ids: Sequence[int],
    vocab_size: Optional[int] = None,
    skip_special: bool = True,
) -> List[str]:
    """Decode UniOMR token IDs using ``vocab_size``'s special-ID layout."""
    vsz = int(vocab_size) if vocab_size is not None else uni_omr.VOCAB_SIZE
    pad_id, bos_id, eos_id, _, _, id_to_tok = _vocab_spec(vsz)
    out: List[str] = []
    for idx in ids:
        i = int(idx)
        if skip_special and i in (pad_id, bos_id, eos_id):
            if i == eos_id:
                break
            continue
        tok = id_to_tok.get(i)
        if tok is not None:
            out.append(tok)
    return out


def trim_truncated_tokens(toks: Sequence[str]) -> List[str]:
    """Trim trailing incomplete note/chord/rest/voice tokens on max_len truncation.

    Walks ``toks`` with ``grammar.PrefixState(strict=False)`` (resetting on
    ``<sys>``) and cuts back to the last complete bar or completed event
    boundary so ``bekern-decode`` does not crash on partial trailing tokens
    (e.g. ``quarter C`` -> ``missing step in measure 0``).
    """
    st = grammar.PrefixState(strict=False)
    last_complete = 0
    for i, tok in enumerate(toks):
        if tok == uni_omr.SYS_TOKEN:
            st = grammar.PrefixState(strict=False)
            continue
        try:
            st.step(tok)
        except ValueError:
            break
        _, eos_ok = st.allowed()
        if not eos_ok:
            continue
        if (
            tok in grammar.BAR_LIKES
            or tok == grammar.BAR_REST
            or tok == grammar.TIE_STOP
            or tok == grammar.TUPLET_STOP
            or st.mode in ("pitch-tail", "tiestart", "rest-tail")
        ):
            last_complete = i + 1
    if last_complete > 0:
        return list(toks[:last_complete])
    return []


def _allowed_ids_for_state(state: grammar.PrefixState, vocab_size: int) -> List[int]:
    """Return legal token IDs for ``state`` in the UniOMR vocabulary.

    Allows ``sys_id`` and ``eos_id`` whenever ``PrefixState`` permits
    end-of-system (``eos_ok`` or immediately after a barline). To avoid
    degenerate empty systems, ``sys_id`` is only offered once at least
    one token has been emitted in the current system.
    """
    pad_id, bos_id, eos_id, sys_id, tok_to_id, _ = _vocab_spec(vocab_size)
    if state.broken:
        # Fail open over all non-PAD/non-BOS tokens.
        return [
            i for i in range(vocab_size)
            if i not in (pad_id, bos_id)
        ]

    allow_set, eos_ok = state.allowed()
    allow_ids = {i for i in allow_set if 0 <= i < pad_id}

    # Suppress tight-loop cycle extension in BOTH strict=False and strict=True
    # modes whenever alternatives exist.
    cyc_tok = state.cycle_forbidden()
    if cyc_tok is not None:
        cyc_id = tok_to_id.get(cyc_tok)
        if cyc_id is not None and len(allow_ids) > 1:
            allow_ids.discard(cyc_id)

    after_bar = bool(state.recent) and (state.recent[-1] in grammar.BAR_LIKES)
    can_end_sys = bool(eos_ok or after_bar)
    if can_end_sys:
        allow_ids.add(eos_id)
        if state.recent:
            allow_ids.add(sys_id)

    # Never emit PAD or BOS as generated tokens.
    allow_ids.discard(pad_id)
    allow_ids.discard(bos_id)
    return sorted(i for i in allow_ids if 0 <= i < vocab_size)


def _select_constrained_token(
    step_logits_1d: torch.Tensor,
    state: grammar.PrefixState,
) -> int:
    """Pick argmax token under ``state``'s grammar mask; safe fallback if empty/NaN."""
    vocab_size = int(step_logits_1d.shape[-1])
    allowed = _allowed_ids_for_state(state, vocab_size)
    if not allowed:
        return int(step_logits_1d.argmax(dim=-1).item())

    idx_t = torch.tensor(allowed, dtype=torch.long, device=step_logits_1d.device)
    sub_logits = step_logits_1d.index_select(0, idx_t)
    if not bool(torch.isfinite(sub_logits).any()):
        return int(step_logits_1d.argmax(dim=-1).item())

    best_pos = int(sub_logits.argmax(dim=-1).item())
    return int(allowed[best_pos])


def _advance_grammar_state(
    state: grammar.PrefixState,
    tok_id: int,
    strict: bool,
    vocab_size: Optional[int] = None,
) -> grammar.PrefixState:
    """Advance ``PrefixState`` by ``tok_id``; reset cleanly on ``sys_id``."""
    vsz = int(vocab_size) if vocab_size is not None else (len(state.vocab_map) + 4)
    pad_id, bos_id, eos_id, sys_id, tok_to_id, id_to_tok = _vocab_spec(vsz)
    if tok_id in (pad_id, bos_id, eos_id):
        return state
    if tok_id == sys_id:
        return grammar.PrefixState(strict=strict, vocab_map=tok_to_id)
    tok_str = id_to_tok.get(tok_id)
    if tok_str is None:
        state.broken = True
        return state
    try:
        state.step(tok_str)
    except ValueError:
        state.broken = True
    return state


def _truncate_past_kvs(
    past_kvs: List[Any],
    keep_len: int,
) -> List[Any]:
    """Slice decoder self-attention KV cache to ``keep_len`` (cross-attn untouched)."""
    trimmed: List[Any] = []
    for self_kv, cross_kv in past_kvs:
        k, v = self_kv
        trimmed.append(
            (
                (k[:, :, :keep_len, :].contiguous(), v[:, :, :keep_len, :].contiguous()),
                cross_kv,
            )
        )
    return trimmed


# ---------------------------------------------------------------------------
# 1. FSA Grammar-Constrained Greedy Decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_constrained(
    model: uni_omr.UniOMRNet,
    images: torch.Tensor,
    max_len: int = 1024,
    strict: bool = False,
    keep_ratio_override: Optional[float] = None,
    return_info: bool = False,
) -> Any:
    """KV-cached greedy decoding with per-item ``grammar.PrefixState`` masking.

    Resets ``PrefixState`` cleanly whenever ``SYS_ID`` (``<sys>``) is emitted,
    and falls back safely to unconstrained argmax if a mask ever empties.
    """
    mem, _, router_info = model.encode_image(
        images,
        keep_ratio_override=keep_ratio_override,
        return_aux=False,
    )
    B = int(images.shape[0])
    device = images.device
    vocab_size = int(model.cfg.vocab_size)
    pad_id, bos_id, eos_id, _, tok_to_id, _ = _vocab_spec(vocab_size)

    cur_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    past_kvs: Optional[List[Any]] = None
    finished = [False] * B
    states = [grammar.PrefixState(strict=strict, vocab_map=tok_to_id) for _ in range(B)]
    out_tokens: List[torch.Tensor] = [cur_ids]

    for _ in range(max_len):
        logits, _, past_kvs = model.decode_tokens(
            cur_ids,
            memory=mem if past_kvs is None else None,
            return_mtp=False,
            past_kvs=past_kvs,
            use_cache=True,
        )
        step_logits = logits[:, -1, :]  # (B, V)
        next_ids: List[int] = []
        for b in range(B):
            if finished[b]:
                next_ids.append(pad_id)
                continue
            tok_id = _select_constrained_token(step_logits[b], states[b])
            next_ids.append(tok_id)
            if tok_id == eos_id:
                finished[b] = True
            else:
                states[b] = _advance_grammar_state(
                    states[b], tok_id, strict=strict, vocab_size=vocab_size
                )

        next_tok = torch.tensor(next_ids, dtype=torch.long, device=device)
        out_tokens.append(next_tok.unsqueeze(1))
        if all(finished):
            break
        cur_ids = next_tok.unsqueeze(1)

    seq = torch.cat(out_tokens, dim=1)
    if return_info:
        return seq, router_info
    return seq


# ---------------------------------------------------------------------------
# 2. MTP-4 Self-Speculative Decoding (Exact Greedy Parity)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_mtp_spec(
    model: uni_omr.UniOMRNet,
    images: torch.Tensor,
    max_len: int = 1024,
    use_grammar: bool = False,
    strict_grammar: bool = False,
    keep_ratio_override: Optional[float] = None,
    return_stats: bool = True,
) -> Any:
    """MTP-k self-speculative KV-cached decoding using ``model.mtp_heads``.

    Drafts ``M = len(model.mtp_heads)`` lookahead tokens in parallel, verifies
    them in a single parallel KV-cached decoder forward pass, accepts the
    longest matching prefix (truncating KV cache cleanly to the accepted
    length), and guarantees exact token-for-token parity with standard greedy
    ``model.generate()`` (or ``generate_constrained()`` when ``use_grammar=True``).

    Returns:
        ``(seq_ids, stats)`` when ``return_stats=True``, else ``seq_ids``.
    """
    t0 = time.perf_counter()
    mem, _, router_info = model.encode_image(
        images,
        keep_ratio_override=keep_ratio_override,
        return_aux=False,
    )
    B = int(images.shape[0])
    device = images.device
    n_mtp = len(model.mtp_heads)
    vocab_size = int(model.cfg.vocab_size)
    pad_id, bos_id, eos_id, _, tok_to_id, _ = _vocab_spec(vocab_size)

    bos = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    if max_len <= 0:
        stats = {
            "mtp_k": 1 + n_mtp,
            "draft_proposed": 0,
            "draft_accepted": 0,
            "acceptance_rate": 0.0,
            "verify_passes": 0,
            "tokens_generated": 0,
            "mean_tokens_per_pass": 0.0,
            "elapsed_secs": max(1e-6, time.perf_counter() - t0),
            "tokens_per_sec": 0.0,
            "router_info": router_info,
        }
        return (bos, stats) if return_stats else bos

    finished = [False] * B
    states = (
        [grammar.PrefixState(strict=strict_grammar, vocab_map=tok_to_id) for _ in range(B)]
        if use_grammar
        else None
    )
    out_tokens: List[torch.Tensor] = [bos]

    def _pick_step(step_logits: torch.Tensor) -> torch.Tensor:
        if not use_grammar:
            tok = step_logits.argmax(dim=-1)
            for b in range(B):
                if finished[b]:
                    tok[b] = pad_id
            return tok
        assert states is not None
        picked: List[int] = []
        for b in range(B):
            if finished[b]:
                picked.append(pad_id)
            else:
                picked.append(_select_constrained_token(step_logits[b], states[b]))
        return torch.tensor(picked, dtype=torch.long, device=device)

    def _commit_token(tok: torch.Tensor) -> None:
        out_tokens.append(tok.unsqueeze(1))
        for b in range(B):
            if finished[b]:
                continue
            tid = int(tok[b].item())
            if tid == eos_id:
                finished[b] = True
            elif use_grammar:
                assert states is not None
                states[b] = _advance_grammar_state(
                    states[b], tid, strict=strict_grammar, vocab_size=vocab_size
                )

    def _extract_drafts(mtp_logits_list: List[torch.Tensor], pos_idx: int) -> List[torch.Tensor]:
        drafts_out: List[torch.Tensor] = []
        for m_logits in mtp_logits_list:
            d = m_logits[:, pos_idx, :].argmax(dim=-1)
            for b in range(B):
                if finished[b]:
                    d[b] = pad_id
            drafts_out.append(d)
        return drafts_out

    # Step 0 (prefill BOS): get first verified token + initial M drafts.
    logits0, mtp0, past_kvs = model.decode_tokens(
        bos,
        memory=mem,
        return_mtp=(n_mtp > 0),
        past_kvs=None,
        use_cache=True,
    )
    cur_tok = _pick_step(logits0[:, -1, :])
    _commit_token(cur_tok)
    gen_len = 1
    drafts = _extract_drafts(mtp0, -1) if n_mtp > 0 else []

    draft_proposed = 0
    draft_accepted = 0
    verify_passes = 1  # counts prefill pass

    # Invariant at top of loop:
    #   - out_tokens has [BOS, tok_1, ..., tok_{gen_len}]
    #   - past_kvs has length gen_len (covers [BOS, tok_1, ..., tok_{gen_len-1}])
    #   - cur_tok == tok_{gen_len} (verified & committed, not yet in past_kvs)
    #   - drafts has M speculative tokens for tok_{gen_len+1} .. tok_{gen_len+M}
    while gen_len < max_len and not all(finished):
        rem = max_len - gen_len
        m = min(len(drafts), rem - 1) if n_mtp > 0 else 0
        past_len = int(past_kvs[0][0][0].shape[2])

        if m > 0:
            verify_ids = torch.stack([cur_tok] + drafts[:m], dim=1)  # (B, 1 + m)
        else:
            verify_ids = cur_tok.unsqueeze(1)  # (B, 1)

        v_logits, v_mtp, new_past_kvs = model.decode_tokens(
            verify_ids,
            memory=None,
            return_mtp=(n_mtp > 0),
            past_kvs=past_kvs,
            use_cache=True,
        )
        verify_passes += 1
        draft_proposed += m

        n_acc = 0
        mismatched = False
        for j in range(m):
            target_j = _pick_step(v_logits[:, j, :])
            draft_j = drafts[j]
            # For already-finished items, force draft_j == pad_id so they match target_j.
            if any(finished):
                draft_j = draft_j.clone()
                for b in range(B):
                    if finished[b]:
                        draft_j[b] = pad_id

            if bool((target_j == draft_j).all()):
                _commit_token(draft_j)
                gen_len += 1
                n_acc += 1
                draft_accepted += 1
                if all(finished):
                    past_kvs = _truncate_past_kvs(new_past_kvs, past_len + 1 + n_acc)
                    mismatched = True
                    break
            else:
                # First mismatch at draft j: commit corrected target_j and truncate KV cache.
                _commit_token(target_j)
                gen_len += 1
                cur_tok = target_j
                drafts = _extract_drafts(v_mtp, j) if n_mtp > 0 else []
                past_kvs = _truncate_past_kvs(new_past_kvs, past_len + 1 + n_acc)
                mismatched = True
                break

        if not mismatched and not all(finished):
            # All m drafts matched: emit the free bonus token from position m!
            bonus_tok = _pick_step(v_logits[:, m, :])
            _commit_token(bonus_tok)
            gen_len += 1
            cur_tok = bonus_tok
            drafts = _extract_drafts(v_mtp, m) if n_mtp > 0 else []
            past_kvs = new_past_kvs

    elapsed = max(1e-6, time.perf_counter() - t0)
    seq = torch.cat(out_tokens, dim=1)
    # Count non-PAD generated tokens (excluding BOS).
    total_gen_toks = int((seq[:, 1:] != pad_id).sum().item())

    stats = {
        "mtp_k": 1 + n_mtp,
        "draft_proposed": int(draft_proposed),
        "draft_accepted": int(draft_accepted),
        "acceptance_rate": round(draft_accepted / max(1, draft_proposed), 6) if draft_proposed > 0 else 0.0,
        "verify_passes": int(verify_passes),
        "tokens_generated": total_gen_toks,
        "mean_tokens_per_pass": round(total_gen_toks / max(1, verify_passes), 3),
        "elapsed_secs": round(elapsed, 6),
        "tokens_per_sec": round(total_gen_toks / elapsed, 2),
        "router_info": router_info,
    }
    if return_stats:
        return seq, stats
    return seq


# ---------------------------------------------------------------------------
# 3. Manifest Resolution, Stitching, and Checkpoint Loading
# ---------------------------------------------------------------------------

def resolve_manifest_pieces(manifest_path: str | Path) -> Tuple[Path, List[Dict[str, Any]]]:
    """Load ``suite31_hw_manifest.json`` and resolve ``clean``/``photo``/``gt`` paths."""
    mpath = Path(manifest_path)
    base_dir = mpath.parent
    raw = json.loads(mpath.read_text(encoding="utf-8"))
    resolved: List[Dict[str, Any]] = []

    def _entry_path(entry: Any) -> Path:
        if isinstance(entry, dict):
            return base_dir / entry["file"]
        return base_dir / str(entry)

    for p in raw["pieces"]:
        resolved.append({
            "slug": str(p["slug"]),
            "handwritten": bool(p.get("handwritten", False)),
            "stratum": esg.classify_suite31_stratum(p),
            "clean": _entry_path(p["clean"]),
            "photo": _entry_path(p["photo"]),
            "gt": _entry_path(p["gt"]),
        })
    return base_dir, resolved


def stitch_uni_tokens(pred_toks: Sequence[str]) -> Tuple[List[str], int]:
    """Split UniOMR full-page tokens on ``<sys>`` and stitch into page bekern.

    Uses ``eval_sys_gate.stitch_page_toks`` so header deduplication and
    canonical tail ordering match the 2-stage ``suite31_hw`` evaluator.
    Returns ``(stitched_tokens, n_systems)``.
    """
    chunks: List[List[str]] = []
    cur: List[str] = []
    for t in pred_toks:
        if t == uni_omr.SYS_TOKEN:
            if cur:
                chunks.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        chunks.append(cur)

    if not chunks:
        return [], 0
    if len(chunks) == 1:
        return esg.canonicalize_bekern_tail_order(chunks[0]), 1
    return esg.stitch_page_toks(chunks), len(chunks)


def load_uni_checkpoint(
    ckpt_path: str | Path,
    device: torch.device,
) -> Tuple[uni_omr.UniOMRNet, uni_omr.UniOMRConfig, Dict[str, Any]]:
    """Load a ``UniOMRNet`` checkpoint onto ``device`` in eval mode."""
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    cfg_dict = ck.get("config", {}) if isinstance(ck, dict) else {}
    valid_fields = {f.name for f in fields(uni_omr.UniOMRConfig)}
    filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    cfg = uni_omr.UniOMRConfig(**filtered_cfg)
    model = uni_omr.UniOMRNet(cfg)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    inc = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    meta = {
        "step": ck.get("step") if isinstance(ck, dict) else None,
        "val_ser": ck.get("val_ser") if isinstance(ck, dict) else None,
        "missing_keys": list(inc.missing_keys),
        "unexpected_keys": list(inc.unexpected_keys),
    }
    return model, cfg, meta


# ---------------------------------------------------------------------------
# 4. Full Suite31+HW Evaluation Loop & Reporting
# ---------------------------------------------------------------------------

def _summarize_slice(rows: Sequence[Dict[str, Any]], domain: str) -> Dict[str, Any]:
    """Aggregate Token & Note metrics for one domain ('clean' or 'photo')."""
    def _mean(vals: List[Optional[float]]) -> Optional[float]:
        xs = [float(v) for v in vals if v is not None]
        return round(sum(xs) / len(xs), 6) if xs else None

    def _med(vals: List[Optional[float]]) -> Optional[float]:
        xs = [float(v) for v in vals if v is not None]
        return round(float(statistics.median(xs)), 2) if xs else None

    ok = [r for r in rows if not r.get("skipped")]
    p = domain + "_"
    n = len(ok)
    dec_ok = sum(1 for r in ok if r.get(p + "decode_ok"))
    draft_prop = sum(int(r.get(p + "draft_proposed", 0) or 0) for r in ok)
    draft_acc = sum(int(r.get(p + "draft_accepted", 0) or 0) for r in ok)

    return {
        "n_pieces": len(rows),
        "n_scored": n,
        "tok_f1_mean": _mean([r.get(p + "tok_f1") for r in ok]),
        "tok_ser_mean": _mean([r.get(p + "tok_ser") for r in ok]),
        "note_f1_mean": _mean([r.get(p + "note_f1") for r in ok]),
        "note_ser_mean": _mean([r.get(p + "note_ser") for r in ok]),
        "fixes_median": _med([r.get(p + "proxy_fixes") for r in ok]),
        "decode_ok_rate": round(dec_ok / max(1, n), 4) if n else None,
        "tok_per_sec_mean": _mean([r.get(p + "tok_per_sec") for r in ok]),
        "mtp_acceptance_rate": round(draft_acc / draft_prop, 4) if draft_prop > 0 else None,
    }


def evaluate_suite31(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    model, cfg, ckpt_meta = load_uni_checkpoint(args.ckpt, device=device)
    _, pieces = resolve_manifest_pieces(args.manifest)

    if args.pieces:
        wanted = {s.strip() for s in args.pieces.split(",") if s.strip()}
        pieces = [p for p in pieces if p["slug"] in wanted]
    if args.limit_pieces is not None:
        pieces = pieces[: int(args.limit_pieces)]

    use_bf16 = bool(getattr(args, "bf16", False) and device.type == "cuda")
    mode_tag = (
        ("mtp-spec" if args.use_mtp_spec else "greedy")
        + ("+grammar" if args.use_grammar else "")
        + ("(strict)" if (args.use_grammar and args.strict_grammar) else "")
    )
    print(
        f"[UniOMR suite31_hw] ckpt={args.ckpt} (step={ckpt_meta['step']}) | "
        f"device={device} | mode={mode_tag} | max_dim={args.max_dim} | "
        f"max_len={args.max_len} | pieces={len(pieces)}",
        flush=True,
    )

    workdir = Path(tempfile.mkdtemp(prefix="uni_s31_"))
    piece_rows: List[Dict[str, Any]] = []

    for idx, piece in enumerate(pieces, start=1):
        slug = piece["slug"]
        is_hw = piece["handwritten"]
        tag = "HW" if is_hw else "PR"
        prow: Dict[str, Any] = {
            "piece": slug,
            "handwritten": is_hw,
            "stratum": piece["stratum"],
        }
        try:
            ref_page_toks, page_ref_notes = esg.suite31_page_refs(
                piece["gt"], "uni-" + slug, workdir
            )
            prow["ref_tokens"] = len(ref_page_toks)
        except Exception as e:  # noqa: BLE001
            prow["skipped"] = True
            prow["reason"] = f"ref-error: {str(e)[:160]}"
            piece_rows.append(prow)
            print(f"[{idx:02d}/{len(pieces):02d}] {slug:34s} [{tag}] SKIP ref-error: {e}", flush=True)
            continue

        for domain in ("clean", "photo"):
            img_path: Path = piece[domain]
            with Image.open(img_path) as raw_im:
                pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), args.max_dim)
            img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            mtp_stats: Optional[Dict[str, Any]] = None
            with torch.no_grad():
                ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16
                    else _nullcontext()
                )
                with ctx:
                    if args.use_mtp_spec:
                        seq, mtp_stats = generate_mtp_spec(
                            model,
                            img_t,
                            max_len=args.max_len,
                            use_grammar=args.use_grammar,
                            strict_grammar=args.strict_grammar,
                            return_stats=True,
                        )
                    elif args.use_grammar:
                        seq = generate_constrained(
                            model,
                            img_t,
                            max_len=args.max_len,
                            strict=args.strict_grammar,
                        )
                    elif int(cfg.vocab_size) != uni_omr.VOCAB_SIZE:
                        seq = generate_mtp_spec(
                            model,
                            img_t,
                            max_len=args.max_len,
                            use_grammar=False,
                            return_stats=False,
                        )
                    else:
                        seq = model.generate(img_t, max_len=args.max_len)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = max(1e-6, time.perf_counter() - t0)

            seq_ids = seq[0].detach().cpu().tolist()
            _, _, eos_id, _, _, _ = _vocab_spec(int(cfg.vocab_size))
            pred_raw_toks = decode_uni_compat(
                seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
            )
            if eos_id not in seq_ids[1:]:
                pred_raw_toks = trim_truncated_tokens(pred_raw_toks)
            hyp_page_toks, n_sys = stitch_uni_tokens(pred_raw_toks)

            # 1. Token-level F1 & SER
            tok_edits = uni_omr.levenshtein_distance(ref_page_toks, hyp_page_toks)
            tok_ser = tok_edits / max(1, len(ref_page_toks))
            _, _, tok_f1 = uni_omr.compute_token_f1(ref_page_toks, hyp_page_toks)

            # 2. Note-level F1 & SER via assembly-serialize + eval-harness
            hyp_txt = workdir / f"uni-{slug}-{domain}-hyp-bekern.txt"
            hyp_txt.write_text(" ".join(hyp_page_toks) + "\n", encoding="utf-8")
            hyp_notes = workdir / f"uni-{slug}-{domain}-hyp-notes.json"
            p_dec = esg.run_cli([
                esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes),
            ])
            if p_dec.returncode != 0 or not hyp_notes.exists():
                note_f1 = 0.0
                note_ser = 1.0
                proxy_fixes = None
                decode_ok = False
                zero_reason = f"bekern-decode-rc-{p_dec.returncode}"
            else:
                sc = esg.score_notes_pair(
                    page_ref_notes,
                    hyp_notes,
                    workdir,
                    f"uni-{slug}-{domain}",
                    esg.EHARNESS,
                    esg.ASER,
                )
                note_f1 = float(sc.get("f1", 0.0))
                note_ser = float(sc.get("ser", 1.0))
                proxy_fixes = sc.get("proxy_fixes")
                decode_ok = not bool(sc.get("zeroed"))
                zero_reason = sc.get("zero_reason")

            n_gen = max(1, len(pred_raw_toks) + 1)
            tok_per_sec = n_gen / elapsed

            pfx = domain + "_"
            prow[pfx + "tok_f1"] = round(tok_f1, 6)
            prow[pfx + "tok_ser"] = round(tok_ser, 6)
            prow[pfx + "note_f1"] = round(note_f1, 6)
            prow[pfx + "note_ser"] = round(note_ser, 6)
            prow[pfx + "proxy_fixes"] = proxy_fixes
            prow[pfx + "decode_ok"] = decode_ok
            prow[pfx + "zero_reason"] = zero_reason
            prow[pfx + "n_sys"] = n_sys
            prow[pfx + "n_pred_toks"] = len(hyp_page_toks)
            prow[pfx + "elapsed_secs"] = round(elapsed, 4)
            prow[pfx + "tok_per_sec"] = round(tok_per_sec, 2)
            if mtp_stats is not None:
                prow[pfx + "draft_proposed"] = mtp_stats["draft_proposed"]
                prow[pfx + "draft_accepted"] = mtp_stats["draft_accepted"]
                prow[pfx + "mtp_acceptance_rate"] = mtp_stats["acceptance_rate"]

            mtp_str = (
                f" mtp_acc={mtp_stats['acceptance_rate']:.2f}"
                if mtp_stats is not None
                else ""
            )
            dec_str = "OK" if decode_ok else f"FAIL({zero_reason})"
            print(
                f"[{idx:02d}/{len(pieces):02d}] {slug:34s} [{tag}|{domain:5s}] "
                f"tokF1={tok_f1:.3f} tokSER={tok_ser:.3f} | "
                f"noteF1={note_f1:.3f} noteSER={note_ser:.3f} ({dec_str}) | "
                f"sys={n_sys} tok={len(hyp_page_toks):4d} {tok_per_sec:5.1f}tok/s{mtp_str}",
                flush=True,
            )

        piece_rows.append(prow)

    printed_rows = [r for r in piece_rows if not r["handwritten"]]
    hw_rows = [r for r in piece_rows if r["handwritten"]]

    summary = {
        "printed_clean": _summarize_slice(printed_rows, "clean"),
        "printed_photo": _summarize_slice(printed_rows, "photo"),
        "hw_clean": _summarize_slice(hw_rows, "clean"),
        "hw_photo": _summarize_slice(hw_rows, "photo"),
        "all45_clean": _summarize_slice(piece_rows, "clean"),
        "all45_photo": _summarize_slice(piece_rows, "photo"),
    }

    _print_summary_table(summary, mode_tag=mode_tag)

    report = {
        "ckpt": str(args.ckpt),
        "ckpt_meta": ckpt_meta,
        "model_config": asdict(cfg),
        "eval_args": {
            "manifest": str(args.manifest),
            "max_dim": int(args.max_dim),
            "max_len": int(args.max_len),
            "use_grammar": bool(args.use_grammar),
            "strict_grammar": bool(args.strict_grammar),
            "use_mtp_spec": bool(args.use_mtp_spec),
            "device": str(device),
        },
        "summary": summary,
        "pieces": piece_rows,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote JSON report -> {out_path}", flush=True)
    return report


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return False


def _fmt_f(val: Optional[float], width: int = 6, prec: int = 4) -> str:
    if val is None:
        return f"{'n/a':>{width}s}"
    return f"{val:{width}.{prec}f}"


def _print_summary_table(summary: Dict[str, Dict[str, Any]], mode_tag: str) -> None:
    print("=" * 94, flush=True)
    print(f"UniOMR Suite31+HW Summary ({mode_tag})", flush=True)
    print("=" * 94, flush=True)
    print(
        f"{'Slice':15s} | {'N':>3s} | {'Tok F1':>7s} | {'Tok SER':>7s} | "
        f"{'Note F1':>7s} | {'Note SER':>8s} | {'DecOK':>6s} | {'Tok/s':>6s} | {'MTP Acc':>7s}",
        flush=True,
    )
    print("-" * 94, flush=True)
    for key in (
        "printed_clean",
        "printed_photo",
        "hw_clean",
        "hw_photo",
        "all45_clean",
        "all45_photo",
    ):
        s = summary[key]
        dec_pct = f"{s['decode_ok_rate'] * 100.0:5.1f}%" if s["decode_ok_rate"] is not None else "   n/a"
        print(
            f"{key:15s} | {s['n_scored']:3d} | {_fmt_f(s['tok_f1_mean'], 7, 4)} | "
            f"{_fmt_f(s['tok_ser_mean'], 7, 4)} | {_fmt_f(s['note_f1_mean'], 7, 4)} | "
            f"{_fmt_f(s['note_ser_mean'], 8, 4)} | {dec_pct:>6s} | "
            f"{_fmt_f(s['tok_per_sec_mean'], 6, 1)} | {_fmt_f(s['mtp_acceptance_rate'], 7, 3)}",
            flush=True,
        )
    print("=" * 94, flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate UniOMR on Suite31+HW (45 full pages)")
    ap.add_argument("--ckpt", required=True, help="Path to UniOMR .pt checkpoint")
    ap.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to suite31_hw_manifest.json",
    )
    ap.add_argument("--max-dim", type=int, default=1024, help="Full-page resize max dimension")
    ap.add_argument("--max-len", type=int, default=1024, help="Max autoregressive decode tokens")
    ap.add_argument("--use-grammar", action="store_true", help="Enable FSA grammar-constrained decoding")
    ap.add_argument("--strict-grammar", action="store_true", help="Enable strict bar-duration grammar rules")
    ap.add_argument("--use-mtp-spec", action="store_true", help="Enable MTP-4 self-speculative decoding")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
        help="Inference device (cuda or cpu)",
    )
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast on CUDA")
    ap.add_argument("--out", default=None, help="Optional output JSON report path")
    ap.add_argument("--limit-pieces", type=int, default=None, help="Cap number of pieces (smoke test)")
    ap.add_argument("--pieces", default=None, help="Comma-separated slugs to evaluate")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    evaluate_suite31(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

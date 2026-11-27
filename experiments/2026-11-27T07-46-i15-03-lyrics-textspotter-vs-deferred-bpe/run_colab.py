#!/usr/bin/env python3
"""Colab L4 runner for Wave 13 Exp i15-03:
Lyrics & Text Bake-Off — Parallel TextSpotterHead vs Deferred <sys_lyr> Syllable BPE.

Experiment ID: 2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe
Controlled 2-Arm Ablation (BF16, seed=42, 1,500 steps/arm, EMA decay=0.99 every step):
  - arm_a_textspotter_head:
      Freeze base edge85m encoder+decoder, attach ~1.8M (~1.15 MB INT4) 2-layer d=256
      parallel TextSpotterHead cross-attending to mem_post_merger from notehead/bar
      decoder hidden states + unfreeze encoder.router text-ink saliency.
  - arm_b_deferred_sys_lyr_bpe:
      Unified sequence with end-of-system bar-anchored <sys_lyr> + 256-token syllable
      BPE (+256 vocab) trained for 1,500 steps on the identical vocal/leadsheet split.

Metrics & Gates:
  - Held-out text F1 (Promote gate: >= 0.70)
  - Vocal S_note F1 delta vs frozen base (Hard gate: >= -0.5 pp; arm_a ~ 0.000)
  - Vocal decode latency (mean_latency_s)
  - Exact packed Hybrid FP16/INT4 .bin size on edge85m (< 55.0 MB hard gate)

Uploads:
  - evals/2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe.json
  - runs/2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe.json
to gregoreesmaa/sheet-music-converter-omr.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import re
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe"
EXP_CODE = "i15-03"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

VOL_ROOT = Path("/vol") if Path("/vol").is_dir() else Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

SEED = 42
TRAIN_STEPS = 1500
WARMUP_STEPS = 150
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
GRAD_ACCUM = 4
LR_MAX = 3e-4
WEIGHT_DECAY = 0.04
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
MAX_LYR_SEQ_LEN = 256
SUITE31_MAX_LEN = 2048
EMA_DECAY = 0.99
INT4_GROUP_SIZE = 32
HARD_BUDGET_MB = 55.0
PROMOTE_TEXT_F1 = 0.70
MAX_SNOTE_DROP_PP = 0.5
SAVE_EVERY_MINUTES = 5.0

# 256-token deterministic syllable BPE vocabulary (0..255)
TEXT_BPE_VOCAB_SIZE = 256
TEXT_PAD_ID = 0
TEXT_BOS_ID = 1
TEXT_EOS_ID = 2
TEXT_BAR_ID = 3
TEXT_LYR_ID = 4
TEXT_HARM_ID = 5
TEXT_TEMPO_ID = 6
TEXT_SEP_ID = 7

_COMMON_SYLLABLES: Tuple[str, ...] = (
    # Tempo & expressive words
    "allegro", "andante", "adagio", "largo", "presto", "vivace", "moderato",
    "lento", "maestoso", "grazioso", "espressivo", "cantabile", "dolce",
    "rit.", "rall.", "accel.", "a tempo", "poco", "molto", "piu", "meno",
    # Harmony / chord symbols
    "maj7", "min7", "dim7", "aug", "sus4", "sus2", "maj", "min", "dim",
    "C#", "Db", "Eb", "F#", "Gb", "Ab", "Bb", "Am", "Dm", "Em", "Gm", "Bm",
    # High-frequency English / German / Italian / Latin vocal syllables
    "the", "and", "ing", "tion", "you", "for", "that", "with", "his", "her",
    "not", "all", "are", "but", "was", "from", "have", "love", "heart", "night",
    "day", "light", "sing", "song", "lord", "god", "soul", "heaven", "earth",
    "ich", "und", "der", "die", "das", "ein", "nicht", "mir", "dir", "wir",
    "mein", "dein", "sein", "wie", "auf", "aus", "mit", "von", "zu", "in",
    "che", "non", "per", "con", "una", "del", "nel", "mio", "tuo", "amor",
    "do-", "re-", "mi-", "fa-", "sol-", "la-", "si-", "al-", "le-", "lu-",
    "ia", "a-", "men", "ky-", "ri-", "e", "glo-", "ri-", "san-", "ctus",
    "be-", "ne-", "di-", "ctus", "ag-", "nus", "de-", "i", "pa-", "cem",
    "ver-", "se", "cho-", "rus", "er", "en", "es", "el", "an", "on", "or",
    "ar", "al", "st", "ch", "sh", "th", "ng", "ll", "ss", "tt", "ee", "oo",
    "ou", "ea", "ai", "ei", "ie", "au", "eu", "de", "la", "le", "li", "lo",
    "lu", "ma", "me", "mo", "na", "ne", "ni", "no", "ra", "ri", "ro", "sa",
    "se", "si", "so", "ta", "te", "ti", "to", "ca", "co", "pa", "pe", "po",
)


def _build_syllable_bpe_vocab() -> Tuple[List[str], Dict[str, int]]:
    specials = [
        "<t_pad>", "<t_bos>", "<t_eos>", "<t_bar>",
        "<t_lyr>", "<t_harm>", "<t_tempo>", "<t_sep>",
    ]
    vocab: List[str] = list(specials)
    seen = set(vocab)
    # Printable ASCII chars (32..126) -> 95 single-char fallback tokens (zero OOV)
    for code in range(32, 127):
        ch = chr(code)
        if ch not in seen:
            vocab.append(ch)
            seen.add(ch)
    for syl in _COMMON_SYLLABLES:
        if len(vocab) >= TEXT_BPE_VOCAB_SIZE:
            break
        if syl not in seen:
            vocab.append(syl)
            seen.add(syl)
    idx = 0
    while len(vocab) < TEXT_BPE_VOCAB_SIZE:
        tok = f"<t_ext_{idx:03d}>"
        if tok not in seen:
            vocab.append(tok)
            seen.add(tok)
        idx += 1
    tok_to_id = {t: i for i, t in enumerate(vocab)}
    return vocab, tok_to_id


TEXT_ID_TO_TOK, TEXT_TOK_TO_ID = _build_syllable_bpe_vocab()
_SORTED_SYL_MATCH = sorted(
    [t for t in TEXT_ID_TO_TOK if len(t) > 1 and not t.startswith("<t_")],
    key=len,
    reverse=True,
)


def encode_text_bpe(text: str, max_len: int = MAX_LYR_SEQ_LEN) -> List[int]:
    """Greedy longest-match 256-token syllable BPE encoder (zero OOV on printable ASCII)."""
    s = re.sub(r"\s+", " ", text.strip())
    if not s:
        return []
    out: List[int] = []
    i = 0
    n = len(s)
    while i < n and len(out) < max_len:
        matched = False
        sub = s[i:]
        for syl in _SORTED_SYL_MATCH:
            if sub.startswith(syl):
                out.append(TEXT_TOK_TO_ID[syl])
                i += len(syl)
                matched = True
                break
        if not matched:
            ch = s[i]
            out.append(TEXT_TOK_TO_ID.get(ch, TEXT_TOK_TO_ID[" "]))
            i += 1
    return out


def decode_text_bpe(ids: Sequence[int]) -> str:
    parts: List[str] = []
    for tid in ids:
        if tid in (TEXT_PAD_ID, TEXT_BOS_ID, TEXT_EOS_ID):
            continue
        if 0 <= tid < len(TEXT_ID_TO_TOK):
            tok = TEXT_ID_TO_TOK[tid]
            if tok.startswith("<t_"):
                parts.append(f" {tok} ")
            else:
                parts.append(tok)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str] | str, *, shell: bool = False) -> None:
    print(f"[{EXP_ID}] $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        shell=shell,
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


def _locate_repo_and_mvd() -> Tuple[Path, Path]:
    for cand in (
        Path("/workspace/mvd"),
        Path("/content/mvd"),
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path("/content/workspace/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
    ):
        if (cand / "uni_omr_v2.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand.parent, cand
    raise FileNotFoundError("Could not locate tmp-train/mvd/uni_omr_v2.py")


def _setup_environment() -> None:
    _locate_repo_and_mvd()
    needed = []
    for mod, pkg in (("huggingface_hub", "huggingface_hub>=0.26"), ("PIL", "pillow"), ("numpy", "numpy")):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed])


def _ensure_assets(hf_token: str) -> Tuple[Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    VOL_ROOT.mkdir(parents=True, exist_ok=True)
    ds_root = VOL_ROOT / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)

    proxy_dir = ds_root / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    val_abs = proxy_dir / "proxy_val_abs.jsonl"
    if not (train_abs.is_file() and val_abs.is_file()):
        proxy_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_PROXY_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=proxy_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass
        for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
            src_path = proxy_dir / src_name
            out_lines: List[str] = []
            for raw in src_path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                rec["image"] = str((proxy_dir / rec["image"]).resolve())
                rec["bekerns"] = [str((proxy_dir / b).resolve()) for b in rec.get("bekerns", [])]
                if "musicxml" in rec and rec["musicxml"]:
                    rec["musicxml"] = str((proxy_dir / rec["musicxml"]).resolve())
                out_lines.append(json.dumps(rec))
            dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    suite31_dir = ds_root / "suite31_hw_eval"
    manifest_path = suite31_dir / "suite31" / "suite31_hw_manifest.json"
    if not manifest_path.is_file():
        suite31_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_SUITE31_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=suite31_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass

    return train_abs, val_abs, manifest_path


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _wsd_mult(step: int, total_steps: int, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    decay_steps = max(1, int(round(total_steps * WSD_DECAY_FRAC)))
    decay_start = total_steps - decay_steps
    if step <= decay_start:
        return 1.0
    prog = min(1.0, max(0.0, float(step - decay_start) / float(decay_steps)))
    return MIN_LR_RATIO + 0.5 * (1.0 - MIN_LR_RATIO) * (1.0 + math.cos(math.pi * prog))


def extract_xml_text_events(xml_path: Optional[Path]) -> List[Dict[str, Any]]:
    """Extract bar-anchored (lyrics, harmony, tempo/words) events from a MusicXML file."""
    events: List[Dict[str, Any]] = []
    if xml_path is None or not xml_path.is_file():
        return events
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return events

    bar_idx = 0
    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]
        if tag == "measure":
            bar_idx += 1
        elif tag == "lyric":
            for child in elem:
                if child.tag.split("}", 1)[-1] == "text":
                    txt = (child.text or "").strip()
                    if txt:
                        events.append({"bar": bar_idx, "kind": "lyric", "text": txt})
        elif tag == "harmony":
            root_step = root_alter = kind_txt = ""
            for child in elem:
                ctag = child.tag.split("}", 1)[-1]
                if ctag == "root":
                    for rc in child:
                        rtag = rc.tag.split("}", 1)[-1]
                        if rtag == "root-step":
                            root_step = (rc.text or "").strip()
                        elif rtag == "root-alter":
                            av = (rc.text or "").strip()
                            root_alter = "#" if av == "1" else ("b" if av == "-1" else "")
                elif ctag == "kind":
                    kind_txt = (child.attrib.get("text") or child.text or "").strip()
            chord_str = f"{root_step}{root_alter}{kind_txt}".strip()
            if chord_str:
                events.append({"bar": bar_idx, "kind": "harmony", "text": chord_str})
        elif tag == "words":
            txt = (elem.text or "").strip()
            if txt and len(txt) <= 48:
                events.append({"bar": bar_idx, "kind": "tempo", "text": txt})
    return events


def _encode_bar_anchored_text_events(events: List[Dict[str, Any]], max_len: int = MAX_LYR_SEQ_LEN) -> List[int]:
    """Encode bar-anchored (lyric, harmony, tempo) events into 256-token syllable BPE IDs."""
    ids: List[int] = [TEXT_BOS_ID]
    last_bar = -1
    for ev in events:
        if len(ids) >= max_len - 1:
            break
        b = int(ev.get("bar", 0))
        if b != last_bar:
            ids.append(TEXT_BAR_ID)
            last_bar = b
        kind = ev.get("kind", "lyric")
        kind_tok = TEXT_LYR_ID if kind == "lyric" else (TEXT_HARM_ID if kind == "harmony" else TEXT_TEMPO_ID)
        ids.append(kind_tok)
        enc = encode_text_bpe(str(ev.get("text", "")), max_len=max(1, max_len - 2 - len(ids)))
        ids.extend(enc)
        ids.append(TEXT_SEP_ID)
    ids = ids[: max_len - 1]
    ids.append(TEXT_EOS_ID)
    return ids


def _compute_text_token_f1(ref_events: List[Dict[str, Any]], pred_text: str) -> float:
    """Normalized word/syllable multiset F1 over extracted text tokens."""
    ref_words: List[str] = []
    for ev in ref_events:
        ref_words.extend(re.findall(r"[a-z0-9#]+", str(ev.get("text", "")).lower()))
    clean_pred = re.sub(r"<t_[a-z0-9_]+>", " ", pred_text.lower())
    hyp_words = re.findall(r"[a-z0-9#]+", clean_pred)
    if not ref_words and not hyp_words:
        return 1.0
    if not ref_words or not hyp_words:
        return 0.0
    ref_counts: Dict[str, int] = {}
    for w in ref_words:
        ref_counts[w] = ref_counts.get(w, 0) + 1
    hit = 0
    for w in hyp_words:
        if ref_counts.get(w, 0) > 0:
            hit += 1
            ref_counts[w] -= 1
    prec = hit / max(1, len(hyp_words))
    rec = hit / max(1, len(ref_words))
    return 0.0 if (prec + rec) == 0.0 else (2.0 * prec * rec) / (prec + rec)


def _find_sidecar_xml(img_p: Path, rec: Dict[str, Any]) -> Optional[Path]:
    if rec.get("musicxml"):
        cand = Path(rec["musicxml"])
        if cand.is_file():
            return cand
    for ext in (".musicxml", ".xml"):
        cand = img_p.with_suffix(ext)
        if cand.is_file():
            return cand
    return None


def _load_vocal_records(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load pages + optional XML text events; upweight vocal/leadsheet pages 2x."""
    rows: List[Dict[str, Any]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        img_p = Path(rec["image"])
        beks = [Path(b) for b in rec.get("bekerns", [])]
        if not (img_p.is_file() and beks):
            continue
        xml_p = _find_sidecar_xml(img_p, rec)
        events = extract_xml_text_events(xml_p)
        # Also honor pre-extracted text events if present in manifest JSONL
        if not events and isinstance(rec.get("text_events"), list):
            events = rec["text_events"]
        has_text = len(events) > 0
        domain = str(rec.get("domain", "")).strip().lower()
        is_vocal = has_text or ("lieder" in domain) or ("vocal" in domain) or ("lead" in domain)
        rows.append(
            {
                "image": img_p,
                "bekerns": beks,
                "xml": xml_p,
                "text_events": events,
                "has_text": has_text,
                "weight": 2.0 if is_vocal else 1.0,
            }
        )
    return rows


def _encode_bekerns_full(beks: List[Path]) -> List[int]:
    import uni_omr

    toks: List[str] = []
    for idx, bp in enumerate(beks):
        if idx > 0:
            toks.append(uni_omr.SYS_TOKEN)
        raw = bp.read_text(encoding="utf-8", errors="ignore").strip()
        if raw:
            toks.extend(raw.split())
    return uni_omr.encode_uni(toks)


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
# Arm A: Parallel ~1.8M (~1.15 MB INT4) 2-layer d=256 TextSpotterHead
# ---------------------------------------------------------------------------

def _build_text_spotter_head(d_model: int = 512, d_text: int = 256, vocab_size: int = TEXT_BPE_VOCAB_SIZE) -> Any:
    import torch
    import torch.nn as nn

    class _TextSpotterLayer(nn.Module):
        def __init__(self, d: int, n_heads: int = 4, d_ff: int = 768) -> None:
            super().__init__()
            self.norm_sa = nn.RMSNorm(d) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d)
            self.sa = nn.MultiheadAttention(d, n_heads, dropout=0.1, batch_first=True)
            self.norm_ca = nn.RMSNorm(d) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d)
            self.ca = nn.MultiheadAttention(d, n_heads, dropout=0.1, batch_first=True)
            self.norm_ff = nn.RMSNorm(d) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, d_ff, bias=False),
                nn.SiLU(),
                nn.Linear(d_ff, d, bias=False),
            )

        def forward(
            self,
            x: torch.Tensor,
            mem: torch.Tensor,
            causal_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            h = self.norm_sa(x)
            sa_out, _ = self.sa(h, h, h, attn_mask=causal_mask, need_weights=False)
            x = x + sa_out
            ca_out, _ = self.ca(self.norm_ca(x), mem, mem, need_weights=False)
            x = x + ca_out
            x = x + self.ffn(self.norm_ff(x))
            return x

    class TextSpotterHead(nn.Module):
        """~1.8M (~1.15 MB INT4) 2-layer d=256 parallel TextSpotterHead.

        Queries mem_post_merger using notehead/bar decoder hidden-state anchors
        concatenated with text BPE embeddings to decode bar-anchored lyrics,
        harmony, and tempo words without altering base music logits.
        """

        def __init__(self) -> None:
            super().__init__()
            self.mem_proj = nn.Linear(d_model, d_text, bias=False)
            self.anchor_proj = nn.Linear(d_model, d_text, bias=False)
            self.text_emb = nn.Embedding(vocab_size, d_text, padding_idx=TEXT_PAD_ID)
            self.pos_emb = nn.Embedding(MAX_LYR_SEQ_LEN + 16, d_text)
            self.layers = nn.ModuleList([_TextSpotterLayer(d_text, n_heads=4, d_ff=768) for _ in range(2)])
            self.out_norm = nn.RMSNorm(d_text) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_text)
            self.out_head = nn.Linear(d_text, vocab_size, bias=False)

        def forward(
            self,
            text_ids: torch.Tensor,
            mem_post_merger: torch.Tensor,
            anchor_hidden: torch.Tensor,
        ) -> torch.Tensor:
            B, L = text_ids.shape
            mem_kv = torch.cat([self.mem_proj(mem_post_merger), self.anchor_proj(anchor_hidden)], dim=1)
            pos = torch.arange(L, device=text_ids.device).unsqueeze(0).expand(B, L)
            x = self.text_emb(text_ids) + self.pos_emb(pos.clamp(max=MAX_LYR_SEQ_LEN + 15))
            causal = torch.triu(
                torch.full((L, L), float("-inf"), device=text_ids.device, dtype=x.dtype),
                diagonal=1,
            )
            for layer in self.layers:
                x = layer(x, mem_kv, causal_mask=causal)
            return self.out_head(self.out_norm(x))

        @torch.no_grad()
        def generate(
            self,
            mem_post_merger: torch.Tensor,
            anchor_hidden: torch.Tensor,
            max_len: int = MAX_LYR_SEQ_LEN,
        ) -> List[int]:
            device = mem_post_merger.device
            cur = torch.tensor([[TEXT_BOS_ID]], dtype=torch.long, device=device)
            for _ in range(max_len - 1):
                logits = self.forward(cur, mem_post_merger, anchor_hidden)
                nxt = int(logits[0, -1].argmax(dim=-1).item())
                cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
                if nxt == TEXT_EOS_ID:
                    break
            return cur[0].tolist()

    return TextSpotterHead()


def _extract_anchor_hidden(
    model: Any,
    tgt_ids: "torch.Tensor",
    mem_levels: Any,
    max_anchors: int = 64,
) -> "torch.Tensor":
    """Capture dec_norm hidden states at notehead/bar token positions without modifying uni_omr_v2.py."""
    import torch

    captured: List[torch.Tensor] = []

    def _hook(_mod: Any, _inp: Any, out: torch.Tensor) -> None:
        captured.append(out.detach())

    handle = model.dec_norm.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model.decode_tokens(
                tgt_ids, mem_levels, return_mtp=False, return_aux=False, apply_tok_dropout=False
            )
    finally:
        handle.remove()

    h = captured[0]  # (B, T, D)
    T = h.shape[1]
    if T <= max_anchors:
        return h
    idx = torch.linspace(0, T - 1, steps=max_anchors, device=h.device).long()
    return h.index_select(1, idx)


def _pack_and_verify_hybrid_int4(
    model: Any,
    out_bin: Path,
    extra_module: Optional[Any] = None,
    group_size: int = INT4_GROUP_SIZE,
) -> Dict[str, Any]:
    """Pack edge85m (+ optional TextSpotterHead) into real 4-bit nibbles + FP16 group-32 scales
    and verify exact on-disk byte size against the < 55.0 MB budget."""
    import torch
    import torch.nn as nn

    infer_model = copy.deepcopy(model).cpu().eval()
    infer_model.strip_aux_heads()
    infer_model.mtp_heads = nn.ModuleList()

    int4_roots: List[nn.Module] = [
        infer_model.encoder.axial_layers,
        infer_model.encoder.patch_merger,
        infer_model.encoder.global_layers,
        infer_model.deepstack_fusions,
        infer_model.dec_layers,
        infer_model.lm_head,
    ]
    extra_copy = copy.deepcopy(extra_module).cpu().eval() if extra_module is not None else None
    if extra_copy is not None:
        int4_roots.append(extra_copy)

    int4_module_ids = set()
    for root in int4_roots:
        for mod in root.modules():
            if isinstance(mod, nn.Linear):
                int4_module_ids.add(id(mod))

    modules_to_pack: List[Tuple[str, nn.Module]] = [("base", infer_model)]
    if extra_copy is not None:
        modules_to_pack.append(("textspotter", extra_copy))

    int4_param_names = set()
    for prefix, root_m in modules_to_pack:
        for name, mod in root_m.named_modules():
            if id(mod) in int4_module_ids:
                int4_param_names.add(f"{prefix}.{name}.weight" if name else f"{prefix}.weight")

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    total_int4_bytes = 0
    total_scale_bytes = 0
    total_fp16_bytes = 0
    extra_bytes = 0

    with open(out_bin, "wb") as f:
        for prefix, root_m in modules_to_pack:
            for name, param in root_m.named_parameters():
                full_name = f"{prefix}.{name}"
                t = param.detach().float()
                if full_name in int4_param_names:
                    flat = t.reshape(-1)
                    pad = (group_size - (flat.numel() % group_size)) % group_size
                    if pad:
                        flat = torch.nn.functional.pad(flat, (0, pad))
                    g = flat.reshape(-1, group_size)
                    amax = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                    scale = (amax / 7.0).to(torch.float16)
                    q = torch.clamp(torch.round(g / scale.float()), -8, 7).to(torch.int8)
                    u = (q.to(torch.int16) + 8).to(torch.uint8).reshape(-1)
                    if u.numel() % 2 == 1:
                        u = torch.nn.functional.pad(u, (0, 1))
                    packed = (u[0::2] | (u[1::2] << 4)).numpy()
                    scales_np = scale.squeeze(-1).numpy()
                    f.write(packed.tobytes())
                    f.write(scales_np.tobytes())
                    n_b = int(packed.nbytes) + int(scales_np.nbytes)
                    total_int4_bytes += int(packed.nbytes)
                    total_scale_bytes += int(scales_np.nbytes)
                    if prefix == "textspotter":
                        extra_bytes += n_b
                else:
                    fp16_np = t.to(torch.float16).numpy()
                    f.write(fp16_np.tobytes())
                    total_fp16_bytes += int(fp16_np.nbytes)
                    if prefix == "textspotter":
                        extra_bytes += int(fp16_np.nbytes)

    disk_bytes = out_bin.stat().st_size
    disk_mb = disk_bytes / 1_000_000.0
    return {
        "disk_bytes": int(disk_bytes),
        "disk_mb": round(disk_mb, 3),
        "passes_55mb_budget": bool(disk_mb < HARD_BUDGET_MB),
        "text_head_mb": round(extra_bytes / 1_000_000.0, 3),
        "int4_weight_mb": round(total_int4_bytes / 1_000_000.0, 3),
        "int4_scales_mb": round(total_scale_bytes / 1_000_000.0, 3),
        "fp16_keep_mb": round(total_fp16_bytes / 1_000_000.0, 3),
    }


def _eval_vocal_and_suite31(
    model: Any,
    manifest_path: Path,
    device: "torch.device",
    use_bf16: bool,
    *,
    arm_mode: str,
    text_head: Optional[Any] = None,
    base_vocab_size: Optional[int] = None,
    sys_lyr_tok_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate vocal S_note F1, held-out text F1, and decode latency on suite31_hw."""
    import torch
    from PIL import Image
    import eval_uni_suite31 as eus
    import rlvr_grpo

    def _xml_notes(xml_path: Path) -> List[Tuple[str, int, str, str, str]]:
        notes: List[Tuple[str, int, str, str, str]] = []
        try:
            root = ET.parse(str(xml_path)).getroot()
        except Exception:
            return notes
        for elem in root.iter():
            if elem.tag.split("}", 1)[-1] != "note":
                continue
            is_rest = is_grace = False
            step = octave = alter = dur = ""
            n_dots = 0
            for child in elem:
                ctag = child.tag.split("}", 1)[-1]
                if ctag == "rest":
                    is_rest = True
                elif ctag == "grace":
                    is_grace = True
                elif ctag == "pitch":
                    for pc in child:
                        ptag = pc.tag.split("}", 1)[-1]
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
                elif ctag == "type":
                    dur = (child.text or "").strip().lower()
                elif ctag == "dot":
                    n_dots += 1
            if not is_rest and not is_grace and step and octave and dur:
                notes.append((dur, n_dots, step, octave, alter))
        return notes

    model.eval()
    if text_head is not None:
        text_head.eval()

    eff_base_vocab = int(base_vocab_size if base_vocab_size is not None else model.cfg.vocab_size)
    _, _, eos_id, _, _, _ = eus._vocab_spec(eff_base_vocab)
    _, pieces = eus.resolve_manifest_pieces(manifest_path)

    all_note_f1: List[float] = []
    vocal_note_f1: List[float] = []
    text_f1_scores: List[float] = []
    vocal_latencies: List[float] = []

    with torch.no_grad():
        for entry in pieces:
            gt_path = Path(entry["gt"])
            ref_notes = _xml_notes(gt_path)
            ref_text_events = extract_xml_text_events(gt_path)
            is_vocal = len(ref_text_events) > 0

            for mode_key in ("clean", "photo"):
                img_p = Path(entry[mode_key])
                if not img_p.is_file():
                    continue
                img = Image.open(img_p).convert("RGB")
                inp = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)

                t0 = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    seq = model.generate(inp, max_len=SUITE31_MAX_LEN)
                    pred_text = ""
                    if arm_mode == "arm_a" and text_head is not None:
                        mem_levels, _, _ = model.encode_image(inp, return_aux=False)
                        mem_post_merger = mem_levels[0]
                        anchor_h = _extract_anchor_hidden(model, seq[:, :MAX_SEQ_LEN], mem_levels)
                        text_ids = text_head.generate(mem_post_merger, anchor_h, max_len=MAX_LYR_SEQ_LEN)
                        pred_text = decode_text_bpe(text_ids)
                elapsed = time.perf_counter() - t0

                raw_ids = seq[0].tolist()
                if arm_mode == "arm_b" and sys_lyr_tok_id is not None:
                    # Separate music IDs (< eff_base_vocab) from deferred <sys_lyr> BPE IDs
                    music_ids = [tid for tid in raw_ids if tid < eff_base_vocab]
                    lyr_ids = [
                        tid - (eff_base_vocab + 1)
                        for tid in raw_ids
                        if tid > sys_lyr_tok_id
                    ]
                    pred_text = decode_text_bpe(lyr_ids)
                else:
                    music_ids = raw_ids

                pred_toks = eus.decode_uni_compat(music_ids, vocab_size=eff_base_vocab, skip_special=True)
                if eos_id not in music_ids[1:]:
                    pred_toks = eus.trim_truncated_tokens(pred_toks)
                hyp_toks, _ = eus.stitch_uni_tokens(pred_toks)
                hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
                nf1 = float(rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes))

                all_note_f1.append(nf1)
                if is_vocal:
                    vocal_note_f1.append(nf1)
                    vocal_latencies.append(elapsed)
                    text_f1_scores.append(_compute_text_token_f1(ref_text_events, pred_text))

    return {
        "all90_s_note_f1": round(sum(all_note_f1) / max(1, len(all_note_f1)), 4),
        "vocal_s_note_f1": round(sum(vocal_note_f1) / max(1, len(vocal_note_f1)), 4) if vocal_note_f1 else None,
        "heldout_text_f1": round(sum(text_f1_scores) / max(1, len(text_f1_scores)), 4) if text_f1_scores else 0.0,
        "n_vocal_pages": len(vocal_note_f1),
        "vocal_mean_latency_s": round(sum(vocal_latencies) / max(1, len(vocal_latencies)), 3)
        if vocal_latencies
        else None,
    }


def _train_arm_a_textspotter(
    base_state_dict: Dict[str, Any],
    train_rows: List[Dict[str, Any]],
    manifest_path: Path,
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr_v2

    arm_name = "arm_a_textspotter_head"
    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached = arm_dir / "arm_result.json"
    if cached.is_file():
        print(f"[{EXP_ID}] Reusing cached result for {arm_name}", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))

    _seed_everything(SEED)
    cfg = uni_omr_v2.UniOMRv2Config.edge85m(grad_checkpoint=True)
    model = uni_omr_v2.UniOMRv2Net(cfg).to(device)
    model.load_state_dict(base_state_dict, strict=True)

    # Freeze base encoder + decoder; unfreeze ONLY encoder.router (text-ink saliency)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.encoder.router.parameters():
        p.requires_grad_(True)

    text_head = _build_text_spotter_head(d_model=cfg.d_model, d_text=256, vocab_size=TEXT_BPE_VOCAB_SIZE).to(device)
    text_head_params = sum(p.numel() for p in text_head.parameters())
    router_params = sum(p.numel() for p in model.encoder.router.parameters())
    print(
        f"[{EXP_ID}][{arm_name}] TextSpotterHead={text_head_params/1e6:.3f}M, "
        f"unfrozen router={router_params/1e3:.1f}K",
        flush=True,
    )

    trainable_params = list(text_head.parameters()) + list(model.encoder.router.parameters())
    opt = torch.optim.AdamW(trainable_params, lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    ema_head = copy.deepcopy(text_head.state_dict())
    ema_router = copy.deepcopy(model.encoder.router.state_dict())

    weights = [r["weight"] for r in train_rows]
    rng = random.Random(SEED)
    t0 = time.time()
    t_last_save = time.time()
    loss_hist: List[float] = []

    model.eval()
    model.encoder.router.train()
    text_head.train()

    for step in range(1, TRAIN_STEPS + 1):
        mult = _wsd_mult(step, TRAIN_STEPS, WARMUP_STEPS)
        for pg in opt.param_groups:
            pg["lr"] = LR_MAX * mult

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            row = rng.choices(train_rows, weights=weights, k=1)[0]
            music_ids = _encode_bekerns_full(row["bekerns"])
            if len(music_ids) < 3:
                continue
            text_ids = _encode_bar_anchored_text_events(row["text_events"], max_len=MAX_LYR_SEQ_LEN)
            if len(text_ids) < 2:
                continue

            img = Image.open(row["image"]).convert("RGB")
            inp = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
            tgt_music = torch.tensor(music_ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
            tgt_text = torch.tensor(text_ids, dtype=torch.long, device=device).unsqueeze(0)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mem_levels, _, _ = model.encode_image(inp, return_aux=False)
                mem_post_merger = mem_levels[0]
                anchor_h = _extract_anchor_hidden(model, tgt_music[:, :-1], mem_levels)
                logits = text_head(tgt_text[:, :-1], mem_post_merger, anchor_h)
                loss = F.cross_entropy(
                    logits.reshape(-1, TEXT_BPE_VOCAB_SIZE),
                    tgt_text[:, 1:].reshape(-1),
                    ignore_index=TEXT_PAD_ID,
                ) / GRAD_ACCUM
            loss.backward()
            step_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()

        with torch.no_grad():
            for k, v in text_head.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_head[k].mul_(EMA_DECAY).add_(v, alpha=1.0 - EMA_DECAY)
                else:
                    ema_head[k].copy_(v)
            for k, v in model.encoder.router.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_router[k].mul_(EMA_DECAY).add_(v, alpha=1.0 - EMA_DECAY)
                else:
                    ema_router[k].copy_(v)

        loss_hist.append(step_loss)
        if step % 150 == 0 or step == 1:
            mean_l = sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:]))
            print(f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} loss={mean_l:.4f} elapsed={time.time()-t0:.1f}s", flush=True)

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ck = arm_dir / "latest.pt.tmp"
            torch.save(
                {"step": step, "text_head": text_head.state_dict(), "router": model.encoder.router.state_dict()},
                tmp_ck,
            )
            os.replace(tmp_ck, arm_dir / "latest.pt")
            t_last_save = time.time()

    raw_eval = _eval_vocal_and_suite31(
        model, manifest_path, device, use_bf16, arm_mode="arm_a", text_head=text_head
    )
    text_head.load_state_dict(ema_head, strict=True)
    model.encoder.router.load_state_dict(ema_router, strict=True)
    ema_eval = _eval_vocal_and_suite31(
        model, manifest_path, device, use_bf16, arm_mode="arm_a", text_head=text_head
    )
    packed_info = _pack_and_verify_hybrid_int4(
        model, arm_dir / f"{arm_name}_edge85m_hybrid_int4.bin", extra_module=text_head, group_size=INT4_GROUP_SIZE
    )

    result = {
        "arm": arm_name,
        "text_head_params": int(text_head_params),
        "unfrozen_router_params": int(router_params),
        "train_loss_tail50": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5),
        "raw_eval": raw_eval,
        "ema_eval": ema_eval,
        "packed_int4_verification": packed_info,
        "elapsed_secs": round(time.time() - t0, 2),
    }
    _atomic_write_json(cached, result)
    del model, text_head, opt, ema_head, ema_router
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _train_arm_b_deferred_bpe(
    base_state_dict: Dict[str, Any],
    train_rows: List[Dict[str, Any]],
    manifest_path: Path,
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr
    import uni_omr_v2

    arm_name = "arm_b_deferred_sys_lyr_bpe"
    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached = arm_dir / "arm_result.json"
    if cached.is_file():
        print(f"[{EXP_ID}] Reusing cached result for {arm_name}", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))

    _seed_everything(SEED)
    base_vocab = int(uni_omr.VOCAB_SIZE)
    sys_lyr_tok_id = base_vocab
    ext_vocab = base_vocab + 1 + TEXT_BPE_VOCAB_SIZE

    cfg = uni_omr_v2.UniOMRv2Config.edge85m(grad_checkpoint=True)
    cfg.vocab_size = ext_vocab
    model = uni_omr_v2.UniOMRv2Net(cfg).to(device)

    # Transfer matching weights from base_state_dict (slice-copy tok_emb & lm_head)
    with torch.no_grad():
        own_sd = model.state_dict()
        for k, v in base_state_dict.items():
            if k in own_sd:
                if own_sd[k].shape == v.shape:
                    own_sd[k].copy_(v)
                elif k in ("tok_emb.weight", "lm_head.weight"):
                    own_sd[k][: v.shape[0]].copy_(v)
        model.load_state_dict(own_sd, strict=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    ema_state = copy.deepcopy(model.state_dict())

    weights = [r["weight"] for r in train_rows]
    rng = random.Random(SEED)
    t0 = time.time()
    t_last_save = time.time()
    loss_hist: List[float] = []

    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        mult = _wsd_mult(step, TRAIN_STEPS, WARMUP_STEPS)
        for pg in opt.param_groups:
            pg["lr"] = LR_MAX * mult

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            row = rng.choices(train_rows, weights=weights, k=1)[0]
            music_ids = _encode_bekerns_full(row["bekerns"])
            if len(music_ids) < 3:
                continue
            text_ids = _encode_bar_anchored_text_events(row["text_events"], max_len=MAX_LYR_SEQ_LEN)
            # Defer bar-anchored <sys_lyr> + 256-token syllable BPE right before final EOS
            if music_ids[-1] == uni_omr.EOS_ID:
                body = music_ids[:-1]
            else:
                body = music_ids
            shifted_lyr = [sys_lyr_tok_id] + [base_vocab + 1 + tid for tid in text_ids[1:-1]]
            unified_ids = (body[: MAX_SEQ_LEN - len(shifted_lyr) - 1] + shifted_lyr + [uni_omr.EOS_ID])[:MAX_SEQ_LEN]

            img = Image.open(row["image"]).convert("RGB")
            inp = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
            tgt = torch.tensor(unified_ids, dtype=torch.long, device=device).unsqueeze(0)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(inp, tgt[:, :-1], return_aux=False, return_mtp=True)
                losses = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)
                loss = losses["loss"] / GRAD_ACCUM
            loss.backward()
            step_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_state[k].mul_(EMA_DECAY).add_(v, alpha=1.0 - EMA_DECAY)
                else:
                    ema_state[k].copy_(v)

        loss_hist.append(step_loss)
        if step % 150 == 0 or step == 1:
            mean_l = sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:]))
            print(f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} loss={mean_l:.4f} elapsed={time.time()-t0:.1f}s", flush=True)

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ck = arm_dir / "latest.pt.tmp"
            torch.save({"step": step, "model_state_dict": model.state_dict(), "ema_state_dict": ema_state}, tmp_ck)
            os.replace(tmp_ck, arm_dir / "latest.pt")
            t_last_save = time.time()

    raw_eval = _eval_vocal_and_suite31(
        model,
        manifest_path,
        device,
        use_bf16,
        arm_mode="arm_b",
        base_vocab_size=base_vocab,
        sys_lyr_tok_id=sys_lyr_tok_id,
    )
    model.load_state_dict(ema_state, strict=True)
    ema_eval = _eval_vocal_and_suite31(
        model,
        manifest_path,
        device,
        use_bf16,
        arm_mode="arm_b",
        base_vocab_size=base_vocab,
        sys_lyr_tok_id=sys_lyr_tok_id,
    )
    packed_info = _pack_and_verify_hybrid_int4(
        model, arm_dir / f"{arm_name}_edge85m_hybrid_int4.bin", extra_module=None, group_size=INT4_GROUP_SIZE
    )

    result = {
        "arm": arm_name,
        "base_vocab_size": base_vocab,
        "extended_vocab_size": ext_vocab,
        "added_vocab_tokens": 1 + TEXT_BPE_VOCAB_SIZE,
        "train_loss_tail50": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5),
        "raw_eval": raw_eval,
        "ema_eval": ema_eval,
        "packed_int4_verification": packed_info,
        "elapsed_secs": round(time.time() - t0, 2),
    }
    _atomic_write_json(cached, result)
    del model, opt, ema_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    t_start = time.time()
    started_iso = _utc_now()
    hf_token = _read_and_purge_hf_token()
    try:
        _setup_environment()
        import torch
        from huggingface_hub import HfApi
        import uni_omr_v2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

        train_abs, _, manifest_path = _ensure_assets(hf_token)
        train_rows = _load_vocal_records(train_abs)
        n_with_text = sum(1 for r in train_rows if r["has_text"])
        print(
            f"[{EXP_ID}] Loaded {len(train_rows)} train pages ({n_with_text} with XML text events) on {gpu_name}",
            flush=True,
        )

        # Shared deterministic base edge85m initialization + baseline vocal S_note measurement
        _seed_everything(SEED)
        base_cfg = uni_omr_v2.UniOMRv2Config.edge85m(grad_checkpoint=True)
        base_model = uni_omr_v2.UniOMRv2Net(base_cfg).to(device)
        base_eval = _eval_vocal_and_suite31(
            base_model, manifest_path, device, use_bf16, arm_mode="base"
        )
        base_sd = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
        del base_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        arm_a = _train_arm_a_textspotter(base_sd, train_rows, manifest_path, device, use_bf16)
        arm_b = _train_arm_b_deferred_bpe(base_sd, train_rows, manifest_path, device, use_bf16)

        def _delta_pp(val: Optional[float], ref: Optional[float]) -> Optional[float]:
            if val is None or ref is None:
                return None
            return round((float(val) - float(ref)) * 100.0, 3)

        base_vocal_snote = base_eval.get("vocal_s_note_f1")
        comparison = {
            "arm_a_heldout_text_f1": arm_a["ema_eval"]["heldout_text_f1"],
            "arm_b_heldout_text_f1": arm_b["ema_eval"]["heldout_text_f1"],
            "arm_a_passes_text_f1_070": bool(arm_a["ema_eval"]["heldout_text_f1"] >= PROMOTE_TEXT_F1),
            "arm_b_passes_text_f1_070": bool(arm_b["ema_eval"]["heldout_text_f1"] >= PROMOTE_TEXT_F1),
            "arm_a_vocal_snote_delta_pp": _delta_pp(arm_a["ema_eval"]["vocal_s_note_f1"], base_vocal_snote),
            "arm_b_vocal_snote_delta_pp": _delta_pp(arm_b["ema_eval"]["vocal_s_note_f1"], base_vocal_snote),
            "arm_a_vocal_latency_s": arm_a["ema_eval"]["vocal_mean_latency_s"],
            "arm_b_vocal_latency_s": arm_b["ema_eval"]["vocal_mean_latency_s"],
            "arm_a_packed_mb": arm_a["packed_int4_verification"]["disk_mb"],
            "arm_b_packed_mb": arm_b["packed_int4_verification"]["disk_mb"],
            "arm_a_passes_55mb": arm_a["packed_int4_verification"]["passes_55mb_budget"],
            "arm_b_passes_55mb": arm_b["packed_int4_verification"]["passes_55mb_budget"],
        }

        report = {
            "exp_id": EXP_ID,
            "exp_code": EXP_CODE,
            "wave": 13,
            "status": "completed",
            "started_utc": started_iso,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"colab-{gpu_name}",
            "bf16": use_bf16,
            "base_eval": base_eval,
            "arm_a_textspotter_head": arm_a,
            "arm_b_deferred_sys_lyr_bpe": arm_b,
            "comparison": comparison,
        }

        eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        run_local = VOL_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(
            run_local,
            {
                "exp_id": EXP_ID,
                "exp_code": EXP_CODE,
                "wave": 13,
                "status": "completed",
                "started_utc": started_iso,
                "completed_utc": report["completed_utc"],
                "elapsed_secs": report["elapsed_secs"],
                "eval_path": HF_EVAL_PATH,
                "comparison": comparison,
            },
        )

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload Wave-13 lyrics TextSpotterHead vs deferred BPE bake-off",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload Wave-13 lyrics bake-off run ledger",
        )
        print(f"[{EXP_ID}] UPLOADED {HF_EVAL_PATH} & {HF_RUN_PATH} TO HF!", flush=True)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()

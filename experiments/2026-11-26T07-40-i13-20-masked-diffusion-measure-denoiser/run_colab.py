#!/usr/bin/env python3
"""Google Colab L4 runner for Iter 13 Moonshot i13-20:
Bidirectional Masked-Diffusion Measure Denoiser (MDM-MeasureRefiner).

Experiment ID: 2026-11-26T07-40-i13-20-masked-diffusion-measure-denoiser
Controlled 2-Arm Ablation (BF16, seed=42) on 1x Colab L4:
  - Arm A (Control): Standard causal UniOMR v2 (small_45m, ~41.3M) fine-tuned
    1,000 steps on iter10_5_proxy_2k, evaluated with causal FSA greedy (K=0).
  - Arm B (Treatment): Identical UniOMR v2 + MDMMeasureRefiner (+2.75M params,
    4-layer non-causal GQA cross-attention denoiser sharing tok_emb & lm_head,
    total ~44.1M / < 33 MB Hybrid INT4):
      * Trained jointly with L_ar + 0.5 * L_mdm (50% i.i.d. token masking +
        50% measure-span masking, t ~ U(0.15, 0.70), 1/t importance weighting).
      * At inference: runs causal FSA draft -> masks low-confidence tokens
        (p_max < tau, capped at 35%) -> K=4/6 parallel cosine-schedule
        bidirectional denoising steps conditioned on mem_final + FSA guard.

Uploads:
  - evals/2026-11-26T07-40-i13-20-masked-diffusion-measure-denoiser.json
  - runs/2026-11-26T07-40-i13-20-masked-diffusion-measure-denoiser.json
to gregoreesmaa/sheet-music-converter-omr.
"""

from __future__ import annotations

import copy
import difflib
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T07-40-i13-20-masked-diffusion-measure-denoiser"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt",
)
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

VOL_ROOT = Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

SEED = 42
TRAIN_STEPS = 1000
WARMUP_STEPS = 100
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
GRAD_ACCUM = 4
LR_BACKBONE = 3e-5
LR_MDM = 3e-4
WEIGHT_DECAY = 0.04
MDM_LOSS_WEIGHT = 0.50
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
SUITE31_MAX_LEN = 2048
EMA_DECAY = 0.995
SAVE_EVERY_MINUTES = 5.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _locate_repo_and_mvd() -> Tuple[Path, Path]:
    for repo in (
        Path("/content/sheet-music-converter"),
        Path("/workspace/sheet-music-converter"),
        Path("/content/workspace/sheet-music-converter"),
        Path.cwd(),
    ):
        mvd = repo / "tmp-train" / "mvd"
        if (mvd / "uni_omr_v2.py").is_file():
            if str(mvd) not in sys.path:
                sys.path.insert(0, str(mvd))
            return repo, mvd

    for bundle in (Path("/content/ov_mvd_bundle.tar.gz"), Path("/content/bundle.tar.gz")):
        if bundle.is_file():
            extract_root = Path("/content/workspace")
            extract_root.mkdir(parents=True, exist_ok=True)
            print(f"[{EXP_ID}] Extracting {bundle} -> {extract_root}...", flush=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=extract_root)
            for repo, mvd in (
                (extract_root / "sheet-music-converter", extract_root / "sheet-music-converter" / "tmp-train" / "mvd"),
                (extract_root, extract_root / "mvd"),
            ):
                if (mvd / "uni_omr_v2.py").is_file():
                    if str(mvd) not in sys.path:
                        sys.path.insert(0, str(mvd))
                    return repo, mvd

    raise FileNotFoundError("Could not locate tmp-train/mvd/uni_omr_v2.py")


def _setup_environment() -> Tuple[Path, Optional[Path], Optional[Path]]:
    repo_root, _ = _locate_repo_and_mvd()
    needed = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed])

    aser_manifest = repo_root / "converter" / "assembly-serialize" / "Cargo.toml"
    eharness_manifest = repo_root / "converter" / "eval-harness" / "Cargo.toml"
    aser_bin = aser_manifest.parent / "target" / "release" / "assembly-serialize"
    eharness_bin = eharness_manifest.parent / "target" / "release" / "eval-harness"

    if aser_manifest.is_file() and eharness_manifest.is_file():
        cargo_bin_dir = Path.home() / ".cargo" / "bin"
        env = os.environ.copy()
        env["PATH"] = f"{cargo_bin_dir}:{env.get('PATH', '')}"
        if shutil.which("cargo", path=env["PATH"]) is None:
            _run_cmd(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
                "| sh -s -- -y --default-toolchain stable --profile minimal",
                shell=True,
                env=env,
            )
        if not aser_bin.is_file():
            _run_cmd(["cargo", "build", "--release", "--manifest-path", str(aser_manifest)], env=env)
        if not eharness_bin.is_file():
            _run_cmd(["cargo", "build", "--release", "--manifest-path", str(eharness_manifest)], env=env)
        if aser_bin.is_file() and eharness_bin.is_file():
            return repo_root, aser_bin, eharness_bin

    return repo_root, None, None


def _ensure_assets(hf_token: str) -> Tuple[Optional[Path], Optional[str], Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    VOL_ROOT.mkdir(parents=True, exist_ok=True)
    ds_root = VOL_ROOT / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)
    ckpt_root = VOL_ROOT / "pretrained"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    ckpt_local: Optional[Path] = None
    ckpt_source: Optional[str] = None
    for cand in HF_CKPT_CANDIDATES:
        target = ckpt_root / Path(cand).name
        if target.is_file():
            ckpt_local = target
            ckpt_source = cand
            break
        try:
            dl_p = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=cand,
                repo_type="dataset",
                token=hf_token,
                local_dir="/tmp/hf_dl",
            )
            shutil.copy2(dl_p, target)
            try:
                os.remove(dl_p)
            except OSError:
                pass
            ckpt_local = target
            ckpt_source = cand
            print(f"[{EXP_ID}] Using warm-start checkpoint {cand}", flush=True)
            break
        except Exception as exc:
            print(f"[{EXP_ID}] Checkpoint {cand} unavailable ({exc}); trying next...", flush=True)

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

    return ckpt_local, ckpt_source, train_abs, val_abs, manifest_path


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


def _encode_bekerns_full(bek_paths: Sequence[Path]) -> List[int]:
    import data
    import uni_omr

    full_toks: List[str] = []
    for idx, bp in enumerate(bek_paths):
        toks = data.read_bekern_tokens(bp)
        if not toks:
            continue
        if idx > 0:
            full_toks.append(uni_omr.SYS_TOKEN)
        full_toks.extend(toks)
    return uni_omr.encode_uni(full_toks)


def _build_mdm_modules():
    """Define 4-layer Bidirectional Masked-Diffusion Measure Denoiser (~2.75M params)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from uni_omr_v2 import GroupedQueryAttention, RMSNorm, SwiGLU

    class BiDirGQALayer(nn.Module):
        def __init__(self, d_model: int = 384, n_heads: int = 8, n_kv_heads: int = 2, d_ff: int = 768) -> None:
            super().__init__()
            self.norm1 = RMSNorm(d_model)
            self.self_attn = GroupedQueryAttention(d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, use_rope=True)
            self.norm2 = RMSNorm(d_model)
            self.cross_attn = GroupedQueryAttention(d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, use_rope=False)
            self.norm3 = RMSNorm(d_model)
            self.ffn = SwiGLU(d_model, d_ff, dropout=0.05)
            self.ada = nn.Linear(d_model, d_model * 2, bias=True)
            nn.init.zeros_(self.ada.weight)
            nn.init.zeros_(self.ada.bias)

        def forward(
            self,
            x: torch.Tensor,
            memory: torch.Tensor,
            t_emb: torch.Tensor,
            rope_cos: torch.Tensor,
            rope_sin: torch.Tensor,
        ) -> torch.Tensor:
            scale, shift = self.ada(t_emb).unsqueeze(1).chunk(2, dim=-1)
            h = self.norm1(x) * (1.0 + scale) + shift
            # Non-causal bidirectional self-attention!
            x = x + self.self_attn(h, h, rope_cos=rope_cos, rope_sin=rope_sin, is_causal=False)
            x = x + self.cross_attn(self.norm2(x), memory, is_causal=False)
            x = x + self.ffn(self.norm3(x))
            return x

    class MDMMeasureRefiner(nn.Module):
        """Bidirectional Masked-Diffusion Denoiser sharing tok_emb & lm_head with UniOMR v2."""

        def __init__(self, base_model: nn.Module, n_layers: int = 4, d_ff: int = 768) -> None:
            super().__init__()
            self.base = base_model
            d_model = int(base_model.cfg.d_model)
            n_heads = int(base_model.cfg.n_heads)
            n_kv = int(base_model.cfg.n_kv_heads)
            self.mask_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.time_mlp = nn.Sequential(
                nn.Linear(1, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            self.layers = nn.ModuleList(
                [BiDirGQALayer(d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv, d_ff=d_ff) for _ in range(n_layers)]
            )
            self.out_norm = RMSNorm(d_model)

        def forward_denoise(
            self,
            tok_ids: torch.Tensor,
            mask_bool: torch.Tensor,
            t_ratio: torch.Tensor,
            mem_final: torch.Tensor,
        ) -> torch.Tensor:
            # tok_ids: (B, T), mask_bool: (B, T) True where masked, t_ratio: (B, 1)
            x = self.base.tok_emb(tok_ids)
            mask_vec = self.mask_embed.to(x.dtype).expand_as(x)
            x = torch.where(mask_bool.unsqueeze(-1), mask_vec, x)
            t_emb = self.time_mlp(t_ratio.to(x.dtype))
            cos1d, sin1d = self.base.rope1d(tok_ids.shape[1], device=x.device, dtype=x.dtype)
            for layer in self.layers:
                x = layer(x, mem_final, t_emb, cos1d, sin1d)
            x = self.out_norm(x)
            return self.base.lm_head(x)

        @torch.no_grad()
        def refine_draft(
            self,
            draft_ids: List[int],
            draft_probs: List[float],
            mem_final: torch.Tensor,
            *,
            k_steps: int = 4,
            tau: float = 0.65,
            max_mask_frac: float = 0.35,
            device: torch.device,
            use_bf16: bool,
        ) -> List[int]:
            import uni_omr

            if k_steps <= 0 or len(draft_ids) < 6:
                return list(draft_ids)

            seq = torch.tensor(draft_ids, dtype=torch.long, device=device).unsqueeze(0)
            probs = torch.tensor(draft_probs, dtype=torch.float32, device=device)
            T = len(draft_ids)
            # Never mask special delimiters (<bos>, <eos>, <pad>, <sys>)
            protect = (
                (seq[0] == uni_omr.BOS_ID)
                | (seq[0] == uni_omr.EOS_ID)
                | (seq[0] == uni_omr.PAD_ID)
                | (seq[0] == uni_omr.SYS_ID)
            )
            cand_mask = (probs < tau) & (~protect)
            max_m = max(1, int(round(T * max_mask_frac)))
            if int(cand_mask.sum().item()) > max_m:
                # Mask only the lowest-confidence max_m tokens
                masked_scores = torch.where(protect, torch.full_like(probs, 2.0), probs)
                _, low_idx = torch.topk(masked_scores, k=max_m, largest=False)
                cand_mask = torch.zeros_like(cand_mask)
                cand_mask[low_idx] = True

            cur_mask = cand_mask.unsqueeze(0)  # (1, T)
            n_init_mask = int(cur_mask.sum().item())
            if n_init_mask == 0:
                return list(draft_ids)

            cur_ids = seq.clone()
            for step_i in range(k_steps):
                t_val = float(cur_mask.sum().item()) / float(max(1, T))
                t_tensor = torch.tensor([[t_val]], device=device, dtype=torch.float32)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    logits = self.forward_denoise(cur_ids, cur_mask, t_tensor, mem_final)
                p = torch.softmax(logits[0].float(), dim=-1)
                conf, pred = p.max(dim=-1)

                if step_i == k_steps - 1:
                    cur_ids[0] = torch.where(cur_mask[0], pred, cur_ids[0])
                    cur_mask.zero_()
                    break

                # MaskGIT cosine unmasking schedule
                frac_keep_masked = math.cos(0.5 * math.pi * float(step_i + 1) / float(k_steps))
                n_rem = int(round(n_init_mask * frac_keep_masked))
                n_unmask = max(1, int(cur_mask.sum().item()) - n_rem)
                masked_pos = torch.nonzero(cur_mask[0], as_tuple=False).squeeze(-1)
                if masked_pos.numel() == 0:
                    break
                pos_conf = conf[masked_pos]
                _, top_local = torch.topk(pos_conf, k=min(n_unmask, masked_pos.numel()), largest=True)
                chosen = masked_pos[top_local]
                cur_ids[0, chosen] = pred[chosen]
                cur_mask[0, chosen] = False

            return [int(x) for x in cur_ids[0].tolist()]

    return MDMMeasureRefiner


def _sample_mdm_mask(ids: "torch.Tensor", rng: random.Random) -> Tuple["torch.Tensor", float]:
    import torch
    import uni_omr

    B, T = ids.shape
    t = rng.uniform(0.15, 0.70)
    protect = (
        (ids == uni_omr.BOS_ID)
        | (ids == uni_omr.EOS_ID)
        | (ids == uni_omr.PAD_ID)
        | (ids == uni_omr.SYS_ID)
    )
    if rng.random() < 0.50:
        # i.i.d. token masking
        mask = (torch.rand((B, T), device=ids.device) < t) & (~protect)
    else:
        # Contiguous measure-span masking (1-3 spans covering ~t of sequence)
        mask = torch.zeros((B, T), dtype=torch.bool, device=ids.device)
        span_total = max(2, int(round(T * t)))
        n_spans = rng.randint(1, 3)
        span_len = max(2, span_total // n_spans)
        for b in range(B):
            for _ in range(n_spans):
                s0 = rng.randint(1, max(1, T - span_len - 1))
                mask[b, s0 : min(T, s0 + span_len)] = True
        mask = mask & (~protect)
    if int(mask.sum().item()) == 0 and T > 3:
        mask[:, 1] = True
    return mask, t


def _build_base_model(ckpt_path: Optional[Path], device: "torch.device") -> Any:
    import torch
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net

    if ckpt_path is not None and ckpt_path.is_file():
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        raw_cfg = ck.get("cfg") or ck.get("config") or {}
        if isinstance(raw_cfg, UniOMRv2Config):
            cfg = raw_cfg
        elif isinstance(raw_cfg, dict) and raw_cfg:
            valid = set(UniOMRv2Config.__dataclass_fields__.keys())
            filtered = {k: v for k, v in raw_cfg.items() if k in valid}
            for k in ("backbone_channels", "backbone_blocks"):
                if k in filtered and isinstance(filtered[k], list):
                    filtered[k] = tuple(filtered[k])
            cfg = UniOMRv2Config(**filtered)
        else:
            cfg = UniOMRv2Config.small_45m()
        model = UniOMRv2Net(cfg)
        sd = ck.get("ema_state_dict") or ck.get("model_state_dict") or ck.get("state_dict") or ck
        model.load_state_dict(sd, strict=False)
        return model.to(device)
    return UniOMRv2Net(UniOMRv2Config.small_45m()).to(device)


def _token_f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    sm = difflib.SequenceMatcher(a=list(gold), b=list(pred), autojunk=False)
    matches = sum(b.size for b in sm.get_matching_blocks())
    return (2.0 * matches) / float(len(pred) + len(gold))


def _generate_draft_with_probs(
    base: Any, mem_levels: Any, max_len: int, device: "torch.device", use_bf16: bool
) -> Tuple[List[int], List[float]]:
    import torch
    import eval_uni_suite31 as eus
    import uni_omr

    ids = eus.generate_constrained(base, mem_levels, max_len=max_len, use_grammar=True, device=device)
    if len(ids) < 2:
        return list(ids), [1.0] * len(ids)
    # Single fast teacher-forced pass to extract exact per-token draft probabilities
    seq = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        out = base.decode_tokens(seq[:, :-1], mem_levels, return_mtp=False, return_aux=False, apply_tok_dropout=False)
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.softmax(logits[0].float(), dim=-1)
        tok_p = probs.gather(-1, seq[0, 1:].unsqueeze(-1)).squeeze(-1).cpu().tolist()
    return [int(x) for x in ids], [1.0] + [float(p) for p in tok_p]


def _eval_proxy_val(
    wrapper_or_base: Any,
    val_pairs: List[Tuple[Path, List[Path], str]],
    *,
    has_mdm: bool,
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr

    wrapper_or_base.eval()
    base = wrapper_or_base.base if has_mdm else wrapper_or_base
    ar_ces: List[float] = []
    mdm_ces: List[float] = []
    oracle_acc: List[float] = []
    rng = random.Random(SEED + 77)

    with torch.no_grad():
        for img_p, beks, _ in val_pairs[:32]:
            img = Image.open(img_p).convert("RGB")
            full_ids = _encode_bekerns_full(beks)
            if len(full_ids) < 6:
                continue
            inp_t = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
            tgt = torch.tensor(full_ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mem, _, _ = base.encode_image(inp_t, return_aux=False)
                out = base.decode_tokens(tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False)
                logits = out[0] if isinstance(out, tuple) else out
                ar_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID)
            ar_ces.append(float(ar_loss.item()))

            if has_mdm:
                mask_b, t_val = _sample_mdm_mask(tgt, rng)
                t_tensor = torch.tensor([[t_val]], device=device, dtype=torch.float32)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    m_logits = wrapper_or_base.forward_denoise(tgt, mask_b, t_tensor, mem[-1])
                    m_loss = F.cross_entropy(m_logits[mask_b], tgt[mask_b])
                mdm_ces.append(float(m_loss.item()))
                acc = float((m_logits[mask_b].argmax(dim=-1) == tgt[mask_b]).float().mean().item())
                oracle_acc.append(acc)

    return {
        "val_ar_ce": round(sum(ar_ces) / max(1, len(ar_ces)), 5),
        "val_mdm_ce": round(sum(mdm_ces) / max(1, len(mdm_ces)), 5) if mdm_ces else None,
        "val_oracle_mask_acc": round(sum(oracle_acc) / max(1, len(oracle_acc)), 4) if oracle_acc else None,
        "n_pages": len(ar_ces),
    }


def _eval_suite31(
    wrapper_or_base: Any,
    manifest_path: Path,
    *,
    has_mdm: bool,
    k_steps: int,
    tau: float,
    device: "torch.device",
    use_bf16: bool,
    aser_bin: Optional[Path],
    eharness_bin: Optional[Path],
) -> Dict[str, Any]:
    import torch
    from PIL import Image
    import data
    import eval_uni_suite31 as eus
    import uni_omr

    wrapper_or_base.eval()
    base = wrapper_or_base.base if has_mdm else wrapper_or_base
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent

    buckets: Dict[str, List[Dict[str, float]]] = {
        "printed_clean": [],
        "printed_photo": [],
        "hw_clean": [],
        "hw_photo": [],
    }
    latencies: List[float] = []

    with torch.no_grad():
        for entry in manifest.get("pieces", []):
            is_hw = bool(entry.get("is_handwritten", False)) or str(entry.get("domain", "")).lower().startswith("hw")
            gt_beks = [base_dir / p for p in entry.get("bekerns", [])]
            gt_toks: List[str] = []
            for idx, bp in enumerate(gt_beks):
                if bp.is_file():
                    if idx > 0:
                        gt_toks.append(uni_omr.SYS_TOKEN)
                    gt_toks.extend(data.read_bekern_tokens(bp))

            for mode_key, img_rel in (("clean", entry.get("clean_image")), ("photo", entry.get("photo_image"))):
                if not img_rel:
                    continue
                img_p = base_dir / img_rel
                if not img_p.is_file():
                    continue
                img = Image.open(img_p).convert("RGB")
                inp_t = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
                t0 = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    mem, _, _ = base.encode_image(inp_t, return_aux=False)
                draft_ids, draft_probs = _generate_draft_with_probs(base, mem, SUITE31_MAX_LEN, device, use_bf16)
                if has_mdm and k_steps > 0:
                    final_ids = wrapper_or_base.refine_draft(
                        draft_ids, draft_probs, mem[-1], k_steps=k_steps, tau=tau, device=device, use_bf16=use_bf16
                    )
                else:
                    final_ids = draft_ids
                latencies.append(time.perf_counter() - t0)

                pred_toks = eus.decode_uni_compat(final_ids, vocab_size=base.cfg.vocab_size)
                tok_f1 = _token_f1(pred_toks, gt_toks)
                note_f1 = tok_f1
                dec_ok = 1.0 if len(pred_toks) > 0 else 0.0
                if aser_bin is not None and eharness_bin is not None and entry.get("gt_musicxml"):
                    gt_xml = base_dir / entry["gt_musicxml"]
                    if gt_xml.is_file():
                        try:
                            with tempfile.TemporaryDirectory() as td:
                                bek_f = Path(td) / "pred.bekern"
                                xml_f = Path(td) / "pred.musicxml"
                                bek_f.write_text(" ".join(eus.trim_truncated_tokens(pred_toks)), encoding="utf-8")
                                r1 = subprocess.run(
                                    [str(aser_bin), "bekern-decode", "--input", str(bek_f), "--out", str(xml_f)],
                                    capture_output=True, text=True, timeout=15,
                                )
                                if r1.returncode == 0 and xml_f.is_file():
                                    r2 = subprocess.run(
                                        [str(eharness_bin), "--gt", str(gt_xml), "--pred", str(xml_f), "--json"],
                                        capture_output=True, text=True, timeout=15,
                                    )
                                    if r2.returncode == 0:
                                        res = json.loads(r2.stdout)
                                        note_f1 = float(res.get("strict_note_f1", res.get("note_f1", tok_f1)))
                                        dec_ok = 1.0
                                else:
                                    dec_ok = 0.0
                        except Exception:
                            dec_ok = 0.0

                bkey = f"{'hw' if is_hw else 'printed'}_{mode_key}"
                buckets[bkey].append({"tok_f1": tok_f1, "note_f1": note_f1, "dec_ok": dec_ok})

    def _agg(rows: List[Dict[str, float]]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0, "tok_f1": None, "note_f1": None, "dec_ok": None}
        return {
            "n": len(rows),
            "tok_f1": round(sum(r["tok_f1"] for r in rows) / len(rows), 4),
            "note_f1": round(sum(r["note_f1"] for r in rows) / len(rows), 4),
            "dec_ok": round(sum(r["dec_ok"] for r in rows) / len(rows), 4),
        }

    return {
        "k_steps": k_steps,
        "tau": tau,
        "printed_clean": _agg(buckets["printed_clean"]),
        "printed_photo": _agg(buckets["printed_photo"]),
        "hw_clean": _agg(buckets["hw_clean"]),
        "hw_photo": _agg(buckets["hw_photo"]),
        "mean_latency_s": round(sum(latencies) / max(1, len(latencies)), 3),
    }


def _train_arm(
    arm_name: str,
    *,
    has_mdm: bool,
    ckpt_path: Optional[Path],
    train_pairs: List[Tuple[Path, List[Path], str]],
    val_pairs: List[Tuple[Path, List[Path], str]],
    manifest_path: Path,
    device: "torch.device",
    use_bf16: bool,
    aser_bin: Optional[Path],
    eharness_bin: Optional[Path],
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr

    _seed_everything(SEED)
    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached_res = arm_dir / "arm_result.json"
    if cached_res.is_file():
        return json.loads(cached_res.read_text(encoding="utf-8"))

    base = _build_base_model(ckpt_path, device)
    MDMCls = _build_mdm_modules()
    model = MDMCls(base, n_layers=4, d_ff=768).to(device) if has_mdm else base

    if has_mdm:
        base_ids = {id(p) for p in model.base.parameters()}
        mdm_params = [p for p in model.parameters() if id(p) not in base_ids and p.requires_grad]
        bb_params = [p for p in model.base.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            [
                {"params": bb_params, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
                {"params": mdm_params, "lr": LR_MDM, "weight_decay": 0.02},
            ]
        )
        extra_params = sum(p.numel() for p in mdm_params)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR_BACKBONE, weight_decay=WEIGHT_DECAY)
        extra_params = 0

    ema_state = copy.deepcopy(model.state_dict())
    rng = random.Random(SEED)
    t_last_save = time.time()
    t_arm0 = time.time()
    loss_hist: List[float] = []

    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        mult = _wsd_mult(step, TRAIN_STEPS, WARMUP_STEPS)
        opt.param_groups[0]["lr"] = LR_BACKBONE * mult
        if has_mdm:
            opt.param_groups[1]["lr"] = LR_MDM * mult

        opt.zero_grad(set_to_none=True)
        step_loss_val = 0.0
        for _ in range(GRAD_ACCUM):
            img_p, beks, _ = train_pairs[rng.randrange(len(train_pairs))]
            img = Image.open(img_p).convert("RGB")
            full_ids = _encode_bekerns_full(beks)
            if len(full_ids) < 6:
                continue
            inp_t = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
            tgt = torch.tensor(full_ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                cur_base = model.base if has_mdm else model
                mem, _, _ = cur_base.encode_image(inp_t, return_aux=False)
                out = cur_base.decode_tokens(tgt[:, :-1], mem, return_mtp=False, return_aux=False)
                logits = out[0] if isinstance(out, tuple) else out
                loss_ar = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID
                )
                if has_mdm:
                    mask_b, t_val = _sample_mdm_mask(tgt, rng)
                    t_tensor = torch.tensor([[t_val]], device=device, dtype=torch.float32)
                    m_logits = model.forward_denoise(tgt, mask_b, t_tensor, mem[-1])
                    imp_w = min(4.0, 1.0 / max(0.15, t_val))
                    loss_mdm = imp_w * F.cross_entropy(m_logits[mask_b], tgt[mask_b])
                    loss = (loss_ar + MDM_LOSS_WEIGHT * loss_mdm) / GRAD_ACCUM
                else:
                    loss = loss_ar / GRAD_ACCUM

            loss.backward()
            step_loss_val += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_state[k].mul_(EMA_DECAY).add_(v, alpha=1.0 - EMA_DECAY)
                else:
                    ema_state[k].copy_(v)

        loss_hist.append(step_loss_val)
        if step % 100 == 0 or step == 1:
            mean_l = sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:]))
            print(f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} loss={mean_l:.4f} elapsed={time.time()-t_arm0:.1f}s", flush=True)

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ck = arm_dir / "latest.pt.tmp"
            torch.save({"step": step, "ema_state_dict": ema_state}, tmp_ck)
            os.replace(tmp_ck, arm_dir / "latest.pt")
            t_last_save = time.time()

    model.load_state_dict(ema_state, strict=True)
    proxy_val = _eval_proxy_val(model, val_pairs, has_mdm=has_mdm, device=device, use_bf16=use_bf16)
    s31_k0 = _eval_suite31(
        model, manifest_path, has_mdm=has_mdm, k_steps=0, tau=0.65, device=device, use_bf16=use_bf16,
        aser_bin=aser_bin, eharness_bin=eharness_bin,
    )
    s31_k4 = None
    s31_k6 = None
    if has_mdm:
        s31_k4 = _eval_suite31(
            model, manifest_path, has_mdm=True, k_steps=4, tau=0.65, device=device, use_bf16=use_bf16,
            aser_bin=aser_bin, eharness_bin=eharness_bin,
        )
        s31_k6 = _eval_suite31(
            model, manifest_path, has_mdm=True, k_steps=6, tau=0.75, device=device, use_bf16=use_bf16,
            aser_bin=aser_bin, eharness_bin=eharness_bin,
        )

    result = {
        "arm": arm_name,
        "has_mdm": has_mdm,
        "extra_params": extra_params,
        "total_params": sum(p.numel() for p in model.parameters()),
        "train_loss_tail50": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5),
        "proxy_val": proxy_val,
        "suite31_k0": s31_k0,
        "suite31_k4": s31_k4,
        "suite31_k6": s31_k6,
        "elapsed_secs": round(time.time() - t_arm0, 2),
    }
    _atomic_write_json(cached_res, result)
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
        _, aser_bin, eharness_bin = _setup_environment()
        import torch
        from huggingface_hub import HfApi

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

        ckpt_local, ckpt_source, train_abs, val_abs, manifest_path = _ensure_assets(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)

        arm_a = _train_arm(
            "arm_a_control_causal_ar",
            has_mdm=False,
            ckpt_path=ckpt_local,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            manifest_path=manifest_path,
            device=device,
            use_bf16=use_bf16,
            aser_bin=aser_bin,
            eharness_bin=eharness_bin,
        )
        arm_b = _train_arm(
            "arm_b_mdm_measure_refiner",
            has_mdm=True,
            ckpt_path=ckpt_local,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            manifest_path=manifest_path,
            device=device,
            use_bf16=use_bf16,
            aser_bin=aser_bin,
            eharness_bin=eharness_bin,
        )

        def _delta(b: Optional[float], a: Optional[float]) -> Optional[float]:
            return None if (b is None or a is None) else round(float(b) - float(a), 5)

        b_k4 = arm_b["suite31_k4"]
        b_k0 = arm_b["suite31_k0"]
        a_k0 = arm_a["suite31_k0"]
        comparison = {
            "k4_vs_k0_hw_clean_note_f1_delta": _delta(b_k4["hw_clean"]["note_f1"], b_k0["hw_clean"]["note_f1"]),
            "k4_vs_arm_a_hw_clean_note_f1_delta": _delta(b_k4["hw_clean"]["note_f1"], a_k0["hw_clean"]["note_f1"]),
            "k4_vs_arm_a_hw_photo_note_f1_delta": _delta(b_k4["hw_photo"]["note_f1"], a_k0["hw_photo"]["note_f1"]),
            "k4_vs_arm_a_printed_clean_note_f1_delta": _delta(b_k4["printed_clean"]["note_f1"], a_k0["printed_clean"]["note_f1"]),
            "k4_hw_dec_ok": b_k4["hw_clean"]["dec_ok"],
            "k4_printed_dec_ok": b_k4["printed_clean"]["dec_ok"],
            "k4_latency_overhead_ratio": round(b_k4["mean_latency_s"] / max(1e-6, b_k0["mean_latency_s"]), 3),
            "val_oracle_mask_acc": arm_b["proxy_val"]["val_oracle_mask_acc"],
        }

        report = {
            "exp_id": EXP_ID,
            "exp_code": "i13-20",
            "status": "completed",
            "warmstart_ckpt": ckpt_source or "scratch_small_45m",
            "started_utc": started_iso,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"colab-{gpu_name}",
            "bf16": use_bf16,
            "arm_a_control": arm_a,
            "arm_b_mdm_refiner": arm_b,
            "comparison": comparison,
        }

        eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        run_local = VOL_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(
            run_local,
            {
                "exp_id": EXP_ID,
                "status": "completed",
                "platform": f"colab-{gpu_name}",
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
            commit_message=f"eval({EXP_ID}): upload MDM-MeasureRefiner ablation",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload MDM-MeasureRefiner run ledger",
        )
        print(f"[{EXP_ID}] UPLOADED {HF_EVAL_PATH} & {HF_RUN_PATH} TO HF!", flush=True)
        print(json.dumps(comparison, indent=2), flush=True)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()

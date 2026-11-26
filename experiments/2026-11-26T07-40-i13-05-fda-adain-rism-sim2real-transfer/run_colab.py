#!/usr/bin/env python3
"""Iter 13 Exp i13-05 on Google Colab L4:
Unpaired RISM-5707 Sim-to-Real Ink/Paper Domain Transfer (FDA 2D-FFT Low-Freq
Amplitude Swap & Chromatic-AdaIN) during Supervised Fine-Tuning of uni_omr_soup.pt.

4 matched 120-step fine-tuning arms (identical seed=42, identical 480 labeled
iter10_5_proxy_2k page draws, AdamW lr=1.5e-5 -> 3e-6, label_smoothing=0.02):
  1. arm_ctrl_standard : Standard mild contrast/brightness jitter (control)
  2. arm_fda_b002      : Unpaired RISM-5707 2D-FFT FDA (beta=0.02, p=0.5)
  3. arm_fda_b006      : Unpaired RISM-5707 2D-FFT FDA (beta=0.06, p=0.5)
  4. arm_color_adain   : Unpaired RISM-5707 Chromatic-AdaIN moment match (alpha~U[0.4,0.8], p=0.5)

Evaluated on suite31_hw (14 HW + 12 stratified printed = 52 pages, max_dim=1024,
eval_max_len=1536, MTP-4 + FSA grammar).

Expects:
  - /content/bundle.tar.gz (or pre-extracted workspace)
  - /tmp/.hf_token (Hugging Face token; chmod 600, deleted immediately and in finally)
"""

from __future__ import annotations

from datetime import datetime, timezone
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

EXP_ID = "2026-11-26T07-40-i13-05-fda-adain-rism-sim2real-transfer"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_PATH = "checkpoints/iter10_soup_baseline/uni_omr_soup.pt"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_RISM_TAR = "datasets/rism_hw_5707.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

ARMS: List[Dict[str, Any]] = [
    {"key": "arm_ctrl_standard", "mode": "ctrl", "beta": 0.0, "prob": 0.0},
    {"key": "arm_fda_b002", "mode": "fda", "beta": 0.02, "prob": 0.5},
    {"key": "arm_fda_b006", "mode": "fda", "beta": 0.06, "prob": 0.5},
    {"key": "arm_color_adain", "mode": "adain", "beta": 0.0, "prob": 0.5},
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _purge_hf_token() -> None:
    if TOKEN_PATH.exists():
        try:
            TOKEN_PATH.unlink()
        except OSError:
            pass


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and TOKEN_PATH.exists():
        try:
            os.chmod(TOKEN_PATH, 0o600)
        except OSError:
            pass
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    _purge_hf_token()
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
            bundle = BUNDLE_PATH if BUNDLE_PATH.exists() else Path("/content/repo_bundle.tar.gz")
            if not bundle.exists():
                raise FileNotFoundError(f"Missing bundle at {BUNDLE_PATH}")
            print(f"[{EXP_ID}] Extracting {bundle} -> {WORK_ROOT}...", flush=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORK_ROOT)

    needed = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)
    return mvd_dir


def _extract_tar_once(tar_path: Path, dest_dir: Path) -> None:
    sentinel = dest_dir / ".extracted"
    if sentinel.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=dest_dir)
    sentinel.write_text(_utc_now() + "\n", encoding="utf-8")


def _ensure_assets(hf_token: str) -> Tuple[Path, Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    ckpt_path = CACHE_ROOT / HF_CKPT_PATH
    if not ckpt_path.exists():
        print(f"[{EXP_ID}] Downloading {HF_CKPT_PATH}...", flush=True)
        hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", filename=HF_CKPT_PATH,
            local_dir=str(CACHE_ROOT), token=hf_token,
        )

    proxy_dir = CACHE_ROOT / "datasets" / "iter10_5_proxy_2k"
    if not (proxy_dir / ".extracted").exists():
        print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR}...", flush=True)
        tar_p = Path(hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", filename=HF_PROXY_TAR,
            local_dir=str(CACHE_ROOT / "tars"), token=hf_token,
        ))
        _extract_tar_once(tar_p, proxy_dir)
        tar_p.unlink(missing_ok=True)

    rism_dir = CACHE_ROOT / "datasets" / "rism_hw"
    if not (rism_dir / ".extracted").exists():
        print(f"[{EXP_ID}] Downloading {HF_RISM_TAR}...", flush=True)
        tar_p = Path(hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", filename=HF_RISM_TAR,
            local_dir=str(CACHE_ROOT / "tars"), token=hf_token,
        ))
        _extract_tar_once(tar_p, rism_dir)
        tar_p.unlink(missing_ok=True)

    s31_dir = CACHE_ROOT / "datasets" / "suite31_hw_eval"
    if not (s31_dir / ".extracted").exists():
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR}...", flush=True)
        tar_p = Path(hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", filename=HF_SUITE31_TAR,
            local_dir=str(CACHE_ROOT / "tars"), token=hf_token,
        ))
        _extract_tar_once(tar_p, s31_dir)
        tar_p.unlink(missing_ok=True)

    manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json under {s31_dir}")
    return ckpt_path, proxy_dir, rism_dir, manifests[0]


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


def _select_eval_pieces(pieces: Sequence[Dict[str, Any]], n_printed: int = 12) -> List[Dict[str, Any]]:
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


def _apply_fda_2d(src_t: Any, tgt_t: Any, beta: float) -> Any:
    """Fourier Domain Adaptation (Yang & Soatto): swap low-frequency 2D-FFT amplitude
    spectrum from unpaired target archival RISM page onto source labeled page while
    preserving 100% of source phase (exact staff lines, noteheads, GT alignment).
    Both tensors are `(1, 3, H, W)` float32 in `[0, 1]`.
    """
    import torch
    import torch.nn.functional as F

    _, _, h, w = src_t.shape
    if tgt_t.shape[-2:] != (h, w):
        tgt_t = F.interpolate(tgt_t, size=(h, w), mode="bilinear", align_corners=False)

    fft_src = torch.fft.fft2(src_t.float(), dim=(-2, -1))
    fft_tgt = torch.fft.fft2(tgt_t.float(), dim=(-2, -1))

    amp_src = torch.fft.fftshift(torch.abs(fft_src), dim=(-2, -1))
    pha_src = torch.angle(fft_src)
    amp_tgt = torch.fft.fftshift(torch.abs(fft_tgt), dim=(-2, -1))

    b = max(1, int(math.floor(min(h, w) * float(beta))))
    cy, cx = h // 2, w // 2
    y1, y2 = max(0, cy - b), min(h, cy + b + 1)
    x1, x2 = max(0, cx - b), min(w, cx + b + 1)

    amp_mix = amp_src.clone()
    amp_mix[..., y1:y2, x1:x2] = amp_tgt[..., y1:y2, x1:x2]
    amp_mix = torch.fft.ifftshift(amp_mix, dim=(-2, -1))

    fft_mix = torch.polar(amp_mix, pha_src)
    out = torch.fft.ifft2(fft_mix, dim=(-2, -1)).real
    return out.clamp(0.0, 1.0)


def _apply_color_adain(src_t: Any, tgt_t: Any, alpha: float) -> Any:
    """Per-channel mean/std archival ink-parchment moment transfer (AdaIN in RGB space)."""
    s = src_t.float()
    t = tgt_t.float()
    s_mean = s.mean(dim=(-2, -1), keepdim=True)
    s_std = s.std(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
    t_mean = t.mean(dim=(-2, -1), keepdim=True)
    t_std = t.std(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
    matched = (s - s_mean) / s_std * t_std + t_mean
    blended = (1.0 - float(alpha)) * s + float(alpha) * matched
    return blended.clamp(0.0, 1.0)


def _load_proxy_labeled_records(
    proxy_dir: Path,
    tok_to_id: Dict[str, int],
    bos_id: int,
    eos_id: int,
    max_seq_len: int = 1536,
) -> List[Dict[str, Any]]:
    manifest_p = proxy_dir / "proxy_train.jsonl"
    if not manifest_p.is_file():
        raise FileNotFoundError(f"Missing {manifest_p}")
    records: List[Dict[str, Any]] = []
    for raw in manifest_p.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        img_p = (proxy_dir / rec["image"]).resolve()
        if not img_p.is_file():
            continue
        bekern_paths = [(proxy_dir / b).resolve() for b in rec.get("bekerns", [])]
        toks: List[str] = []
        ok = True
        for bp in bekern_paths:
            if not bp.is_file():
                ok = False
                break
            txt = bp.read_text(encoding="utf-8").strip()
            if txt:
                toks.extend(txt.split())
        if not ok or len(toks) < 12:
            continue
        ids = [bos_id]
        for tk in toks:
            tid = tok_to_id.get(tk)
            if tid is not None:
                ids.append(tid)
        ids.append(eos_id)
        if len(ids) > max_seq_len:
            ids = ids[: max_seq_len - 1] + [eos_id]
        if len(ids) >= 16:
            records.append({"image": str(img_p), "seq_ids": ids})
    return records


def _evaluate_pieces_python(
    model: Any,
    cfg: Any,
    pieces: Sequence[Dict[str, Any]],
    device: Any,
    *,
    tag: str,
    max_dim: int = 1024,
    max_len: int = 1536,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    from PIL import Image
    import torch
    import eval_uni_suite31 as eval_s31
    import rlvr_grpo
    import uni_omr

    _, _, eos_id, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))
    model.eval()
    rows: List[Dict[str, Any]] = []
    t_all = time.perf_counter()

    for idx, piece in enumerate(pieces, start=1):
        slug = piece["slug"]
        is_hw = bool(piece["handwritten"])
        ref_notes = _extract_musicxml_notes(piece["gt"])
        prow: Dict[str, Any] = {"piece": slug, "handwritten": is_hw}
        for domain in ("clean", "photo"):
            with Image.open(piece[domain]) as raw_im:
                pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
            img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
            with torch.no_grad():
                if use_bf16 and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        seq, _ = eval_s31.generate_mtp_spec(
                            model, img_t, max_len=max_len, use_grammar=True,
                            strict_grammar=False, return_stats=True,
                        )
                else:
                    seq, _ = eval_s31.generate_mtp_spec(
                        model, img_t, max_len=max_len, use_grammar=True,
                        strict_grammar=False, return_stats=True,
                    )
            seq_ids = seq[0].detach().cpu().tolist()
            has_eos = eos_id in seq_ids[1:]
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
            pfx = domain + "_"
            prow[pfx + "note_f1_py"] = round(float(note_f1), 6)
            prow[pfx + "r_rhythm"] = round(float(struct["r_rhythm"]), 4)
            prow[pfx + "r_noloop"] = round(float(struct["r_noloop"]), 4)
        rows.append(prow)
        if idx % 10 == 0 or idx == len(pieces):
            print(f"[eval:{tag}] {idx:02d}/{len(pieces):02d} ({time.perf_counter() - t_all:.1f}s)", flush=True)

    def _agg(sub: Sequence[Dict[str, Any]], dom: str) -> Dict[str, Any]:
        p = dom + "_"
        n = len(sub)
        if n == 0:
            return {"n": 0}
        def _m(k: str) -> Optional[float]:
            vals = [float(r[p + k]) for r in sub if r.get(p + k) is not None]
            return round(sum(vals) / len(vals), 6) if vals else None
        return {
            "n": n,
            "note_f1_py_mean": _m("note_f1_py"),
            "r_rhythm_mean": _m("r_rhythm"),
            "r_noloop_mean": _m("r_noloop"),
        }

    pr_rows = [r for r in rows if not r["handwritten"]]
    hw_rows = [r for r in rows if r["handwritten"]]
    return {
        "summary": {
            "printed_clean": _agg(pr_rows, "clean"),
            "printed_photo": _agg(pr_rows, "photo"),
            "hw_clean": _agg(hw_rows, "clean"),
            "hw_photo": _agg(hw_rows, "photo"),
        },
        "rows": rows,
    }


def main() -> None:
    t0 = time.perf_counter()
    started_utc = _utc_now()
    ft_steps = 120
    grad_accum = 4
    ft_lr = 1.5e-5
    max_dim = 1024
    eval_max_len = 1536
    seed = 42

    try:
        hf_token = _read_and_purge_hf_token()
        mvd_dir = _setup_environment()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        from huggingface_hub import HfApi
        from PIL import Image, ImageEnhance
        import torch
        import torch.nn.functional as F
        import eval_uni_suite31 as eval_s31
        import uni_omr

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        use_bf16 = bool(device.type == "cuda" and cc[0] >= 8)
        env_tag = "kaggle" if Path("/kaggle").exists() else ("colab" if Path("/content").exists() else "cloud")
        platform_str = f"{env_tag}-{gpu_name}"
        print(f"[{EXP_ID}] {platform_str} (cc={cc[0]}.{cc[1]}, bf16={use_bf16})", flush=True)

        api = HfApi(token=hf_token)
        ckpt_path, proxy_dir, rism_dir, manifest_path = _ensure_assets(hf_token)

        ledger_tmp = CACHE_ROOT / "run_ledger.json"
        _atomic_write_json(ledger_tmp, {
            "exp_id": EXP_ID,
            "status": "RUNNING",
            "started_utc": started_utc,
            "platform": platform_str,
            "arms": [a["key"] for a in ARMS],
            "ft_steps": ft_steps,
        })
        try:
            api.upload_file(
                path_or_fileobj=str(ledger_tmp),
                path_in_repo=HF_RUN_PATH,
                repo_id=HF_REPO,
                repo_type="dataset",
                commit_message=f"{EXP_ID}: mark RUNNING on {platform_str}",
            )
        except Exception as exc:
            print(f"[{EXP_ID}] Warning: initial HF ledger upload failed: {exc}", flush=True)

        model, cfg, ckpt_meta = eval_s31.load_uni_checkpoint(ckpt_path, device=device)
        base_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        pad_id, bos_id, eos_id, _, tok_to_id, _ = eval_s31._vocab_spec(int(cfg.vocab_size))

        train_records = _load_proxy_labeled_records(
            proxy_dir, tok_to_id=tok_to_id, bos_id=bos_id, eos_id=eos_id, max_seq_len=eval_max_len
        )
        rism_images = sorted(
            p for p in rism_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )
        if not train_records or not rism_images:
            raise RuntimeError(f"Insufficient data: train_records={len(train_records)}, rism_images={len(rism_images)}")
        print(f"[{EXP_ID}] Loaded {len(train_records)} labeled proxy pages and {len(rism_images)} RISM style targets", flush=True)

        _, all_pieces = eval_s31.resolve_manifest_pieces(manifest_path)
        eval_pieces = _select_eval_pieces(all_pieces, n_printed=12)

        partial_path = CACHE_ROOT / "evals" / f"{EXP_ID}_partial.json"
        baseline_eval: Optional[Dict[str, Any]] = None
        arm_results: Dict[str, Any] = {}
        if partial_path.exists():
            try:
                prev = json.loads(partial_path.read_text(encoding="utf-8"))
                baseline_eval = prev.get("baseline_eval")
                arm_results = dict(prev.get("arms", {}))
            except Exception:
                pass

        if baseline_eval is None:
            baseline_eval = _evaluate_pieces_python(
                model, cfg, eval_pieces, device,
                tag="baseline", max_dim=max_dim, max_len=eval_max_len, use_bf16=use_bf16,
            )
            _atomic_write_json(partial_path, {"exp_id": EXP_ID, "baseline_eval": baseline_eval, "arms": arm_results})

        for arm in ARMS:
            key = arm["key"]
            if key in arm_results:
                print(f"[{EXP_ID}] Skipping cached arm {key}", flush=True)
                continue

            print(f"[{EXP_ID}] === Training Arm {key} (mode={arm['mode']}, beta={arm['beta']}, p={arm['prob']}) ===", flush=True)
            torch.manual_seed(seed)
            random.seed(seed)
            model.load_state_dict({k: v.to(device) for k, v in base_state_cpu.items()})
            optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=0.01)

            ft_rng = random.Random(seed)
            style_rng = random.Random(seed + 1337)
            t_ft = time.perf_counter()
            losses: List[float] = []
            n_transferred = 0
            warmup = max(1, ft_steps // 10)
            min_lr = ft_lr * 0.2

            for step in range(1, ft_steps + 1):
                if step <= warmup:
                    cur_lr = ft_lr * (step / warmup)
                else:
                    prog = (step - warmup) / max(1, ft_steps - warmup)
                    cur_lr = min_lr + 0.5 * (ft_lr - min_lr) * (1.0 + math.cos(math.pi * prog))
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr

                model.train()
                optimizer.zero_grad(set_to_none=True)
                acc_loss = 0.0
                valid_micro = 0

                for _ in range(grad_accum):
                    rec = train_records[ft_rng.randrange(len(train_records))]
                    seq_ids = rec["seq_ids"]
                    with Image.open(rec["image"]) as raw_im:
                        pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
                    pil_im = ImageEnhance.Contrast(pil_im).enhance(ft_rng.uniform(0.88, 1.12))
                    pil_im = ImageEnhance.Brightness(pil_im).enhance(ft_rng.uniform(0.94, 1.06))
                    img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)

                    if arm["mode"] in ("fda", "adain") and style_rng.random() < float(arm["prob"]):
                        tgt_p = rism_images[style_rng.randrange(len(rism_images))]
                        try:
                            with Image.open(tgt_p) as tgt_raw:
                                tgt_pil = uni_omr._resize_fullpage_pil(tgt_raw.convert("RGB"), max_dim)
                            tgt_t = uni_omr._pil_to_chw_tensor(tgt_pil).unsqueeze(0).to(device)
                            if arm["mode"] == "fda":
                                img_t = _apply_fda_2d(img_t, tgt_t, beta=float(arm["beta"]))
                            else:
                                img_t = _apply_color_adain(img_t, tgt_t, alpha=style_rng.uniform(0.4, 0.8))
                            n_transferred += 1
                        except Exception:
                            pass

                    seq_t = torch.tensor([seq_ids], dtype=torch.long, device=device)
                    if use_bf16 and device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            out = model(img_t, seq_t[:, :-1], return_aux=False, return_mtp=False)
                            logits = out["logits"].float()
                            loss = F.cross_entropy(
                                logits.reshape(-1, logits.shape[-1]),
                                seq_t[:, 1:].reshape(-1),
                                ignore_index=pad_id,
                                label_smoothing=0.02,
                            ) / float(grad_accum)
                    else:
                        out = model(img_t, seq_t[:, :-1], return_aux=False, return_mtp=False)
                        logits = out["logits"].float()
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            seq_t[:, 1:].reshape(-1),
                            ignore_index=pad_id,
                            label_smoothing=0.02,
                        ) / float(grad_accum)

                    if torch.isfinite(loss):
                        loss.backward()
                        acc_loss += float(loss.detach().item())
                        valid_micro += 1

                if valid_micro > 0:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if math.isfinite(float(gn.detach().item())):
                        optimizer.step()
                    losses.append(round(acc_loss, 5))

                if step == 1 or step % 30 == 0 or step == ft_steps:
                    print(f"[{key} {step:03d}/{ft_steps:03d}] loss={losses[-1] if losses else 0.0:.4f} lr={cur_lr:.2e}", flush=True)

            arm_eval = _evaluate_pieces_python(
                model, cfg, eval_pieces, device,
                tag=key, max_dim=max_dim, max_len=eval_max_len, use_bf16=use_bf16,
            )
            deltas_vs_base: Dict[str, Optional[float]] = {}
            for sl in ("printed_clean", "printed_photo", "hw_clean", "hw_photo"):
                b = baseline_eval["summary"][sl]["note_f1_py_mean"]
                a = arm_eval["summary"][sl]["note_f1_py_mean"]
                deltas_vs_base[sl] = round(a - b, 6) if (a is not None and b is not None) else None

            hw_avg_delta = 0.5 * ((deltas_vs_base["hw_clean"] or 0.0) + (deltas_vs_base["hw_photo"] or 0.0))
            arm_results[key] = {
                "config": arm,
                "n_transferred": n_transferred,
                "ft_elapsed_secs": round(time.perf_counter() - t_ft, 2),
                "final_loss": losses[-1] if losses else None,
                "summary": arm_eval["summary"],
                "deltas_vs_base": deltas_vs_base,
                "hw_avg_delta_vs_base": round(hw_avg_delta, 6),
            }
            _atomic_write_json(partial_path, {"exp_id": EXP_ID, "baseline_eval": baseline_eval, "arms": arm_results})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ctrl_summary = arm_results["arm_ctrl_standard"]["summary"]
        vs_ctrl: Dict[str, Any] = {}
        for key, res in arm_results.items():
            if key == "arm_ctrl_standard":
                continue
            d_ctrl = {}
            for sl in ("printed_clean", "printed_photo", "hw_clean", "hw_photo"):
                c = ctrl_summary[sl]["note_f1_py_mean"]
                a = res["summary"][sl]["note_f1_py_mean"]
                d_ctrl[sl] = round(a - c, 6) if (a is not None and c is not None) else None
            d_ctrl["hw_avg"] = round(0.5 * ((d_ctrl["hw_clean"] or 0.0) + (d_ctrl["hw_photo"] or 0.0)), 6)
            vs_ctrl[key] = d_ctrl

        report = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_utc,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.perf_counter() - t0, 2),
            "platform": platform_str,
            "bf16": use_bf16,
            "ckpt_meta": ckpt_meta,
            "ft_steps": ft_steps,
            "baseline_summary": baseline_eval["summary"],
            "arms": arm_results,
            "deltas_vs_ctrl": vs_ctrl,
        }
        eval_local = CACHE_ROOT / "evals" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(ledger_tmp, {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "platform": platform_str,
            "elapsed_secs": report["elapsed_secs"],
            "deltas_vs_ctrl": vs_ctrl,
        })

        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload unpaired RISM FDA/AdaIN sim-to-real eval",
        )
        api.upload_file(
            path_or_fileobj=str(ledger_tmp),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): mark COMPLETED",
        )
        print(f"[{EXP_ID}] COMPLETE & UPLOADED TO HF!", flush=True)
    finally:
        _purge_hf_token()


if __name__ == "__main__":
    main()

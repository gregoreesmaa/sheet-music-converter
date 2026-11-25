"""Colab L4 Runner — Academic Experiment C: GEVT (V-bekern) + 2-Step Masked Refinement.

Controlled 2-arm ablation on UniOMR v2 small_45m (BF16, seed=42, LR=3e-4, 1,000 steps each, dim=1280):
  - Arm 0 (Control):   Standard semantic bekern (vocab=237)
  - Arm 1 (Treatment): Bijective V-bekern (vocab=294) + Deterministic Gauge-Lifter + MaskedRefinerHead (581K)

Evaluates BOTH arms apples-to-apples in lifted semantic bekern space on:
  1. Held-out proxy_val.jsonl (64 pages)
  2. Unseen-Clef / Rare-Key OOD Probe (C-clef / 8vb / |fifths|>=3 vs standard Treble/Bass)
  3. suite31_hw OOD sample (20 pieces: 10 Printed + 10 Real HW, clean + photo = 40 pages)

Security: reads HF_TOKEN from /tmp/.hf_token (chmod 600), NEVER prints it, deletes it in finally.
"""

from __future__ import annotations

import difflib
import gc
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

EXP_ID = "2026-11-26T01-30-academic-gevt-vbkern-mr"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
SUITE31_TAR = "datasets/suite31_hw_eval.tar"
TOKEN_FILE = Path("/tmp/.hf_token")
WORK_DIR = Path("/content/exp_c")
CODE_DIR = WORK_DIR / "mvd"
OUT_DIR = WORK_DIR / "out"

STEPS = 1000
SEED = 42
LR = 3e-4
MIN_LR = 1.5e-5
WARMUP_STEPS = 100
BATCH_SIZE = 2
GRAD_ACCUM = 2
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
EVAL_MAX_LEN = 512
MR_LOSS_WEIGHT = 0.25


def _ensure_deps() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.26"])


def _read_hf_token() -> str:
    if TOKEN_FILE.is_file():
        try:
            os.chmod(TOKEN_FILE, 0o600)
        except OSError:
            pass
        tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        raise RuntimeError("HF_TOKEN missing (expected /tmp/.hf_token or env HF_TOKEN)")
    return tok


def _unpack_code_bundle() -> None:
    bundle = Path("/content/bundle.tar.gz")
    if not bundle.is_file():
        raise FileNotFoundError(f"Expected uploaded bundle at {bundle}")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tf:
        tf.extractall(path=WORK_DIR)
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    print(f"[setup] Unpacked code bundle to {CODE_DIR}", flush=True)


def _download_and_unpack(hf_token: str) -> Tuple[Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = WORK_DIR / "datasets" / "iter10_5_proxy_2k"
    s31_dir = WORK_DIR / "datasets" / "suite31_hw"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"

    if not (train_abs.is_file() and val_abs.is_file()):
        ds_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        print(f"[data] Downloading {PROXY_TAR}...", flush=True)
        local_tar = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=PROXY_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        with tarfile.open(local_tar, "r") as tf:
            tf.extractall(path=ds_dir)
        try:
            os.remove(local_tar)
        except OSError:
            pass
        for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
            src_path = ds_dir / src_name
            out_lines: List[str] = []
            for raw in src_path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                rec["image"] = str((ds_dir / rec["image"]).resolve())
                rec["bekerns"] = [str((ds_dir / b).resolve()) for b in rec.get("bekerns", [])]
                out_lines.append(json.dumps(rec))
            dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"[data] Proxy dataset ready in {time.perf_counter() - t0:.1f}s", flush=True)

    if not (s31_dir / "suite31" / "suite31_hw_manifest.json").is_file():
        s31_dir.mkdir(parents=True, exist_ok=True)
        t1 = time.perf_counter()
        print(f"[data] Downloading {SUITE31_TAR}...", flush=True)
        local_s31 = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=SUITE31_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        with tarfile.open(local_s31, "r") as tf:
            tf.extractall(path=s31_dir)
        try:
            os.remove(local_s31)
        except OSError:
            pass
        print(f"[data] suite31_hw ready in {time.perf_counter() - t1:.1f}s", flush=True)

    return train_abs, val_abs, s31_dir / "suite31"


def _load_manifest_records(manifest_path: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        img_p = Path(rec["image"])
        beks = [Path(b) for b in rec.get("bekerns", []) if Path(b).is_file()]
        if img_p.is_file() and beks:
            recs.append({"image": img_p, "bekerns": beks, "domain": str(rec.get("domain", ""))})
    return recs


def _read_page_bekern(bek_paths: Sequence[Path]) -> List[str]:
    import uni_omr

    toks: List[str] = []
    for idx, bp in enumerate(bek_paths):
        if idx > 0:
            toks.append(uni_omr.SYS_TOKEN)
        raw = bp.read_text(encoding="utf-8").strip()
        if raw:
            toks.extend(raw.split())
    return toks


def _is_rare_clef_or_key(bekern_toks: Sequence[str]) -> bool:
    """True if page contains C-clef, octave-transposing clef, or key signature with |fifths| >= 3."""
    import vbkern as vb

    for t in bekern_toks:
        if t.startswith("clef-C-") or t.endswith("-8vb") or t.endswith("-8va"):
            return True
        m = vb.KEY_PATTERN.match(t)
        if m and abs(int(m.group(1))) >= 3:
            return True
    return False


def _token_lcs_f1(ref: Sequence[str], hyp: Sequence[str]) -> float:
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    matches = sum(b.size for b in sm.get_matching_blocks())
    return (2.0 * matches) / (len(ref) + len(hyp))


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    prog = float(step - warmup) / float(max(1, total_steps - warmup))
    prog = max(0.0, min(1.0, prog))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


def _make_dataset_and_collate(
    records: List[Dict[str, Any]],
    *,
    mode: str,
    max_dim: int,
    max_seq_len: int,
):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    import uni_omr
    import vbkern as vb

    pad_id = uni_omr.PAD_ID if mode == "semantic" else vb.VB_PAD_ID

    class _ExpCDataset(Dataset):
        def __init__(self, items: List[Dict[str, Any]]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            rec = self.items[idx]
            with Image.open(rec["image"]) as im:
                pil = uni_omr._resize_fullpage_pil(im.convert("RGB"), max_dim)
            img_t = uni_omr._pil_to_chw_tensor(pil)  # float32 [0,1] (3, H, W)
            sem_toks = _read_page_bekern(rec["bekerns"])
            if mode == "semantic":
                ids = [uni_omr.BOS_ID]
                for t in sem_toks:
                    tid = uni_omr.UNI_TOKEN_TO_ID.get(t)
                    if tid is not None:
                        ids.append(tid)
                ids = ids[: max_seq_len - 1] + [uni_omr.EOS_ID]
            else:
                vb_toks = vb.bekern_to_vbkern(sem_toks)
                ids = [vb.VB_BOS_ID]
                for t in vb_toks:
                    tid = vb.VBKERN_TOKEN_TO_ID.get(t)
                    if tid is not None:
                        ids.append(tid)
                ids = ids[: max_seq_len - 1] + [vb.VB_EOS_ID]
            return {
                "image": img_t,
                "seq": torch.tensor(ids, dtype=torch.long),
                "sem_toks": sem_toks,
                "domain": rec["domain"],
            }

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_h = max(int(b["image"].shape[1]) for b in batch)
        max_w = max(int(b["image"].shape[2]) for b in batch)
        # Round up to multiple of 16 for backbone/merger
        max_h = ((max_h + 15) // 16) * 16
        max_w = ((max_w + 15) // 16) * 16
        max_t = max(int(b["seq"].shape[0]) for b in batch)

        imgs = torch.ones((len(batch), 3, max_h, max_w), dtype=torch.float32)
        seqs = torch.full((len(batch), max_t), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            _, h, w = b["image"].shape
            imgs[i, :, :h, :w] = b["image"]
            t = b["seq"].shape[0]
            seqs[i, :t] = b["seq"]
        return {
            "images": imgs,
            "seq": seqs,
            "sem_toks": [b["sem_toks"] for b in batch],
            "domains": [b["domain"] for b in batch],
        }

    return _ExpCDataset(records), _collate


def _compute_arm_loss(
    model,
    refiner,
    images,
    seq,
    *,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from uni_omr_v2 import compute_stroke_hog_loss

    inp = seq[:, :-1]
    tgt = seq[:, 1:]

    captured_h: List[torch.Tensor] = []

    def _hook(_mod, _inp, output):
        captured_h.append(output)

    handle = model.dec_norm.register_forward_hook(_hook) if refiner is not None else None
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(images, inp, return_aux=True, return_mtp=True)
            logits = out["logits"]
            V = logits.shape[-1]
            ce_loss = F.cross_entropy(
                logits.reshape(-1, V),
                tgt.reshape(-1),
                ignore_index=pad_id,
                label_smoothing=0.05,
            )

            mtp_terms = []
            T = tgt.shape[1]
            for idx, m_logits in enumerate(out.get("mtp_logits", [])):
                shift = idx + 1
                if T > shift:
                    m_loss = F.cross_entropy(
                        m_logits[:, :-shift, :].reshape(-1, V),
                        tgt[:, shift:].reshape(-1),
                        ignore_index=pad_id,
                        label_smoothing=0.05,
                    )
                    mtp_terms.append(torch.nan_to_num(m_loss, nan=0.0))
            mtp_loss = torch.stack(mtp_terms).mean() if mtp_terms else logits.new_zeros(())

            rinfo = out.get("router_info") or {}
            sal_logits = rinfo.get("saliency_logits")
            ink_target = rinfo.get("ink_target")
            if sal_logits is not None and ink_target is not None:
                sal_loss = F.binary_cross_entropy_with_logits(
                    sal_logits.float(), ink_target.float()
                ).to(logits.dtype)
            else:
                sal_loss = logits.new_zeros(())

            hog_pred = out.get("stroke_hog_pred")
            if hog_pred is not None:
                hog_loss = compute_stroke_hog_loss(hog_pred, images)
            else:
                hog_loss = logits.new_zeros(())

            if refiner is not None and captured_h:
                h_dec = captured_h[0]  # (B, T, d_model) aligned with inp -> predicts tgt
                mr_loss = refiner.compute_refinement_loss(
                    h_dec,
                    tgt,
                    model.lm_head,
                    pad_id=pad_id,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    mask_prob=0.15,
                    label_smoothing=0.05,
                )
            else:
                mr_loss = logits.new_zeros(())

            total = ce_loss + 0.30 * mtp_loss + 0.10 * sal_loss + 0.15 * hog_loss + MR_LOSS_WEIGHT * mr_loss
    finally:
        if handle is not None:
            handle.remove()

    return {
        "loss": total,
        "ce_loss": ce_loss,
        "mtp_loss": mtp_loss,
        "mr_loss": mr_loss,
    }


@torch.no_grad()
def _generate_greedy_with_probs(
    model,
    images,
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int = EVAL_MAX_LEN,
    use_bf16: bool = True,
):
    import torch
    import torch.nn.functional as F

    device = images.device
    B = images.shape[0]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        mem, _, _ = model.encode_image(images, return_aux=False)
        past_kvs = model.precompute_cross_kv(mem)
        cur_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        out_tokens = []
        out_probs = []
        for _ in range(max_len):
            logits, _, past_kvs = model.decode_tokens(
                cur_ids, memory=None, return_mtp=False, return_aux=False,
                past_kvs=past_kvs, use_cache=True,
            )
            step_probs = F.softmax(logits[:, -1, :].float(), dim=-1)
            conf, next_tok = step_probs.max(dim=-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)
            conf = torch.where(finished, torch.ones_like(conf), conf)
            out_tokens.append(next_tok.unsqueeze(1))
            out_probs.append(conf.unsqueeze(1))
            finished = finished | (next_tok == eos_id)
            if bool(finished.all()):
                break
            cur_ids = next_tok.unsqueeze(1)

    draft_ids = torch.cat(out_tokens, dim=1) if out_tokens else torch.full((B, 1), eos_id, dtype=torch.long, device=device)
    draft_probs = torch.cat(out_probs, dim=1) if out_probs else torch.ones((B, 1), dtype=torch.float32, device=device)
    return draft_ids, draft_probs, mem


@torch.no_grad()
def _refine_draft(
    model,
    refiner,
    mem,
    draft_ids,
    draft_probs,
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    use_bf16: bool = True,
):
    import torch

    B = draft_ids.shape[0]
    bos_col = torch.full((B, 1), bos_id, dtype=torch.long, device=draft_ids.device)
    dec_inp = torch.cat([bos_col, draft_ids[:, :-1]], dim=1)
    captured_h: List[torch.Tensor] = []

    def _hook(_mod, _inp, output):
        captured_h.append(output)

    handle = model.dec_norm.register_forward_hook(_hook)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            _ = model.decode_tokens(dec_inp, mem, return_mtp=False, return_aux=False, use_cache=False)
            h_dec = captured_h[0]
            refined_ids, mr_stats = refiner.refine_2step(
                h_dec,
                draft_ids,
                draft_probs,
                model.lm_head,
                pad_id=pad_id,
                bos_id=bos_id,
                eos_id=eos_id,
            )
    finally:
        handle.remove()
    return refined_ids, mr_stats


def _score_predictions(pred_sem_list: List[List[str]], gt_sem_list: List[List[str]]) -> Dict[str, Any]:
    import numpy as np
    import vbkern as vb

    f1s, lcs_f1s, pitch_accs, rhy_accs, note_accs = [], [], [], [], []
    tot_pitch_err = 0
    tot_gauge_err = 0
    tot_vis_sp_err = 0
    tot_vis_acc_err = 0

    for pred, gt in zip(pred_sem_list, gt_sem_list):
        f1s.append(vb.compute_token_f1_multiset(pred, gt))
        lcs_f1s.append(_token_lcs_f1(gt, pred))
        dec = vb.gauge_error_decomposition(pred, gt)
        pitch_accs.append(dec["pitch_acc"])
        rhy_accs.append(dec["rhythm_acc"])
        note_accs.append(dec["note_exact_acc"])
        tot_pitch_err += dec["total_pitch_errors"]
        tot_gauge_err += dec["gauge_transport_errors"]
        tot_vis_sp_err += dec["visual_sp_errors"]
        tot_vis_acc_err += dec["visual_acc_errors"]

    return {
        "n_pages": len(gt_sem_list),
        "token_f1": round(float(np.mean(f1s)) if f1s else 0.0, 5),
        "token_lcs_f1": round(float(np.mean(lcs_f1s)) if lcs_f1s else 0.0, 5),
        "pitch_acc": round(float(np.mean(pitch_accs)) if pitch_accs else 0.0, 5),
        "rhythm_acc": round(float(np.mean(rhy_accs)) if rhy_accs else 0.0, 5),
        "note_exact_acc": round(float(np.mean(note_accs)) if note_accs else 0.0, 5),
        "total_pitch_errors": int(tot_pitch_err),
        "gauge_transport_errors": int(tot_gauge_err),
        "visual_sp_errors": int(tot_vis_sp_err),
        "visual_acc_errors": int(tot_vis_acc_err),
        "gauge_error_share": round(tot_gauge_err / max(1, tot_pitch_err), 5),
    }


def _evaluate_arm(
    model,
    refiner,
    val_records: List[Dict[str, Any]],
    s31_dir: Path,
    s31_refs: Dict[str, List[str]],
    *,
    mode: str,
    device,
) -> Dict[str, Any]:
    import torch
    from PIL import Image
    import uni_omr
    import vbkern as vb

    model.eval()
    if refiner is not None:
        refiner.eval()

    pad_id = uni_omr.PAD_ID if mode == "semantic" else vb.VB_PAD_ID
    bos_id = uni_omr.BOS_ID if mode == "semantic" else vb.VB_BOS_ID
    eos_id = uni_omr.EOS_ID if mode == "semantic" else vb.VB_EOS_ID

    def _decode_ids_to_sem(id_row: List[int]) -> List[str]:
        if mode == "semantic":
            return uni_omr.decode_uni(id_row, skip_special=True)
        vb_toks = vb.decode_vbkern(id_row, skip_special=True)
        return vb.vbkern_to_bekern(vb_toks)

    # 1. Held-out proxy_val.jsonl (64 pages) + Rare-Clef/Key stratification
    val_gt_all: List[List[str]] = []
    val_pred_greedy: List[List[str]] = []
    val_pred_mr2: List[List[str]] = []
    val_is_rare: List[bool] = []
    mr_changed_total = 0
    mr_masked_total = 0
    gen_toks_total = 0
    t_gen0 = time.perf_counter()

    for idx_v, rec in enumerate(val_records, start=1):
        with Image.open(rec["image"]) as im:
            pil = uni_omr._resize_fullpage_pil(im.convert("RGB"), FULLPAGE_MAX_DIM)
        img_t = uni_omr._pil_to_chw_tensor(pil).unsqueeze(0).to(device)
        gt_sem = _read_page_bekern(rec["bekerns"])
        val_gt_all.append(gt_sem)
        val_is_rare.append(_is_rare_clef_or_key(gt_sem))

        draft_ids, draft_probs, mem = _generate_greedy_with_probs(
            model, img_t, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id, max_len=EVAL_MAX_LEN
        )
        gen_toks_total += int(draft_ids.shape[1])
        val_pred_greedy.append(_decode_ids_to_sem(draft_ids[0].tolist()))

        if refiner is not None:
            ref_ids, mrs = _refine_draft(
                model, refiner, mem, draft_ids, draft_probs,
                bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
            )
            mr_changed_total += int(mrs["total_changed"])
            mr_masked_total += int(mrs["total_masked"])
            val_pred_mr2.append(_decode_ids_to_sem(ref_ids[0].tolist()))
        else:
            val_pred_mr2.append(val_pred_greedy[-1])
        if idx_v % 16 == 0 or idx_v == len(val_records):
            print(f"  [eval val] {idx_v}/{len(val_records)} pages ({time.perf_counter()-t_gen0:.1f}s)", flush=True)

    val_gen_secs = time.perf_counter() - t_gen0

    rare_idx = [i for i, r in enumerate(val_is_rare) if r]
    std_idx = [i for i, r in enumerate(val_is_rare) if not r]

    val_summary = {
        "all64_greedy": _score_predictions(val_pred_greedy, val_gt_all),
        "all64_final": _score_predictions(val_pred_mr2, val_gt_all),
        "rare_clef_or_key": _score_predictions(
            [val_pred_mr2[i] for i in rare_idx], [val_gt_all[i] for i in rare_idx]
        ),
        "standard_clef_and_key": _score_predictions(
            [val_pred_mr2[i] for i in std_idx], [val_gt_all[i] for i in std_idx]
        ),
        "gen_secs": round(val_gen_secs, 2),
        "gen_toks_per_sec": round(gen_toks_total / max(1e-6, val_gen_secs), 2),
        "mr_masked_total": mr_masked_total,
        "mr_changed_total": mr_changed_total,
    }

    # 2. suite31_hw OOD sample: 10 Printed (incl. C-clef & 8vb) + 10 Real HW, clean + photo = 40 pages
    manifest = json.loads((s31_dir / "suite31_hw_manifest.json").read_text(encoding="utf-8"))
    pieces = manifest["pieces"]
    # Pick C-clef/rare-key printed pieces first + standard printed, plus 10 real HW
    pr_pieces = [p for p in pieces if not p.get("handwritten") and p["slug"] in s31_refs]
    hw_pieces = [p for p in pieces if p.get("handwritten") and p["slug"] in s31_refs]
    pr_rare = [p for p in pr_pieces if _is_rare_clef_or_key(s31_refs[p["slug"]])]
    pr_std = [p for p in pr_pieces if not _is_rare_clef_or_key(s31_refs[p["slug"]])]
    selected_pr = (pr_rare[:6] + pr_std[:4])[:10]
    selected_hw = hw_pieces[:10]
    selected = selected_pr + selected_hw

    buckets: Dict[str, Tuple[List[List[str]], List[List[str]]]] = {
        "printed_clean": ([], []),
        "printed_photo": ([], []),
        "hw_clean": ([], []),
        "hw_photo": ([], []),
        "s31_rare_clef_key": ([], []),
        "s31_std_clef_key": ([], []),
        "all_greedy": ([], []),
        "all_final": ([], []),
    }

    for idx_s, p in enumerate(selected, start=1):
        slug = p["slug"]
        gt_sem = s31_refs[slug]
        is_hw = bool(p.get("handwritten"))
        is_rare = _is_rare_clef_or_key(gt_sem)
        for dom in ("clean", "photo"):
            img_path = s31_dir / p[dom]["file"]
            if not img_path.is_file():
                continue
            with Image.open(img_path) as im:
                pil = uni_omr._resize_fullpage_pil(im.convert("RGB"), FULLPAGE_MAX_DIM)
            img_t = uni_omr._pil_to_chw_tensor(pil).unsqueeze(0).to(device)

            draft_ids, draft_probs, mem = _generate_greedy_with_probs(
                model, img_t, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id, max_len=EVAL_MAX_LEN
            )
            pred_g = _decode_ids_to_sem(draft_ids[0].tolist())
            if refiner is not None:
                ref_ids, _ = _refine_draft(
                    model, refiner, mem, draft_ids, draft_probs,
                    bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
                )
                pred_f = _decode_ids_to_sem(ref_ids[0].tolist())
            else:
                pred_f = pred_g

            key = f"{'hw' if is_hw else 'printed'}_{dom}"
            buckets[key][0].append(pred_f)
            buckets[key][1].append(gt_sem)
            buckets["all_greedy"][0].append(pred_g)
            buckets["all_greedy"][1].append(gt_sem)
            buckets["all_final"][0].append(pred_f)
            buckets["all_final"][1].append(gt_sem)
            rk = "s31_rare_clef_key" if is_rare else "s31_std_clef_key"
            buckets[rk][0].append(pred_f)
            buckets[rk][1].append(gt_sem)
        if idx_s % 5 == 0 or idx_s == len(selected):
            print(f"  [eval suite31_hw] {idx_s}/{len(selected)} pieces", flush=True)

    s31_summary = {k: _score_predictions(v[0], v[1]) for k, v in buckets.items()}
    s31_summary["selected_slugs"] = [p["slug"] for p in selected]

    return {
        "proxy_val": val_summary,
        "suite31_hw_sample": s31_summary,
    }


def _run_arm(
    arm_name: str,
    mode: str,
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    s31_dir: Path,
    s31_refs: Dict[str, List[str]],
    device,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import uni_omr
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net
    import vbkern as vb

    print(f"\n==================== {arm_name} ({mode}) ====================", flush=True)
    _set_seed(SEED)

    vocab_size = uni_omr.VOCAB_SIZE if mode == "semantic" else vb.VB_VOCAB_SIZE
    pad_id = uni_omr.PAD_ID if mode == "semantic" else vb.VB_PAD_ID
    bos_id = uni_omr.BOS_ID if mode == "semantic" else vb.VB_BOS_ID
    eos_id = uni_omr.EOS_ID if mode == "semantic" else vb.VB_EOS_ID

    cfg = UniOMRv2Config.small_45m(
        vocab_size=vocab_size,
        grad_checkpoint=True,
        use_sys_y_aux=False,
        use_sys_count_aux=False,
        use_staff_seg_aux=False,
        use_stroke_hog_head=True,
    )
    model = UniOMRv2Net(cfg).to(device)
    model = model.to(memory_format=torch.channels_last)
    refiner = (
        vb.MaskedRefinerHead(d_model=cfg.d_model, vocab_size=vocab_size).to(device)
        if mode == "vbkern_mr"
        else None
    )

    params = list(model.parameters()) + (list(refiner.parameters()) if refiner is not None else [])
    n_total_params = sum(p.numel() for p in params)
    n_refiner_params = refiner.num_params() if refiner is not None else 0
    print(f"[{arm_name}] vocab={vocab_size} | total_params={n_total_params:,} | refiner_params={n_refiner_params:,}", flush=True)

    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.05, betas=(0.9, 0.95), fused=True)
    train_ds, collate_fn = _make_dataset_and_collate(
        train_records, mode=mode, max_dim=FULLPAGE_MAX_DIM, max_seq_len=MAX_SEQ_LEN
    )
    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    arm_out = OUT_DIR / arm_name
    arm_out.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    if refiner is not None:
        refiner.train()

    data_iter = iter(loader)
    history: List[Dict[str, float]] = []
    t_start = time.perf_counter()
    last_ckpt_t = t_start
    seq_lens: List[int] = []

    for step in range(1, STEPS + 1):
        lr_now = _cosine_lr(step - 1, STEPS, LR, MIN_LR, WARMUP_STEPS)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        opt.zero_grad(set_to_none=True)

        step_loss = 0.0
        step_ce = 0.0
        step_mr = 0.0
        for _ in range(GRAD_ACCUM):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            images = batch["images"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            seq = batch["seq"].to(device, non_blocking=True)
            seq_lens.append(int(seq.shape[1]))
            losses = _compute_arm_loss(
                model, refiner, images, seq,
                pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, use_bf16=True,
            )
            (losses["loss"] / GRAD_ACCUM).backward()
            step_loss += float(losses["loss"].item()) / GRAD_ACCUM
            step_ce += float(losses["ce_loss"].item()) / GRAD_ACCUM
            step_mr += float(losses["mr_loss"].item()) / GRAD_ACCUM

        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item())
        opt.step()
        history.append({"step": step, "loss": step_loss, "ce": step_ce, "mr": step_mr, "gnorm": gnorm, "lr": lr_now})

        if step == 1 or step % 25 == 0 or step == STEPS:
            el = time.perf_counter() - t_start
            print(
                f"[{arm_name}] step {step:4d}/{STEPS} | loss={step_loss:.4f} ce={step_ce:.4f} "
                f"mr={step_mr:.4f} gnorm={gnorm:.2f} lr={lr_now:.2e} | {el:.1f}s ({step/max(1e-6,el):.2f} it/s)",
                flush=True,
            )

        if (time.perf_counter() - last_ckpt_t) >= 300.0 or step == STEPS:
            ckpt_tmp = arm_out / "latest.pt.tmp"
            ckpt_dst = arm_out / "latest.pt"
            payload = {
                "step": step,
                "mode": mode,
                "model": model.state_dict(),
                "refiner": refiner.state_dict() if refiner is not None else None,
            }
            torch.save(payload, ckpt_tmp)
            os.replace(ckpt_tmp, ckpt_dst)
            last_ckpt_t = time.perf_counter()

    train_secs = time.perf_counter() - t_start
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[{arm_name}] Training complete in {train_secs:.1f}s | peak_vram={peak_vram_gb:.2f} GB | Evaluating...", flush=True)

    eval_res = _evaluate_arm(
        model, refiner, val_records, s31_dir, s31_refs, mode=mode, device=device
    )
    print(
        f"[{arm_name}] Eval DONE | val_tok_f1={eval_res['proxy_val']['all64_final']['token_f1']:.4f} "
        f"val_pitch={eval_res['proxy_val']['all64_final']['pitch_acc']:.4f} "
        f"rare_f1={eval_res['proxy_val']['rare_clef_or_key']['token_f1']:.4f} "
        f"s31_all_f1={eval_res['suite31_hw_sample']['all_final']['token_f1']:.4f}",
        flush=True,
    )

    # Summarize per-200-step buckets
    buckets = []
    for s0 in range(0, len(history), 200):
        sl = history[s0 : s0 + 200]
        buckets.append({
            "step_start": int(sl[0]["step"]),
            "step_end": int(sl[-1]["step"]),
            "loss_mean": round(float(np.mean([r["loss"] for r in sl])), 5),
            "ce_mean": round(float(np.mean([r["ce"] for r in sl])), 5),
            "mr_mean": round(float(np.mean([r["mr"] for r in sl])), 5),
            "gnorm_mean": round(float(np.mean([r["gnorm"] for r in sl])), 4),
        })

    del model, refiner, opt
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "arm": arm_name,
        "mode": mode,
        "vocab_size": vocab_size,
        "total_params": n_total_params,
        "refiner_params": n_refiner_params,
        "steps": STEPS,
        "train_secs": round(train_secs, 2),
        "steps_per_sec": round(STEPS / max(1e-6, train_secs), 3),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "mean_batch_seq_len": round(float(np.mean(seq_lens)), 1),
        "final_100_ce": round(float(np.mean([r["ce"] for r in history[-100:]])), 5),
        "per_200_steps": buckets,
        "eval": eval_res,
    }


def main() -> None:
    import torch
    from huggingface_hub import HfApi

    _ensure_deps()
    hf_token = _read_hf_token()
    try:
        print(
            f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
            f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
            flush=True,
        )
        _unpack_code_bundle()
        train_abs, val_abs, s31_dir = _download_and_unpack(hf_token)

        s31_refs_path = WORK_DIR / "suite31_hw_refs.json"
        s31_refs: Dict[str, List[str]] = json.loads(s31_refs_path.read_text(encoding="utf-8"))
        train_records = _load_manifest_records(train_abs)
        val_records = _load_manifest_records(val_abs)
        print(
            f"[data] train={len(train_records)} pages | val={len(val_records)} pages | "
            f"s31_refs={len(s31_refs)} pieces",
            flush=True,
        )

        # CPU round-trip & compression summary on all 45 suite31_hw refs
        import vbkern as vb
        rt_exact = 0
        tot_sem = 0
        tot_vb = 0
        for slug, toks in s31_refs.items():
            v = vb.bekern_to_vbkern(toks)
            rt = vb.vbkern_to_bekern(v)
            tot_sem += len(toks)
            tot_vb += len(v)
            if rt == toks:
                rt_exact += 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arm0 = _run_arm("arm0_semantic_bekern", "semantic", train_records, val_records, s31_dir, s31_refs, device)
        arm1 = _run_arm("arm1_gevt_vbkern_mr", "vbkern_mr", train_records, val_records, s31_dir, s31_refs, device)

        report = {
            "experiment_id": EXP_ID,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "torch_version": torch.__version__,
            "hyperparams": {
                "preset": "small_45m",
                "steps": STEPS,
                "seed": SEED,
                "lr": LR,
                "min_lr": MIN_LR,
                "warmup_steps": WARMUP_STEPS,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "fullpage_max_dim": FULLPAGE_MAX_DIM,
                "max_seq_len": MAX_SEQ_LEN,
                "eval_max_len": EVAL_MAX_LEN,
                "bf16": True,
                "mr_loss_weight": MR_LOSS_WEIGHT,
            },
            "roundtrip_audit_suite31_hw": {
                "n_pieces": len(s31_refs),
                "exact_pieces": rt_exact,
                "total_bekern_tokens": tot_sem,
                "total_vbkern_tokens": tot_vb,
                "compression_ratio": round(tot_vb / max(1, tot_sem), 5),
                "pct_shorter": round((1.0 - tot_vb / max(1, tot_sem)) * 100.0, 2),
            },
            "arm0_semantic_bekern": arm0,
            "arm1_gevt_vbkern_mr": arm1,
            "deltas_arm1_minus_arm0": {
                "val_token_f1": round(
                    arm1["eval"]["proxy_val"]["all64_final"]["token_f1"]
                    - arm0["eval"]["proxy_val"]["all64_final"]["token_f1"],
                    5,
                ),
                "val_pitch_acc": round(
                    arm1["eval"]["proxy_val"]["all64_final"]["pitch_acc"]
                    - arm0["eval"]["proxy_val"]["all64_final"]["pitch_acc"],
                    5,
                ),
                "val_rhythm_acc": round(
                    arm1["eval"]["proxy_val"]["all64_final"]["rhythm_acc"]
                    - arm0["eval"]["proxy_val"]["all64_final"]["rhythm_acc"],
                    5,
                ),
                "val_rare_clef_key_token_f1": round(
                    arm1["eval"]["proxy_val"]["rare_clef_or_key"]["token_f1"]
                    - arm0["eval"]["proxy_val"]["rare_clef_or_key"]["token_f1"],
                    5,
                ),
                "val_rare_clef_key_pitch_acc": round(
                    arm1["eval"]["proxy_val"]["rare_clef_or_key"]["pitch_acc"]
                    - arm0["eval"]["proxy_val"]["rare_clef_or_key"]["pitch_acc"],
                    5,
                ),
                "s31_all_token_f1": round(
                    arm1["eval"]["suite31_hw_sample"]["all_final"]["token_f1"]
                    - arm0["eval"]["suite31_hw_sample"]["all_final"]["token_f1"],
                    5,
                ),
                "s31_printed_clean_token_f1": round(
                    arm1["eval"]["suite31_hw_sample"]["printed_clean"]["token_f1"]
                    - arm0["eval"]["suite31_hw_sample"]["printed_clean"]["token_f1"],
                    5,
                ),
                "s31_hw_clean_token_f1": round(
                    arm1["eval"]["suite31_hw_sample"]["hw_clean"]["token_f1"]
                    - arm0["eval"]["suite31_hw_sample"]["hw_clean"]["token_f1"],
                    5,
                ),
                "s31_rare_clef_key_token_f1": round(
                    arm1["eval"]["suite31_hw_sample"]["s31_rare_clef_key"]["token_f1"]
                    - arm0["eval"]["suite31_hw_sample"]["s31_rare_clef_key"]["token_f1"],
                    5,
                ),
                "mr2_val_token_f1_gain": round(
                    arm1["eval"]["proxy_val"]["all64_final"]["token_f1"]
                    - arm1["eval"]["proxy_val"]["all64_greedy"]["token_f1"],
                    5,
                ),
                "train_speedup_x": round(arm0["train_secs"] / max(1e-6, arm1["train_secs"]), 3),
            },
        }

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        local_json = OUT_DIR / f"{EXP_ID}.json"
        local_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

        api = HfApi(token=hf_token)
        for remote_path in (f"evals/{EXP_ID}.json", f"runs/{EXP_ID}.json"):
            api.upload_file(
                path_or_fileobj=str(local_json),
                path_in_repo=remote_path,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"exp({EXP_ID}): upload controlled 2-arm GEVT V-bekern + 2-step MR results",
            )
            print(f"[hf] Uploaded {remote_path}", flush=True)

        print("\n=== FINAL DELTAS (Arm 1 V-bekern+MR minus Arm 0 Semantic bekern) ===", flush=True)
        print(json.dumps(report["deltas_arm1_minus_arm0"], indent=2), flush=True)
    finally:
        if TOKEN_FILE.is_file():
            try:
                TOKEN_FILE.unlink()
                print("[security] Deleted /tmp/.hf_token", flush=True)
            except OSError:
                pass


if __name__ == "__main__":
    main()

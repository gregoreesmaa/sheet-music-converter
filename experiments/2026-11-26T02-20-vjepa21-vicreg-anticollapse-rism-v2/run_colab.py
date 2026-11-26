#!/usr/bin/env python3
"""Google Colab L4 runner for Exp 2b: V-JEPA 2.1 + VICReg Anti-Collapse + KL-Router RISM SSL & Transfer Probe.

Experiment ID: 2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2
DAG successor to: 2026-11-26T00-54-vjepa21-gram-rism-v2 (ITERATE: p3/mem rank collapse)

Stage 1: 600-step RISM 5,707 SSL pretraining with anti-collapse fixes:
  --vicreg-var-weight 1.0 --vicreg-cov-weight 0.04 --router-loss-mode kl --disable-random-gram
Stage 2: Effective-rank health probe (p2/p3/mem_erank, cross_page_cos) on 6 RISM pages.
Stage 3: Matched 800-step UniOMR v2 small_45m supervised transfer probe on iter10_5_proxy_2k
  (arm_a_scratch vs arm_b_vjepa_vicreg_warmstart) evaluated on 45-piece suite31_hw.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_RISM_TAR = "datasets/rism_hw_5707.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")

SSL_STEPS = 600
TRAIN_STEPS = 800
BATCH_SIZE = 2
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
LR = 3e-4
WARMUP_STEPS = 80
EVAL_EVERY = 400
SAVE_EVERY_MINUTES = 5.0
SUITE31_MAX_DIM = 1280
SUITE31_MAX_LEN = 768
SEED = 42


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str] | str, *, shell: bool = False, env: Optional[Dict[str, str]] = None) -> None:
    label = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"[{EXP_ID}] $ {label}", flush=True)
    proc = subprocess.Popen(
        cmd if shell else list(cmd),
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
        raise RuntimeError(f"Command failed (rc={rc}): {label}")


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and HF_TOKEN_FILE.exists():
        token = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var")
    return token


def _locate_or_extract_repo_root() -> Path:
    candidates = [
        WORKSPACE_ROOT / "sheet-music-converter",
        Path("/content/sheet-music-converter"),
        Path.cwd(),
    ]
    for cand in candidates:
        if (cand / "tmp-train" / "mvd" / "pretrain_ijepa_rism_v2.py").is_file():
            return cand
    for bundle in (Path("/content/repo_bundle.tar.gz"), Path("/content/bundle.tar.gz")):
        if bundle.is_file():
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORKSPACE_ROOT)
            for cand in candidates:
                if (cand / "tmp-train" / "mvd" / "pretrain_ijepa_rism_v2.py").is_file():
                    return cand
    raise FileNotFoundError("Could not locate sheet-music-converter/tmp-train/mvd/pretrain_ijepa_rism_v2.py")


def _ensure_datasets(hf_token: str) -> Tuple[Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_root = VOL_ROOT / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)

    # 1. RISM 5,707
    rism_dir = ds_root / "rism_hw_5707"
    if not (rism_dir / ".unpacked_ok").is_file():
        rism_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(HF_REPO_ID, HF_RISM_TAR, repo_type="dataset", token=hf_token, local_dir="/tmp/hf_dl")
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=rism_dir)
        os.remove(tar_p)
        (rism_dir / ".unpacked_ok").write_text("ok\n", encoding="utf-8")

    # 2. iter10_5_proxy_2k
    proxy_dir = ds_root / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    if not train_abs.is_file():
        proxy_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(HF_REPO_ID, HF_PROXY_TAR, repo_type="dataset", token=hf_token, local_dir="/tmp/hf_dl")
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=proxy_dir)
        os.remove(tar_p)
        out_lines = []
        for raw in (proxy_dir / "proxy_train.jsonl").read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            rec["image"] = str((proxy_dir / rec["image"]).resolve())
            rec["bekerns"] = [str((proxy_dir / b).resolve()) for b in rec.get("bekerns", [])]
            out_lines.append(json.dumps(rec))
        train_abs.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # 3. suite31_hw_eval
    suite31_dir = ds_root / "suite31_hw_eval"
    manifest_path = suite31_dir / "suite31" / "suite31_hw_manifest.json"
    if not manifest_path.is_file():
        suite31_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(HF_REPO_ID, HF_SUITE31_TAR, repo_type="dataset", token=hf_token, local_dir="/tmp/hf_dl")
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=suite31_dir)
        os.remove(tar_p)

    return rism_dir, train_abs, manifest_path


def _probe_ssl_rank(mvd_dir: Path, ckpt_path: Path, rism_dir: Path) -> Dict[str, float]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(mvd_dir))
    from pretrain_ijepa_rism_v2 import discover_unlabeled_images, load_image_resized
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net

    imgs = discover_unlabeled_images(rism_dir)[:6]
    tensors = [load_image_resized(p, 512).unsqueeze(0) for p in imgs]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = UniOMRv2Net(UniOMRv2Config.small_45m())
    net.load_state_dict(ckpt["model"], strict=False)
    net.eval()

    def _erank(mat: torch.Tensor) -> float:
        c = mat - mat.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(c)
        pr = (s * s) / ((s * s).sum() + 1e-12)
        return float(math.exp(float(-(pr * (pr + 1e-12).log()).sum().item())))

    p2_r, p3_r, mem_r, stds, wcos, vecs = [], [], [], [], [], []
    with torch.no_grad():
        for t in tensors:
            feat_2d, p2, _ = net.backbone(t, return_pyramid=True)
            p2f = p2[0].flatten(1).T.float()
            p2_r.append(_erank(p2f[torch.linspace(0, p2f.shape[0] - 1, min(256, p2f.shape[0])).long()]))
            p3f = feat_2d[0].flatten(1).T.float()
            p3_r.append(_erank(p3f[torch.linspace(0, p3f.shape[0] - 1, min(256, p3f.shape[0])).long()]))
            enc_out = net.encoder(feat_2d, images=t)
            mem = enc_out[0] if isinstance(enc_out, (tuple, list)) else enc_out
            if isinstance(mem, (tuple, list)):
                mem = mem[-1]
            x = mem[0].float()
            stds.append(float(x.std(dim=0).mean().item()))
            idx = torch.linspace(0, x.shape[0] - 1, min(256, x.shape[0])).long()
            xs = F.normalize(x[idx], dim=-1)
            cm = xs @ xs.T
            wcos.append(float(((cm.sum() - cm.trace()) / (xs.shape[0] * (xs.shape[0] - 1))).item()))
            mem_r.append(_erank(x[idx]))
            vecs.append(x.mean(dim=0))
    mv = F.normalize(torch.stack(vecs), dim=-1)
    ccm = mv @ mv.T
    cross_cos = float(((ccm.sum() - ccm.trace()) / (mv.shape[0] * (mv.shape[0] - 1))).item())
    return {
        "p2_erank": round(float(np.mean(p2_r)), 2),
        "p3_erank": round(float(np.mean(p3_r)), 2),
        "mem_erank": round(float(np.mean(mem_r)), 2),
        "mem_tok_std": round(float(np.mean(stds)), 4),
        "within_page_cos": round(float(np.mean(wcos)), 4),
        "cross_page_cos": round(cross_cos, 4),
    }


def _extract_suite31_summary(eval_json_path: Path) -> Dict[str, Any]:
    raw = json.loads(eval_json_path.read_text(encoding="utf-8"))
    summary = raw.get("summary") or raw.get("splits") or raw
    out: Dict[str, Any] = {}
    for k in ("hw_clean", "hw_photo", "printed_clean", "printed_photo", "all45_clean", "all45_photo"):
        if isinstance(summary, dict) and k in summary and isinstance(summary[k], dict):
            s = summary[k]
            out[k] = {
                "note_f1": s.get("note_f1", s.get("strict_note_f1")),
                "tok_f1": s.get("tok_f1", s.get("token_f1")),
                "ser": s.get("ser"),
                "n": s.get("n", s.get("count")),
            }
    return out


def _train_and_eval_arm(
    arm_name: str,
    mvd_dir: Path,
    train_manifest: Path,
    suite31_manifest: Path,
    init_encoder_from: Optional[Path],
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    arm_dir = VOL_ROOT / "ckpts" / "exp2b_vicreg" / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(mvd_dir / "train_uni_v2.py"),
        "--preset", "small_45m",
        "--curriculum", "fullpage",
        "--fullpage-manifest", str(train_manifest),
        "--out-dir", str(arm_dir),
        "--steps", str(TRAIN_STEPS),
        "--batch-size", str(BATCH_SIZE),
        "--grad-accum", str(GRAD_ACCUM),
        "--fullpage-max-dim", str(FULLPAGE_MAX_DIM),
        "--max-seq-len", str(MAX_SEQ_LEN),
        "--optimizer", "muon",
        "--lr", str(LR),
        "--lr-schedule", "wsd",
        "--warmup-steps", str(WARMUP_STEPS),
        "--bf16", "--grad-checkpoint",
        "--eval-every", str(EVAL_EVERY),
        "--save-every-minutes", str(SAVE_EVERY_MINUTES),
        "--log-every", "25",
        "--seed", str(SEED),
    ]
    if init_encoder_from is not None:
        cmd.extend(["--init-encoder-from", str(init_encoder_from)])
    _run_cmd(cmd)
    train_secs = round(time.perf_counter() - t0, 2)

    ckpt_path = next(
        (arm_dir / n for n in ("uni_omr_v2_ema_best.pt", "uni_omr_v2_ema_latest.pt", "uni_omr_v2_latest.pt") if (arm_dir / n).is_file()),
        None,
    )
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint produced in {arm_dir}")

    suite31_out = arm_dir / f"{arm_name}_suite31_hw.json"
    t_eval = time.perf_counter()
    _run_cmd([
        sys.executable, "-u", str(mvd_dir / "eval_uni_suite31.py"),
        "--ckpt", str(ckpt_path),
        "--manifest", str(suite31_manifest),
        "--max-dim", str(SUITE31_MAX_DIM),
        "--max-len", str(SUITE31_MAX_LEN),
        "--bf16",
        "--out", str(suite31_out),
    ])
    return {
        "arm": arm_name,
        "init_encoder_from": str(init_encoder_from) if init_encoder_from else None,
        "evaluated_ckpt": ckpt_path.name,
        "train_secs": train_secs,
        "eval_secs": round(time.perf_counter() - t_eval, 2),
        "suite31_hw": _extract_suite31_summary(suite31_out),
    }


def main() -> None:
    t_start = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    hf_token = _read_and_purge_hf_token()
    repo_root = _locate_or_extract_repo_root()
    mvd_dir = repo_root / "tmp-train" / "mvd"

    import torch
    from huggingface_hub import HfApi

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    rism_dir, train_abs, suite31_manifest = _ensure_datasets(hf_token)

    # Stage 1: Anti-collapse V-JEPA 2.1 + VICReg + KL-router SSL
    ssl_dir = VOL_ROOT / "ckpts" / "exp2b_vicreg" / "ssl"
    ssl_dir.mkdir(parents=True, exist_ok=True)
    _run_cmd([
        sys.executable, "-u", str(mvd_dir / "pretrain_ijepa_rism_v2.py"),
        "--preset", "small_45m",
        "--image-dir", str(rism_dir),
        "--out-dir", str(ssl_dir),
        "--steps", str(SSL_STEPS),
        "--batch-size", "4", "--grad-accum", "2", "--max-dim", "1024",
        "--bf16", "--optimizer", "muon",
        "--gram-weight", "1.0", "--disable-random-gram",
        "--stroke-hog-weight", "0.25",
        "--router-cons-weight", "0.2", "--router-loss-mode", "kl",
        "--vicreg-var-weight", "1.0", "--vicreg-cov-weight", "0.04",
        "--save-every-minutes", str(SAVE_EVERY_MINUTES),
        "--log-every", "25", "--seed", str(SEED),
    ])
    ssl_ckpt = ssl_dir / "uni_omr_v2_ijepa_last.pt"
    ssl_rank_probe = _probe_ssl_rank(mvd_dir, ssl_ckpt, rism_dir)
    print(f"[{EXP_ID}] SSL rank probe: {json.dumps(ssl_rank_probe)}", flush=True)

    # Stage 2: Supervised transfer probe (scratch vs VICReg-SSL warmstart)
    arm_a = _train_and_eval_arm("arm_a_scratch", mvd_dir, train_abs, suite31_manifest, None)
    arm_b = _train_and_eval_arm("arm_b_vjepa_vicreg_warmstart", mvd_dir, train_abs, suite31_manifest, ssl_ckpt)

    comparison: Dict[str, Any] = {}
    for k in ("hw_clean", "hw_photo", "printed_clean", "printed_photo", "all45_clean", "all45_photo"):
        sa, sb = arm_a["suite31_hw"].get(k, {}), arm_b["suite31_hw"].get(k, {})
        if isinstance(sa, dict) and isinstance(sb, dict):
            if sa.get("note_f1") is not None and sb.get("note_f1") is not None:
                comparison[f"{k}_note_f1_delta"] = round(float(sb["note_f1"]) - float(sa["note_f1"]), 6)
            if sa.get("tok_f1") is not None and sb.get("tok_f1") is not None:
                comparison[f"{k}_tok_f1_delta"] = round(float(sb["tok_f1"]) - float(sa["tok_f1"]), 6)

    report = {
        "exp_id": EXP_ID,
        "status": "completed",
        "started_utc": started_iso,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_secs": round(time.time() - t_start, 2),
        "platform": f"colab-{gpu_name}",
        "ssl_steps": SSL_STEPS,
        "ssl_rank_probe": ssl_rank_probe,
        "arm_a_scratch": arm_a,
        "arm_b_vjepa_vicreg_warmstart": arm_b,
        "comparison_warmstart_minus_scratch": comparison,
    }
    eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
    run_local = VOL_ROOT / "runs" / f"{EXP_ID}.json"
    _atomic_write_json(eval_local, report)
    _atomic_write_json(run_local, {k: report[k] for k in ("exp_id", "status", "platform", "elapsed_secs", "ssl_rank_probe", "comparison_warmstart_minus_scratch")})

    api = HfApi(token=hf_token)
    api.upload_file(path_or_fileobj=str(ssl_ckpt), path_in_repo=f"checkpoints/{EXP_ID}/uni_omr_v2_ijepa_last.pt", repo_id=HF_REPO_ID, repo_type="dataset", commit_message=f"ckpt({EXP_ID}): upload VICReg anti-collapse SSL encoder")
    if (ssl_dir / "vjepa21_train_log.json").is_file():
        api.upload_file(path_or_fileobj=str(ssl_dir / "vjepa21_train_log.json"), path_in_repo=f"checkpoints/{EXP_ID}/vjepa21_train_log.json", repo_id=HF_REPO_ID, repo_type="dataset", commit_message=f"log({EXP_ID}): upload SSL log")
    api.upload_file(path_or_fileobj=str(eval_local), path_in_repo=HF_EVAL_PATH, repo_id=HF_REPO_ID, repo_type="dataset", commit_message=f"eval({EXP_ID}): upload anti-collapse SSL transfer probe")
    api.upload_file(path_or_fileobj=str(run_local), path_in_repo=HF_RUN_PATH, repo_id=HF_REPO_ID, repo_type="dataset", commit_message=f"run({EXP_ID}): upload run ledger")
    print(f"[{EXP_ID}] COMPLETE & UPLOADED TO HF!", flush=True)


if __name__ == "__main__":
    main()

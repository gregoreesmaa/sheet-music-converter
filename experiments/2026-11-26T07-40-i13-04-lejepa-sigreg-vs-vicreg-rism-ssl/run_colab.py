#!/usr/bin/env python3
"""Iter 13 Exp i13-04 on Google Colab L4:
LeJEPA SIGReg (Epps-Pulley Sketched Isotropic Gaussian Regularization) vs VICReg
Anti-Collapse Regularization for RISM 5,707 V-JEPA 2.1 SSL & Downstream Transfer.

Stage 1: 3 matched 400-step RISM 5,707 SSL arms (small_45m, KL router, no random Gram,
         Sobel-HOG=0.25, Muon, seed=42):
           - arm_vicreg     : VICReg (var_w=1.0, cov_w=0.04) [Exp 2b baseline]
           - arm_sigreg_w05 : LeJEPA SIGReg (weight=0.5, K=64 random unit slices, 17 knots)
           - arm_sigreg_w15 : LeJEPA SIGReg (weight=1.5, K=64 random unit slices, 17 knots)
Stage 2: SVD effective-rank & Gaussianity audit on 12 real RISM pages.
Stage 3: Matched 600-step UniOMR v2 small_45m supervised transfer probe on
         iter10_5_proxy_2k (arm_vicreg vs best arm_sigreg_*) evaluated on suite31_hw.

Expects:
  - /content/bundle.tar.gz (or /content/repo_bundle.tar.gz or pre-extracted workspace)
  - /tmp/.hf_token (Hugging Face token; chmod 600, deleted immediately and in finally)
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T07-40-i13-04-lejepa-sigreg-vs-vicreg-rism-ssl"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_RISM_TAR = "datasets/rism_hw_5707.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORKSPACE_ROOT = Path("/content/workspace")
VOL_ROOT = Path("/content/vol")

SSL_STEPS = 400
TRAIN_STEPS = 600
BATCH_SIZE = 2
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
LR = 3e-4
WARMUP_STEPS = 60
EVAL_EVERY = 300
SAVE_EVERY_MINUTES = 5.0
SUITE31_MAX_DIM = 1280
SUITE31_MAX_LEN = 768
SEED = 42

SSL_ARMS: List[Dict[str, Any]] = [
    {"key": "arm_vicreg", "mode": "vicreg", "var_w": 1.0, "cov_w": 0.04, "sigreg_w": 0.0},
    {"key": "arm_sigreg_w05", "mode": "sigreg", "var_w": 0.5, "cov_w": 0.0, "sigreg_w": 0.5},
    {"key": "arm_sigreg_w15", "mode": "sigreg", "var_w": 1.5, "cov_w": 0.0, "sigreg_w": 1.5},
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _purge_hf_token() -> None:
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and HF_TOKEN_FILE.exists():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        token = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    _purge_hf_token()
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var")
    return token


def _run_cmd(cmd: Sequence[str], *, env: Optional[Dict[str, str]] = None) -> None:
    label = " ".join(cmd)
    print(f"[{EXP_ID}] $ {label}", flush=True)
    proc = subprocess.Popen(
        list(cmd),
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


def _setup_environment() -> Path:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    VOL_ROOT.mkdir(parents=True, exist_ok=True)

    candidates = [
        WORKSPACE_ROOT / "sheet-music-converter",
        Path("/workspace/sheet-music-converter"),
        Path("/content/sheet-music-converter"),
        Path.cwd(),
    ]
    repo_root: Optional[Path] = None
    for cand in candidates:
        if (cand / "tmp-train" / "mvd" / "pretrain_ijepa_rism_v2.py").is_file():
            repo_root = cand
            break

    if repo_root is None:
        for bundle in (Path("/content/bundle.tar.gz"), Path("/content/repo_bundle.tar.gz")):
            if bundle.is_file():
                print(f"[{EXP_ID}] Extracting {bundle} -> {WORKSPACE_ROOT}...", flush=True)
                with tarfile.open(bundle, "r:*") as tf:
                    tf.extractall(path=WORKSPACE_ROOT)
                for cand in candidates:
                    if (cand / "tmp-train" / "mvd" / "pretrain_ijepa_rism_v2.py").is_file():
                        repo_root = cand
                        break
                if repo_root is not None:
                    break

    if repo_root is None:
        raise FileNotFoundError("Could not locate sheet-music-converter/tmp-train/mvd/pretrain_ijepa_rism_v2.py")

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
    return repo_root


def _ensure_datasets(hf_token: str) -> Tuple[Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_root = VOL_ROOT / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)

    rism_dir = ds_root / "rism_hw_5707"
    if not (rism_dir / ".unpacked_ok").is_file():
        rism_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(HF_REPO_ID, HF_RISM_TAR, repo_type="dataset", token=hf_token, local_dir="/tmp/hf_dl")
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=rism_dir)
        os.remove(tar_p)
        (rism_dir / ".unpacked_ok").write_text(_utc_now() + "\n", encoding="utf-8")

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

    suite31_dir = ds_root / "suite31_hw_eval"
    manifests = sorted(suite31_dir.rglob("suite31_hw_manifest.json")) if suite31_dir.exists() else []
    if not manifests:
        suite31_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(HF_REPO_ID, HF_SUITE31_TAR, repo_type="dataset", token=hf_token, local_dir="/tmp/hf_dl")
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=suite31_dir)
        os.remove(tar_p)
        manifests = sorted(suite31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json under {suite31_dir}")

    return rism_dir, train_abs, manifests[0]


def make_sigreg_epps_pulley_fn(num_slices: int = 64, num_knots: int = 17, t_max: float = 3.0):
    """Build a drop-in SIGReg (LeJEPA, Balestriero & LeCun 2025) regularizer matching
    `compute_vicreg_loss(feats, var_weight, cov_weight, gamma, max_tokens)` signature.

    Projects `(N, D)` tokens onto `K` random unit directions and integrates the squared
    distance between the empirical 1D characteristic function and `exp(-t^2 / 2)` via
    weighted trapezoidal quadrature (Epps-Pulley test) in O(N * K * M) linear time.
    """
    import torch
    import torch.nn.functional as F

    def _compute_sigreg_loss(
        feats: Any,
        var_weight: float = 1.0,
        cov_weight: float = 0.0,
        gamma: float = 1.0,
        max_tokens: int = 512,
    ) -> Any:
        del cov_weight, gamma
        if float(var_weight) <= 0.0:
            return feats.new_zeros((), dtype=torch.float32)
        bsz, k_tok, dim = feats.shape
        x = feats.reshape(bsz * k_tok, dim).float()
        if x.shape[0] > max_tokens:
            stride = max(1, x.shape[0] // max_tokens)
            x = x[::stride][:max_tokens]
        if x.shape[0] < 4:
            return feats.new_zeros((), dtype=torch.float32)

        # Random unit projection directions A: (D, K)
        proj_dirs = torch.randn(dim, num_slices, device=x.device, dtype=torch.float32)
        proj_dirs = F.normalize(proj_dirs, p=2, dim=0, eps=1e-6)
        u = x @ proj_dirs  # (N, K)

        t = torch.linspace(-float(t_max), float(t_max), int(num_knots), device=x.device, dtype=torch.float32)
        dt = (2.0 * float(t_max)) / float(max(1, num_knots - 1))
        phi_target = torch.exp(-0.5 * t * t)  # (M,)
        quad_w = phi_target * dt  # Gaussian integration weight exp(-t^2 / 2)

        # Empirical characteristic function: (N, K, M) -> mean over N -> (K, M)
        phases = u.unsqueeze(-1) * t.view(1, 1, -1)
        ecf_real = torch.cos(phases).mean(dim=0)
        ecf_imag = torch.sin(phases).mean(dim=0)

        err_sq = (ecf_real - phi_target.view(1, -1)).pow(2) + ecf_imag.pow(2)
        # Scale by N^(1/2) damping? Epps-Pulley statistic scales with N; mean-normalized
        # integral over slices stays O(1) and stable across variable token counts.
        ep_stat = (err_sq * quad_w.view(1, -1)).sum(dim=-1).mean()
        return float(var_weight) * ep_stat

    return _compute_sigreg_loss


def _run_ssl_arm(
    arm: Dict[str, Any],
    mvd_dir: Path,
    rism_dir: Path,
    out_dir: Path,
    use_bf16: bool,
) -> Tuple[Path, Dict[str, Any]]:
    import torch

    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))
    import pretrain_ijepa_rism_v2 as pij

    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        "--preset", "small_45m",
        "--image-dir", str(rism_dir),
        "--out-dir", str(out_dir),
        "--steps", str(SSL_STEPS),
        "--batch-size", "4",
        "--grad-accum", "2",
        "--max-dim", "1024",
        "--optimizer", "muon",
        "--lr", str(LR),
        "--gram-weight", "1.0",
        "--disable-random-gram",
        "--stroke-hog-weight", "0.25",
        "--router-cons-weight", "0.2",
        "--router-loss-mode", "kl",
        "--vicreg-var-weight", str(arm["var_w"]),
        "--vicreg-cov-weight", str(arm["cov_w"]),
        "--save-every-minutes", str(SAVE_EVERY_MINUTES),
        "--log-every", "25",
        "--seed", str(SEED),
    ]
    if use_bf16:
        argv.append("--bf16")

    args = pij.build_argparser().parse_args(argv)
    orig_vicreg = pij.compute_vicreg_loss
    t0 = time.perf_counter()
    try:
        if arm["mode"] == "sigreg":
            pij.compute_vicreg_loss = make_sigreg_epps_pulley_fn(num_slices=64, num_knots=17, t_max=3.0)
        summary = pij.run_pretrain(args)
    finally:
        pij.compute_vicreg_loss = orig_vicreg
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ssl_ckpt = out_dir / "uni_omr_v2_ijepa_last.pt"
    if not ssl_ckpt.is_file():
        raise FileNotFoundError(f"Expected SSL checkpoint missing: {ssl_ckpt}")
    return ssl_ckpt, {
        "arm": arm["key"],
        "mode": arm["mode"],
        "elapsed_secs": round(time.perf_counter() - t0, 2),
        "final_ema_loss": summary.get("final_ema_loss"),
        "oom_skips": summary.get("oom_skips", 0),
    }


def _probe_ssl_rank(mvd_dir: Path, ckpt_path: Path, rism_dir: Path) -> Dict[str, float]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))
    from pretrain_ijepa_rism_v2 import discover_unlabeled_images, load_image_resized
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net

    imgs = discover_unlabeled_images(rism_dir)[:12]
    tensors = [load_image_resized(p, 512).unsqueeze(0) for p in imgs]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = UniOMRv2Net(UniOMRv2Config.small_45m())
    net.load_state_dict(ckpt["model"], strict=False)
    net.eval()

    sigreg_probe = make_sigreg_epps_pulley_fn(num_slices=64, num_knots=17, t_max=3.0)

    def _erank(mat: torch.Tensor) -> float:
        c = mat - mat.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(c)
        pr = (s * s) / ((s * s).sum() + 1e-12)
        return float(math.exp(float(-(pr * (pr + 1e-12).log()).sum().item())))

    p2_r, p3_r, mem_r, stds, wcos, ep_stats, vecs = [], [], [], [], [], [], []
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
            ep_stats.append(float(sigreg_probe(mem, var_weight=1.0).item()))
            x = mem[0].float()
            stds.append(float(x.std(dim=0).mean().item()))
            idx = torch.linspace(0, x.shape[0] - 1, min(256, x.shape[0])).long()
            xs = F.normalize(x[idx], dim=-1)
            cm = xs @ xs.T
            wcos.append(float(((cm.sum() - cm.trace()) / max(1, xs.shape[0] * (xs.shape[0] - 1))).item()))
            mem_r.append(_erank(x[idx]))
            vecs.append(x.mean(dim=0))
    mv = F.normalize(torch.stack(vecs), dim=-1)
    ccm = mv @ mv.T
    cross_cos = float(((ccm.sum() - ccm.trace()) / max(1, mv.shape[0] * (mv.shape[0] - 1))).item())
    return {
        "p2_erank": round(float(np.mean(p2_r)), 2),
        "p3_erank": round(float(np.mean(p3_r)), 2),
        "mem_erank": round(float(np.mean(mem_r)), 2),
        "mem_tok_std": round(float(np.mean(stds)), 4),
        "within_page_cos": round(float(np.mean(wcos)), 4),
        "cross_page_cos": round(cross_cos, 4),
        "epps_pulley_stat": round(float(np.mean(ep_stats)), 5),
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


def _train_and_eval_transfer(
    arm_name: str,
    mvd_dir: Path,
    train_manifest: Path,
    suite31_manifest: Path,
    init_encoder_from: Path,
    use_bf16: bool,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    arm_dir = VOL_ROOT / "ckpts" / "i13_04_transfer" / arm_name
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
        "--grad-checkpoint",
        "--eval-every", str(EVAL_EVERY),
        "--save-every-minutes", str(SAVE_EVERY_MINUTES),
        "--log-every", "25",
        "--seed", str(SEED),
        "--init-encoder-from", str(init_encoder_from),
    ]
    if use_bf16:
        cmd.append("--bf16")
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
    eval_cmd = [
        sys.executable, "-u", str(mvd_dir / "eval_uni_suite31.py"),
        "--ckpt", str(ckpt_path),
        "--manifest", str(suite31_manifest),
        "--max-dim", str(SUITE31_MAX_DIM),
        "--max-len", str(SUITE31_MAX_LEN),
        "--out", str(suite31_out),
    ]
    if use_bf16:
        eval_cmd.append("--bf16")
    _run_cmd(eval_cmd)
    return {
        "arm": arm_name,
        "init_encoder_from": str(init_encoder_from),
        "evaluated_ckpt": ckpt_path.name,
        "train_secs": train_secs,
        "eval_secs": round(time.perf_counter() - t_eval, 2),
        "suite31_hw": _extract_suite31_summary(suite31_out),
    }


def main() -> None:
    t_start = time.time()
    started_utc = _utc_now()
    hf_token = ""
    try:
        hf_token = _read_and_purge_hf_token()
        repo_root = _setup_environment()
        mvd_dir = repo_root / "tmp-train" / "mvd"

        import torch
        from huggingface_hub import HfApi

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        use_bf16 = bool(device.type == "cuda" and cc[0] >= 8)
        env_tag = "kaggle" if Path("/kaggle").exists() else ("colab" if Path("/content").exists() else "cloud")
        platform_str = f"{env_tag}-{gpu_name}"
        print(f"[{EXP_ID}] {platform_str} (cc={cc[0]}.{cc[1]}, bf16={use_bf16})", flush=True)

        api = HfApi(token=hf_token)
        rism_dir, train_abs, suite31_manifest = _ensure_datasets(hf_token)

        ledger_tmp = VOL_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(ledger_tmp, {
            "exp_id": EXP_ID,
            "status": "RUNNING",
            "started_utc": started_utc,
            "platform": platform_str,
            "ssl_steps": SSL_STEPS,
            "train_steps": TRAIN_STEPS,
        })
        try:
            api.upload_file(
                path_or_fileobj=str(ledger_tmp),
                path_in_repo=HF_RUN_PATH,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"{EXP_ID}: mark RUNNING on {platform_str}",
            )
        except Exception as exc:
            print(f"[{EXP_ID}] Warning: initial HF ledger upload failed: {exc}", flush=True)

        ssl_results: Dict[str, Any] = {}
        ssl_ckpts: Dict[str, Path] = {}
        for arm in SSL_ARMS:
            key = arm["key"]
            arm_ssl_dir = VOL_ROOT / "ckpts" / "i13_04_ssl" / key
            ckpt_p, train_meta = _run_ssl_arm(arm, mvd_dir, rism_dir, arm_ssl_dir, use_bf16=use_bf16)
            rank_probe = _probe_ssl_rank(mvd_dir, ckpt_p, rism_dir)
            ssl_ckpts[key] = ckpt_p
            ssl_results[key] = {**train_meta, "rank_probe": rank_probe}
            print(f"[{EXP_ID}] {key} rank_probe={json.dumps(rank_probe)}", flush=True)

        # Pick best SIGReg arm by combined (p3_erank + mem_erank)
        sigreg_keys = [a["key"] for a in SSL_ARMS if a["mode"] == "sigreg"]
        best_sigreg_key = max(
            sigreg_keys,
            key=lambda k: ssl_results[k]["rank_probe"]["p3_erank"] + ssl_results[k]["rank_probe"]["mem_erank"],
        )
        print(f"[{EXP_ID}] Selected SIGReg arm for transfer probe: {best_sigreg_key}", flush=True)

        transfer_vicreg = _train_and_eval_transfer(
            "transfer_vicreg", mvd_dir, train_abs, suite31_manifest, ssl_ckpts["arm_vicreg"], use_bf16=use_bf16
        )
        transfer_sigreg = _train_and_eval_transfer(
            f"transfer_{best_sigreg_key}", mvd_dir, train_abs, suite31_manifest, ssl_ckpts[best_sigreg_key], use_bf16=use_bf16
        )

        deltas: Dict[str, Any] = {}
        for k in ("hw_clean", "hw_photo", "printed_clean", "printed_photo", "all45_clean", "all45_photo"):
            sa = transfer_vicreg["suite31_hw"].get(k, {})
            sb = transfer_sigreg["suite31_hw"].get(k, {})
            if isinstance(sa, dict) and isinstance(sb, dict):
                if sa.get("note_f1") is not None and sb.get("note_f1") is not None:
                    deltas[f"{k}_note_f1_delta"] = round(float(sb["note_f1"]) - float(sa["note_f1"]), 6)
                if sa.get("tok_f1") is not None and sb.get("tok_f1") is not None:
                    deltas[f"{k}_tok_f1_delta"] = round(float(sb["tok_f1"]) - float(sa["tok_f1"]), 6)

        report = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_utc,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": platform_str,
            "bf16": use_bf16,
            "ssl_steps": SSL_STEPS,
            "train_steps": TRAIN_STEPS,
            "ssl_arms": ssl_results,
            "selected_sigreg_arm": best_sigreg_key,
            "transfer_vicreg": transfer_vicreg,
            "transfer_sigreg": transfer_sigreg,
            "comparison_sigreg_minus_vicreg": deltas,
        }

        eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(ledger_tmp, {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "platform": platform_str,
            "elapsed_secs": report["elapsed_secs"],
            "selected_sigreg_arm": best_sigreg_key,
            "comparison_sigreg_minus_vicreg": deltas,
        })

        api.upload_file(
            path_or_fileobj=str(ssl_ckpts[best_sigreg_key]),
            path_in_repo=f"checkpoints/{EXP_ID}/best_ssl_encoder.pt",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"ckpt({EXP_ID}): upload best SIGReg SSL encoder ({best_sigreg_key})",
        )
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload SIGReg vs VICReg SSL eval",
        )
        api.upload_file(
            path_or_fileobj=str(ledger_tmp),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): mark COMPLETED",
        )
        print(f"[{EXP_ID}] COMPLETE & UPLOADED TO HF!", flush=True)
    finally:
        _purge_hf_token()


if __name__ == "__main__":
    main()

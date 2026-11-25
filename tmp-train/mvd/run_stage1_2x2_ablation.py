r"""Controlled 2x2 Stage-1 (StaffUNet2D) ablation runner:
  Axis 1: Handwriting (NO = 7,500 printed only | YES = +1,500 clean-paper legible Hero HW)
  Axis 2: Unsupervised EMA (NO | YES = 200-step EMA on 17.1k unlabeled + RISM per epoch)

All 4 arms warm-start from `v3 Ep 15` (`staff-unet2d-v3-dynres/staff_unet2d.pt`) with
pure BCE+Dice, gentle lr=5e-5, hw-hard-neg-prob=0.0, clean-frac=0.25, dynamic-res, row-ctx.
Every arm (+ baseline) is evaluated on all 45 pieces of `suite31_hw` (31 printed + 14 HW)
paired with the frozen `exp2B @ 20k` Stage-2 model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
PY = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")

BASE_S1_CKPT = Path(r"D:\tmp\mvd-runs\staff-unet2d-v3-dynres\staff_unet2d.pt")
S2_CKPT = Path(r"D:\tmp\mvd-runs\exp2B-s18-comer-gdcm-v1\ckpt-20000.pt")
SUITE31_HW_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"

MANIFEST_PRINTED = Path(r"D:\tmp\staff-unet-data\manifest.jsonl")
MANIFEST_HW_LEGIBLE = Path(r"D:\tmp\staff-unet-data\manifest_with_hwsim_v03_leg1500.jsonl")
UNLABELED_DIRS = r"D:\tmp\unlabeled-10k-real,D:\tmp\rism-pdm-hw\images"

OUT_ROOT = Path(r"D:\tmp\ablation-stage1-2x2")


def run_cmd(cmd: List[str], label: str) -> float:
    print(f"\n=== [{label}] ===\n  {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"[{label}] failed with exit code {proc.returncode} after {dt:.1f}s")
    print(f"=== [{label}] OK ({dt:.1f}s) ===", flush=True)
    return dt


def eval_arm(arm_name: str, s1_ckpt: Path, eval_json: Path) -> Dict[str, Any]:
    if not eval_json.exists():
        cmd = [
            PY, "-u", "tmp-train/mvd/eval_sys_gate.py",
            "--ckpt", str(S2_CKPT),
            "--staff-unet-ckpt", str(s1_ckpt),
            "--suite31-manifest", str(SUITE31_HW_MANIFEST),
            "--suite31-only",
            "--blind-photo",
            "--dynamic-staff-unet",
            "--stride", "4x8",
            "--height", "256",
            "--out", str(eval_json),
        ]
        run_cmd(cmd, f"eval:{arm_name}")
    doc = json.loads(eval_json.read_text(encoding="utf-8"))
    s = doc.get("summary", doc)
    return {
        "arm": arm_name,
        "s1_ckpt": str(s1_ckpt),
        "printed_clean_f1": round(float(s.get("suite31_typeset_clean_f1_mean", 0.0)), 4),
        "printed_photo_f1": round(float(s.get("suite31_typeset_photo_f1_mean", 0.0)), 4),
        "hw_clean_f1": round(float(s.get("suite31_hw_clean_f1_mean", 0.0)), 4),
        "hw_photo_f1": round(float(s.get("suite31_hw_photo_f1_mean", 0.0)), 4),
        "all45_clean_f1": round(float(s.get("suite31_clean_f1_mean", 0.0)), 4),
        "all45_photo_f1": round(float(s.get("suite31_photo_f1_mean", 0.0)), 4),
        "detected_rate": round(float(s.get("blind_photo_detected_rate", 0.0)), 4),
    }


def print_table(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 96, flush=True)
    print(f"{'Arm':26s} | {'PrClean':>8s} | {'PrPhoto':>8s} | {'HWClean':>8s} | {'HWPhoto':>8s} | {'All45Ph':>8s} | {'Det':>5s}", flush=True)
    print("-" * 96, flush=True)
    for r in rows:
        print(
            f"{r['arm']:26s} | {r['printed_clean_f1']:8.4f} | {r['printed_photo_f1']:8.4f} | "
            f"{r['hw_clean_f1']:8.4f} | {r['hw_photo_f1']:8.4f} | {r['all45_photo_f1']:8.4f} | {r['detected_rate']:5.2f}",
            flush=True,
        )
    print("=" * 96 + "\n", flush=True)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    # 0. Exact Baseline (v3 Ep 15)
    rows.append(eval_arm("0_baseline_v3_ep15", BASE_S1_CKPT, OUT_ROOT / "eval_0_baseline_v3_ep15.json"))
    print_table(rows)

    arms = [
        ("A_printed_only", MANIFEST_PRINTED, False),
        ("B_printed_plus_unsup", MANIFEST_PRINTED, True),
        ("C_printed_plus_hw_clean", MANIFEST_HW_LEGIBLE, False),
        ("D_printed_hw_plus_unsup", MANIFEST_HW_LEGIBLE, True),
    ]

    for arm_name, manifest_path, use_unsup in arms:
        arm_dir = OUT_ROOT / arm_name
        arm_ckpt = arm_dir / "staff_unet2d.pt"
        if not arm_ckpt.exists():
            cmd = [
                PY, "-u", "tmp-train/mvd/staff_unet2d.py",
                "--manifest", str(manifest_path),
                "--out-dir", str(arm_dir),
                "--init-from", str(BASE_S1_CKPT),
                "--sp-mode", "log_hm",
                "--row-ctx",
                "--epochs", "2",
                "--batch-size", "4",
                "--lr", "5e-5",
                "--bf16",
                "--dynamic-res",
                "--hw-hard-neg-prob", "0.0",
                "--clean-frac", "0.25",
                "--num-workers", "4",
            ]
            if use_unsup:
                cmd += [
                    "--unlabeled-dirs", UNLABELED_DIRS,
                    "--unsup-round-every", "1",
                    "--unsup-round-steps", "200",
                ]
            run_cmd(cmd, f"train:{arm_name}")
        rows.append(eval_arm(arm_name, arm_ckpt, OUT_ROOT / f"eval_{arm_name}.json"))
        (OUT_ROOT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print_table(rows)

    print(f"[DONE] Saved full 2x2 ablation summary to {OUT_ROOT / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

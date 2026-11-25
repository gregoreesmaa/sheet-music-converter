#!/usr/bin/env python3
"""Experiment 7 (Modal L4, BF16): Render-and-Diff Visual Self-Feedback RLVR.

Tests 100% label-free closed-loop visual render-and-diff GRPO post-training
(`tmp-train/mvd/trial_2027_render_diff_rlvr.py`) starting from
`checkpoints/iter10_soup_baseline/uni_omr_soup.pt` for 60 GRPO updates on
unlabeled pages from `datasets/iter10_5_proxy_2k.tar`, with 5-minute atomic
checkpoints to Modal Volume `/vol/ckpts/exp7_render_diff_rlvr/` and before vs.
after 45-piece `suite31_hw` OOD evaluation on `datasets/suite31_hw_eval.tar`.

Launch detached from repo root:
    .\\.venv\\Scripts\\modal.exe run --detach experiments/2026-11-26T00-55-trial-render-diff-rlvr/run_modal.py
"""

from __future__ import annotations

from pathlib import Path
import modal

EXP_ID = "2026-11-26T00-55-trial-render-diff-rlvr"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"

REPO_ROOT = Path(__file__).resolve().parents[2]
MSCZ_ROOT = REPO_ROOT.parent / "mscz-rs"

app = modal.App("exp7-trial-render-diff-rlvr")
vol = modal.Volume.from_name("omr-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "tar",
        "curl",
        "build-essential",
        "ca-certificates",
        "pkg-config",
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal",
        "ln -sf /root/.cargo/bin/cargo /usr/local/bin/cargo",
        "ln -sf /root/.cargo/bin/rustc /usr/local/bin/rustc",
        "rustc --version && cargo --version",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "huggingface_hub>=0.26",
        "pillow",
        "numpy",
        "tqdm",
        "verovio",
        "resvg-py",
    )
    .add_local_dir(
        str(MSCZ_ROOT),
        remote_path="/workspace/mscz-rs",
        copy=True,
        ignore=[".git", "target", "**/.git", "**/target"],
    )
    .add_local_dir(
        str(REPO_ROOT / "converter" / "assembly-serialize"),
        remote_path="/workspace/sheet-music-converter/converter/assembly-serialize",
        copy=True,
        ignore=["target", "**/target"],
    )
    .add_local_dir(
        str(REPO_ROOT / "converter" / "eval-harness"),
        remote_path="/workspace/sheet-music-converter/converter/eval-harness",
        copy=True,
        ignore=["target", "**/target"],
    )
    .run_commands(
        "cargo build --release --manifest-path /workspace/sheet-music-converter/converter/assembly-serialize/Cargo.toml",
        "cargo build --release --manifest-path /workspace/sheet-music-converter/converter/eval-harness/Cargo.toml",
        "ln -sf /workspace/sheet-music-converter/converter/assembly-serialize/target/release/assembly-serialize /workspace/sheet-music-converter/converter/assembly-serialize/target/release/assembly-serialize.exe",
        "ln -sf /workspace/sheet-music-converter/converter/eval-harness/target/release/eval-harness /workspace/sheet-music-converter/converter/eval-harness/target/release/eval-harness.exe",
    )
    .add_local_dir(
        str(REPO_ROOT / "tmp-train" / "mvd"),
        remote_path="/workspace/sheet-music-converter/tmp-train/mvd",
        ignore=["__pycache__", "*.pyc", "*.pt", "*.pth"],
    )
)


def _atomic_save_ckpt(path: Path, payload: dict) -> None:
    import os
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(str(tmp), str(path))


def _download_and_unpack(api_token: str, hf_path: str, tar_dest: Path, unpack_dir: Path) -> Path:
    import tarfile
    from huggingface_hub import hf_hub_download

    marker = unpack_dir / ".unpacked_ok"
    if marker.exists():
        print(f"[cache hit] {unpack_dir}", flush=True)
        return unpack_dir

    tar_dest.parent.mkdir(parents=True, exist_ok=True)
    if not tar_dest.exists():
        print(f"[hf] downloading {hf_path} -> {tar_dest}", flush=True)
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=hf_path,
            token=api_token,
            local_dir=str(tar_dest.parent.parent),
        )
        dl_path = Path(downloaded)
        if dl_path.resolve() != tar_dest.resolve() and not tar_dest.exists():
            tar_dest.symlink_to(dl_path)

    unpack_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unpack] {tar_dest} -> {unpack_dir}", flush=True)
    with tarfile.open(str(tar_dest), "r:*") as tf:
        tf.extractall(path=str(unpack_dir))
    marker.write_text("ok\n", encoding="utf-8")
    vol.commit()
    return unpack_dir


def _upload_to_hf(api, local_path: Path, path_in_repo: str, commit_message: str) -> None:
    print(f"[hf upload] {local_path} -> {HF_REPO}:{path_in_repo}", flush=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message=commit_message,
    )


@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def run_exp7(
    steps: int = 60,
    group_size: int = 4,
    batch_size: int = 1,
    max_len: int = 512,
    train_max_dim: int = 1024,
    eval_max_dim: int = 1536,
    eval_max_len: int = 2048,
    lr: float = 2e-5,
    kl_beta: float = 0.04,
    temperature: float = 0.8,
    top_p: float = 0.95,
    save_every_minutes: float = 5.0,
    seed: int = 42,
    resume: bool = True,
) -> dict:
    import argparse
    import copy
    from dataclasses import asdict
    from datetime import datetime, timezone
    import json
    import os
    import random
    import sys
    import time
    import numpy as np
    from PIL import Image
    import torch
    from huggingface_hub import HfApi, hf_hub_download

    sys.path.insert(0, "/workspace/sheet-music-converter/tmp-train/mvd")
    import eval_sys_gate as esg
    import eval_uni_suite31 as eval_s31
    import rlvr_grpo
    import trial_2027_render_diff_rlvr as rd_rlvr
    import uni_omr

    # Guard against cuDNN SDPA variable-shape recompilation stalls
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    aser_bin = Path("/workspace/sheet-music-converter/converter/assembly-serialize/target/release/assembly-serialize")
    eharness_bin = Path("/workspace/sheet-music-converter/converter/eval-harness/target/release/eval-harness")
    assert aser_bin.exists(), f"Missing assembly-serialize at {aser_bin}"
    assert eharness_bin.exists(), f"Missing eval-harness at {eharness_bin}"
    esg.ASER = aser_bin
    esg.EHARNESS = eharness_bin

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
    use_bf16 = device.type == "cuda" and cc[0] >= 8
    print(
        f"[exp7] device={device} ({gpu_name}, cc={cc}) bf16={use_bf16} "
        f"steps={steps} G={group_size} lr={lr} kl_beta={kl_beta}",
        flush=True,
    )

    hf_token = os.environ["HF_TOKEN"]
    api = HfApi(token=hf_token)

    # 1. Pull datasets & baseline checkpoint into /vol
    vol_root = Path("/vol")
    proxy_dir = _download_and_unpack(
        hf_token,
        "datasets/iter10_5_proxy_2k.tar",
        vol_root / "hf_cache" / "datasets" / "iter10_5_proxy_2k.tar",
        vol_root / "datasets" / "iter10_5_proxy_2k",
    )
    eval_dir = _download_and_unpack(
        hf_token,
        "datasets/suite31_hw_eval.tar",
        vol_root / "hf_cache" / "datasets" / "suite31_hw_eval.tar",
        vol_root / "datasets" / "suite31_hw_eval",
    )

    base_ckpt = vol_root / "checkpoints" / "iter10_soup_baseline" / "uni_omr_soup.pt"
    if not base_ckpt.exists():
        base_ckpt.parent.mkdir(parents=True, exist_ok=True)
        print("[hf] downloading checkpoints/iter10_soup_baseline/uni_omr_soup.pt", flush=True)
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename="checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
            token=hf_token,
            local_dir=str(vol_root),
        )
        vol.commit()
    assert base_ckpt.exists(), f"Baseline checkpoint missing at {base_ckpt}"

    manifests = list(eval_dir.rglob("suite31_hw_manifest.json"))
    assert manifests, f"suite31_hw_manifest.json not found under {eval_dir}"
    suite31_manifest = manifests[0]
    print(f"[exp7] suite31 manifest: {suite31_manifest}", flush=True)

    ckpt_dir = vol_root / "ckpts" / "exp7_render_diff_rlvr"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "uni_omr_latest.pt"
    final_ckpt = ckpt_dir / "uni_omr_final.pt"
    before_eval_json = ckpt_dir / "suite31_hw_before.json"
    after_eval_json = ckpt_dir / "suite31_hw_after.json"
    combined_eval_json = ckpt_dir / f"{EXP_ID}.json"
    run_ledger_json = ckpt_dir / f"{EXP_ID}_run.json"

    def _run_suite31_eval(ckpt_path: Path, out_path: Path) -> dict:
        ns = argparse.Namespace(
            ckpt=str(ckpt_path),
            manifest=str(suite31_manifest),
            max_dim=int(eval_max_dim),
            max_len=int(eval_max_len),
            use_grammar=True,
            strict_grammar=False,
            use_mtp_spec=True,
            beam_size=1,
            beam_length_penalty=0.6,
            measure_beam_size=0,
            max_keep_ratio_override=None,
            device=str(device),
            bf16=use_bf16,
            out=str(out_path),
            limit_pieces=None,
            pieces=None,
        )
        return eval_s31.evaluate_suite31(ns)

    # 2. Baseline suite31_hw evaluation BEFORE RLVR (cached on volume if resuming)
    if before_eval_json.exists():
        print(f"[exp7] loading cached baseline eval: {before_eval_json}", flush=True)
        before_report = json.loads(before_eval_json.read_text(encoding="utf-8"))
    else:
        print("[exp7] running BEFORE RLVR suite31_hw baseline eval...", flush=True)
        before_report = _run_suite31_eval(base_ckpt, before_eval_json)
        vol.commit()

    # 3. Discover unlabeled training pages from iter10_5_proxy_2k
    all_imgs = rd_rlvr.discover_unlabeled_images(proxy_dir, limit=100000)
    assert all_imgs, f"No images found in {proxy_dir}"
    rng = random.Random(int(seed))
    rng.shuffle(all_imgs)
    print(f"[exp7] discovered {len(all_imgs)} proxy pages in {proxy_dir}", flush=True)

    # 4. Load model + frozen reference policy (with resume support)
    torch.manual_seed(int(seed))
    ref_model, cfg, base_meta = eval_s31.load_uni_checkpoint(base_ckpt, device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    start_step = 0
    history: list[dict] = []
    if resume and latest_ckpt.exists():
        print(f"[exp7] resuming from {latest_ckpt}", flush=True)
        model, cfg, _ = eval_s31.load_uni_checkpoint(latest_ckpt, device=device)
        ck = torch.load(str(latest_ckpt), map_location="cpu", weights_only=True)
        start_step = int(ck.get("rlvr_step", 0))
        if run_ledger_json.exists():
            prev = json.loads(run_ledger_json.read_text(encoding="utf-8"))
            history = list(prev.get("history", []))[:start_step]
    else:
        model = copy.deepcopy(ref_model).to(device)
        for p in model.parameters():
            p.requires_grad_(True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.01)
    grpo_cfg = rlvr_grpo.GRPOConfig(
        group_size=int(group_size),
        max_len=int(max_len),
        temperature=float(temperature),
        top_p=float(top_p),
        kl_beta=float(kl_beta),
        use_grammar=True,
        strict_grammar=False,
    )

    step_vis_records: list[dict] = []

    def _label_free_reward_fn(pred_toks, ref_img_gray):
        res = rd_rlvr.compute_label_free_render_diff_reward(
            ref_img_gray, pred_toks, aser_bin=aser_bin
        )
        step_vis_records.append(res)
        return res

    save_every_s = float(save_every_minutes) * 60.0
    last_save_t = time.monotonic()
    last_hf_sync_t = time.monotonic()
    t_train0 = time.perf_counter()

    def _save_progress(cur_step: int, sync_hf: bool = False) -> None:
        payload = {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": asdict(cfg),
            "step": int(base_meta.get("step") or 0) + cur_step,
            "rlvr_step": cur_step,
            "exp_id": EXP_ID,
        }
        _atomic_save_ckpt(latest_ckpt, payload)
        ledger = {
            "exp_id": EXP_ID,
            "status": "running" if cur_step < steps else "evaluating",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": gpu_name,
            "bf16": use_bf16,
            "config": {
                "steps": steps,
                "group_size": group_size,
                "batch_size": batch_size,
                "max_len": max_len,
                "train_max_dim": train_max_dim,
                "eval_max_dim": eval_max_dim,
                "eval_max_len": eval_max_len,
                "lr": lr,
                "kl_beta": kl_beta,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
            },
            "completed_steps": cur_step,
            "elapsed_train_s": round(time.perf_counter() - t_train0, 2),
            "history": history,
        }
        tmp_l = run_ledger_json.with_suffix(".json.tmp")
        tmp_l.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        os.replace(str(tmp_l), str(run_ledger_json))
        vol.commit()
        print(f"[ckpt] saved atomic checkpoint at step {cur_step}/{steps} -> {latest_ckpt}", flush=True)
        if sync_hf:
            try:
                _upload_to_hf(
                    api,
                    latest_ckpt,
                    f"checkpoints/{EXP_ID}/uni_omr_latest.pt",
                    f"exp7: rolling ckpt step {cur_step}/{steps}",
                )
                _upload_to_hf(
                    api,
                    run_ledger_json,
                    f"runs/{EXP_ID}.json",
                    f"exp7: rolling ledger step {cur_step}/{steps}",
                )
            except Exception as exc:
                print(f"[hf warn] rolling sync failed (will retry): {exc}", flush=True)

    # 5. Run 60 GRPO updates with 5-minute atomic checkpoints
    for step_idx in range(start_step, int(steps)):
        t_step0 = time.perf_counter()
        batch_paths = [
            all_imgs[(step_idx * batch_size + b) % len(all_imgs)]
            for b in range(int(batch_size))
        ]
        tensors = []
        grays = []
        for p in batch_paths:
            with Image.open(p) as raw_im:
                rgb_im = raw_im.convert("RGB")
                raw_gray = np.asarray(rgb_im.convert("L"), dtype=np.float32) / 255.0
                pil_im = uni_omr._resize_fullpage_pil(rgb_im, int(train_max_dim))
                t_img = uni_omr._pil_to_chw_tensor(pil_im)
            tensors.append(t_img)
            grays.append(raw_gray)
        images = torch.stack(tensors, dim=0).to(device)

        step_vis_records.clear()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else torch.no_grad()  # unused when cpu; handled below
        )
        if use_bf16:
            with autocast_ctx:
                loss, stats = rlvr_grpo.grpo_step(
                    model,
                    ref_model,
                    images,
                    grays,
                    cfg=grpo_cfg,
                    reward_fn=_label_free_reward_fn,
                )
        else:
            loss, stats = rlvr_grpo.grpo_step(
                model,
                ref_model,
                images,
                grays,
                cfg=grpo_cfg,
                reward_fn=_label_free_reward_fn,
            )

        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()

        n_rec = max(1, len(step_vis_records))
        mean_r_vis = sum(float(r.get("r_visual", 0.0)) for r in step_vis_records) / n_rec
        render_ok_rate = sum(1.0 for r in step_vis_records if r.get("render_ok")) / n_rec
        mean_h_corr = sum(float(r.get("horizontal_ink_profile_corr", 0.0)) for r in step_vis_records) / n_rec
        mean_v_corr = sum(float(r.get("vertical_ink_profile_corr", 0.0)) for r in step_vis_records) / n_rec
        mean_ms_cos = sum(float(r.get("multi_scale_pooled_ink_cosine", 0.0)) for r in step_vis_records) / n_rec

        row = {
            "step": step_idx + 1,
            "images": [p.name for p in batch_paths],
            "loss": round(float(stats["loss"]), 6),
            "pg_loss": round(float(stats["pg_loss"]), 6),
            "kl_loss": round(float(stats["kl_loss"]), 6),
            "mean_reward": round(float(stats["mean_reward"]), 6),
            "std_reward": round(float(stats["std_reward"]), 6),
            "mean_r_visual": round(mean_r_vis, 6),
            "mean_h_corr": round(mean_h_corr, 6),
            "mean_v_corr": round(mean_v_corr, 6),
            "mean_ms_cos": round(mean_ms_cos, 6),
            "render_ok_rate": round(render_ok_rate, 4),
            "mean_r_parse": round(float(stats["mean_r_parse"]), 4),
            "mean_r_meter": round(float(stats["mean_r_meter"]), 4),
            "mean_r_noloop": round(float(stats["mean_r_noloop"]), 4),
            "seq_len": int(stats["seq_len"]),
            "grad_norm": round(grad_norm, 4),
            "step_s": round(time.perf_counter() - t_step0, 2),
        }
        history.append(row)
        print(
            f"[step {row['step']:02d}/{steps}] loss={row['loss']:.4f} pg={row['pg_loss']:.4f} "
            f"kl={row['kl_loss']:.4f} R={row['mean_reward']:.4f}±{row['std_reward']:.4f} "
            f"vis={row['mean_r_visual']:.3f} rend={row['render_ok_rate']:.2f} "
            f"parse={row['mean_r_parse']:.2f} meter={row['mean_r_meter']:.2f} "
            f"len={row['seq_len']} ({row['step_s']:.1f}s)",
            flush=True,
        )

        now_t = time.monotonic()
        if (now_t - last_save_t) >= save_every_s:
            do_hf = (now_t - last_hf_sync_t) >= 900.0
            _save_progress(step_idx + 1, sync_hf=do_hf)
            last_save_t = now_t
            if do_hf:
                last_hf_sync_t = now_t

    # Save final checkpoint atomically
    final_payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": asdict(cfg),
        "step": int(base_meta.get("step") or 0) + int(steps),
        "rlvr_step": int(steps),
        "exp_id": EXP_ID,
    }
    _atomic_save_ckpt(final_ckpt, final_payload)
    _atomic_save_ckpt(latest_ckpt, final_payload)
    vol.commit()

    # Free training memory before suite31_hw eval
    del model, ref_model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 6. Post-RLVR suite31_hw evaluation AFTER 60 GRPO updates
    print("[exp7] running AFTER RLVR suite31_hw evaluation...", flush=True)
    after_report = _run_suite31_eval(final_ckpt, after_eval_json)
    vol.commit()

    # 7. Compute before vs after deltas and write combined eval + final ledger
    deltas = {}
    for sl in ("printed_clean", "printed_photo", "hw_clean", "hw_photo", "all45_clean", "all45_photo"):
        b_s = before_report["summary"][sl]
        a_s = after_report["summary"][sl]
        deltas[sl] = {
            "note_f1_before": b_s["note_f1_mean"],
            "note_f1_after": a_s["note_f1_mean"],
            "note_f1_delta": round((a_s["note_f1_mean"] or 0.0) - (b_s["note_f1_mean"] or 0.0), 6),
            "tok_f1_before": b_s["tok_f1_mean"],
            "tok_f1_after": a_s["tok_f1_mean"],
            "tok_f1_delta": round((a_s["tok_f1_mean"] or 0.0) - (b_s["tok_f1_mean"] or 0.0), 6),
            "decode_ok_before": b_s["decode_ok_rate"],
            "decode_ok_after": a_s["decode_ok_rate"],
        }

    combined_eval = {
        "exp_id": EXP_ID,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_name,
        "baseline_ckpt": "checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
        "final_ckpt": f"checkpoints/{EXP_ID}/uni_omr_final.pt",
        "rlvr_steps": int(steps),
        "before_summary": before_report["summary"],
        "after_summary": after_report["summary"],
        "deltas": deltas,
        "before_report": before_report,
        "after_report": after_report,
    }
    combined_eval_json.write_text(json.dumps(combined_eval, indent=2), encoding="utf-8")

    final_ledger = {
        "exp_id": EXP_ID,
        "status": "completed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_name,
        "bf16": use_bf16,
        "completed_steps": int(steps),
        "elapsed_train_s": round(time.perf_counter() - t_train0, 2),
        "deltas": deltas,
        "history": history,
    }
    run_ledger_json.write_text(json.dumps(final_ledger, indent=2), encoding="utf-8")
    vol.commit()

    # 8. Upload final checkpoint, eval JSON, and run ledger to HF
    _upload_to_hf(
        api,
        final_ckpt,
        f"checkpoints/{EXP_ID}/uni_omr_final.pt",
        f"exp7: final checkpoint after {steps} render-diff RLVR steps",
    )
    _upload_to_hf(
        api,
        combined_eval_json,
        f"evals/{EXP_ID}.json",
        f"exp7: before/after suite31_hw eval after {steps} render-diff RLVR steps",
    )
    _upload_to_hf(
        api,
        run_ledger_json,
        f"runs/{EXP_ID}.json",
        f"exp7: final run ledger ({steps} steps)",
    )

    print("=== EXP7 BEFORE vs AFTER SUITE31_HW DELTAS ===", flush=True)
    print(json.dumps(deltas, indent=2), flush=True)
    return {"exp_id": EXP_ID, "status": "completed", "deltas": deltas}


@app.local_entrypoint()
def main(
    steps: int = 60,
    group_size: int = 4,
    lr: float = 2e-5,
    kl_beta: float = 0.04,
) -> None:
    res = run_exp7.remote(steps=steps, group_size=group_size, lr=lr, kl_beta=kl_beta)
    print("Exp 7 finished:", res)

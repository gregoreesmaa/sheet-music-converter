#!/usr/bin/env python3
"""Iter 13 Proposal i13-08: EAGLE-3 Multi-Layer Speculative Drafter + FSA vs MTP-4.

Experiment ID: 2026-11-26T07-40-i13-08-eagle3-fsa-spec-decode
Academic basis:
  - EAGLE-3 (Li et al., arXiv:2503.01840, 2025): multi-layer hidden feature fusion
    (low=0, mid=L//2, high=L-1) + Training-Time Test (TTT) multi-step KL distillation.
  - XGrammar / Grammar-Constrained Speculative Decoding (Dong et al., 2024-2025):
    PrefixState FSA token masking inside both drafter and target verifier for 100%
    bit-exact greedy parity and higher speculative acceptance length.

Controlled 5-arm decode benchmark on 16 real suite31_hw pages (8 printed + 8 HW):
  1. greedy_fsa_baseline
  2. mtp4_fsa
  3. eagle3_k4_no_fsa
  4. eagle3_k4_fsa
  5. eagle3_k6_fsa

Uploads evals/<EXP_ID>.json and runs/<EXP_ID>.json to HF Private Dataset Hub.
Zero edits to train_uni_v2.py or uni_omr_v2.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
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

EXP_ID = "2026-11-26T07-40-i13-08-eagle3-fsa-spec-decode"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_CANDIDATES = (
    Path("/content/bundle.tar.gz"),
    Path("/content/ov_bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
)
WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")

SEED = 1308
BASE_STEPS = 600
BASE_LR = 3e-4
TTT_STEPS = 300
TTT_LR = 5e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
BENCH_MAX_LEN = 512
SAVE_EVERY_MINUTES = 5.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and TOKEN_PATH.exists():
        try:
            os.chmod(TOKEN_PATH, 0o600)
        except OSError:
            pass
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    if TOKEN_PATH.exists():
        try:
            TOKEN_PATH.unlink()
        except OSError:
            pass
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var")
    return token


def _setup_environment() -> Path:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    VOL_ROOT.mkdir(parents=True, exist_ok=True)

    candidates = [
        WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
        Path("/content/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
    ]
    mvd_dir = next((c for c in candidates if (c / "eagle3_drafter.py").is_file()), None)
    if mvd_dir is None:
        for b in BUNDLE_CANDIDATES:
            if b.is_file():
                print(f"[{EXP_ID}] Extracting {b} -> {WORKSPACE_ROOT}...", flush=True)
                with tarfile.open(b, "r:*") as tf:
                    tf.extractall(path=WORKSPACE_ROOT)
                mvd_dir = next((c for c in candidates if (c / "eagle3_drafter.py").is_file()), None)
                if mvd_dir is not None:
                    break
    if mvd_dir is None:
        raise FileNotFoundError("Could not locate tmp-train/mvd/eagle3_drafter.py")

    needed = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)
    return mvd_dir


def _ensure_datasets(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    proxy_dir = VOL_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    if not train_abs.is_file():
        proxy_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO, filename=HF_PROXY_TAR, repo_type="dataset",
            token=hf_token, local_dir="/tmp/hf_dl",
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=proxy_dir)
        Path(tar_p).unlink(missing_ok=True)
        out_lines: List[str] = []
        for raw in (proxy_dir / "proxy_train.jsonl").read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            rec["image"] = str((proxy_dir / rec["image"]).resolve())
            rec["bekerns"] = [str((proxy_dir / b).resolve()) for b in rec.get("bekerns", [])]
            out_lines.append(json.dumps(rec))
        train_abs.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    s31_dir = VOL_ROOT / "datasets" / "suite31_hw_eval"
    if not (s31_dir / ".extracted").exists():
        s31_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO, filename=HF_SUITE31_TAR, repo_type="dataset",
            token=hf_token, local_dir="/tmp/hf_dl",
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=s31_dir)
        Path(tar_p).unlink(missing_ok=True)
        (s31_dir / ".extracted").write_text(_utc_now() + "\n", encoding="utf-8")

    manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json under {s31_dir}")
    return train_abs, manifests[0]


def _load_train_batch(
    pairs: Sequence[Tuple[Path, List[Path]]],
    rng: random.Random,
    batch_size: int,
    device: Any,
) -> Tuple[Any, Any, Any]:
    from PIL import Image
    import torch
    import uni_omr

    imgs: List[Any] = []
    seqs: List[List[int]] = []
    for _ in range(batch_size):
        img_p, beks = rng.choice(pairs)
        with Image.open(img_p) as im:
            rgb = im.convert("RGB")
        imgs.append(uni_omr._pil_to_chw_tensor(uni_omr._resize_fullpage_pil(rgb, FULLPAGE_MAX_DIM)))
        toks: List[str] = []
        for idx, bp in enumerate(beks):
            if idx > 0:
                toks.append(uni_omr.SYS_TOKEN)
            toks.extend(bp.read_text(encoding="utf-8").split())
        ids = [uni_omr.BOS_ID] + [
            uni_omr.UNI_TOKEN_TO_ID[t] for t in toks if t in uni_omr.UNI_TOKEN_TO_ID
        ] + [uni_omr.EOS_ID]
        seqs.append(ids[:MAX_SEQ_LEN])

    max_h = max(im.shape[1] for im in imgs)
    max_w = max(im.shape[2] for im in imgs)
    img_batch = torch.ones((batch_size, 3, max_h, max_w), dtype=torch.float32)
    for i, im in enumerate(imgs):
        img_batch[i, :, : im.shape[1], : im.shape[2]] = im

    max_l = max(len(s) for s in seqs)
    tok_batch = torch.full((batch_size, max_l), uni_omr.PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        tok_batch[i, : len(s)] = torch.tensor(s, dtype=torch.long)

    return img_batch.to(device), tok_batch[:, :-1].to(device), tok_batch[:, 1:].to(device)


def _train_base_and_eagle3(
    train_pairs: Sequence[Tuple[Path, List[Path]]],
    device: Any,
    amp_dtype: Any,
    use_scaler: bool,
    ckpt_dir: Path,
) -> Tuple[Any, Any, Dict[str, Any]]:
    import torch
    import eagle3_drafter as e3
    import uni_omr_v2 as v2

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt = ckpt_dir / "base_fp_600.pt"
    e3_ckpt = ckpt_dir / "eagle3_ttt_300.pt"

    cfg = v2.UniOMRv2Config.small_45m(grad_checkpoint=True, mtp_k=4)
    model = v2.UniOMRv2Net(cfg).to(device)
    rng = random.Random(SEED)

    base_wall = 0.0
    if base_ckpt.is_file():
        print(f"[{EXP_ID}] Loading cached base checkpoint {base_ckpt}...", flush=True)
        model.load_state_dict(torch.load(base_ckpt, map_location="cpu", weights_only=True))
    else:
        t_b = time.perf_counter()
        opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.04)
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        model.train()
        last_save = time.perf_counter()
        for step in range(1, BASE_STEPS + 1):
            lr = BASE_LR * min(1.0, step / 40.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / BASE_STEPS)))
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.zero_grad(set_to_none=True)
            for _ in range(GRAD_ACCUM):
                imgs, inp_ids, tgt_ids = _load_train_batch(train_pairs, rng, BATCH_SIZE, device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                    out = model(imgs, inp_ids)
                    loss_d = model.compute_loss(out, tgt_ids, images=imgs, mtp_loss_weight=0.30)
                    loss = loss_d["loss"] / float(GRAD_ACCUM)
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if step % 100 == 0 or step == BASE_STEPS:
                print(f"[{EXP_ID}] base step {step}/{BASE_STEPS} loss={loss_d['loss'].item():.4f}", flush=True)
            if (time.perf_counter() - last_save) >= SAVE_EVERY_MINUTES * 60.0:
                tmp = base_ckpt.with_suffix(".pt.tmp")
                torch.save(model.state_dict(), tmp)
                os.replace(tmp, base_ckpt)
                last_save = time.perf_counter()
        tmp = base_ckpt.with_suffix(".pt.tmp")
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, base_ckpt)
        base_wall = time.perf_counter() - t_b
        del opt

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    drafter = e3.Eagle3DraftHead.from_model(model).to(device)
    ttt_wall = 0.0
    ttt_final_loss = 0.0
    if e3_ckpt.is_file():
        print(f"[{EXP_ID}] Loading cached EAGLE-3 drafter {e3_ckpt}...", flush=True)
        drafter.load_state_dict(torch.load(e3_ckpt, map_location="cpu", weights_only=True))
    else:
        t_d = time.perf_counter()
        opt_d = torch.optim.AdamW(drafter.parameters(), lr=TTT_LR, weight_decay=0.01)
        scaler_d = torch.amp.GradScaler("cuda", enabled=use_scaler)
        drafter.train()
        last_save = time.perf_counter()
        for step in range(1, TTT_STEPS + 1):
            lr = TTT_LR * min(1.0, step / 25.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / TTT_STEPS)))
            for pg in opt_d.param_groups:
                pg["lr"] = lr
            opt_d.zero_grad(set_to_none=True)
            step_loss_val = 0.0
            for _ in range(GRAD_ACCUM):
                imgs, inp_ids, _ = _load_train_batch(train_pairs, rng, BATCH_SIZE, device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                    res = e3.compute_eagle3_ttt_loss(
                        model, drafter, inp_ids, images=imgs, k_draft=4, step_decay=0.85, temperature=1.0
                    )
                    loss = res["loss"] / float(GRAD_ACCUM)
                scaler_d.scale(loss).backward()
                step_loss_val += float(res["loss"].item()) / float(GRAD_ACCUM)
            scaler_d.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
            scaler_d.step(opt_d)
            scaler_d.update()
            ttt_final_loss = step_loss_val
            if step % 50 == 0 or step == TTT_STEPS:
                print(f"[{EXP_ID}] eagle3 TTT step {step}/{TTT_STEPS} kl_loss={step_loss_val:.4f}", flush=True)
            if (time.perf_counter() - last_save) >= SAVE_EVERY_MINUTES * 60.0:
                tmp = e3_ckpt.with_suffix(".pt.tmp")
                torch.save(drafter.state_dict(), tmp)
                os.replace(tmp, e3_ckpt)
                last_save = time.perf_counter()
        tmp = e3_ckpt.with_suffix(".pt.tmp")
        torch.save(drafter.state_dict(), tmp)
        os.replace(tmp, e3_ckpt)
        ttt_wall = time.perf_counter() - t_d
        del opt_d

    drafter.eval()
    return model, drafter, {
        "base_wall_sec": round(base_wall, 2),
        "ttt_wall_sec": round(ttt_wall, 2),
        "ttt_final_kl_loss": round(ttt_final_loss, 5),
        "drafter_params": int(drafter.param_count()),
        "drafter_int4_mb": round(drafter.param_count() * 0.5625 / 1e6, 3),
    }


def _mtp_speculative_generate(
    model: Any,
    img_t: Any,
    *,
    max_len: int,
    k_draft: int = 4,
    use_grammar: bool = True,
) -> Tuple[List[int], Dict[str, float]]:
    import torch
    import eagle3_drafter as e3
    import grammar
    import uni_omr

    mem, _, _ = model.encode_image(img_t, return_aux=False)
    state = grammar.PrefixState(strict=False) if use_grammar else None
    seq: List[int] = [uni_omr.BOS_ID]
    past_kvs: Optional[List[Any]] = None
    cur_tok = torch.tensor([[uni_omr.BOS_ID]], dtype=torch.long, device=img_t.device)

    total_proposed = 0
    total_accepted = 0
    rounds = 0

    with torch.no_grad():
        logits, mtp_logits, past_kvs = model.decode_tokens(
            cur_tok, mem, return_mtp=True, past_kvs=past_kvs, use_cache=True, return_aux=False
        )
        first_tok = e3._select_greedy_token(logits[0, -1], state)
        seq.append(first_tok)
        if state is not None:
            e3._advance_grammar_state(state, first_tok)

        while len(seq) - 1 < max_len and seq[-1] != uni_omr.EOS_ID:
            rounds += 1
            k_eff = min(int(k_draft), len(mtp_logits), max_len - (len(seq) - 1))
            if k_eff <= 0:
                break
            draft_state = state.clone() if (state is not None and hasattr(state, "clone")) else None
            draft_toks: List[int] = []
            for j in range(k_eff):
                dt = e3._select_greedy_token(mtp_logits[j][0, -1], draft_state)
                draft_toks.append(dt)
                if draft_state is not None:
                    e3._advance_grammar_state(draft_state, dt)
                if dt == uni_omr.EOS_ID:
                    break
            total_proposed += len(draft_toks)
            verify_inp = torch.tensor([[seq[-1]] + draft_toks], dtype=torch.long, device=img_t.device)
            kv_len_before = past_kvs[0][0].shape[2] if past_kvs is not None else 0
            v_logits, v_mtp, new_kvs = model.decode_tokens(
                verify_inp, mem, return_mtp=True, past_kvs=past_kvs, use_cache=True, return_aux=False
            )
            accepted = 0
            bonus_tok = uni_omr.EOS_ID
            for j, dt in enumerate(draft_toks):
                tgt_tok = e3._select_greedy_token(v_logits[0, j], state)
                if tgt_tok == dt:
                    seq.append(dt)
                    if state is not None:
                        e3._advance_grammar_state(state, dt)
                    accepted += 1
                    if dt == uni_omr.EOS_ID or (len(seq) - 1) >= max_len:
                        break
                else:
                    bonus_tok = tgt_tok
                    break
            else:
                if (len(seq) - 1) < max_len and seq[-1] != uni_omr.EOS_ID:
                    bonus_tok = e3._select_greedy_token(v_logits[0, len(draft_toks)], state)
            total_accepted += accepted
            if accepted < len(draft_toks) and (len(seq) - 1) < max_len and seq[-1] != uni_omr.EOS_ID:
                seq.append(bonus_tok)
                if state is not None:
                    e3._advance_grammar_state(state, bonus_tok)
            elif accepted == len(draft_toks) and (len(seq) - 1) < max_len and seq[-1] != uni_omr.EOS_ID:
                seq.append(bonus_tok)
                if state is not None:
                    e3._advance_grammar_state(state, bonus_tok)
            keep_len = kv_len_before + accepted + 1
            past_kvs = e3._truncate_target_past_kvs(new_kvs, keep_len)
            mtp_logits = [ml[:, accepted : accepted + 1, :] for ml in v_mtp]

    return seq, {
        "accept_rate": float(total_accepted) / float(max(1, total_proposed)),
        "mean_accepted_per_round": float(total_accepted) / float(max(1, rounds)),
        "rounds": float(rounds),
    }


def main() -> None:
    t0 = time.perf_counter()
    try:
        hf_token = _read_and_purge_hf_token()
        mvd_dir = _setup_environment()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        from PIL import Image
        import torch
        from huggingface_hub import HfApi
        import eagle3_drafter as e3
        import eval_uni_suite31 as eval_s31
        import uni_omr

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
        amp_dtype = torch.bfloat16 if (device.type == "cuda" and cc[0] >= 8) else (
            torch.float16 if device.type == "cuda" else None
        )
        use_scaler = (device.type == "cuda" and cc[0] < 8)
        train_abs, manifest_path = _ensure_datasets(hf_token)

        train_pairs: List[Tuple[Path, List[Path]]] = []
        for raw in train_abs.read_text(encoding="utf-8").splitlines():
            rec = json.loads(raw)
            img_p = Path(rec["image"])
            beks = [Path(b) for b in rec.get("bekerns", [])]
            if img_p.is_file() and beks:
                train_pairs.append((img_p, beks))

        ckpt_dir = VOL_ROOT / "ckpts" / EXP_ID
        model, drafter, train_stats = _train_base_and_eagle3(
            train_pairs, device, amp_dtype, use_scaler, ckpt_dir
        )

        _, all_pieces = eval_s31.resolve_manifest_pieces(manifest_path)
        hw_p = [p for p in all_pieces if p["handwritten"]][:8]
        pr_p = [p for p in all_pieces if not p["handwritten"]][:8]
        bench_pages: List[Tuple[str, bool, Path]] = []
        for p in pr_p + hw_p:
            bench_pages.append((p["slug"], bool(p["handwritten"]), p["clean"]))

        # Reference greedy FSA sequences for 100% bit-exact parity check
        ref_seqs: Dict[str, List[int]] = {}
        arms_cfg = [
            {"name": "greedy_fsa_baseline", "mode": "greedy", "k_draft": 1, "use_grammar": True},
            {"name": "mtp4_fsa", "mode": "mtp4", "k_draft": 4, "use_grammar": True},
            {"name": "eagle3_k4_no_fsa", "mode": "eagle3", "k_draft": 4, "use_grammar": False},
            {"name": "eagle3_k4_fsa", "mode": "eagle3", "k_draft": 4, "use_grammar": True},
            {"name": "eagle3_k6_fsa", "mode": "eagle3", "k_draft": 6, "use_grammar": True},
        ]

        arm_metrics: Dict[str, Any] = {}
        baseline_tps = 1.0
        for arm in arms_cfg:
            total_toks = 0
            total_sec = 0.0
            exact_matches = 0
            acc_rates: List[float] = []
            acc_per_round: List[float] = []

            for idx, (slug, _is_hw, img_p) in enumerate(bench_pages):
                with Image.open(img_p) as im:
                    rgb = im.convert("RGB")
                img_t = uni_omr._pil_to_chw_tensor(uni_omr._resize_fullpage_pil(rgb, FULLPAGE_MAX_DIM)).unsqueeze(0).to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_p = time.perf_counter()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                    if arm["mode"] == "greedy":
                        seq_t = (
                            eval_s31.generate_constrained(model, img_t, max_len=BENCH_MAX_LEN, strict=False)
                            if arm["use_grammar"]
                            else model.generate(img_t, max_len=BENCH_MAX_LEN)
                        )
                        seq = seq_t[0].tolist()
                        stats = {"accept_rate": 1.0, "mean_accepted_per_round": 1.0}
                    elif arm["mode"] == "mtp4":
                        seq, stats = _mtp_speculative_generate(
                            model, img_t, max_len=BENCH_MAX_LEN, k_draft=arm["k_draft"], use_grammar=arm["use_grammar"]
                        )
                    else:
                        seq_t, stats = e3.eagle3_speculative_generate(
                            model, drafter, img_t, max_len=BENCH_MAX_LEN, k_draft=arm["k_draft"],
                            use_grammar=arm["use_grammar"], return_stats=True,
                        )
                        seq = seq_t[0].tolist()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t_p

                if arm["name"] == "greedy_fsa_baseline":
                    ref_seqs[slug] = seq
                if slug in ref_seqs and seq == ref_seqs[slug]:
                    exact_matches += 1
                if idx > 0:  # skip warmup page 0 from timing
                    total_toks += max(0, len(seq) - 1)
                    total_sec += dt
                acc_rates.append(float(stats.get("accept_rate", stats.get("acceptance_rate", 0.0))))
                acc_per_round.append(float(stats.get("mean_accepted_per_round", stats.get("mean_accepted_tokens", 0.0))))

            tps = float(total_toks) / max(1e-6, total_sec)
            if arm["name"] == "greedy_fsa_baseline":
                baseline_tps = tps
            arm_metrics[arm["name"]] = {
                "tok_per_sec": round(tps, 2),
                "speedup_x": round(tps / max(1e-6, baseline_tps), 3),
                "exact_match_rate": round(float(exact_matches) / float(len(bench_pages)), 4),
                "mean_accept_rate": round(sum(acc_rates) / max(1, len(acc_rates)), 4),
                "mean_accepted_per_round": round(sum(acc_per_round) / max(1, len(acc_per_round)), 3),
                "total_tokens_timed": int(total_toks),
            }
            print(f"[{EXP_ID}] {arm['name']}: {arm_metrics[arm['name']]}", flush=True)

        payload = {
            "exp_id": EXP_ID,
            "completed_at": _utc_now(),
            "wall_sec": round(time.perf_counter() - t0, 2),
            "train_stats": train_stats,
            "n_bench_pages": len(bench_pages),
            "arms": arm_metrics,
        }
        out_eval = VOL_ROOT / "eval_out" / f"{EXP_ID}.json"
        out_run = VOL_ROOT / "eval_out" / f"{EXP_ID}_run.json"
        _atomic_write_json(out_eval, payload)
        _atomic_write_json(out_run, {"exp_id": EXP_ID, "status": "COMPLETED", "completed_at": payload["completed_at"], "wall_sec": payload["wall_sec"]})

        api = HfApi(token=hf_token)
        api.upload_file(path_or_fileobj=str(out_eval), path_in_repo=HF_EVAL_PATH, repo_id=HF_REPO, repo_type="dataset")
        api.upload_file(path_or_fileobj=str(out_run), path_in_repo=HF_RUN_PATH, repo_id=HF_REPO, repo_type="dataset")
        print(f"[{EXP_ID}] Uploaded {HF_EVAL_PATH} and {HF_RUN_PATH} successfully!", flush=True)
    finally:
        if TOKEN_PATH.exists():
            try:
                TOKEN_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()

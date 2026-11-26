"""Re-register ITERATE follow-up experiments that were dropped from queue.json.

A stale in-memory snapshot in the old run_queue_colab.py overwrote queue.json,
dropping follow-ups appended by analysts. The experiment folders are intact, so
rebuild entries from disk. Idempotent; uses the shared .queue.lock protocol.
"""
import glob
import json
import os
import re
import time

from huggingface_hub import HfApi

LOCK = "experiments/overnight_queue/.queue.lock"
QUEUE = "experiments/overnight_queue/queue.json"
REPO = "gregoreesmaa/sheet-music-converter-omr"

# folder-name substring -> list of predecessor folder-name substrings
DEPS = {
    "ov-34b-": ["ov-34-measure-beam"],
    "ov-35b-": ["ov-35-maxlen"],
    "ov-36b-": ["ov-36-hw-multiview"],
    "ov-37b-": ["ov-37-rlvr-kl"],
    "ov-38b-": ["ov-38-scribe-ttt"],
    "ov-15b-": ["ov-15-ema-decay"],
    "exp4b-": ["ablation-muon-vs-adamw", "ov-13-muon-lr"],
    "vjepa21-vicreg": ["vjepa21-gram-rism-v2"],
    "vir-router-bucket-parity": ["ablation-vir-vico-router"],
    "academic-dual-verifier-v2": ["academic-dual-verifier-rlvr-ttc"],
    "academic-gevt-vbkern-4k": ["academic-gevt-vbkern-mr"],
    # hard-depends on a converged v2 checkpoint; blocked until Iter 12 completes
    "academic-gauge-rope-converged-v2": ["academic-gauge-rope", "iter12-v2-strip-warmup-24k"],
}


def acquire(timeout=120):
    t0 = time.time()
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - t0 > timeout:
                raise TimeoutError("queue lock busy")
            time.sleep(0.3)


def main():
    api = HfApi()
    evals = {f for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith("evals/")}
    acquire()
    try:
        with open(QUEUE, encoding="utf-8") as f:
            q = json.load(f)
        have = {e["id"] for e in q}

        def resolve(sub):
            return [i for i in have if sub in i]

        added = []
        for script in sorted(glob.glob("experiments/*/run_colab.py")):
            folder = os.path.basename(os.path.dirname(script))
            if folder in have:
                continue
            key = next((k for k in DEPS if k in folder), None)
            if key is None:
                continue  # only restore known follow-ups
            deps = []
            for sub in DEPS[key]:
                hits = resolve(sub)
                # unknown predecessor (e.g. iter12) -> use its folder name if it exists
                if not hits:
                    cand = [os.path.basename(p) for p in glob.glob(f"experiments/*{sub}*")]
                    hits = cand
                deps.extend(sorted(hits))
            target = f"evals/{folder}.json"
            done = target in evals or os.path.exists(f"experiments/{folder}/OUTCOME.md")
            entry = {
                "id": folder,
                "wave": 9,
                "priority": 40,
                "provider": "colab_l4",
                "gpu": "L4",
                "est_minutes": 30,
                "script": script.replace("\\", "/"),
                "hf_eval_target": target,
                "status": "COMPLETED" if done else "QUEUED",
                "depends_on": deps,
                "restored": "queue clobber 2026-11-26",
            }
            q.append(entry)
            have.add(folder)
            added.append((folder, entry["status"], deps))
        tmp = QUEUE + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(q, f, indent=2)
            f.write("\n")
        os.replace(tmp, QUEUE)
        print(f"RESTORED {len(added)} entries; total={len(q)}")
        for a in added:
            print("  +", a[0], a[1], "deps=", a[2])
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)


if __name__ == "__main__":
    main()

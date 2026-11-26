"""Atomically merge experiments/iter13_proposals/batch_*.json into overnight_queue/queue.json.

Handles list-shaped batches and dict-shaped batches (entries under an
'experiments' / 'entries' / 'queue' / 'items' key). Idempotent: skips ids
already in the queue. Uses the shared .queue.lock + tmp/os.replace protocol.
"""
import glob
import json
import os
import time
from collections import Counter

LOCK = "experiments/overnight_queue/.queue.lock"
QUEUE = "experiments/overnight_queue/queue.json"
LIST_KEYS = ("experiments", "entries", "queue", "items", "jobs")


def entries_of(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in LIST_KEYS:
            if isinstance(obj.get(k), list):
                return obj[k]
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "id" in v[0]:
                return v
    return []


def acquire(timeout=60):
    t0 = time.time()
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - t0 > timeout:
                raise TimeoutError("queue lock busy")
            time.sleep(0.2)


def main():
    acquire()
    try:
        with open(QUEUE, encoding="utf-8") as f:
            q = json.load(f)
        seen = {e["id"] for e in q}
        added, skipped = [], 0
        for bp in sorted(glob.glob("experiments/iter13_proposals/batch_*.json")):
            with open(bp, encoding="utf-8") as f:
                items = entries_of(json.load(f))
            if not items:
                print(f"WARN no entries parsed from {bp}")
            for it in items:
                if not isinstance(it, dict) or "id" not in it:
                    continue
                if it["id"] in seen:
                    skipped += 1
                    continue
                it.setdefault("status", "QUEUED")
                it.setdefault("depends_on", [])
                it.setdefault("gpu", "L4")
                q.append(it)
                seen.add(it["id"])
                added.append(it["id"])
        tmp = QUEUE + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(q, f, indent=2)
            f.write("\n")
        os.replace(tmp, QUEUE)
        print(f"MERGED {len(added)} new, skipped {skipped} dupes, total={len(q)}")
        print("status:", dict(Counter(e.get("status") for e in q)))
        for i in added:
            print("  +", i)
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)


if __name__ == "__main__":
    main()

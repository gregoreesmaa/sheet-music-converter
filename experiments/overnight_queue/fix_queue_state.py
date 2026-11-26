import json, os, time
LOCK = "experiments/overnight_queue/.queue.lock"
QUEUE = "experiments/overnight_queue/queue.json"
FALSE_DONE = ("ov-16-", "ov-17-", "ov-19-", "ov-20-", "ov-21-")
t0 = time.time()
while True:
    try:
        os.close(os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)); break
    except FileExistsError:
        if time.time() - t0 > 120: raise
        time.sleep(0.3)
try:
    q = json.load(open(QUEUE, encoding="utf-8"))
    fixed_self, reset = 0, []
    for e in q:
        if e["id"] in e.get("depends_on", []):
            e["depends_on"] = [d for d in e["depends_on"] if d != e["id"]]
            fixed_self += 1
        if e.get("status") == "COMPLETED" and any(k in e["id"] for k in FALSE_DONE):
            for k in ("started_at", "completed_at", "elapsed_sec", "returncode", "session", "log_path", "blocked_by"):
                e.pop(k, None)
            e["status"] = "QUEUED"
            e["note"] = "reset: false COMPLETED (inner rc=1 hidden by colab exec)"
            reset.append(e["id"])
    tmp = QUEUE + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(q, f, indent=2); f.write("\n")
    os.replace(tmp, QUEUE)
    print("self-deps removed:", fixed_self); print("reset to QUEUED:", reset)
finally:
    if os.path.exists(LOCK): os.remove(LOCK)

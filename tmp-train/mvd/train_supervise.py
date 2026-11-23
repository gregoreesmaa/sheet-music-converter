"""Restart-on-crash supervisor for long unattended training runs.

S13 context: the VRAM battle is won (width-cap skip + defrag, both
verified on disk), but this box kills GPU contexts every few hours
with no warning (nvlddmkm Event 153 habit, gaming overlap) — deaths
that no training-code change can prevent. This wrapper relaunches
``train.py --resume`` after any nonzero exit, so each incident costs
at most the maintenance-cadence window (<= 500 steps), never the run.

Contract (exit codes): 0 = training finished (train.py exited 0),
1 = STOP file present (user-requested hold), 2 = retry budget
exhausted, 3 = launcher misuse. Numerics-neutral: resume is exact
(same batches, optimizer+scheduler sidecars), so restarts change
nothing but wall-clock.

Usage:
    python train_supervise.py --tries 20 --cooldown 60
        --stop-file D:/tmp/t32s13/STOP -- <train.py args...>
"""
import subprocess
import sys
import time
from pathlib import Path


def decide(exit_code, attempt, tries, stop_exists):
    """One restart decision. ``attempt`` counts attempts used so far."""
    if exit_code == 0:
        return 'done'
    if stop_exists:
        return 'stop'
    if attempt >= tries:
        return 'exhausted'
    return 'retry'


def cooldown_for(consec_early, base=120):
    """Seconds to wait after a crash.

    A sick driver (poisoned CUDA context, SIGKILLed VRAM holder,
    nvlddm wobble) needs minutes — not the base cooldown — before
    a retry can succeed; hammering it burns the try budget in a
    death spiral. Doubles per consecutive early death, cap 30 min.
    """
    return min(base * (2 ** max(0, consec_early - 1)), 1800)


def ensure_resume(train_args):
    """Return a copy of ``train_args`` containing exactly one --resume (no --init-from)."""
    args = []
    i = 0
    while i < len(train_args):
        if train_args[i] == '--resume':
            i += 1
        elif train_args[i] == '--init-from':
            i += 2  # drop --init-from <ckpt> once resuming
        else:
            args.append(train_args[i])
            i += 1
    args.append('--resume')
    return args


def _has_init_from(train_args):
    return '--init-from' in train_args


def main(argv):
    try:
        sep = argv.index('--')
    except ValueError:
        print('usage: train_supervise.py [--tries N] [--cooldown S] '
              '[--stop-file P] -- <train.py args...>', flush=True)
        return 3
    opts, train_args = argv[:sep], argv[sep + 1:]
    tries, cooldown, stop_file = 20, 60, None
    i = 0
    while i < len(opts):
        if opts[i] == '--tries':
            tries = int(opts[i + 1])
            i += 2
        elif opts[i] == '--cooldown':
            cooldown = int(opts[i + 1])
            i += 2
        elif opts[i] == '--stop-file':
            stop_file = opts[i + 1]
            i += 2
        else:
            print('unknown supervisor option: %s' % opts[i],
                  flush=True)
            return 3
    if not train_args:
        print('no train.py args after --', flush=True)
        return 3
    first_args = [a for a in train_args if a != '--resume'] if _has_init_from(train_args) else ensure_resume(train_args)
    resume_args = ensure_resume(train_args)
    here = Path(__file__).resolve().parent
    attempt = 0
    consec_early = 0
    while True:
        attempt += 1
        cur_args = first_args if attempt == 1 else resume_args
        print('supervisor: attempt %d/%d: python %s'
              % (attempt, tries,
                 ' '.join(['train.py'] + cur_args)), flush=True)
        born = time.time()
        proc = subprocess.run(
            [sys.executable, str(here / 'train.py')] + cur_args)
        lived = time.time() - born
        # An attempt that survives 10+ min died of something else
        # than startup sickness; only chain-escalate fast repeats.
        consec_early = consec_early + 1 if lived < 600 else 0
        action = decide(proc.returncode, attempt, tries,
                        stop_file is not None
                        and Path(stop_file).exists())
        print('supervisor: exit=%s action=%s lived=%.0fs'
              % (proc.returncode, action, lived), flush=True)
        if action == 'done':
            return 0
        if action == 'stop':
            return 1
        if action == 'exhausted':
            return 2
        wait = cooldown_for(consec_early, base=cooldown)
        print('supervisor: cooling down %ds' % wait, flush=True)
        time.sleep(wait)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

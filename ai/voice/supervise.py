"""
Autonomous overnight SER training supervisor.

Runs two stages, self-healing CUDA OOM by shrinking the batch / clip length:

  v0    — train + eval on whatever commercial-clean data is ready now
  full  — wait for the Bangla/Kannada/Russian downloads, then re-prepare the
          whole manifest and train + eval the real multilingual model

Everything (timeline, every error, every fix, final metrics) is written to
``ai/training/ser/RUN_REPORT.json`` as it happens. Progress lines are printed as
``SUPERVISOR| ...`` for the watching monitor.

    nohup .venv/bin/python -m ai.voice.supervise &
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = str(ROOT / ".venv" / "bin" / "python")
SER = ROOT / "ai" / "training" / "ser"
CKPT = SER / "checkpoints"
MODEL_DIR = CKPT / "xls-r-300m-ser"
REPORT = SER / "RUN_REPORT.json"

CKPT.mkdir(parents=True, exist_ok=True)

report: dict = {
    "started": datetime.now().isoformat(timespec="seconds"),
    "backbone": "facebook/wav2vec2-xls-r-300m",
    "events": [],
    "errors_and_fixes": [],
    "stages": {},
}


def save() -> None:
    report["updated"] = datetime.now().isoformat(timespec="seconds")
    REPORT.write_text(json.dumps(report, indent=2))


def log(msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} | {msg}"
    print(line, flush=True)
    report["events"].append(line)
    save()


def emit(msg: str) -> None:
    print(f"SUPERVISOR| {msg}", flush=True)


LIVE_LOG = CKPT / "live.log"          # always the currently-running step, tail -f this


def run(mod_args: list[str], log_name: str | None = None, timeout: int | None = None) -> tuple[int, str]:
    log(f"run: python -m {' '.join(mod_args)}")
    dest = CKPT / log_name if log_name else LIVE_LOG
    lines: list[str] = []
    start = time.time()
    with open(dest, "w", encoding="utf-8") as fout, open(LIVE_LOG, "w", encoding="utf-8") as flive:
        proc = subprocess.Popen(
            [PY, "-u", "-m", *mod_args], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            for f in (fout, flive):
                f.write(line)
                f.flush()
            if timeout and time.time() - start > timeout:
                proc.kill()
                lines.append("\n<timeout — killed>\n")
                break
        proc.wait()
    return proc.returncode, "".join(lines)


def manifest_summary() -> dict:
    import csv
    from collections import Counter

    mpath = SER / "manifest.csv"
    if not mpath.exists():
        return {}
    rows = list(csv.DictReader(open(mpath, encoding="utf-8")))
    return {
        "clips": len(rows),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_language": dict(Counter(r["language"] for r in rows)),
        "by_emotion": dict(Counter(r["emotion"] for r in rows)),
        "by_corpus": dict(Counter(r["corpus"] for r in rows)),
    }


def train_with_heal(name: str) -> bool:
    batch, accum, maxsec = 4, 8, 6
    for attempt in range(1, 7):
        log(f"[{name}] train attempt {attempt}: batch={batch} grad_accum={accum} max_seconds={maxsec}")
        rc, out = run(
            ["ai.voice.train", "--epochs", "4", "--batch", str(batch),
             "--grad-accum", str(accum), "--max-seconds", str(maxsec)],
            log_name=f"{name}_train_attempt{attempt}.log",
        )
        if "saved to" in out and rc == 0:
            log(f"[{name}] training succeeded on attempt {attempt}")
            report["stages"].setdefault(name, {})
            report["stages"][name].update(
                {"trained": True, "attempt": attempt, "batch": batch,
                 "grad_accum": accum, "max_seconds": maxsec}
            )
            save()
            return True

        low = out.lower()
        if "out of memory" in low or "cuda out of memory" in low or "outofmemory" in low:
            if batch > 1:
                batch //= 2
                accum *= 2
                fix = f"CUDA OOM -> batch={batch}, grad_accum={accum}"
            elif maxsec > 4:
                maxsec -= 1
                fix = f"CUDA OOM -> max_seconds={maxsec}"
            else:
                report["errors_and_fixes"].append(
                    {"stage": name, "attempt": attempt, "error": "CUDA OOM",
                     "fix": "hit the floor (batch=1, max_seconds=4) — cannot shrink further",
                     "resolved": False})
                save()
                emit(f"FAIL {name}: OOM floor")
                return False
            report["errors_and_fixes"].append(
                {"stage": name, "attempt": attempt, "error": "CUDA OOM", "fix": fix, "resolved": None})
            log(f"[{name}] {fix}")
            save()
            continue

        # unknown failure
        report["errors_and_fixes"].append(
            {"stage": name, "attempt": attempt, "error": "unknown training failure",
             "fix": "needs manual review", "resolved": False,
             "log": f"checkpoints/{name}_train_attempt{attempt}.log",
             "tail": out[-2500:]})
        save()
        log(f"[{name}] UNKNOWN failure — see checkpoints/{name}_train_attempt{attempt}.log")
        emit(f"FAIL {name}: unknown error, needs review")
        return False
    return False


def evaluate(name: str) -> None:
    rc, out = run(["ai.voice.evaluate"], log_name=f"{name}_eval.log")
    metrics_path = MODEL_DIR / "test_metrics.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        (CKPT / f"{name}_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    report["stages"].setdefault(name, {})
    report["stages"][name]["eval"] = metrics or {"error": "evaluate produced no metrics", "rc": rc}
    save()
    emit(f"DONE {name}: macro_f1={metrics.get('macro_f1', '?')}")


def wait_for_downloads(max_minutes: int = 180) -> None:
    log("waiting for background dataset downloads to finish ...")
    for _ in range(max_minutes):
        r = subprocess.run(["pgrep", "-f", "ser.download"], capture_output=True, text=True)
        if not r.stdout.strip():
            log("no download process running")
            return
        time.sleep(60)
    log("download wait timed out — proceeding with what is on disk")


def main() -> int:
    log("==== supervisor start ====")
    emit("START")

    # ---------------- Stage v0 ----------------
    rc, out = run(["ai.voice.prepare", "--corpus", "ravdess", "crema_d", "urdu"],
                  log_name="v0_prepare.log")
    report["stages"]["v0"] = {"prepare": manifest_summary()}
    save()
    if (SER / "manifest.csv").exists() and train_with_heal("v0"):
        evaluate("v0")
    else:
        emit("v0 training did not complete")

    # keep the v0 model aside
    if MODEL_DIR.exists():
        v0_dir = CKPT / "xls-r-300m-ser-v0"
        if v0_dir.exists():
            import shutil
            shutil.rmtree(v0_dir)
        MODEL_DIR.rename(CKPT / "xls-r-300m-ser-v0")

    # ---------------- Stage full ----------------
    wait_for_downloads()
    run(["ai.voice.download", "--corpus", "subesco", "kannada", "resd"],
        log_name="full_download.log")
    run(["ai.voice.prepare", "--corpus", "all"], log_name="full_prepare.log")
    report["stages"]["full"] = {"prepare": manifest_summary()}
    save()

    if (SER / "manifest.csv").exists() and train_with_heal("full"):
        evaluate("full")
    else:
        emit("full training did not complete")

    report["finished"] = datetime.now().isoformat(timespec="seconds")
    save()
    log("==== supervisor finished ====")
    emit("ALL DONE — see ai/training/ser/RUN_REPORT.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        report["errors_and_fixes"].append(
            {"stage": "supervisor", "error": f"crashed: {type(exc).__name__}: {exc}", "resolved": False})
        save()
        emit(f"SUPERVISOR CRASHED: {exc}")
        raise

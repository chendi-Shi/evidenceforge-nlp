"""Launch one resumable, time-bounded local run with hidden Windows workers.

The process owns only its child model server. It never stops unrelated processes,
changes power settings, provisions cloud compute, or silently restarts itself.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-hours", type=float, default=11.5)
    args = parser.parse_args()
    if os.name != "nt" or not 0 < args.max_hours <= 12:
        raise ValueError("This detached launcher targets Windows and a budget no greater than 12 hours")
    root = Path(__file__).resolve().parents[1]
    output = root / "results/distillation"
    if not (output / "protocol.json").exists():
        raise RuntimeError("Run the pilot and register the protocol before launching the long experiment")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runtime.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "evidenceforge.distill_experiment", "run", "--max-hours", str(args.max_hours)],
            cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB,
        )
    print(json.dumps({"pid": process.pid, "max_hours": args.max_hours,
                      "status": str(output / "live_status.json"), "log": str(output / "runtime.log")}))
    history_path = output / "launch_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    history.append({"pid": process.pid, "max_hours": args.max_hours,
                    "started_at_utc": datetime.now(timezone.utc).isoformat()})
    temporary = history_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, history_path)


if __name__ == "__main__":
    main()

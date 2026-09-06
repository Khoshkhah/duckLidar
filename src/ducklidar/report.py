"""One line per stage: the pipeline's flight recorder.

Every stage, on completion, appends a single summary line to the pipeline
log — timestamp, stage name, its key facts. The log reads as the history
of the build (and greps as a table):

    2026-08-02T15:41:07 stage2   partitions=64 points=191,950,769 edges=447,506,102 elapsed=723s
    2026-08-02T16:02:11 stage4   points=168,377,550 components=9,548 largest=167,023,319 elapsed=1810s

The path comes from the LIDAR_REPORT environment variable, or the `log`
argument; the line is also printed, so tee'd run logs carry it too.
"""
import datetime
import os
from pathlib import Path

__all__ = ["stage_report"]


def stage_report(stage, facts, log=None):
    """Append one summary line for `stage`. `facts` is an ordered dict of
    key → value; ints are thousands-separated. Returns the line."""
    log = Path(log or os.environ.get("LIDAR_REPORT", "pipeline_report.log"))
    body = " ".join(f"{k}={v:,}" if isinstance(v, int) and not isinstance(v, bool)
                    else f"{k}={v}" for k, v in facts.items())
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {stage:<9} {body}"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)
    return line

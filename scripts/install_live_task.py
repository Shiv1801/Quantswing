"""Register the live paper updates in Windows Task Scheduler: every ``poll_minutes`` from
``poll_start`` to ``poll_end``, Monday–Friday (config ``paper.live``). Holidays are skipped
by the script itself. The PC must be on and you logged in to Windows.

Usage::

    python scripts/install_live_task.py           # create / update
    python scripts/install_live_task.py --remove  # delete
    python scripts/install_live_task.py --show    # last / next run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from quantswing.core.config import REPO_ROOT, get_config


def ps(cmd: str) -> int:
    """Run a PowerShell command."""
    return subprocess.call(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd]
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)
    if sys.platform != "win32":
        print("Windows only (Task Scheduler).")
        return 1
    lc = get_config().paper.live
    name = lc.task_name
    if args.remove:
        return ps(f"Unregister-ScheduledTask -TaskName '{name}' -Confirm:$false")
    if args.show:
        return ps(
            f"Get-ScheduledTask -TaskName '{name}' | Get-ScheduledTaskInfo | "
            "Format-List LastRunTime,LastTaskResult,NextRunTime"
        )
    py = Path(sys.executable)
    pyw = py.with_name("pythonw.exe")
    exe = pyw if pyw.exists() else py  # no console window
    script = REPO_ROOT / "scripts" / "live_poll.py"
    start = lc.poll_start.strftime("%H:%M")
    span = datetime.combine(datetime.today(), lc.poll_end) - datetime.combine(
        datetime.today(), lc.poll_start
    )
    minutes = int(span.total_seconds() // 60)
    cmd = (
        f"$a = New-ScheduledTaskAction -Execute '{exe}' -Argument '\"{script}\"' -WorkingDirectory '{REPO_ROOT}'; "
        f"$t = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At {start}; "
        f"$r = New-ScheduledTaskTrigger -Once -At {start} -RepetitionInterval (New-TimeSpan -Minutes {lc.poll_minutes}) "
        f"-RepetitionDuration (New-TimeSpan -Minutes {minutes}); "
        "$t.Repetition = $r.Repetition; "
        "$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
        "-MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 4); "
        f"Register-ScheduledTask -TaskName '{name}' -Action $a -Trigger $t -Settings $s -Force | Out-Null; "
        f"Write-Output 'Scheduled: {name} (Mon-Fri every {lc.poll_minutes} min, {start}-{lc.poll_end.strftime('%H:%M')})'"
    )
    return ps(cmd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

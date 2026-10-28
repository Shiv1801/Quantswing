"""Register the daily paper run in Windows Task Scheduler (Mon–Fri, config paper.schedule).

The task runs as you, when you are logged in; if the PC was off/asleep at that time it
runs as soon as possible afterwards ("start when available"), and the engine catches up.

Usage::

    python scripts/install_schedule.py           # create / update the task
    python scripts/install_schedule.py --remove  # delete it
    python scripts/install_schedule.py --show    # show its status
"""

from __future__ import annotations

import argparse
import subprocess
import sys
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
    cfg = get_config().paper.schedule
    name = cfg.task_name
    if args.remove:
        return ps(f"Unregister-ScheduledTask -TaskName '{name}' -Confirm:$false")
    if args.show:
        return ps(
            f"Get-ScheduledTask -TaskName '{name}' | Get-ScheduledTaskInfo | "
            "Format-List LastRunTime,LastTaskResult,NextRunTime"
        )
    py = Path(sys.executable)
    pyw = py.with_name("pythonw.exe")
    exe = pyw if pyw.exists() else py  # pythonw: no console window pops up
    script = REPO_ROOT / "scripts" / "paper_daily.py"
    at = cfg.time.strftime("%H:%M")
    cmd = (
        f"$a = New-ScheduledTaskAction -Execute '{exe}' -Argument '\"{script}\"' "
        f"-WorkingDirectory '{REPO_ROOT}'; "
        f"$t = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At {at}; "
        "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 6); "
        f"Register-ScheduledTask -TaskName '{name}' -Action $a -Trigger $t -Settings $s -Force | Out-Null; "
        f"Write-Output 'Scheduled: {name} (Mon-Fri {at})'"
    )
    return ps(cmd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

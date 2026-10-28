"""Background-process helpers for the app's buttons (Windows and Linux, no extra packages)."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys


def pid_alive(pid: int | None) -> bool:
    """True if a process with this id is still running."""
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def kill_tree(pid: int | None) -> bool:
    """Stop a process and its children. True if it is gone afterwards."""
    if not pid_alive(pid):
        return True
    if sys.platform == "win32":
        subprocess.call(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=0x08000000,  # no console window
        )
    else:
        with contextlib.suppress(OSError):
            os.kill(int(pid), signal.SIGKILL)  # type: ignore[arg-type]
    return not pid_alive(pid)


def kill_scripts(*names: str) -> None:
    """Stop every Python process running one of these scripts (safety net for jobs started
    by older versions, which did not record their process id)."""
    if sys.platform == "win32":
        pat = "|".join(n.replace(".", r"\.") for n in names)
        cmd = (
            "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
            f"Where-Object {{ $_.CommandLine -match '{pat}' }} | "
            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
        )
        subprocess.call(
            ["powershell", "-NoProfile", "-Command", cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=0x08000000,
        )
    else:
        for n in names:
            subprocess.call(
                ["pkill", "-f", n], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

"""How much RAM the process can still take, from the standard library alone (no psutil).

Loading a ~2GB model (the bge-m3 embedder, the bge-reranker-v2-m3 cross-encoder) into a process the
machine has no room left for doesn't raise a Python MemoryError: the Rust tokenizer aborts the whole
process ("memory allocation of 8388608 bytes failed", Windows exit code 3221226505 = 0xC0000409),
taking the app down with it. Callers check `available_bytes()` first and skip what's optional.
"""

from __future__ import annotations

import ctypes
import sys


def available_bytes() -> int | None:
    """Bytes the process can still allocate without the OS refusing, or None when unknown (then
    callers don't guard). Windows: the smaller of the free physical memory and the commit charge
    left (an allocation past the commit limit fails even with physical RAM free). Linux:
    MemAvailable. Elsewhere: None."""
    try:
        if sys.platform == "win32":
            return _windows_available()
        if sys.platform.startswith("linux"):
            return _linux_available()
    except Exception:  # noqa: BLE001 -- an unreadable figure just means no guard
        return None
    return None


def _windows_available() -> int | None:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        return None
    return min(status.ullAvailPhys, status.ullAvailPageFile)


def _linux_available() -> int | None:
    with open("/proc/meminfo", encoding="ascii") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def has_room(needed_gb: float | None) -> tuple[bool, int | None]:
    """(whether `needed_gb` GB is available -- True when unknown or no bar is set, the bytes free)."""
    free = available_bytes()
    if not needed_gb or free is None:
        return True, free
    return free >= needed_gb * 1024**3, free

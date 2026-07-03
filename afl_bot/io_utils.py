"""Atomic file-write utilities.

Every JSON / report / CSV write in this project uses ``atomic_write_text``
so a killed process never leaves a half-written file.  The pattern is:

    write to <path>.tmp-<pid>  →  fsync  →  os.replace(<tmp>, <path>)

``os.replace`` is atomic on both POSIX (rename(2)) and Windows (MoveFileEx
with MOVEFILE_REPLACE_EXISTING), so the destination is either the old file or
the new one — never a partial write.
"""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Write *text* to *path* atomically.

    Writes to ``<path>.tmp-<pid>`` in the same directory (so the rename stays
    on the same filesystem), fsyncs, then replaces the target.  On any error
    the temp file is cleaned up and the original is untouched.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp-{os.getpid()}")
    try:
        data = text.encode(encoding)
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise

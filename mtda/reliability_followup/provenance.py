"""Seal prior baselines and hash the independent follow-up runtime."""

import hashlib
from pathlib import Path

from mtda.experiment import PROJECT, file_sha
from mtda.reliability.experiment import implementation_hashes as previous_hashes

SEALED_RELIABILITY = "f33008a054906a428f079d001a7eb84df87a97c8bd6789d231662b39c6ff5bb2"


def implementation_hashes():
    baseline, reliability = previous_hashes()
    if reliability != SEALED_RELIABILITY:
        raise ValueError("preserved R0–R6 runtime changed")
    files = [PROJECT / "train_b_followup.py", *sorted(Path(__file__).parent.glob("*.py"))]
    return baseline, hashlib.sha256((reliability + "".join(file_sha(p) for p in files)).encode()).hexdigest()

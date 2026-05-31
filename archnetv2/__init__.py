"""ArchNetv2 — multiscale object detection for architectural floor plans.

Reimplementation of:
    Xu, Jha, Mehadi, Mandal. "Multiscale object detection on complex
    architectural floor plans." Automation in Construction 165 (2024)
    105486. https://doi.org/10.1016/j.autcon.2024.105486

Configured for openings (door + window). See README in this folder.
"""
from .model import AC_CBAM, build_archnetv2  # noqa: F401

"""Readers for competition inputs whose use is permitted by the applicable data terms."""

from possession_lab.io.atd import atd_paths, check_alignment_raw, frame_id_of, load_atd
from possession_lab.io.homography import load_homographies, project_pitch_to_image
from possession_lab.io.smart_tagging import check_kickoff_alignment, load_smart_tagging

__all__ = [
    "atd_paths",
    "check_alignment_raw",
    "check_kickoff_alignment",
    "frame_id_of",
    "load_atd",
    "load_homographies",
    "load_smart_tagging",
    "project_pitch_to_image",
]

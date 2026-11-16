"""Tests enforcing strict data firewall and zero leakage of audit/validation matches (N020)."""

import pytest

from possession_lab.neural.pathcrf.labels import generate_training_path_annotations


def test_denylist_raises_value_error():
    """Access to validation, audit, or challenge matches for training labels must be blocked."""
    forbidden = [
        "J03WN1",
        "idsse_J03WN1",
        "J03WR9",
        "idsse_J03WR9",
        "J03WQQ",
        "idsse_J03WQQ",
        "metrica_2",
        "metrica_3",
        "challenge",
    ]
    for match_id in forbidden:
        with pytest.raises(ValueError, match="SECURITY / LEAKAGE VIOLATION"):
            generate_training_path_annotations(match_id, n_frames=100)


def test_approved_train_matches_allowed():
    """Approved training matches must pass the firewall."""
    approved = [
        "J03WMX",
        "idsse_J03WMX",
        "J03WOH",
        "idsse_J03WOH",
        "J03WOY",
        "idsse_J03WOY",
        "J03WPY",
        "idsse_J03WPY",
    ]
    for match_id in approved:
        ann = generate_training_path_annotations(match_id, n_frames=100)
        assert ann.source_match in match_id


def test_unapproved_matches_return_none_available():
    """Matches without approved strong paths return confident=False without fabricating labels."""
    ann = generate_training_path_annotations("skillcorner_1874553", n_frames=100)
    assert ann.annotation_method == "none_available"
    assert not ann.confident.any()
    assert (ann.edge_id == -1).all()

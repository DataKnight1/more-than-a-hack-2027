"""Tests for node mapping, slot stability, and pitch boundary geometry (N020)."""

import numpy as np

from possession_lab.neural.pathcrf.nodes import (
    OutsideNodeGeometry,
    map_canonical_players_to_26_nodes,
    select_stable_team_slots,
)


def test_outside_node_geometry():
    geo = OutsideNodeGeometry(pitch_x=105.0, pitch_y=68.0)
    std = geo.standard_coords
    assert std.shape == (4, 2)
    # out_left: (0, 34)
    np.testing.assert_allclose(std[0], [0.0, 34.0])
    # out_right: (105, 34)
    np.testing.assert_allclose(std[1], [105.0, 34.0])
    # out_bottom: (52.5, 0)
    np.testing.assert_allclose(std[2], [52.5, 0.0])
    # out_top: (52.5, 68)
    np.testing.assert_allclose(std[3], [52.5, 68.0])

    cen = geo.centered_coords
    assert cen.shape == (4, 2)
    np.testing.assert_allclose(cen[0], [-52.5, 0.0])
    np.testing.assert_allclose(cen[1], [52.5, 0.0])
    np.testing.assert_allclose(cen[2], [0.0, -34.0])
    np.testing.assert_allclose(cen[3], [0.0, 34.0])


def test_select_stable_team_slots_churn():
    # 16 slots with different observation frequencies
    T = 100
    team_pos = np.full((T, 16, 2), np.nan, dtype=np.float32)
    # Give slots 0..10 high observation counts, 11..15 low counts
    for s in range(11):
        team_pos[: 90 - s, s, :] = 1.0
    for s in range(11, 16):
        team_pos[:10, s, :] = 1.0

    selected = select_stable_team_slots(team_pos, target_slots=11)
    assert len(selected) == 11
    assert selected == list(range(11))


def test_map_canonical_players_26_nodes():
    T = 50
    home_pos = np.random.uniform(-40, 40, (T, 11, 2)).astype(np.float32)
    away_pos = np.random.uniform(-40, 40, (T, 11, 2)).astype(np.float32)

    # Inject dropouts in slot 5 of home team
    home_pos[10:20, 5, :] = np.nan

    nodes_xy, node_valid = map_canonical_players_to_26_nodes(
        home_pos, away_pos, pitch=(105.0, 68.0), standard_coords=True
    )

    assert nodes_xy.shape == (T, 26, 2)
    assert node_valid.shape == (T, 26)
    assert np.isfinite(nodes_xy).all()

    # Outside nodes must always be valid
    assert node_valid[:, 22:].all()

    # Slot 5 must be invalid at frames 10..19
    assert not node_valid[10:20, 5].any()
    assert node_valid[0:10, 5].all()


def test_all_22_players_absent():
    T = 20
    home_pos = np.full((T, 11, 2), np.nan, dtype=np.float32)
    away_pos = np.full((T, 11, 2), np.nan, dtype=np.float32)

    nodes_xy, node_valid = map_canonical_players_to_26_nodes(
        home_pos, away_pos, pitch=(105.0, 68.0), standard_coords=True
    )
    assert nodes_xy.shape == (T, 26, 2)
    assert not node_valid[:, :22].any()
    assert node_valid[:, 22:].all()
    assert np.isfinite(nodes_xy).all()

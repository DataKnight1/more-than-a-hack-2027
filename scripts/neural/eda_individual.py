"""Individual-player and ball-touch EDA on EXTERNAL TRAINING matches only.

Motivated by ``experiments/research/neural/ERROR_ANALYSIS.md``: the champion misses most possessions shorter
than 2 s (mostly turnovers), and each miss also merges the two neighbouring possessions. Questions:

A. *Ball touches.* What do ball speed, acceleration and turning look like at a possession change,
   and does "team of the nearest player at each detected touch" recover the label changes?
B. *Short possessions.* What distinguishes a turnover that lasts < 2 s from one that lasts?
C. *Roles and marking.* Identity-free roles per frame (most advanced outfield = striker, deepest
   outfield = last defender, deepest wide players = full-backs). Does the defending side *follow*
   the attacking side's movement (lead-lag of each player vs their nearest opponent)?
D. *Pressing and recoveries.* Turnover hazard of a ball carrier as a function of opponent
   pressure; spell duration by recovery zone and by counter-press intensity; team expansion after
   a recovery.
E. *Dribbles.* Carrier speed and how carries end.

Only training-split matches are read (IDSSE 4, SkillCorner 12, Metrica game 1). Output: aggregate
tables in ``experiments/research/neural/EDA_INDIVIDUAL.md``.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "neural"))

import numpy as np  # noqa: E402
from eda_possession import TRAIN, auc  # noqa: E402

from possession_lab.neural.canonical import A, CanonicalMatch, D, H  # noqa: E402
from possession_lab.neural.team_features import attack_direction, last_known, velocity  # noqa: E402

FPS = 25
SHORT = 2 * FPS
LAG = 10  # 0.4 s lead-lag horizon


def nan_hypot(v: np.ndarray) -> np.ndarray:
    """Compute NaN-aware Euclidean magnitudes for coordinate arrays."""
    return np.hypot(v[..., 0], v[..., 1])


def ball_kinematics(ball_xy: np.ndarray) -> dict[str, np.ndarray]:
    """Speed (m/s), acceleration magnitude (m/s^2) and turning angle (deg over +/-0.12 s)."""
    v = velocity(ball_xy, 2)
    acc = nan_hypot(velocity(v, 2))
    k = 3
    v_prev = np.concatenate([np.full((k, 2), np.nan), v[:-k]])
    v_next = np.concatenate([v[k:], np.full((k, 2), np.nan)])
    dot = (v_prev * v_next).sum(1)
    norm = nan_hypot(v_prev) * nan_hypot(v_next)
    turn = np.degrees(np.arccos(np.clip(dot / np.maximum(norm, 1e-6), -1, 1)))
    turn[(nan_hypot(v_prev) < 1.5) & (nan_hypot(v_next) < 1.5)] = 0.0
    return {"speed": nan_hypot(v), "acc": acc, "turn": turn}


def touch_frames(kin: dict[str, np.ndarray], alive: np.ndarray) -> np.ndarray:
    """Candidate touches: local acceleration maxima (+/-3 frames) that are strong or turn the ball."""
    acc = np.nan_to_num(kin["acc"], nan=0.0)
    n = len(acc)
    pad = np.concatenate([np.zeros(3), acc, np.zeros(3)])
    local = np.ones(n, bool)
    for s in range(7):
        if s != 3:
            local &= acc >= pad[s : s + n]
    strong = (acc > 12.0) | ((np.nan_to_num(kin["turn"]) > 25.0) & (acc > 5.0))
    return np.flatnonzero(local & strong & alive & np.isfinite(kin["acc"]))


def nearest(players: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame nearest present player index and distance to ``point`` [T, 2]."""
    d = nan_hypot(players - point[:, None, :])
    d = np.where(np.isnan(d), np.inf, d)
    idx = np.argmin(d, axis=1)
    dist = d[np.arange(len(d)), idx]
    return idx, np.where(np.isinf(dist), np.nan, dist)


def spell_ends(lab: np.ndarray) -> np.ndarray:
    """For each frame, the first frame at which the label differs (exclusive end of its segment)."""
    n = len(lab)
    change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    ends = np.full(n, n)
    pos = np.searchsorted(change, np.arange(n), side="right")
    ok = pos < len(change)
    ends[ok] = change[pos[ok]]
    return ends


def match_events(gt_frames: np.ndarray, pred_frames: np.ndarray, tol: int) -> int:
    """Greedy one-to-one matches within ``tol`` frames (both sorted)."""
    used = np.zeros(len(pred_frames), bool)
    hits = 0
    for g in gt_frames:
        lo, hi = np.searchsorted(pred_frames, [g - tol, g + tol + 1])
        for j in range(lo, hi):
            if not used[j]:
                used[j] = True
                hits += 1
                break
    return hits


def analyse_match(m: CanonicalMatch, acc: dict[str, list]) -> None:  # noqa: PLR0915
    """Compute player and possession diagnostics for one external training match."""
    lab, ok = m.labels, m.period > 0
    alive = np.isin(lab, (H, A)) & ok
    xy = m.ball[:, :2]
    seen = ~np.isnan(xy[:, 0])
    lk, _ = last_known(xy)
    kin = ball_kinematics(xy)
    hdir = attack_direction(m.home, m.away, m.period)
    teams = {H: (m.home, hdir), A: (m.away, -hdir)}
    near = {k: nearest(p, lk) for k, (p, _) in teams.items()}
    ends = spell_ends(lab)
    n = len(lab)

    # ---- A. touches -> label changes
    turn_frames = (
        np.flatnonzero(
            (lab[1:] != lab[:-1])
            & np.isin(lab[1:], (H, A))
            & np.isin(lab[:-1], (H, A))
            & ok[1:]
            & ok[:-1]
        )
        + 1
    )
    touches = touch_frames(kin, alive)
    dh = np.nan_to_num(near[H][1][touches], nan=np.inf)
    da = np.nan_to_num(near[A][1][touches], nan=np.inf)
    t_team = np.where(dh <= da, H, A)
    t_dist = np.minimum(dh, da)
    keep = t_dist < 3.0
    touches, t_team = touches[keep], t_team[keep]
    pred_changes = touches[1:][t_team[1:] != t_team[:-1]]
    # only score GT changes whose +/-0.5 s neighbourhood has the ball seen
    seen_turns = np.array([c for c in turn_frames if seen[max(0, c - 12) : c + 13].mean() > 0.8])
    seen_alive = alive & seen
    acc["touch_gt"].append(len(seen_turns))
    acc["touch_pred"].append(int(seen_alive[pred_changes].sum()) if len(pred_changes) else 0)
    acc["touch_hit"].append(match_events(seen_turns, pred_changes, 12) if len(seen_turns) else 0)
    acc["touch_rate"].append(len(touches) / max(1, seen_alive.sum()) * FPS * 60)
    acc["touch_near_change"].append(
        np.mean([np.min(np.abs(touches - c)) <= 6 for c in seen_turns])
        if len(seen_turns)
        else np.nan
    )

    # ---- A/B. event-locked kinematics + short vs long turnovers
    for c in turn_frames:
        if c < 50 or c + 50 >= n or not seen[c - 5 : c + 6].all():
            continue
        new, old = lab[c], lab[c - 1]
        dur = ends[c] - c
        key = "short" if dur < SHORT else "long"
        acc[f"prof_speed_{key}"].append(kin["speed"][c - 50 : c + 50])
        acc[f"prof_acc_{key}"].append(kin["acc"][c - 50 : c + 50])
        acc[f"prof_newdist_{key}"].append(near[new][1][c - 50 : c + 50])
        acc[f"prof_olddist_{key}"].append(near[old][1][c - 50 : c + 50])
        pn = teams[new][0][c]
        po = teams[old][0][c]
        d_new = nan_hypot(pn - lk[c])
        d_old = nan_hypot(po - lk[c])
        z = m.ball[c, 2]
        acc[f"b_{key}"].append(
            [
                kin["speed"][c],
                np.nanmax(kin["acc"][c - 3 : c + 4]),
                np.nanmax(kin["turn"][c - 3 : c + 4]),
                near[new][1][c],
                near[old][1][c],
                np.nansum(d_new < 3) + np.nansum(d_old < 3),
                np.nansum(d_old < 5),
                z if np.isfinite(z) and m.provider == "idsse" else np.nan,
                kin["speed"][min(n - 1, c + 12)],
            ]
        )

    # ---- C. roles and lead-lag coupling (frames with >= 8 tracked players per team)
    vel = {k: velocity(p, 2) for k, (p, _) in teams.items()}
    sample = np.flatnonzero(alive)[::5]
    sample = sample[(sample + LAG < n)]
    sample = sample[alive[sample + LAG] & (lab[sample + LAG] == lab[sample])]
    for k, (p, d) in teams.items():
        o = A if k == H else H
        po, vo = teams[o][0], vel[o]
        pk, vk = p[sample], vel[k][sample]
        cnt = (~np.isnan(pk[..., 0])).sum(1)
        cnt_o = (~np.isnan(po[sample][..., 0])).sum(1)
        good = (cnt >= 8) & (cnt_o >= 8)
        s, pk, vk = sample[good], pk[good], vk[good]
        if not len(s):
            continue
        dd = np.linalg.norm(pk[:, :, None, :] - po[s][:, None, :, :], axis=-1)
        dd = np.where(np.isnan(dd), np.inf, dd)
        q = np.argmin(dd, axis=2)  # nearest opponent per player
        qd = np.take_along_axis(dd, q[..., None], 2)[..., 0]

        def unit(v: np.ndarray) -> np.ndarray:
            """Convert finite event counts to per-minute rates for this match."""
            sp = nan_hypot(v)[..., None]
            return np.where(sp > 1.0, v / np.maximum(sp, 1e-6), np.nan)

        u_p_now, u_p_lag = unit(vk), unit(vel[k][s + LAG])
        u_q_now = unit(np.take_along_axis(vo[s], q[..., None].repeat(2, -1), 1))
        u_q_lag = unit(np.take_along_axis(vo[s + LAG], q[..., None].repeat(2, -1), 1))
        lead = (u_p_now * u_q_lag).sum(-1) - (u_p_lag * u_q_now).sum(-1)  # >0: opponent follows p
        follow_now = (u_p_now * u_q_now).sum(-1)
        fwd = np.nan_to_num(pk[..., 0], nan=-np.inf) * d[s][:, None]
        striker = np.argmax(fwd, 1)
        fwd_def = np.where(np.isinf(fwd), np.inf, fwd)
        order = np.argsort(fwd_def, 1)  # deepest first (index 0 = goalkeeper)
        last_def = order[:, 1]
        back4 = order[:, 1:5]
        yb = np.take_along_axis(pk[..., 1], back4, 1) * d[s][:, None]  # +y = right of attack
        fb_l = np.take_along_axis(back4, np.nanargmin(yb, 1)[:, None], 1)[:, 0]
        fb_r = np.take_along_axis(back4, np.nanargmax(yb, 1)[:, None], 1)[:, 0]
        r = np.arange(len(s))
        in_pos = lab[s] == k
        rows = {
            "lead_team": np.nanmean(lead, 1),
            "follow_team": np.nanmean(follow_now, 1),
            "lead_striker": lead[r, striker],
            "striker_marker_dist": qd[r, striker],
            "striker_fwd": fwd[r, striker],
            "last_defender_fwd": fwd[r, last_def],
            "fullback_fwd": 0.5 * (fwd[r, fb_l] + fwd[r, fb_r]),
            "fullback_width": np.abs(pk[r, fb_l, 1] - pk[r, fb_r, 1]),
            "mean_marker_dist": np.nanmean(np.where(np.isinf(qd), np.nan, qd), 1),
            "frac_free_5m": np.nanmean(np.where(np.isinf(qd), np.nan, qd > 5), 1),
        }
        prov = m.provider
        for name, v in rows.items():
            acc[f"role_{prov}_{name}_in"].append(v[in_pos])
            acc[f"role_{prov}_{name}_out"].append(v[~in_pos])
            acc[f"role_{prov}_{name}_frames"].append(np.stack([s, np.full(len(s), k), v], 1))

    # ---- D. carrier pressure hazard, recovery zone, counter-press, expansion
    carrier_frames = np.flatnonzero(alive & seen)[::5]
    carrier_frames = carrier_frames[carrier_frames + 2 * FPS < n]
    for t in carrier_frames:
        k = lab[t]
        o = A if k == H else H
        if not near[k][1][t] <= 1.5:
            continue
        cpos = teams[k][0][t, near[k][0][t]]
        dopp = nan_hypot(teams[o][0][t] - cpos)
        press = np.nanmin(dopp) if np.isfinite(dopp).any() else np.nan
        n5 = int(np.nansum(dopp < 5))
        future = lab[t : t + 2 * FPS]
        lost = bool(np.any(future == o))
        cv = vel[k][t, near[k][0][t]]
        acc["carrier"].append([press, n5, lost, nan_hypot(cv), cv[0] * teams[k][1][t]])
    for c in turn_frames:
        if c + 3 * FPS >= n:
            continue
        new = lab[c]
        old = A if new == H else H
        zone = lk[c, 0] * teams[new][1][c]
        dur = (ends[c] - c) / FPS
        cp = nan_hypot(teams[old][0][c + FPS] - lk[c + FPS])
        counter = int(np.nansum(cp < 10))
        wn = [np.nanstd(teams[new][0][c + f, :, 1]) for f in (0, 3 * FPS)]
        wo = [np.nanstd(teams[old][0][c + f, :, 1]) for f in (0, 3 * FPS)]
        acc["recovery"].append(
            [
                zone,
                dur,
                counter,
                wn[1] - wn[0],
                wo[1] - wo[0],
                np.isfinite(wn).all() and np.isfinite(wo).all(),
            ]
        )

    # ---- E. dribbles: same carrier >= 1 s within 1.5 m, speed > 3 m/s
    for k, (_p, d) in teams.items():
        idx, dist = near[k]
        carry = alive & seen & (lab == k) & (dist <= 1.5)
        brk = np.flatnonzero(np.diff(np.concatenate([[0], carry.astype(int), [0]])))
        for a, b in zip(brk[::2], brk[1::2], strict=True):
            if b - a < FPS or len(np.unique(idx[a:b])) > 1:
                continue
            sp = nan_hypot(vel[k][a:b, idx[a]])
            if np.nanmedian(sp) < 3.0:
                continue
            o = A if k == H else H
            after = lab[b : b + FPS]
            acc["dribble"].append(
                [
                    (b - a) / FPS,
                    np.nanmedian(sp),
                    float(np.nanmedian(np.nan_to_num(vel[k][a:b, idx[a], 0]) * d[a])),
                    bool(np.any(after == o)),
                    bool(np.any(after == D)),
                ]
            )


def summarise(acc: dict[str, list]) -> list[str]:  # noqa: PLR0915
    """Pool per-match training diagnostics into summary statistics."""
    L: list[str] = []
    gt, pr, hit = sum(acc["touch_gt"]), sum(acc["touch_pred"]), sum(acc["touch_hit"])
    L += [
        "## A. Ball touches and possession changes",
        "",
        f"Turnovers (team→team) with the ball seen ±0.5 s: {gt:,}. Detected touch candidates: "
        f"{np.mean(acc['touch_rate']):.1f} per minute of seen live ball; "
        f"{np.nanmean(acc['touch_near_change']):.0%} of turnovers have a touch candidate within "
        "±0.24 s.",
        "",
        "Rule *'possession changes when the nearest player (< 3 m) at a touch belongs to the other "
        f"team'*, ±0.5 s tolerance: recall {hit / max(gt, 1):.2f}, precision "
        f"{hit / max(pr, 1):.2f} ({pr:,} predicted changes).",
        "",
    ]
    names = [
        "ball speed m/s",
        "max ball accel m/s² (±0.12 s)",
        "max turn deg",
        "new team nearest m",
        "old team nearest m",
        "players within 3 m",
        "old team within 5 m",
        "ball height m (IDSSE)",
        "ball speed +0.5 s",
    ]
    s = np.array(acc["b_short"], float)
    lo = np.array(acc["b_long"], float)
    L += [
        "## B. Turnovers that last < 2 s vs ≥ 2 s (at the change frame)",
        "",
        f"n short {len(s):,}, long {len(lo):,} (ball seen at the change).",
        "",
        "| feature | median short | median long | AUC short>long |",
        "|---|---|---|---|",
    ]
    for i, nm in enumerate(names):
        a, b = s[:, i], lo[:, i]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) and len(b):
            L.append(f"| {nm} | {np.median(a):.2f} | {np.median(b):.2f} | {auc(a, b):.3f} |")
    L += [
        "",
        "Event-locked means (all turnovers with ball seen), t = 0 is the label change:",
        "",
        "| t (s) | ball speed short / long | ball accel short / long | new-team nearest m short / long | old-team nearest m short / long |",
        "|---|---|---|---|---|",
    ]
    for t in (-1.0, -0.5, -0.24, 0.0, 0.24, 0.5, 1.0):
        i = int(50 + t * FPS)
        cells = []
        for f in ("speed", "acc", "newdist", "olddist"):
            vals = [np.nanmean(np.stack(acc[f"prof_{f}_{k}"])[:, i]) for k in ("short", "long")]
            cells.append(f"{vals[0]:.1f} / {vals[1]:.1f}")
        L.append(f"| {t:+.2f} | " + " | ".join(cells) + " |")

    L += [
        "",
        "## C. Roles and marking (frames with ≥ 8 tracked players per team)",
        "",
        "Roles are assigned per frame from positions only: striker = most advanced, last defender "
        "= deepest outfield, full-backs = widest two of the four deepest outfield players. "
        f"*Lead* = cos(v_player(t), v_marker(t+{LAG / FPS:.1f} s)) − cos(v_player(t+{LAG / FPS:.1f} s), "
        "v_marker(t)) with marker = nearest opponent: > 0 means the opponent copies the player's "
        "movement (the player leads). AUC = in possession vs defending for the same team.",
        "",
    ]
    for prov in ("idsse", "skillcorner", "metrica"):
        keys = sorted(
            {
                k.split(f"role_{prov}_")[1].rsplit("_", 1)[0]
                for k in acc
                if k.startswith(f"role_{prov}_") and k.endswith("_in")
            }
        )
        if not keys:
            continue
        L += [
            f"**{prov}**",
            "",
            "| feature | median in possession | median defending | AUC |",
            "|---|---|---|---|",
        ]
        for nm in keys:
            a = np.concatenate(acc[f"role_{prov}_{nm}_in"])
            b = np.concatenate(acc[f"role_{prov}_{nm}_out"])
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) > 100 and len(b) > 100:
                L.append(f"| {nm} | {np.median(a):.2f} | {np.median(b):.2f} | {auc(a, b):.3f} |")
        L.append("")

    c = np.array(acc["carrier"], float)
    L += [
        "## D. Pressing, recoveries and counter-pressing",
        "",
        f"**Ball carrier under pressure** ({len(c):,} carrier frames, nearest in-possession player "
        "≤ 1.5 m from a seen ball): probability that the opponent has the ball within 2 s.",
        "",
        "| nearest opponent to carrier | n | P(turnover ≤ 2 s) |",
        "|---|---|---|",
    ]
    for lo_, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 99)):
        sel = (c[:, 0] >= lo_) & (c[:, 0] < hi)
        if sel.sum():
            L.append(f"| {lo_}–{hi} m | {sel.sum():,} | {c[sel, 2].mean():.3f} |")
    L += ["", "| opponents within 5 m of carrier | n | P(turnover ≤ 2 s) |", "|---|---|---|"]
    for k in (0, 1, 2, 3):
        sel = c[:, 1] == k if k < 3 else c[:, 1] >= 3
        if sel.sum():
            L.append(f"| {k}{'+' if k == 3 else ''} | {sel.sum():,} | {c[sel, 2].mean():.3f} |")
    r = np.array(acc["recovery"], float)
    L += [
        "",
        f"**Recoveries** ({len(r):,} turnovers). Zone = ball position along the recovering "
        "team's attack (defensive third < −17.5 m).",
        "",
        "| recovery zone | n | median spell s | share < 2 s |",
        "|---|---|---|---|",
    ]
    for nm, lo_, hi in (
        ("defensive third", -99, -17.5),
        ("middle third", -17.5, 17.5),
        ("attacking third", 17.5, 99),
    ):
        sel = (r[:, 0] >= lo_) & (r[:, 0] < hi)
        if sel.sum():
            L.append(
                f"| {nm} | {sel.sum():,} | {np.median(r[sel, 1]):.1f} | {(r[sel, 1] < 2).mean():.2f} |"
            )
    L += [
        "",
        "| losing-team players within 10 m of ball 1 s after the loss (counter-press) | n | median spell s | share < 2 s |",
        "|---|---|---|---|",
    ]
    for lo_, hi in ((0, 1), (1, 2), (2, 3), (3, 99)):
        sel = (r[:, 2] >= lo_) & (r[:, 2] < hi)
        if sel.sum():
            lab_ = f"{lo_}" if hi == lo_ + 1 else f"{lo_}+"
            L.append(
                f"| {lab_} | {sel.sum():,} | {np.median(r[sel, 1]):.1f} | {(r[sel, 1] < 2).mean():.2f} |"
            )
    g = r[r[:, 5] > 0]
    long_ = g[g[:, 1] >= 3]
    L += [
        "",
        f"Width change over the 3 s after a recovery (possessions lasting ≥ 3 s, n={len(long_):,}): "
        f"recovering team {np.median(long_[:, 3]):+.2f} m (y-std), losing team {np.median(long_[:, 4]):+.2f} m.",
        "",
    ]
    d = np.array(acc["dribble"], float)
    if len(d):
        L += [
            "## E. Carries / dribbles (same player ≤ 1.5 m from the ball for ≥ 1 s at > 3 m/s)",
            "",
            f"n {len(d):,}; median duration {np.median(d[:, 0]):.1f} s, carrier speed "
            f"{np.median(d[:, 1]):.1f} m/s, forward speed {np.median(d[:, 2]):.1f} m/s; within 1 s "
            f"after the carry ends the opponent has the ball in {d[:, 3].mean():.0%} and the ball is "
            f"dead in {d[:, 4].mean():.0%} of cases.",
            "",
        ]
    return L


def main() -> int:
    """Run the individual-player EDA on the configured external training split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report", type=Path, default=ROOT / "experiments/research/neural/EDA_INDIVIDUAL.md"
    )
    args = parser.parse_args()
    acc: dict[str, list] = defaultdict(list)
    canon = ROOT / "data" / "external" / "canonical"
    names = [n for prov in TRAIN.values() for n in prov]
    for name in names:
        analyse_match(CanonicalMatch.load(canon / f"{name}.npz"), acc)
        print(f"  {name}", flush=True)
    lines = [
        "# Individual-player and ball-touch EDA — external training matches only",
        "",
        "Generated by `scripts/neural/eda_individual.py` on the training split (IDSSE 4, "
        "SkillCorner 12, Metrica game 1). Validation/test matches and the challenge game are "
        "never read. Motivation: `experiments/research/neural/ERROR_ANALYSIS.md`.",
        "",
    ]
    lines += summarise(acc)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {args.report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Orbit Wars - Phased Strategy Bot

Phases:
  early (0-50):    Aggressive expansion, prioritize high-production neutrals + comets
  mid   (50-300):  Defend + pressure weak enemies
  late  (300+):    Conserve ships, only take clearly profitable captures

Features:
  - Target scoring: production / (distance * sqrt(garrison))
  - Lead prediction for orbiting planets
  - Sun avoidance (skip targets whose path crosses the sun)
  - Defense reserves scaled to incoming enemy fleets
  - Reinforcement from nearest ally when a planet is outgunned
  - Comet bonus early, ignored late
  - Travel-time production accounting for enemy targets
"""

import math
# Named tuples from the env let us write p.x, p.ships instead of p[2], p[5].
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# Board constants. Sun sits dead center with radius 10; fleets crossing it die.
SUN = (50.0, 50.0)
SUN_R = 10.0
# Max fleet speed (units/turn) for the largest fleets, per the env config.
MAX_SPEED = 6.0

# Module-level turn counter. The agent function is called once per turn, so
# we increment this each call. Dict wrapper makes it mutable from inside agent().
_TURN = {"n": -1}


def _get(obs, key, default=None):
    """Read a field from the observation whether it's a dict or a namedtuple-like."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def dist(a, b):
    """Straight-line distance between two (x, y) points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fleet_speed(ships):
    """
    Fleet speed formula from the README:
        speed = 1 + (MAX - 1) * (log(ships) / log(1000)) ^ 1.5
    1 ship = 1.0 unit/turn, ~1000 ships = MAX_SPEED.
    """
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def path_hits_sun(p1, p2, buffer=0.5):
    """
    Does the line segment from p1 to p2 pass close enough to the sun to kill a fleet?

    Geometry: project the sun's center onto the segment, clamp to [0, 1] so we
    only check points that actually lie on the segment (not its extension), then
    measure distance from that closest point to the sun. The small buffer is a
    safety margin since fleets have thickness in practice.
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L2 = dx * dx + dy * dy
    # Degenerate case: p1 and p2 are (nearly) the same point.
    if L2 < 1e-9:
        return False
    # t is the fractional position along the segment of the sun's closest approach.
    t = ((SUN[0] - p1[0]) * dx + (SUN[1] - p1[1]) * dy) / L2
    t = max(0.0, min(1.0, t))
    cx = p1[0] + t * dx
    cy = p1[1] + t * dy
    return math.hypot(cx - SUN[0], cy - SUN[1]) < SUN_R + buffer


def predict_pos(planet, turns_ahead, ang_vel, orbiting_ids):
    """
    Predict where an orbiting planet will be after `turns_ahead` turns.

    Orbiting planets rotate around the sun at `ang_vel` radians/turn. Static
    planets don't move, so we return their current position unchanged.

    Rotation math: translate so sun is origin, apply 2D rotation matrix by
    theta = ang_vel * turns_ahead, translate back.
    """
    if planet.id not in orbiting_ids or ang_vel == 0:
        return (planet.x, planet.y)
    rx, ry = planet.x - SUN[0], planet.y - SUN[1]
    theta = ang_vel * turns_ahead
    c, s = math.cos(theta), math.sin(theta)
    return (SUN[0] + rx * c - ry * s, SUN[1] + rx * s + ry * c)


def agent(obs):
    # -------- Turn counter --------
    # Advance our internal counter; first call lands on step 0.
    _TURN["n"] += 1
    step = _TURN["n"]

    # -------- Unpack the observation --------
    # `player` is our ID (0-3). The rest are raw lists we'll parse into named tuples.
    player = _get(obs, "player", 0)
    raw_planets = _get(obs, "planets", []) or []
    raw_fleets = _get(obs, "fleets", []) or []
    ang_vel = _get(obs, "angular_velocity", 0.0) or 0.0
    initial_planets = _get(obs, "initial_planets", []) or []
    comet_ids = set(_get(obs, "comet_planet_ids", []) or [])

    # Convert raw tuples into Planet / Fleet named tuples for readable access.
    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]

    # -------- Figure out which planets orbit the sun --------
    # A planet rotates if its (distance from sun) + (its radius) < 50. We use
    # the initial positions because current positions have already rotated.
    # Comets are excluded — they follow their own elliptical paths, not circles.
    orbiting_ids = set()
    for ip in initial_planets:
        ipl = Planet(*ip)
        r = math.hypot(ipl.x - SUN[0], ipl.y - SUN[1])
        if r + ipl.radius < 50 and ipl.id not in comet_ids:
            orbiting_ids.add(ipl.id)

    # -------- Split planets into mine vs targets --------
    my_planets = [p for p in planets if p.owner == player]
    # No planets left = we've lost; nothing to do.
    if not my_planets:
        return []
    # Targets include both neutrals (owner == -1) and enemies (0..3, not us).
    targets = [p for p in planets if p.owner != player]

    # -------- Pick the current phase --------
    # The boundaries are heuristic. Early = land grab; mid = fight; late = hoard.
    if step < 50:
        phase = "early"
    elif step < 300:
        phase = "mid"
    else:
        phase = "late"

    # -------- Threat detection --------
    # For each of my planets, estimate how many enemy ships are incoming.
    # We flag a fleet as "heading at" a planet if:
    #   (a) it's within 45 units (close enough to matter soon), and
    #   (b) its heading angle is within ~13° of pointing directly at the planet.
    # We sum ship counts so `incoming[mp.id]` approximates the attack size.
    incoming = {p.id: 0 for p in my_planets}
    for f in fleets:
        if f.owner == player:
            continue
        for mp in my_planets:
            dx, dy = mp.x - f.x, mp.y - f.y
            d = math.hypot(dx, dy)
            if d > 45:
                continue
            aim = math.atan2(dy, dx)
            # Normalize the angle difference into [-pi, pi] before taking abs,
            # otherwise angles near ±pi would look very different when they're close.
            diff = abs((f.angle - aim + math.pi) % (2 * math.pi) - math.pi)
            if diff < 0.22:
                incoming[mp.id] += f.ships

    # The list of moves we'll return at the end.
    moves = []

    # ================= OFFENSIVE PASS =================
    # Each of my planets independently picks its single best target this turn.
    for mine in my_planets:
        # Reserve enough ships to survive incoming attacks + a small buffer.
        # Late game: keep at least half the garrison at home regardless — those
        # ships count toward our final score just for sitting there.
        threat = incoming[mine.id]
        reserve = max(3, threat + 2)
        if phase == "late":
            reserve = max(reserve, int(mine.ships * 0.5))
        available = mine.ships - reserve
        # Not worth a fleet of 1 ship (it moves at speed 1.0 — very slow).
        if available < 2:
            continue

        # Track the best candidate for this launching planet.
        best_score = 0.0
        best_send = 0
        best_angle = 0.0
        picked = False

        for t in targets:
            launch = (mine.x, mine.y)

            # Rough travel-time estimate so we know where an orbiting planet
            # will be when we arrive. We don't know the exact fleet size yet,
            # so we estimate with (target.ships + 5) — close to what we'd send.
            d0 = dist(launch, (t.x, t.y))
            est_ships = max(t.ships + 5, 3)
            est_time = d0 / fleet_speed(est_ships)
            tx, ty = predict_pos(t, est_time, ang_vel, orbiting_ids)
            target_pos = (tx, ty)

            # Skip targets where the direct line crosses the sun — fleets
            # travel in straight lines, so we can't "go around" in one launch.
            if path_hits_sun(launch, target_pos):
                continue

            # Ships needed to capture:
            #   neutral: garrison + 1 (neutrals don't produce)
            #   enemy:   garrison + production_during_travel + 1
            # The +1 is what tips combat in our favor.
            travel_time = dist(launch, target_pos) / fleet_speed(est_ships)
            prod_add = int(t.production * travel_time) if t.owner != -1 else 0
            needed = t.ships + prod_add + 1
            # Skip if we can't afford it or the math produced something silly.
            if needed > available or needed < 1:
                continue

            is_comet = t.id in comet_ids
            is_enemy = t.owner != -1 and t.owner != player

            # Late game: stop picking fights and stop chasing expiring comets.
            # Ships sitting at home are guaranteed score; risky attacks aren't.
            if phase == "late":
                if is_enemy or is_comet:
                    continue

            # Scoring: higher production is better, distance and garrison both
            # drag the score down. Enemy planets are slightly preferred (they
            # deny score to the opponent as well as adding to ours).
            # sqrt on garrison makes medium-defended targets still attractive.
            enemy_bonus = 1.3 if is_enemy else 1.0
            score = (t.production * enemy_bonus) / (
                dist(launch, target_pos) * math.sqrt(t.ships + 1)
            )
            # Comets disappear after a while — grab them early before they leave.
            if is_comet and phase == "early":
                score *= 1.5

            if score > best_score:
                best_score = score
                best_send = needed
                # Final launch angle points at the *predicted* arrival position,
                # not the target's current position — this is the "leading the
                # target" trick every shooter game teaches you.
                best_angle = math.atan2(target_pos[1] - launch[1],
                                        target_pos[0] - launch[0])
                picked = True

        if picked:
            moves.append([mine.id, best_angle, best_send])

    # ================= REINFORCEMENT PASS =================
    # If any of my planets can't defend itself, pull ships from the closest
    # ally that has spare capacity (after subtracting what it already planned
    # to launch offensively). Skip sun-blocked lanes.
    for mp in my_planets:
        threat = incoming[mp.id]
        if threat <= mp.ships:
            continue
        # +2 buffer so reinforcements also leave some survivors.
        deficit = threat - mp.ships + 2
        for ally in sorted(my_planets,
                           key=lambda a: dist((a.x, a.y), (mp.x, mp.y))):
            if ally.id == mp.id or deficit <= 0:
                continue
            # Subtract ships the ally already committed to offense this turn.
            scheduled = sum(m[2] for m in moves if m[0] == ally.id)
            spare = ally.ships - scheduled - 3
            if spare < 2:
                continue
            if path_hits_sun((ally.x, ally.y), (mp.x, mp.y)):
                continue
            send = min(spare, deficit)
            angle = math.atan2(mp.y - ally.y, mp.x - ally.x)
            moves.append([ally.id, angle, send])
            deficit -= send

    return moves

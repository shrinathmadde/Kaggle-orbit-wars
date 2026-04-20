"""
Orbit Wars - Advanced Tactical Bot

Core ideas that go beyond the phased bot:
  1. Fleet-arrival simulation
       Every in-flight fleet is projected forward turn-by-turn until it hits
       a planet, leaves the board, or crosses the sun. This gives us a
       *time-aware* view of incoming combat instead of a crude angle cone.

  2. Defensive combat simulation
       For each of my planets we walk forward up to HORIZON turns, adding
       production and resolving friend/foe arrivals, to find (a) whether the
       planet will fall and (b) how many extra ships we need to save it.

  3. Global target allocation
       Targets are scored once by (future_value / capture_cost) where
       future_value = production * remaining_turns and capture_cost accounts
       for travel-time production and ships already en route. The launcher
       closest to each target (that can afford it) claims the prize — no
       more two planets wasting ships on the same neutral while a juicier
       one goes uncontested.

  4. Value-over-time scoring
       A planet captured at turn 50 is worth far more than the same planet
       captured at turn 450. Weighting by remaining turns naturally tapers
       aggression toward the end of the game.

  5. Arrival-time-aware defense
       Reinforcements only launch if they can actually arrive *before* the
       defender falls, using the fleet-speed formula on the ship count we
       intend to send.
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# ---------- Board & game constants ----------
SUN = (50.0, 50.0)
SUN_R = 10.0
MAX_SPEED = 6.0
BOARD = 100.0
TOTAL_STEPS = 500
# How many turns we simulate forward when predicting fleet arrivals / defense.
HORIZON = 60

# Module-level turn counter. The agent is called once per turn.
_TURN = {"n": -1}


# ---------- Tiny helpers ----------
def _get(obs, key, default=None):
    """Read a field from obs whether it's a dict or a namedtuple-like object."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fleet_speed(ships):
    """Per-turn speed from the env formula: 1 ship = 1.0; ~1000 ships = MAX."""
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def path_hits_sun(p1, p2, buffer=0.5):
    """True if the segment p1→p2 passes within SUN_R + buffer of the sun."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return False
    t = ((SUN[0] - p1[0]) * dx + (SUN[1] - p1[1]) * dy) / L2
    t = max(0.0, min(1.0, t))
    cx = p1[0] + t * dx
    cy = p1[1] + t * dy
    return math.hypot(cx - SUN[0], cy - SUN[1]) < SUN_R + buffer


def predict_pos(planet, turns_ahead, ang_vel, orbiting_ids):
    """Rotate an orbiting planet around the sun; static planets stay put."""
    if planet.id not in orbiting_ids or ang_vel == 0:
        return (planet.x, planet.y)
    rx, ry = planet.x - SUN[0], planet.y - SUN[1]
    theta = ang_vel * turns_ahead
    c, s = math.cos(theta), math.sin(theta)
    return (SUN[0] + rx * c - ry * s, SUN[1] + rx * s + ry * c)


# ---------- Fleet arrival prediction ----------
def fleet_arrival(f, planets, ang_vel, orbiting_ids, horizon=HORIZON):
    """
    Walk a fleet forward in 1-turn steps. Return (planet_id, arrival_turn) on
    first collision, or None if the fleet flies off the board / into the sun
    without hitting anything within the horizon.

    Notes:
      - Planet collisions are checked at *predicted* planet positions so this
        correctly handles orbiting planets sweeping up fleets.
      - Sun is checked against the full frame segment (continuous), matching
        the env's collision model.
    """
    sp = fleet_speed(f.ships)
    dxu = sp * math.cos(f.angle)
    dyu = sp * math.sin(f.angle)
    for t in range(1, horizon + 1):
        nx = f.x + dxu * t
        ny = f.y + dyu * t
        # Out of bounds → fleet is destroyed, never arrives.
        if not (0 <= nx <= BOARD and 0 <= ny <= BOARD):
            return None
        prev = (f.x + dxu * (t - 1), f.y + dyu * (t - 1))
        if path_hits_sun(prev, (nx, ny), buffer=0):
            return None
        # Check collision with each planet's predicted position this turn.
        for p in planets:
            ppos = predict_pos(p, t, ang_vel, orbiting_ids)
            if math.hypot(nx - ppos[0], ny - ppos[1]) < p.radius + 0.3:
                return (p.id, t)
    return None


# ---------- Defense simulation ----------
def simulate_defense(mp, arrivals, player, horizon=HORIZON):
    """
    Walk turn-by-turn on planet `mp`. Return (deficit, deadline_turn). If
    deficit == 0, the planet survives. Otherwise `deficit` is roughly the
    extra ships we need to deliver by `deadline_turn`.

    This is an intentional simplification of the env's combat rules:
      - We treat friendly arrivals as straight reinforcement (+ships).
      - We treat enemy arrivals as straight subtraction from garrison.
      - We stop simulating once the planet flips (we've already lost).
    The real rules are more complex (largest-vs-second-largest resolution),
    but this conservative view is good enough for reinforcement decisions.
    """
    garrison = mp.ships
    arrivals_sorted = sorted(arrivals, key=lambda a: a[0])
    idx = 0
    for t in range(1, horizon + 1):
        # Production happens first each turn.
        garrison += mp.production
        # Process every arrival scheduled for this turn.
        while idx < len(arrivals_sorted) and arrivals_sorted[idx][0] == t:
            _, ships, owner = arrivals_sorted[idx]
            garrison += ships if owner == player else -ships
            idx += 1
        if garrison < 0:
            return (-garrison, t)
    return (0, 0)


def agent(obs):
    # --- 0. Turn bookkeeping ---
    _TURN["n"] += 1
    step = _TURN["n"]
    remaining_turns = max(1, TOTAL_STEPS - step)

    # --- 1. Unpack observation ---
    player = _get(obs, "player", 0)
    raw_planets = _get(obs, "planets", []) or []
    raw_fleets = _get(obs, "fleets", []) or []
    ang_vel = _get(obs, "angular_velocity", 0.0) or 0.0
    initial_planets = _get(obs, "initial_planets", []) or []
    comet_ids = set(_get(obs, "comet_planet_ids", []) or [])

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    by_id = {p.id: p for p in planets}

    # Identify orbiting planets from their *initial* positions. Comets are
    # excluded — their paths are elliptical and not handled here.
    orbiting_ids = set()
    for ip in initial_planets:
        ipl = Planet(*ip)
        r = math.hypot(ipl.x - SUN[0], ipl.y - SUN[1])
        if r + ipl.radius < 50 and ipl.id not in comet_ids:
            orbiting_ids.add(ipl.id)

    my_planets = [p for p in planets if p.owner == player]
    if not my_planets:
        return []

    # --- 2. Predict every in-flight fleet's target ---
    # planet_arrivals[pid] = list of (arrival_turn, ships, owner)
    planet_arrivals = {p.id: [] for p in planets}
    for f in fleets:
        hit = fleet_arrival(f, planets, ang_vel, orbiting_ids)
        if hit:
            pid, turn = hit
            planet_arrivals[pid].append((turn, f.ships, f.owner))

    # --- 3. Defensive assessment ---
    # For each of my planets, find (how many extra ships I need, by when).
    defense_info = {}  # planet_id -> (deficit, deadline)
    for mp in my_planets:
        deficit, deadline = simulate_defense(mp, planet_arrivals[mp.id], player)
        if deficit > 0:
            defense_info[mp.id] = (deficit, deadline)

    moves = []
    # How many ships each of my planets has already pledged this turn.
    committed = {p.id: 0 for p in my_planets}

    # --- 4. Reinforcement pass ---
    # Most urgent (soonest deadline) first. For each threatened planet, walk
    # allies by distance and send ships that will *arrive before the deadline*.
    for pid, (deficit, deadline) in sorted(defense_info.items(), key=lambda x: x[1][1]):
        mp = by_id[pid]
        remaining_need = deficit
        candidates = []
        for ally in my_planets:
            if ally.id == pid:
                continue
            if path_hits_sun((ally.x, ally.y), (mp.x, mp.y)):
                continue
            # Ally keeps reserves for its own defense.
            ally_need = defense_info.get(ally.id, (0, 0))[0]
            reserve = max(3, ally_need + 2)
            spare = ally.ships - committed[ally.id] - reserve
            if spare < 2:
                continue
            d = dist((ally.x, ally.y), (mp.x, mp.y))
            candidates.append((d, ally, spare))
        candidates.sort()  # closest ally first
        for d, ally, spare in candidates:
            if remaining_need <= 0:
                break
            send = min(spare, remaining_need)
            # Will this fleet arrive in time?
            sp = fleet_speed(send)
            eta = d / sp
            if eta >= deadline:
                continue  # too slow; try next ally
            tpos = predict_pos(mp, eta, ang_vel, orbiting_ids)
            if path_hits_sun((ally.x, ally.y), tpos):
                continue
            angle = math.atan2(tpos[1] - ally.y, tpos[0] - ally.x)
            moves.append([ally.id, angle, send])
            committed[ally.id] += send
            remaining_need -= send

    # --- 5. Score all potential targets globally ---
    # value = production * remaining_turns (ships we'd accumulate by game end)
    # cost  = garrison + production_during_travel + enemy_inflight - my_inflight + 1
    # score = value / cost
    targets = [p for p in planets if p.owner != player]
    target_scores = []
    for t in targets:
        # Find the nearest launcher with a clear path (for travel-time estimate).
        closest_d = float("inf")
        closest_travel = 0.0
        for mp in my_planets:
            if path_hits_sun((mp.x, mp.y), (t.x, t.y)):
                continue
            d = dist((mp.x, mp.y), (t.x, t.y))
            if d < closest_d:
                closest_d = d
                est_ships = max(t.ships + 10, 20)
                closest_travel = d / fleet_speed(est_ships)
        if closest_d == float("inf"):
            continue  # no one can reach this target

        # Production accumulates during flight for owned planets (enemy or me).
        prod_add = int(t.production * closest_travel) if t.owner != -1 else 0
        # Ships already headed to this target.
        my_inflight = sum(s for (_, s, o) in planet_arrivals[t.id] if o == player)
        enemy_inflight = sum(
            s for (_, s, o) in planet_arrivals[t.id]
            if o != player and o != t.owner and o != -1
        )
        eff_garrison = t.ships + prod_add + enemy_inflight - my_inflight
        cost = max(2, eff_garrison + 1)

        is_comet = t.id in comet_ids
        is_enemy = t.owner != -1 and t.owner != player

        # Value: how much score we expect this planet to give us.
        if is_comet:
            # Comets expire in ~50-100 turns, have production 1. Skip distant ones.
            if closest_d > 30:
                continue
            value = t.production * min(60, remaining_turns)
        else:
            value = t.production * remaining_turns
        # Taking from an enemy also denies them — small extra weight.
        if is_enemy:
            value *= 1.25

        # Late game: sitting on ships is better than risky attacks.
        if step > 400 and (is_enemy or is_comet):
            continue

        target_scores.append((value / cost, t, cost))

    target_scores.sort(reverse=True)

    # --- 6. Assign one launcher per target in score order ---
    # We require a single launcher that can afford the full cost, because
    # multi-launch attacks arrive at different turns and each one fights the
    # garrison separately (env combat resolves per arrival-turn).
    for _, t, cost in target_scores:
        best_launcher = None
        best_launcher_dist = float("inf")
        for mp in my_planets:
            # Reserve to cover this planet's own defense.
            own_need = defense_info.get(mp.id, (0, 0))[0]
            reserve = max(3, own_need + 2)
            spare = mp.ships - committed[mp.id] - reserve
            if spare < cost:
                continue
            if path_hits_sun((mp.x, mp.y), (t.x, t.y)):
                continue
            d = dist((mp.x, mp.y), (t.x, t.y))
            if d < best_launcher_dist:
                best_launcher = mp
                best_launcher_dist = d
        if best_launcher is None:
            continue

        send = cost
        sp = fleet_speed(send)
        eta = best_launcher_dist / sp
        tpos = predict_pos(t, eta, ang_vel, orbiting_ids)
        # Rare: predicted arrival point passes the sun even if launch point didn't.
        if path_hits_sun((best_launcher.x, best_launcher.y), tpos):
            continue
        angle = math.atan2(tpos[1] - best_launcher.y, tpos[0] - best_launcher.x)
        moves.append([best_launcher.id, angle, send])
        committed[best_launcher.id] += send

    return moves

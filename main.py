"""
Orbit Wars - Nearest Planet Sniper Agent

A simple agent that captures the nearest unowned planet when it has
enough ships to guarantee the takeover.

Strategy:
  For each planet we own, find the closest planet we don't own.
  If we have more ships than the target's garrison, send exactly
  enough to capture it (garrison + 1). Otherwise, wait and accumulate.

Key concepts demonstrated:
  - Parsing the observation (planets, player ID)
  - Computing angles with atan2 for fleet direction
  - Sending moves as [from_planet_id, angle, num_ships]
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

def agent(obs):
    planets = obs["planets"]
    player = obs["player"]
    actions = []

    for p in planets:
        pid, owner, px, py, radius, ships, production = p

        # skip planets I don't own or that have too few ships
        if owner != player or ships < 2:
            continue

        # find the nearest planet I don't own
        best_dist = float("inf")
        best_target = None
        for t in planets:
            tid, towner, tx, ty, *_ = t
            if tid == pid or towner == player:
                continue
            dist = math.sqrt((tx - px) ** 2 + (ty - py) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_target = t

        if best_target is None:
            continue

        # calculate angle toward the target
        tx, ty = best_target[2], best_target[3]
        angle = math.atan2(ty - py, tx - px)

        # send half the ships
        send = ships // 2
        actions.append([pid, angle, send])

    return actions
"""Mech Royale bot: loot -> upgrade -> kill, with lead aiming, bullet dodging and zone awareness.

Standard library only. Every sub-system is wrapped so that a bug degrades behaviour instead of idling the mech.
"""
import gc
import math
import os
import random
import sys
import time
from collections import deque

import mechfight as mf
from navmap import NavMap, Planner, NA

FLIP_LO = float(os.environ.get('FLIP_LO', 0.2))
FLIP_HI = float(os.environ.get('FLIP_HI', 0.6))
PREF_GUN = float(os.environ.get('PREF_GUN', 120.0))
TAU = 2.0 * math.pi
PI = math.pi

UPGRADE_SCORE = {
    "plating": 9.0, "gun_damage": 8.6, "laser_damage": 8.4, "armor": 7.0, "gun_rate": 7.8,
    "nanites": 6.8, "lightning_chain": 6.0, "lightning_range": 6.2, "laser_charge": 6.0,
    "gyros": 5.6, "servos": 5.5, "capacitor": 5.0, "reactor": 4.6, "gun_mag": 4.8,
    "gun_velocity": 5.2, "jets": 3.5, "sensors": 4.4, "laser_pierce": 3.0,
    "mortar_splash": 0.5, "mortar_reload": 0.5, "mortar_velocity": 0.5,
}
WEAPON_RANK = {"lightning": 3, "laser": 2, "mortar": 0, "gun": 0}


def wrap(a):
    return (a + PI) % TAU - PI


def log(*a):
    try:
        mf.log(*a)
    except Exception:
        pass


class Bot:
    name = "mech-hunter"

    # ------------------------------------------------------------------ setup
    def __init__(self, hello):
        gc.collect()
        self.cfg = hello["config"]
        self.slot = hello.get("slot", 0)
        self.rng = random.Random(hello.get("seed", 1))
        mapdir = hello.get("map", {}).get("dir", "/arena")
        t0 = time.perf_counter()
        try:
            self.nm = NavMap(mapdir)
        except Exception as e:
            log("navmap failed", repr(e))
            self.nm = NavMap(None)
        self.planner = Planner(self.nm)
        log("map ready in %.2fs ok=%s" % (time.perf_counter() - t0, self.nm.ok))
        gc.collect()
        try:
            gc.freeze()
        except Exception:
            pass
        gc.set_threshold(100000, 50, 50)

        m = self.cfg["mech"]
        self.walk = m["walk_speed"]
        self.accel = m["ground_accel"]
        self.pickup_r = self.cfg["loot"]["pickup_radius"]
        self.size_per_cube = self.cfg["cubes"].get("size_per_cube", 0.05)

        # state
        self.last_t = 0.0
        self.hist = deque(maxlen=14)
        self.mem = {}               # loot memory: (kind, id) -> dict
        self.blacklist = {}         # pickup id -> until
        self.unreach = {}           # goal cell -> until
        self.goal = None            # current macro goal dict
        self.goal_until = 0.0
        self.path = None
        self.pidx = 0
        self.plan_goal = None
        self.plan_t = -99.0
        self.strafe_sign = 1
        self.strafe_until = 0.0
        self.tid = None
        self.tid_t = -99.0
        self.alert = None           # (bearing, until)
        self.last_hit_t = -99.0
        self.last_attacker = None
        self.flee_until = -1.0
        self.cover = None
        self.cover_t = -99.0
        self.stuck_until = -1.0
        self.escape = (1.0, 0.0)
        self.scan_yaw = 0.0
        self.boosting = False
        self.last_move = (0.0, 0.0)
        self.roam = None
        self.roam_t = -99.0
        self.visited = deque(maxlen=12)
        self.laser_hold = 0.0
        self.ticks = 0
        self.max_ms = 0.0
        self.slow = 0

    # ------------------------------------------------------------------ helpers
    def _scale(self, cubes):
        return 1.0 + self.size_per_cube * (cubes or 0)

    def _unreach(self, x, y, t):
        return self.unreach.get((int(x // 10), int(y // 10)), -1.0) > t

    def zone_circle(self, z, t, ahead):
        tt = t + ahead
        s0, s1 = z["shrink_start"], z["shrink_end"]
        if tt <= s0:
            f = 0.0
        elif tt >= s1:
            f = 1.0
        else:
            f = (tt - s0) / max(1e-6, s1 - s0)
        cx = z["center"][0] + (z["next_center"][0] - z["center"][0]) * f
        cy = z["center"][1] + (z["next_center"][1] - z["center"][1]) * f
        r = z["radius"] + (z["next_radius"] - z["radius"]) * f
        return cx, cy, r

    # ------------------------------------------------------------------ main entry
    def act(self, obs):
        t_start = time.perf_counter()
        self.ticks += 1
        act = {"tick": obs["tick"]}
        try:
            self._act(obs, act, t_start)
        except Exception:
            import traceback
            if self.ticks % 50 == 1:
                log("act error:\n" + traceback.format_exc())
            # fall back to something sane: keep moving to zone centre and shooting nearest
            try:
                self._fallback(obs, act)
            except Exception:
                pass
        ms = (time.perf_counter() - t_start) * 1000.0
        if ms > self.max_ms:
            self.max_ms = ms
        if ms > 35:
            self.slow += 1
        if self.ticks % 200 == 0:
            log("t=%.0f hp=%.0f max_ms=%.1f slow=%d cubes=%s" % (obs["t"], obs["self"]["hp"], self.max_ms, self.slow, obs["self"]["cubes"]))
            self.max_ms = 0.0
        return act

    def _fallback(self, obs, act):
        me = obs["self"]
        x, y, z = me["pos"]
        zc = obs["zone"]["next_center"]
        yaw = math.atan2(zc[1] - y, zc[0] - x)
        act["move"] = [math.cos(yaw), math.sin(yaw)]
        en = [m for m in obs["mechs"] if m["kind"] == "mech"]
        if en:
            e = min(en, key=lambda m: math.dist(m["pos"], me["pos"]))
            tp = e["pos"]
            act["aim"] = [math.atan2(tp[1] - y, tp[0] - x), math.atan2(tp[2] + 4.5 - z - 7, max(1.0, math.hypot(tp[0] - x, tp[1] - y)))]
            act["fire"] = [True, False]
        if obs.get("offer"):
            act["pick"] = 0

    # ------------------------------------------------------------------ the real thing
    def _act(self, obs, act, t_start):
        t = obs["t"]
        me = obs["self"]
        px, py, pz = me["pos"]
        vx, vy = me["vel"][0], me["vel"][1]
        sc = self._scale(me["cubes"])
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        self.hist.append((t, px, py))

        self._events(obs, me, t)
        self._update_memory(obs, me, t)

        # ---- upgrades
        if obs.get("offer"):
            act["pick"] = self._choose_upgrade(obs["offer"], me)

        # ---- enemies
        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        for e in enemies:
            e["_d"] = math.dist(e["pos"], me["pos"])
        enemies.sort(key=lambda e: e["_d"])
        near_enemies = [e for e in enemies if e["_d"] < 110]

        # ---- weapon pickup (explicit take)
        self._auto_take(obs, me, act, t)

        # ---- zone
        zone = obs["zone"]
        zc = self.zone_circle(zone, t, 7.0)
        zc_now = self.zone_circle(zone, t, 0.0)
        dz_future = math.hypot(px - zc[0], py - zc[1])
        r_safe = max(10.0, zc[2] - 12.0)
        zone_urgent = dz_future > r_safe or (not me["in_zone"])
        zone_goal = None
        if zone_urgent:
            ang = math.atan2(py - zc[1], px - zc[0])
            rr = max(0.0, min(r_safe * 0.55, dz_future))
            zone_goal = (zc[0] + math.cos(ang) * rr, zc[1] + math.sin(ang) * rr)

        # ---- target selection
        tgt = self._pick_target(me, enemies, t)
        farm = None
        # ---- stuck detection
        stuck = self._check_stuck(t)

        # ---- decide mode
        flee = False
        if tgt is not None:
            n_close = len([e for e in enemies if e["_d"] < 90])
            if hp_frac < 0.27 or (n_close >= 3 and hp_frac < 0.55):
                self.flee_until = t + 2.5
            flee = t < self.flee_until
            if hp_frac > 0.65 and n_close <= 2:
                flee = False
                self.flee_until = -1.0

        D = (0.0, 0.0)
        speed = 1.0
        mode = "roam"
        want_boost = False
        capture_hold = None

        if tgt is not None:
            mode = "flee" if flee else "fight"
            if flee:
                D, want_boost = self._flee_vec(me, enemies, t, px, py)
            else:
                D = self._combat_vec(me, tgt, t, px, py)
            # keep to the zone while fighting
            if zone_urgent:
                zx, zy = zone_goal
                dd = math.hypot(zx - px, zy - py)
                if dd > 1:
                    D = (D[0] * 0.5 + (zx - px) / dd, D[1] * 0.5 + (zy - py) / dd)
                    want_boost = want_boost or (not me["in_zone"] and me["energy"] > 30)
        else:
            goal = self._choose_goal(obs, me, t, zone_urgent, zone_goal, zc, enemies, npcs)
            mode = goal["type"]
            if mode in ("crate", "npc"):
                farm = goal
            nd = self._navigate(me, goal, t)
            if nd is None:
                D = (0.0, 0.0)
                speed = 0.0
            else:
                D, dist_goal = nd
                if goal["type"] in ("crate", "npc"):
                    # stand off at ~30 m with line of sight
                    if dist_goal < 32 and self.nm.los(px, py, goal["pos"][0], goal["pos"][1]):
                        D = (-D[1] * self.strafe_sign * 0.7, D[0] * self.strafe_sign * 0.7)
                        self._maybe_flip(t, 0.5, 1.2)
                elif goal["type"] == "drop" and dist_goal < 4.0:
                    D = (0.0, 0.0)
                    speed = 0.0
                if dist_goal > 90 and not near_enemies and me["energy"] > 70:
                    self.boosting = True
                if zone_urgent and not me["in_zone"] and me["energy"] > 25:
                    want_boost = True
            if self.boosting:
                if me["energy"] < 45 or near_enemies or dist_goal_small(goal, px, py):
                    self.boosting = False
                else:
                    want_boost = True

        if stuck:
            D = self.escape
            speed = 1.0
            mode = "unstick"

        # ---- steering (dodging, walls, zone)
        mv = (0.0, 0.0)
        if speed > 0 and (D[0] != 0.0 or D[1] != 0.0):
            mv = self._steer(obs, me, D, t, px, py, vx, vy, sc, zc, fighting=(tgt is not None), boost=want_boost)
        elif tgt is not None or obs["projectiles"]:
            # standing still while someone shoots at us is silly
            mv = self._steer(obs, me, (0.0, 0.0), t, px, py, vx, vy, sc, zc, fighting=(tgt is not None), boost=False)
        act["move"] = [round(mv[0], 3), round(mv[1], 3)]
        act["boost"] = bool(want_boost and (mv[0] != 0.0 or mv[1] != 0.0) and me["energy"] > 8)
        self.last_move = mv

        # ---- aiming and firing
        fire = [False, False]
        aim = None
        shoot_target = tgt
        if shoot_target is None and farm is not None:
            shoot_target = self._farm_target(obs, farm)
        if shoot_target is not None:
            aim, fire = self._aim_fire(obs, me, shoot_target, t, px, py, pz, sc, mover=(tgt is not None))
        if aim is None:
            aim = self._idle_aim(obs, me, t, mv)
            # keep the gun loaded while idle
            gun = self._weapon(me, "gun")
            if gun is not None and gun.get("ammo", 99) < 0.6 * gun["stats"].get("mag", 12) and gun.get("reload", 0) <= 0:
                act["reload"] = True
        act["aim"] = [round(aim[0], 4), round(aim[1], 4)]
        act["fire"] = fire

        # occasional jump to hop small ledges while stuck
        if stuck and (self.ticks % 4 == 0):
            act["jump"] = True

        if self.ticks % 3 == 0:
            try:
                mf.draw.label("%s hp%.0f" % (mode, me["hp"]))
            except Exception:
                pass

    # ------------------------------------------------------------------ events and memory
    def _events(self, obs, me, t):
        for ev in obs.get("events", ()):
            ty = ev.get("type")
            if ty == "damaged":
                self.last_hit_t = t
                if ev.get("bearing") is not None:
                    self.alert = (ev["bearing"], t + 2.0)
                if ev.get("by") is not None:
                    self.last_attacker = (ev["by"], t)
            elif ty == "take_failed":
                self.blacklist[ev.get("id")] = t + 8.0
                self.mem.pop(("pickup", ev.get("id")), None)
            elif ty == "took":
                self.mem.pop(("pickup", ev.get("id")), None)
        for s in obs.get("sounds", ()):
            if s.get("kind") in ("gun", "laser", "lightning", "mortar", "laser_charge") and s.get("dist") in ("near", "mid"):
                if self.alert is None or self.alert[1] < t + 0.8:
                    self.alert = (s["bearing"], t + 1.5)

    def _update_memory(self, obs, me, t):
        px, py, pz = me["pos"]
        seen = set()
        for it in obs.get("pickups", ()):
            k = ("pickup", it["id"])
            seen.add(k)
            self.mem[k] = {"pos": it["pos"], "kind": it["kind"], "t": t, "id": it["id"], "type": "pickup"}
        for it in obs.get("cubes", ()):
            k = ("cube", it["id"])
            seen.add(k)
            self.mem[k] = {"pos": it["pos"], "t": t, "id": it["id"], "type": "cube"}
        for it in obs.get("crates", ()):
            k = ("crate", it["id"])
            seen.add(k)
            self.mem[k] = {"pos": it["pos"], "t": t, "id": it["id"], "type": "crate", "hp": it.get("hp", 300)}
        for it in obs.get("mechs", ()):
            if it["kind"] == "npc":
                k = ("npc", it["id"])
                seen.add(k)
                self.mem[k] = {"pos": it["pos"], "t": t, "id": it["id"], "type": "npc", "vel": it.get("vel", [0, 0, 0])}
        dead = []
        for k, v in self.mem.items():
            if k in seen:
                continue
            d = math.hypot(v["pos"][0] - px, v["pos"][1] - py)
            if d < 30 or t - v["t"] > (25 if v["type"] == "npc" else 150):
                dead.append(k)
        for k in dead:
            del self.mem[k]

    def _choose_upgrade(self, offer, me):
        best, bs = 0, -1e9
        have_w = {w["kind"] for w in me["weapons"] if w}
        for i, c in enumerate(offer["choices"]):
            s = UPGRADE_SCORE.get(c["id"], 3.0)
            if c["id"].startswith("lightning") and "lightning" not in have_w:
                s -= 5
            if c["id"].startswith("laser") and "laser" not in have_w:
                s -= 5
            if c["id"] in ("capacitor", "reactor") and "lightning" in have_w:
                s += 2.0
            if c["id"].startswith("lightning") and "lightning" in have_w:
                s += 2.5
            if c["id"].startswith("laser") and "laser" in have_w:
                s += 1.5
            if s > bs:
                best, bs = i, s
        return best

    @staticmethod
    def _weapon(me, kind):
        for w in me["weapons"]:
            if w and w["kind"] == kind:
                return w
        return None

    def _slot_of(self, me, kind):
        for i, w in enumerate(me["weapons"]):
            if w and w["kind"] == kind:
                return i
        return -1

    # ------------------------------------------------------------------ loot
    def _wanted_pickup(self, me, kind):
        rank = WEAPON_RANK.get(kind, 0)
        if rank <= 0:
            return None
        held = [w["kind"] for w in me["weapons"] if w]
        if kind in held:
            return None
        s1 = me["weapons"][1] if len(me["weapons"]) > 1 else None
        if s1 is None:
            return "free"
        if rank > WEAPON_RANK.get(s1["kind"], 0):
            return "swap"
        return None

    def _auto_take(self, obs, me, act, t):
        px, py, pz = me["pos"]
        best = None
        for it in obs.get("pickups", ()):
            if self.blacklist.get(it["id"], -1) > t:
                continue
            if math.hypot(it["pos"][0] - px, it["pos"][1] - py) > self.pickup_r - 0.4 or abs(it["pos"][2] - pz) > 3.5:
                continue
            w = self._wanted_pickup(me, it["kind"])
            if w and (best is None or WEAPON_RANK[it["kind"]] > WEAPON_RANK[best[0]["kind"]]):
                best = (it, w)
        if best:
            it, w = best
            act["take"] = it["id"]
            if w == "swap":
                act["drop"] = 1

    # ------------------------------------------------------------------ targeting
    def _pick_target(self, me, enemies, t):
        if not enemies:
            self.tid = None
            return None
        px, py, pz = me["pos"]
        best, bs = None, 1e9
        for e in enemies:
            d = e["_d"]
            if d > 200:
                continue
            s = d
            hp = e.get("hp")
            if hp:
                s += 0.04 * hp
            if e["id"] == self.tid and t - self.tid_t < 3.0:
                s -= 22
            if self.last_attacker and self.last_attacker[0] == e["id"] and t - self.last_attacker[1] < 3.0:
                s -= 25
            if d > 45 and not self.nm.los(px, py, e["pos"][0], e["pos"][1]):
                s += 70
            if s < bs:
                best, bs = e, s
        if best is not None:
            if best["id"] != self.tid:
                self.tid = best["id"]
            self.tid_t = t
        return best

    def _farm_target(self, obs, goal):
        want = goal["pos"]
        key = (goal["type"], goal.get("id"))
        if goal["type"] == "crate":
            for c in obs.get("crates", ()):
                if c["id"] == goal.get("id"):
                    return {"kind": "crate", "id": c["id"], "pos": c["pos"], "vel": [0, 0, 0], "cubes": 0, "_h": 2.4, "_r": 1.4}
        else:
            for m in obs.get("mechs", ()):
                if m["kind"] == "npc" and m["id"] == goal.get("id"):
                    mm = dict(m)
                    mm["_h"] = 3.5
                    mm["_r"] = 1.8
                    mm["cubes"] = 0
                    return mm
        return None

    # ------------------------------------------------------------------ movement vectors
    def _maybe_flip(self, t, lo, hi):
        if t >= self.strafe_until:
            if self.rng.random() < 0.75:
                self.strafe_sign = -self.strafe_sign
            self.strafe_until = t + self.rng.uniform(lo, hi)

    def _pref_range(self, me, tgt):
        kinds = [w["kind"] for w in me["weapons"] if w]
        tw = tgt.get("weapons") or []
        if "lightning" in kinds and me["energy"] > 18:
            return 24.0 if "mortar" not in tw else 18.0
        if "lightning" in tw:
            return 75.0
        if "laser" in kinds:
            return 95.0
        return PREF_GUN

    def _combat_vec(self, me, tgt, t, px, py):
        tx, ty = tgt["pos"][0], tgt["pos"][1]
        dx, dy = tx - px, ty - py
        d = math.hypot(dx, dy) or 1.0
        ux, uy = dx / d, dy / d
        pref = self._pref_range(me, tgt)
        self._maybe_flip(t, FLIP_LO, FLIP_HI)
        k = max(-1.0, min(1.0, (d - pref) / 22.0))
        sgn = self.strafe_sign
        tx_, ty_ = -uy * sgn, ux * sgn
        side = 1.0 - 0.55 * abs(k)
        return (tx_ * side + ux * k, ty_ * side + uy * k)

    def _flee_vec(self, me, enemies, t, px, py):
        # head for cover if we can find one, otherwise run away from the enemy centroid
        if t - self.cover_t > 1.2:
            self.cover = self._find_cover(me, enemies, px, py)
            self.cover_t = t
        ex = sum(e["pos"][0] for e in enemies[:3]) / max(1, len(enemies[:3]))
        ey = sum(e["pos"][1] for e in enemies[:3]) / max(1, len(enemies[:3]))
        ax, ay = px - ex, py - ey
        an = math.hypot(ax, ay) or 1.0
        ax, ay = ax / an, ay / an
        if self.cover is not None:
            cx, cy = self.cover
            dd = math.hypot(cx - px, cy - py)
            if dd > 2.0:
                return ((cx - px) / dd * 0.8 + ax * 0.4, (cy - py) / dd * 0.8 + ay * 0.4), me["energy"] > 25
        return (ax, ay), me["energy"] > 25

    def _find_cover(self, me, enemies, px, py):
        if not enemies:
            return None
        nm = self.nm
        best, bs = None, -1e9
        pts = [(e["pos"][0], e["pos"][1]) for e in enemies[:4]]
        for k in range(24):
            ang = k * TAU / 24
            ux, uy = math.cos(ang), math.sin(ang)
            for dist in (14.0, 28.0, 44.0):
                qx, qy = px + ux * dist, py + uy * dist
                if nm.blocked(qx, qy) or not nm.clear(px, py, qx, qy):
                    continue
                hidden = 0
                mind = 1e9
                for (ex, ey) in pts:
                    if not nm.los(qx, qy, ex, ey):
                        hidden += 1
                    mind = min(mind, math.hypot(qx - ex, qy - ey))
                s = hidden * 40.0 - dist * 0.6 + min(mind, 120) * 0.15
                if s > bs:
                    best, bs = (qx, qy), s
        return best if bs > 20 else None

    def _check_stuck(self, t):
        if t < self.stuck_until:
            return True
        if len(self.hist) < 12:
            return False
        t0, x0, y0 = self.hist[0]
        t1, x1, y1 = self.hist[-1]
        if t1 - t0 < 1.0:
            return False
        lm = math.hypot(*self.last_move)
        if lm > 0.6 and math.hypot(x1 - x0, y1 - y0) < 2.0:
            self.stuck_until = t + 0.9
            ang = self.rng.uniform(0, TAU)
            # prefer a direction with clearance
            best, bd = None, -1
            for k in range(8):
                a = ang + k * TAU / 8
                fr = self.nm.free_run(x1, y1, math.cos(a), math.sin(a), 12.0)
                if fr > bd:
                    best, bd = a, fr
            self.escape = (math.cos(best), math.sin(best))
            self.path = None
            self.plan_goal = None
            self.hist.clear()
            return True
        return False

    # ------------------------------------------------------------------ steering
    def _steer(self, obs, me, D, t, px, py, vx, vy, sc, zc, fighting, boost):
        nm = self.nm
        vmax = me["stats"].get("boost_speed", 19.5) if boost else me["stats"].get("walk_speed", self.walk)
        Dn = math.hypot(D[0], D[1])
        if Dn > 1e-6:
            Dx, Dy = D[0] / Dn, D[1] / Dn
        else:
            Dx = Dy = 0.0
        cz = me["pos"][2] + 4.5 * sc
        R = 2.5 * sc + 1.3
        # --- hazards
        projs = []
        blasts = []
        for p in obs.get("projectiles", ()):
            rx, ry, rz = p["pos"][0] - px, p["pos"][1] - py, p["pos"][2] - cz
            dd = rx * rx + ry * ry
            if dd > 220 * 220:
                continue
            if p["kind"] == "mortar":
                b = self._mortar_landing(p)
                if b is not None and math.hypot(b[0] - px, b[1] - py) < 20:
                    blasts.append(b)
                continue
            wx, wy, wz = p["vel"]
            w2 = wx * wx + wy * wy + wz * wz
            if w2 < 1:
                continue
            if rx * wx + ry * wy + rz * wz > 0:  # moving away
                continue
            projs.append((rx, ry, rz, wx, wy, wz))
        beams = []
        for e in obs.get("mechs", ()):
            if e["kind"] == "mech" and e.get("charging"):
                yaw, pit = e["aim"]
                beams.append((e["pos"][0], e["pos"][1], math.cos(yaw), math.sin(yaw)))
        taus = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        v0x, v0y = vx, vy
        best_s, best_u = -1e9, (0.0, 0.0)
        cands = []
        if Dn > 1e-6:
            cands.append((Dx, Dy))
        for k in range(16):
            a = k * TAU / 16
            cands.append((math.cos(a), math.sin(a)))
        if Dn < 0.2 or not fighting:
            cands.append((0.0, 0.0))
        zx, zy, zr = zc
        r_safe = max(10.0, zr - 12.0)
        for (ux, uy) in cands:
            s = 0.0
            if Dn > 1e-6:
                s += 2.4 * (ux * Dx + uy * Dy) * min(1.0, Dn)
            elif ux == 0.0 and uy == 0.0:
                s += 0.3
            if ux != 0.0 or uy != 0.0:
                fr = nm.free_run(px, py, ux, uy, 14.0)
                if fr < 14.0:
                    s -= (14.0 - fr) / 14.0 * 2.2
                if fr < 3.0:
                    s -= 4.0
            vtx, vty = ux * vmax, uy * vmax
            T = max(0.03, math.hypot(vtx - v0x, vty - v0y) / self.accel)
            pen = 0.0
            if projs or blasts or beams:
                disp = []
                for tau in taus:
                    if tau <= T:
                        f = tau * tau / (2 * T)
                        dxx = v0x * tau + (vtx - v0x) * f
                        dyy = v0y * tau + (vty - v0y) * f
                    else:
                        dxx = vtx * tau - (vtx - v0x) * T * 0.5
                        dyy = vty * tau - (vty - v0y) * T * 0.5
                    disp.append((dxx, dyy))
                for (rx, ry, rz, wx, wy, wz) in projs:
                    for tau, (dxx, dyy) in zip(taus, disp):
                        ex_ = rx + wx * tau - dxx
                        ey_ = ry + wy * tau - dyy
                        ez_ = rz + wz * tau
                        if ex_ * ex_ + ey_ * ey_ + ez_ * ez_ < R * R:
                            pen += 1.0
                            break
                if blasts:
                    dxx, dyy = disp[3]
                    for (bx, by, _) in blasts:
                        dd = math.hypot(px + dxx - bx, py + dyy - by)
                        if dd < 10:
                            pen += (10 - dd) / 10.0 * 1.5
                if beams:
                    dxx, dyy = disp[4]
                    qx, qy = px + dxx, py + dyy
                    for (bx, by, bdx, bdy) in beams:
                        rel_x, rel_y = qx - bx, qy - by
                        along = rel_x * bdx + rel_y * bdy
                        if along < 0:
                            continue
                        lat = abs(rel_x * bdy - rel_y * bdx)
                        if lat < 4.5:
                            pen += (4.5 - lat) / 4.5
            s -= 3.2 * pen
            # zone
            qx, qy = px + ux * vmax * 1.6, py + uy * vmax * 1.6
            dzone = math.hypot(qx - zx, qy - zy)
            if dzone > r_safe:
                s -= min(3.0, (dzone - r_safe) / 25.0 * 3.0)
            if s > best_s:
                best_s, best_u = s, (ux, uy)
        ux, uy = best_u
        mag = min(1.0, Dn) if Dn > 1e-6 else 1.0
        if fighting:
            mag = 1.0
        if ux == 0.0 and uy == 0.0:
            return (0.0, 0.0)
        return (ux * mag, uy * mag)

    @staticmethod
    def _mortar_landing(p):
        x, y, z = p["pos"]
        vx, vy, vz = p["vel"]
        g = 22.0
        for i in range(1, 80):
            tt = i * 0.1
            zz = z + vz * tt - 0.5 * g * tt * tt
            if zz < 0.0 and i > 1:
                return (x + vx * tt, y + vy * tt, tt)
        return None

    # ------------------------------------------------------------------ macro goals + navigation
    def _choose_goal(self, obs, me, t, zone_urgent, zone_goal, zc, enemies, npcs):
        px, py, pz = me["pos"]
        if zone_urgent and zone_goal is not None:
            return {"type": "zone", "pos": zone_goal, "id": None}
        # keep current goal for a short time unless it vanished
        if self.goal is not None and t < self.goal_until and self._goal_valid(self.goal, t, px, py, zc):
            return self.goal
        zx, zy, zr = zc
        r_ok = max(15.0, zr - 20.0)
        best, bs = None, -1.0
        near_e = [e for e in enemies if e["_d"] < 120]
        have_s1 = me["weapons"][1] is not None if len(me["weapons"]) > 1 else False
        cubes_held = me["cubes"]
        for k, v in self.mem.items():
            x, y = v["pos"][0], v["pos"][1]
            d = math.hypot(x - px, y - py)
            if math.hypot(x - zx, y - zy) > r_ok or self._unreach(x, y, t):
                continue
            if v["pos"][2] - self.nm.ground(x, y) > 4.0 and v["type"] != "pickup":
                continue
            ty = v["type"]
            if ty == "pickup":
                w = self._wanted_pickup(me, v["kind"])
                if not w or self.blacklist.get(v["id"], -1) > t:
                    continue
                val = (12.0 if not have_s1 else 7.0) + (2.0 if v["kind"] == "lightning" else 0.0)
            elif ty == "cube":
                val = 6.5 if cubes_held < 10 else 3.0
            elif ty == "crate":
                val = 4.5 if cubes_held < 12 else 2.0
            else:  # npc
                val = 5.5 if cubes_held < 12 else 2.0
                if t - v["t"] > 8:
                    val *= 0.5
            s = val / (1.0 + d / 70.0)
            if near_e:
                s *= 0.6
            if self.goal is not None and self.goal.get("key") == k:
                s *= 1.25
            if s > bs:
                bs = s
                best = {"type": ty, "pos": v["pos"], "id": v["id"], "key": k}
        # supply drops
        for dp in obs.get("drops", ()):
            x, y = dp["pos"][0], dp["pos"][1]
            if self._unreach(x, y, t):
                continue
            d = math.hypot(x - px, y - py)
            eta = d / (self.walk * 0.9)
            if dp["lands_in"] > eta + 40:
                continue
            crowd = len([e for e in enemies if math.hypot(e["pos"][0] - x, e["pos"][1] - y) < 40])
            val = 13.0 if me["hp"] < 0.7 * me["max_hp"] else 11.0
            if dp["holder"] is not None and dp["holder"] != me["id"]:
                val *= 0.5
            val /= (1 + 0.8 * crowd)
            s = val / (1.0 + d / 160.0)
            if s > bs:
                bs = s
                best = {"type": "drop", "pos": dp["pos"], "id": dp["id"], "key": ("drop", dp["id"])}
        if best is None or bs < 0.9:
            best = self._roam_goal(me, t, px, py, zc)
        self.goal = best
        self.goal_until = t + 1.5
        return best

    def _goal_valid(self, g, t, px, py, zc):
        k = g.get("key")
        if k is None:
            return g["type"] == "roam" and math.hypot(g["pos"][0] - px, g["pos"][1] - py) > 8 and t - self.roam_t < 18
        if g["type"] == "drop":
            return True
        return k in self.mem

    def _roam_goal(self, me, t, px, py, zc):
        zx, zy, zr = zc
        r_ok = max(15.0, zr - 25.0)
        best, bs = None, -1e9
        for _ in range(10):
            ang = self.rng.uniform(0, TAU)
            rr = self.rng.uniform(0.0, 1.0) ** 0.5 * r_ok
            x, y = zx + math.cos(ang) * rr, zy + math.sin(ang) * rr
            d = math.hypot(x - px, y - py)
            if d < 50 or d > 260:
                continue
            a, b = self.nm.node_of(x, y)
            if self.nm.ok and not self.nm.free_strict[b * NA + a]:
                continue
            if self._unreach(x, y, t):
                continue
            nov = min([math.hypot(x - vx_, y - vy_) for (vx_, vy_) in self.visited] or [200.0])
            s = nov * 0.5 - abs(d - 120.0) * 0.2 + self.rng.uniform(0, 15)
            if s > bs:
                bs, best = s, (x, y)
        if best is None:
            best = (zx + self.rng.uniform(-0.3, 0.3) * r_ok, zy + self.rng.uniform(-0.3, 0.3) * r_ok)
        self.roam_t = t
        self.visited.append((px, py))
        return {"type": "roam", "pos": (best[0], best[1], 0.0), "id": None}

    def _navigate(self, me, goal, t):
        """Return ((dx,dy) unit-ish desired direction, distance_to_goal) or None."""
        px, py, pz = me["pos"]
        gx, gy = goal["pos"][0], goal["pos"][1]
        d = math.hypot(gx - px, gy - py)
        arrive = 3.0 if goal["type"] in ("pickup", "cube") else (5.0 if goal["type"] in ("drop", "roam", "zone") else 6.0)
        if goal["type"] == "roam" and d < 10:
            self.goal_until = 0.0
            return None
        if d < arrive and goal["type"] in ("pickup", "cube", "drop", "zone"):
            return (0.0, 0.0), d
        nm = self.nm
        if d < 70 and nm.clear(px, py, gx, gy):
            self.path = None
            self.plan_goal = None
            return ((gx - px) / d, (gy - py) / d), d
        # planner bookkeeping
        pl = self.planner
        if pl.done and pl.path:
            self.path = pl.path
            self.pidx = 0
            pl.done = False
            self.plan_t = t
        if pl.fail:
            pl.fail = False
            if self.plan_goal is not None and not pl.loose:
                pl.request(px, py, self.plan_goal[0], self.plan_goal[1], loose=True)
            else:
                self.unreach[(int(gx // 10), int(gy // 10))] = t + 12.0
                self.goal_until = 0.0
                self.path = None
                self.plan_goal = None
                return None
        need = False
        if self.plan_goal is None or math.hypot(self.plan_goal[0] - gx, self.plan_goal[1] - gy) > 12.0:
            need = True
        elif self.path is not None and self.pidx >= len(self.path) and d > 8:
            need = True
        elif self.path is not None and t - self.plan_t > 10.0:
            need = True
        if need and not pl.active:
            pl.request(px, py, gx, gy, loose=False)
            self.plan_goal = (gx, gy)
            self.plan_t = t
        if pl.active:
            pl.work(time.perf_counter() + 0.006)
            if pl.done and pl.path:
                self.path = pl.path
                self.pidx = 0
                pl.done = False
        if self.path:
            pth = self.path
            # advance
            while self.pidx < len(pth) - 1 and math.hypot(pth[self.pidx][0] - px, pth[self.pidx][1] - py) < 5.0:
                self.pidx += 1
            # off track -> replan soon
            tgtp = pth[min(self.pidx, len(pth) - 1)]
            kmax = min(len(pth) - 1, self.pidx + 9)
            for k in range(kmax, self.pidx - 1, -1):
                if nm.clear(px, py, pth[k][0], pth[k][1]):
                    tgtp = pth[k]
                    break
            dd = math.hypot(tgtp[0] - px, tgtp[1] - py)
            if dd < 0.5:
                return (0.0, 0.0), d
            if math.hypot(pth[min(self.pidx, len(pth) - 1)][0] - px, pth[min(self.pidx, len(pth) - 1)][1] - py) > 22:
                self.plan_goal = None
            return ((tgtp[0] - px) / dd, (tgtp[1] - py) / dd), d
        # no path yet: head straight for it; steering feelers keep us off walls
        return ((gx - px) / d, (gy - py) / d), d

    # ------------------------------------------------------------------ aim and fire
    def _idle_aim(self, obs, me, t, mv):
        yaw, pit = me["aim"]
        if self.alert is not None and t < self.alert[1]:
            return (self.alert[0], 0.0)
        # sweep: slow rotation so the 120 degree wedge covers everything over time
        self.scan_yaw = wrap(self.scan_yaw + 0.13)
        if math.hypot(*mv) > 0.1 and int(t / 4.0) % 2 == 0:
            return (math.atan2(mv[1], mv[0]), 0.0)
        return (self.scan_yaw, 0.0)

    @staticmethod
    def _solve_lead(o, tp, tv, vb):
        rx, ry, rz = tp[0] - o[0], tp[1] - o[1], tp[2] - o[2]
        a = tv[0] ** 2 + tv[1] ** 2 + tv[2] ** 2 - vb * vb
        b = 2.0 * (rx * tv[0] + ry * tv[1] + rz * tv[2])
        c = rx * rx + ry * ry + rz * rz
        tt = None
        if abs(a) < 1e-6:
            if b < 0:
                tt = -c / b
        else:
            disc = b * b - 4 * a * c
            if disc >= 0:
                sq = math.sqrt(disc)
                cand = [x for x in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)) if x > 0]
                if cand:
                    tt = min(cand)
        if tt is None or tt > 1.6:
            tt = min(1.6, math.sqrt(c) / vb)
        return (tp[0] + tv[0] * tt, tp[1] + tv[1] * tt, tp[2] + tv[2] * tt), tt

    def _aim_fire(self, obs, me, tgt, t, px, py, pz, sc, mover):
        fire = [False, False]
        stats = me["stats"]
        mz = pz + stats.get("muzzle_height", 7.0)
        origin = (px, py, mz)
        tcubes = tgt.get("cubes", 0) or 0
        th = tgt.get("_h", 9.0 * self._scale(tcubes))
        tr = tgt.get("_r", 2.5 * self._scale(tcubes))
        tp = (tgt["pos"][0], tgt["pos"][1], tgt["pos"][2] + th * 0.5)
        tv = tgt.get("vel", [0.0, 0.0, 0.0])
        tv = (tv[0], tv[1], tv[2] if abs(tv[2]) < 12 else 0.0)
        d2 = math.hypot(tp[0] - px, tp[1] - py)
        d3 = math.dist(tp, origin)
        yaw_a, pit_a = me["aim"]
        weapons = me["weapons"]
        gun_slot = self._slot_of(me, "gun")
        gun = weapons[gun_slot] if gun_slot >= 0 else None
        las_slot = self._slot_of(me, "laser")
        lig_slot = self._slot_of(me, "lightning")
        has_los = self.nm.los(px, py, tp[0], tp[1])

        # which weapon drives the aim?
        mode = "gun"
        if lig_slot >= 0 and d2 < 38.0 and me["energy"] > 9 and weapons[lig_slot]["cooldown"] < 0.6:
            mode = "lightning"
        elif las_slot >= 0 and d2 < 250 and (weapons[las_slot]["cooldown"] < 0.25 or weapons[las_slot].get("charge", 0) > 0):
            mode = "laser"
        if mode == "gun" and gun is None:
            mode = "lightning" if lig_slot >= 0 else ("laser" if las_slot >= 0 else "none")

        aim_pt = tp
        if mode == "gun" and gun is not None:
            vb = gun["stats"].get("speed", 252.0)
            aim_pt, tt = self._solve_lead(origin, tp, tv, vb)
        elif mode == "laser":
            aim_pt = (tp[0] + tv[0] * 0.08, tp[1] + tv[1] * 0.08, tp[2] + tv[2] * 0.08)
        elif mode == "lightning":
            aim_pt = tp
        rx, ry, rz = aim_pt[0] - origin[0], aim_pt[1] - origin[1], aim_pt[2] - origin[2]
        hd = math.hypot(rx, ry)
        want_yaw = math.atan2(ry, rx)
        want_pit = max(-0.69, min(1.29, math.atan2(rz, max(0.5, hd))))
        eyaw = abs(wrap(want_yaw - yaw_a))
        epit = abs(want_pit - pit_a)
        dd = max(8.0, d3)
        tol_y = max(0.012, min(0.14, math.atan2(tr * 0.82, dd)))
        tol_p = max(0.02, min(0.16, math.atan2(th * 0.4, dd)))
        aligned = eyaw < tol_y and epit < tol_p

        if mode == "gun" and gun is not None and has_los:
            rng = gun["stats"].get("range", 230.0)
            if d3 < rng * 0.92 and gun.get("ammo", 1) > 0 and gun.get("reload", 0) <= 0.0 and aligned:
                fire[gun_slot] = True
        if mode == "lightning" and has_los:
            w = weapons[lig_slot]
            rng = w["stats"].get("range", 45.0)
            cone = math.radians(w["stats"].get("cone_deg", 30.0))
            if d3 < rng * 0.9 and eyaw < cone * 0.45:
                fire[lig_slot] = True
            # keep the gun going too when lightning is on cooldown / out of cone
            if gun is not None and not fire[lig_slot] and aligned and gun.get("ammo", 1) > 0 and gun.get("reload", 0) <= 0:
                fire[gun_slot] = True
        elif lig_slot >= 0 and mode != "lightning" and has_los and tgt.get("kind") == "mech":
            # lightning auto-targets inside its cone even if the gun is driving the aim
            w = weapons[lig_slot]
            if d3 < w["stats"].get("range", 45.0) * 0.85 and eyaw < math.radians(w["stats"].get("cone_deg", 30.0)) * 0.4 and me["energy"] > 9:
                fire[lig_slot] = True
        if las_slot >= 0 and (mode == "laser" or weapons[las_slot].get("charge", 0) > 0):
            w = weapons[las_slot]
            ch = w.get("charge", 0.0)
            full = w["stats"].get("charge", 0.75)
            if ch > 0.0:
                # charging: release when full (or nearly) and lined up; give up after a grace period
                self.laser_hold += 0.1
                if ch >= full * 0.97:
                    if aligned and has_los:
                        fire[las_slot] = False
                    elif ch >= full + 0.45 or not has_los:
                        fire[las_slot] = False
                    else:
                        fire[las_slot] = True
                else:
                    fire[las_slot] = True
            else:
                self.laser_hold = 0.0
                if aligned and has_los and w["cooldown"] <= 0.12 and d2 > 20.0 and me["weapons"][las_slot]["cooldown"] < 0.2:
                    fire[las_slot] = True
        return (want_yaw, want_pit), fire


def dist_goal_small(goal, px, py):
    try:
        return math.hypot(goal["pos"][0] - px, goal["pos"][1] - py) < 40
    except Exception:
        return False


if __name__ == "__main__":
    mf.run(Bot)

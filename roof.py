"""Rooftop model and climb controller for the mech bot (standard library only).

Surface map: for every 2 m cell, the highest prism top over it (absolute z) and the lowest prism base.
Spots: flat roof points at least ~8 m from any edge or step, 10..48 m above the ground.
Controller: walk to an approach point next to the wall, wait for full energy, jump + jet until the feet
clear the highest edge on the way, move over it, land, walk to the spot and hold.
"""
import json
import math
import os
from array import array

from navmap import FX0, FY0, CS, FNX, FNY

MIN_H = float(os.environ.get("ROOF_MIN_H", 12.0))
MAX_H = float(os.environ.get("ROOF_MAX_H", 45.0))
FLAT_R = 4          # cells (8 m) of flat roof required around a spot
NONE = -1e9


class RoofMap:
    def __init__(self, nm, mapdir):
        self.nm = nm
        self.ok = False
        self.spots = []
        try:
            self._build(mapdir)
            self.ok = True
        except Exception:
            self.spots = []

    def _build(self, mapdir):
        with open(os.path.join(mapdir, "v2", "world.json")) as f:
            world = json.load(f)
        top = array("f", [NONE]) * (FNX * FNY)
        base = array("f", [1e9]) * (FNX * FNY)
        for p in world["prisms"]:
            self._fill(p, top, base)
        self.top = top
        self.base = base
        # flat roof spots
        offs = [(di, dj) for di in range(-FLAT_R, FLAT_R + 1) for dj in range(-FLAT_R, FLAT_R + 1)
                if di * di + dj * dj <= FLAT_R * FLAT_R]
        best = {}
        g = self.nm.ground
        for j in range(FLAT_R, FNY - FLAT_R, 2):
            row = j * FNX
            for i in range(FLAT_R, FNX - FLAT_R, 2):
                h = top[row + i]
                if h == NONE:
                    continue
                x = FX0 + (i + 0.5) * CS
                y = FY0 + (j + 0.5) * CS
                gh = g(x, y)
                if not (MIN_H <= h - gh <= MAX_H):
                    continue
                flat = True
                for di, dj in offs:
                    hh = top[(j + dj) * FNX + i + di]
                    if hh == NONE or abs(hh - h) > 1.0:
                        flat = False
                        break
                if not flat:
                    continue
                key = (i // 6, j // 6)   # one spot per 12 m block
                if key not in best:
                    best[key] = (x, y, h, gh)
        self.spots = list(best.values())

    def _fill(self, p, top, base):
        ptop = p["top"]
        pbase = p["base"]
        edges = []
        for poly in [p["ring"]] + list(p.get("holes") or []):
            pts = list(zip(poly[0::2], poly[1::2]))
            n = len(pts)
            for k in range(n):
                x0, y0 = pts[k]
                x1, y1 = pts[(k + 1) % n]
                if y0 != y1:
                    edges.append((x0, y0, x1, y1))
        if not edges:
            return
        ys = [e[1] for e in edges] + [e[3] for e in edges]
        jmin = max(0, int((min(ys) - FY0) / CS) - 1)
        jmax = min(FNY - 1, int((max(ys) - FY0) / CS) + 1)
        for j in range(jmin, jmax + 1):
            yc = FY0 + (j + 0.5) * CS
            xs = []
            for (x0, y0, x1, y1) in edges:
                if (y0 <= yc < y1) or (y1 <= yc < y0):
                    xs.append(x0 + (yc - y0) * (x1 - x0) / (y1 - y0))
            xs.sort()
            for a in range(0, len(xs) - 1, 2):
                i0 = max(0, int(math.ceil((xs[a] - FX0) / CS - 0.5)))
                i1 = min(FNX - 1, int(math.floor((xs[a + 1] - FX0) / CS - 0.5)))
                for i in range(i0, i1 + 1):
                    k = j * FNX + i
                    if ptop > top[k]:
                        top[k] = ptop
                    if pbase < base[k]:
                        base[k] = pbase

    # ------------------------------------------------------------------ queries
    def surf(self, x, y):
        """Highest solid top at (x, y) or the terrain height if there is none."""
        i = int((x - FX0) / CS)
        j = int((y - FY0) / CS)
        if 0 <= i < FNX and 0 <= j < FNY:
            h = self.top[j * FNX + i]
            if h != NONE:
                return h
        return self.nm.ground(x, y)

    def has_prism(self, x, y):
        i = int((x - FX0) / CS)
        j = int((y - FY0) / CS)
        if 0 <= i < FNX and 0 <= j < FNY:
            return self.top[j * FNX + i] != NONE
        return True

    def overhang_near(self, x, y, r=2):
        i0 = int((x - FX0) / CS)
        j0 = int((y - FY0) / CS)
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                i, j = i0 + di, j0 + dj
                if 0 <= i < FNX and 0 <= j < FNY:
                    k = j * FNX + i
                    if self.top[k] != NONE:
                        gx = FX0 + (i + 0.5) * CS
                        gy = FY0 + (j + 0.5) * CS
                        if self.base[k] > self.nm.ground(gx, gy) + 1.0:
                            return True
        return False

    def approach(self, sx, sy, px, py):
        """Best ground point next to the building to climb towards (sx, sy) from.
        Returns (gx, gy, h_clear, climb) or None."""
        best = None
        bs = 1e18
        for k in range(24):
            a = k * math.pi / 12
            ux, uy = math.cos(a), math.sin(a)
            hmax = NONE
            d = 1.0
            found = None
            sh = self.surf(sx, sy)
            gap = False
            while d < 70.0:
                x, y = sx + ux * d, sy + uy * d
                if self.has_prism(x, y):
                    hh = self.surf(x, y)
                    if hh < sh - 1.5 and d < 66:
                        # a lower part / courtyard between the edge and the spot: is it the outer edge?
                        nx, ny = sx + ux * (d + 1.5), sy + uy * (d + 1.5)
                        if self.has_prism(nx, ny):
                            gap = True
                            break
                    hmax = max(hmax, hh)
                    d += 1.0
                    continue
                # first open ground: step 3 m further out so the mech (r 2.5) stands clear
                gx, gy = sx + ux * (d + 3.0), sy + uy * (d + 3.0)
                if self.nm.blocked(gx, gy) or self.has_prism(gx, gy) or self.overhang_near(gx, gy):
                    break
                found = (gx, gy)
                break
            if gap or found is None or hmax == NONE:
                continue
            gx, gy = found
            climb = hmax - self.nm.ground(gx, gy)
            if climb > MAX_H + 3:
                continue
            s = math.hypot(gx - px, gy - py) + 4.0 * climb
            if s < bs:
                bs = s
                best = (gx, gy, hmax, climb)
        return best

    def max_surf_along(self, x0, y0, x1, y1):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(d / 1.0))
        m = NONE
        for k in range(0, n + 1):
            f = k / n
            m = max(m, self.surf(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
        return m


class RoofCtrl:
    """State machine. act() returns None (let the normal brain drive) or a dict with move/boost/jump."""

    def __init__(self, rm, log=None):
        self.rm = rm
        self.state = "ground"
        self.spot = None          # (x, y, h, ground)
        self.app = None           # (gx, gy, hclear, climb)
        self.fails = {}
        self.bad = set()
        self.air_t = 0.0
        self.last_z = 0.0
        self.stall_t = 0.0
        self.log = log or (lambda *a: None)
        self.pick_t = -99.0
        self.give_up = False

    def _choose(self, px, py, circ, t):
        cx, cy, r = circ
        best, bs = None, 1e18
        for sp in self.rm.spots:
            key = (round(sp[0]), round(sp[1]))
            if key in self.bad:
                continue
            dc = math.hypot(sp[0] - cx, sp[1] - cy)
            if dc > r - 6.0:
                continue
            d = math.hypot(sp[0] - px, sp[1] - py)
            s = d + 2.0 * (sp[2] - sp[3]) + 0.3 * dc
            if s < bs:
                bs, best = s, sp
        if best is None:
            return False
        app = self.rm.approach(best[0], best[1], px, py)
        if app is None:
            self.bad.add((round(best[0]), round(best[1])))
            return self._choose(px, py, circ, t) if len(self.bad) < 60 else False
        self.spot = best
        self.app = app
        self.state = "ground"
        self.pick_t = t
        self.log("roof: chose spot (%.0f,%.0f) h=%.1f climb=%.1f from (%.0f,%.0f)" % (best[0], best[1], best[2] - best[3], app[3], app[0], app[1]))
        return True

    def _fail(self, why, t):
        key = (round(self.spot[0]), round(self.spot[1])) if self.spot else None
        if key is not None:
            self.fails[key] = self.fails.get(key, 0) + 1
            if self.fails[key] >= 2:
                self.bad.add(key)
        self.log("roof: fail (%s) at t=%.1f" % (why, t))
        self.spot = None
        self.app = None
        self.state = "ground"
        if len(self.bad) > 25:
            self.give_up = True

    def on_spot_roof(self, me):
        if self.spot is None:
            return False
        px, py, pz = me["pos"]
        return me.get("grounded", True) and abs(pz - self.spot[2]) < 2.5 and math.hypot(self.spot[0] - px, self.spot[1] - py) < 12.0

    def act(self, me, t, circ_next, circ_safe, navigate):
        """circ_next: the circle we must be in a while from now; circ_safe: same with margin.
        navigate(gx, gy) -> (dx, dy) ground navigation helper from the main brain."""
        if self.give_up or not self.rm.ok or not self.rm.spots:
            return None
        px, py, pz = me["pos"]
        vz = me["vel"][2]
        energy = me["energy"]
        grounded = me.get("grounded", True)
        # spot still good for the zone?
        if self.spot is not None:
            cx, cy, r = circ_next
            if math.hypot(self.spot[0] - cx, self.spot[1] - cy) > r - 4.0:
                self.log("roof: zone moves away from the spot")
                self.spot = None
                self.state = "ground"
        if self.spot is None:
            if t - self.pick_t < 2.0:
                return None
            if not self._choose(px, py, circ_next, t):
                self.pick_t = t
                return None
        sx, sy, sh, sg = self.spot
        gx, gy, hclear, climb = self.app
        dsp = math.hypot(sx - px, sy - py)
        ux, uy = ((sx - px) / dsp, (sy - py) / dsp) if dsp > 0.3 else (0.0, 0.0)
        # ------------------------------------------------ on the roof
        if grounded and pz > sh - 2.5 and dsp < 40:
            self.state = "roof"
            if dsp < 2.0:
                return {"move": [0.0, 0.0], "boost": False}
            # walk to the spot over the flat roof; hop if there is a step
            ahead = self.rm.surf(px + ux * 3, py + uy * 3)
            if ahead > pz + 1.0:
                return {"move": [ux, uy], "boost": True, "jump": True}
            if ahead < pz - 2.0:
                # a gap or edge between us and the spot: this roof is good enough, stay where we are
                back = 2.5
                self.spot = (px - ux * back, py - uy * back, pz, sg)
                return {"move": [-ux * 0.5, -uy * 0.5], "boost": False}
            return {"move": [ux * min(1.0, dsp / 6.0 + 0.3), uy * min(1.0, dsp / 6.0 + 0.3)], "boost": False}
        # ------------------------------------------------ in the air: climb
        if not grounded:
            self.state = "climb"
            look = min(dsp, 7.0)
            need = self.rm.max_surf_along(px, py, px + ux * look, py + uy * look) + 1.2
            if pz < need:
                # still under the edge: keep jetting, keep pushing towards the roof
                if abs(vz) < 0.3 and pz - self.last_z < 0.05:
                    self.stall_t += 0.1
                else:
                    self.stall_t = 0.0
                self.last_z = pz
                if self.stall_t > 0.8 or (energy < 1.0 and vz < 0):
                    self._fail("stalled" if self.stall_t > 0.8 else "out of energy", t)
                    self.stall_t = 0.0
                    return None
                return {"move": [ux, uy], "boost": True}
            self.stall_t = 0.0
            # above everything in the next few metres: stop jetting and drop onto the roof
            k = min(1.0, dsp / 5.0)
            return {"move": [ux * k, uy * k], "boost": bool(vz < -3.0 and pz < need + 0.8 and energy > 2)}
        # ------------------------------------------------ on some surface but not our roof
        if pz > sg + 4.0 and dsp < 60:
            # an intermediate roof: hop up if the way ahead rises
            ahead = self.rm.surf(px + ux * 4, py + uy * 4)
            if ahead > pz + 1.0:
                if energy < 60:
                    return {"move": [0.0, 0.0], "boost": False}
                return {"move": [ux, uy], "boost": True, "jump": True}
            return {"move": [ux, uy], "boost": False}
        # ------------------------------------------------ on the ground
        self.state = "ground"
        dg = math.hypot(gx - px, gy - py)
        if dg > 3.0:
            d = navigate(gx, gy)
            if d is None:
                return None
            return {"move": [d[0], d[1]], "boost": dg > 60 and energy > 70}
        if energy < 92:
            return {"move": [0.0, 0.0], "boost": False}   # wait for a full tank (standing still is silent)
        return {"move": [ux, uy], "boost": True, "jump": True}

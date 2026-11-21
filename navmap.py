"""Static map model for the mech bot: terrain height, a 2 m solid grid, and an anytime A* planner.

Standard library only. Everything is defensive: if the map cannot be loaded the bot still runs with
an empty (all free) map.
"""
import heapq
import json
import math
import os
import sys
import time
from array import array

CS = 2.0                      # fine grid cell (m)
FX0, FY0 = -516.0, -336.0     # fine grid origin
FNX, FNY = 540, 366           # fine grid size
STRIDE = 2                    # planner node = 2x2 fine cells = 4 m
NA, NB = FNX // STRIDE, FNY // STRIDE
NODE = CS * STRIDE


def _dilate(rows, r):
    """Circular dilation of a list of bytearray rows by r cells (bit-parallel with big ints)."""
    if r <= 0:
        return [bytearray(x) for x in rows]
    n = len(rows)
    width = len(rows[0])
    ints = [int.from_bytes(x, "little") for x in rows]
    full = (1 << (8 * width)) - 1
    # horizontal dilations H[k]: OR of shifts -k..k
    H = [ints]
    for k in range(1, r + 1):
        cur = []
        for v in ints:
            acc = v
            for s in range(1, k + 1):
                acc |= (v << (8 * s)) | (v >> (8 * s))
            cur.append(acc & full)
        H.append(cur)
    out = []
    for j in range(n):
        acc = 0
        for dy in range(-r, r + 1):
            jj = j + dy
            if jj < 0 or jj >= n:
                continue
            kmax = int(math.sqrt(max(0.0, r * r + 0.25 - dy * dy)))
            acc |= H[kmax][jj]
        out.append(bytearray(acc.to_bytes(width, "little")))
    return out


class NavMap:
    def __init__(self, mapdir=None):
        self.ok = False
        self.tn = None
        self.solid = [bytearray(FNX) for _ in range(FNY)]
        try:
            self._load(mapdir)
            self.ok = True
        except Exception as e:  # pragma: no cover
            sys.stderr.write("map load failed: %r\n" % (e,))
            self.solid = [bytearray(FNX) for _ in range(FNY)]
        self.d1 = _dilate(self.solid, 1)
        self.d2 = _dilate(self.solid, 2)
        self._build_nodes()

    # ------------------------------------------------------------------ loading
    def _load(self, mapdir):
        wpath = os.path.join(mapdir, "v2", "world.json")
        tpath = os.path.join(mapdir, "v2", "terrain.i16")
        with open(wpath) as f:
            world = json.load(f)
        meta = world["terrain"]
        self.t_minx = meta["minX"]
        self.t_miny = meta["minY"]
        self.t_step = meta["step"]
        self.t_nx = meta["nx"]
        self.t_ny = meta["ny"]
        arr = array("h")
        with open(tpath, "rb") as f:
            arr.frombytes(f.read())
        if sys.byteorder == "big":
            arr.byteswap()
        self.tn = arr
        b = world["bounds"]
        self.bounds = (b["minX"], b["minY"], b["maxX"], b["maxY"])
        # outside bounds = wall
        for j in range(FNY):
            yc = FY0 + (j + 0.5) * CS
            row = self.solid[j]
            for i in range(FNX):
                xc = FX0 + (i + 0.5) * CS
                if xc < b["minX"] + 1 or xc > b["maxX"] - 1 or yc < b["minY"] + 1 or yc > b["maxY"] - 1:
                    row[i] = 1
        # steep terrain
        lim = math.tan(math.radians(world.get("rules", {}).get("maxSlopeDeg", 50) - 2))
        for j in range(FNY):
            yc = FY0 + (j + 0.5) * CS
            row = self.solid[j]
            for i in range(FNX):
                if row[i]:
                    continue
                xc = FX0 + (i + 0.5) * CS
                hx = self._hn(xc + 1.0, yc) - self._hn(xc - 1.0, yc)
                hy = self._hn(xc, yc + 1.0) - self._hn(xc, yc - 1.0)
                if (hx * hx + hy * hy) > (2.0 * lim) ** 2:
                    row[i] = 1
        # prisms
        for p in world["prisms"]:
            self._fill(p)

    def _hn(self, x, y):
        i = int(round((x - self.t_minx) / self.t_step))
        j = int(round((y - self.t_miny) / self.t_step))
        if i < 0:
            i = 0
        elif i >= self.t_nx:
            i = self.t_nx - 1
        if j < 0:
            j = 0
        elif j >= self.t_ny:
            j = self.t_ny - 1
        return self.tn[j * self.t_nx + i] * 0.01

    def ground(self, x, y):
        """Bilinear ground height (0 without terrain)."""
        if self.tn is None:
            return 0.0
        fx = (x - self.t_minx) / self.t_step
        fy = (y - self.t_miny) / self.t_step
        i = int(math.floor(fx))
        j = int(math.floor(fy))
        i = min(max(i, 0), self.t_nx - 2)
        j = min(max(j, 0), self.t_ny - 2)
        ax = min(max(fx - i, 0.0), 1.0)
        ay = min(max(fy - j, 0.0), 1.0)
        t = self.tn
        nx = self.t_nx
        h00 = t[j * nx + i]
        h10 = t[j * nx + i + 1]
        h01 = t[(j + 1) * nx + i]
        h11 = t[(j + 1) * nx + i + 1]
        return 0.01 * ((h00 * (1 - ax) + h10 * ax) * (1 - ay) + (h01 * (1 - ax) + h11 * ax) * ay)

    def _fill(self, p):
        base = p["base"]
        top = p["top"]
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
            row = self.solid[j]
            for a in range(0, len(xs) - 1, 2):
                i0 = max(0, int(math.ceil((xs[a] - FX0) / CS - 0.5)))
                i1 = min(FNX - 1, int(math.floor((xs[a + 1] - FX0) / CS - 0.5)))
                for i in range(i0, i1 + 1):
                    if row[i]:
                        continue
                    g = self._hn(FX0 + (i + 0.5) * CS, yc)
                    if base < g + 9.0 and top > g + 1.2:
                        row[i] = 1

    def _build_nodes(self):
        self.free_strict = bytearray(NA * NB)
        self.free_loose = bytearray(NA * NB)
        d1, s0 = self.d1, self.solid
        for b in range(NB):
            r0, r1 = 2 * b, 2 * b + 1
            for a in range(NA):
                i0, i1 = 2 * a, 2 * a + 1
                if not (d1[r0][i0] or d1[r0][i1] or d1[r1][i0] or d1[r1][i1]):
                    self.free_strict[b * NA + a] = 1
                if not (s0[r0][i0] or s0[r0][i1] or s0[r1][i0] or s0[r1][i1]):
                    self.free_loose[b * NA + a] = 1

    # ------------------------------------------------------------------ queries
    @staticmethod
    def cell(x, y):
        return int((x - FX0) / CS), int((y - FY0) / CS)

    def blocked(self, x, y, grid=None):
        i = int((x - FX0) / CS)
        j = int((y - FY0) / CS)
        if i < 0 or j < 0 or i >= FNX or j >= FNY:
            return True
        return bool((grid or self.d1)[j][i])

    def solid_at(self, x, y):
        return self.blocked(x, y, self.solid)

    def clear(self, x0, y0, x1, y1, grid=None, step=1.5):
        """True if the straight segment crosses no blocked cell of `grid` (default: 1-cell dilated)."""
        g = grid or self.d1
        dx, dy = x1 - x0, y1 - y0
        d = math.hypot(dx, dy)
        n = max(1, int(d / step))
        for k in range(1, n + 1):
            f = k / n
            x = x0 + dx * f
            y = y0 + dy * f
            i = int((x - FX0) / CS)
            j = int((y - FY0) / CS)
            if i < 0 or j < 0 or i >= FNX or j >= FNY or g[j][i]:
                return False
        return True

    def los(self, x0, y0, x1, y1):
        """Rough line of sight / line of fire against buildings (2D, raw solid cells; ignores the map edge)."""
        s = self.solid
        dx, dy = x1 - x0, y1 - y0
        d = math.hypot(dx, dy)
        n = max(1, int(d / 2.0))
        for k in range(1, n):
            f = k / n
            i = int((x0 + dx * f - FX0) / CS)
            j = int((y0 + dy * f - FY0) / CS)
            if 0 <= i < FNX and 0 <= j < FNY and s[j][i]:
                return False
        return True

    def free_run(self, x, y, ux, uy, maxd, step=2.0):
        """Distance one can walk from (x,y) along unit (ux,uy) before hitting a dilated-blocked cell."""
        d = step
        while d <= maxd:
            if self.blocked(x + ux * d, y + uy * d):
                return d - step
            d += step
        return maxd

    # ------------------------------------------------------------------ planning helpers
    @staticmethod
    def node_of(x, y):
        a = int((x - FX0) / NODE)
        b = int((y - FY0) / NODE)
        return min(max(a, 0), NA - 1), min(max(b, 0), NB - 1)

    @staticmethod
    def node_xy(a, b):
        return FX0 + (a + 0.5) * NODE, FY0 + (b + 0.5) * NODE

    def nearest_free(self, a, b, free, rmax=8):
        if free[b * NA + a]:
            return a, b
        for r in range(1, rmax + 1):
            best = None
            bd = 1e9
            for bb in range(max(0, b - r), min(NB - 1, b + r) + 1):
                for aa in range(max(0, a - r), min(NA - 1, a + r) + 1):
                    if max(abs(aa - a), abs(bb - b)) != r:
                        continue
                    if free[bb * NA + aa]:
                        d = (aa - a) ** 2 + (bb - b) ** 2
                        if d < bd:
                            bd = d
                            best = (aa, bb)
            if best:
                return best
        return None


class Planner:
    """Anytime A* over the 4 m node grid. Call request(), then work() every tick until done."""

    SQ2 = math.sqrt(2.0)

    def __init__(self, nm):
        self.nm = nm
        self.active = False
        self.done = False
        self.path = None
        self.goal_xy = None
        self.fail = False

    def request(self, sx, sy, gx, gy, loose=False):
        nm = self.nm
        self.free = nm.free_loose if loose else nm.free_strict
        self.loose = loose
        self.goal_xy = (gx, gy)
        self.start_xy = (sx, sy)
        s = nm.nearest_free(*nm.node_of(sx, sy), self.free)
        g = nm.nearest_free(*nm.node_of(gx, gy), self.free, rmax=12)
        self.active = True
        self.done = False
        self.fail = False
        self.path = None
        self.nodes = 0
        if s is None or g is None:
            self.active = False
            self.fail = True
            return
        self.s, self.g = s, g
        self.sid = s[1] * NA + s[0]
        self.gid = g[1] * NA + g[0]
        self.gscore = {self.sid: 0.0}
        self.parent = {}
        self.open = [(self._h(s[0], s[1]), 0.0, self.sid)]
        self.closed = set()
        if self.sid == self.gid:
            self._finish(self.gid)

    def _h(self, a, b):
        dx = abs(a - self.g[0])
        dy = abs(b - self.g[1])
        return NODE * (max(dx, dy) + (self.SQ2 - 1.0) * min(dx, dy))

    def work(self, deadline, max_nodes=100000):
        if not self.active:
            return
        free = self.free
        gscore = self.gscore
        parent = self.parent
        op = self.open
        closed = self.closed
        gid = self.gid
        cnt = 0
        while op:
            if (cnt & 31) == 31 and time.perf_counter() > deadline:
                return
            f, g, nid = heapq.heappop(op)
            if nid in closed:
                continue
            closed.add(nid)
            cnt += 1
            self.nodes += 1
            if nid == gid:
                self._finish(nid)
                return
            if self.nodes > max_nodes:
                break
            a = nid % NA
            b = nid // NA
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                aa = a + da
                bb = b + db
                if aa < 0 or bb < 0 or aa >= NA or bb >= NB:
                    continue
                nn = bb * NA + aa
                if not free[nn] or nn in closed:
                    continue
                if da and db and not (free[b * NA + aa] and free[bb * NA + a]):
                    continue
                step = NODE * (self.SQ2 if (da and db) else 1.0)
                ng = g + step
                if ng < gscore.get(nn, 1e18):
                    gscore[nn] = ng
                    parent[nn] = nid
                    heapq.heappush(op, (ng + self._h(aa, bb), ng, nn))
        # exhausted / capped
        self.active = False
        self.fail = True

    def _finish(self, nid):
        pts = []
        cur = nid
        parent = self.parent
        while cur in parent:
            pts.append(self.nm.node_xy(cur % NA, cur // NA))
            cur = parent[cur]
        pts.reverse()
        gx, gy = self.goal_xy
        # replace the last node by the exact goal if the goal's own node was free
        if pts and self.nm.node_of(gx, gy) == (self.g[0], self.g[1]):
            pts[-1] = (gx, gy)
        self.path = pts
        self.active = False
        self.done = True

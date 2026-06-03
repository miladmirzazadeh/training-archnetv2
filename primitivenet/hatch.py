"""Universal hatch detection for DXF — run on EVERY dxf (synthetic prep & real
inference) so the transformer sees the same hatch-free primitives both times.

Detection runs HINTS-FIRST, then geometry, unifying everything into labeled
regions (the "mask with labels") + the entity handles to strip. Each later
stage only considers primitives not already claimed by an earlier one:
  1. native   — `HATCH` entities (boundary + pattern; lines OR solid fill)
  2. solid    — filled areas: SOLID / TRACE (gray highlight blocks)
  3. layer    — primitives on known hatch layers (hint, e.g. ["A-WALL-PATT"])
  4. geometric— exploded line hatch: dense clusters of short parallel lines
  5. dots     — stipple hatch: dense clusters of POINTs / tiny circles

    hd = HatchDetector("plan.dxf", hatch_layers=["A-WALL-PATT"])
    hd.regions            # list[HatchRegion]  (vector polygons + pattern + layer)
    clean = list(hd.clean_entities())          # msp entities minus hatch  -> transformer
    mask  = hd.rasterize(W, H)                 # optional labeled raster mask
"""
from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import ezdxf
from ezdxf import bbox as _bbox
from scipy import ndimage as ndi


@dataclass
class HatchRegion:
    boundary: list                       # [(x, y), ...] polygon in WCS/DXF coords
    pattern: Optional[str]               # "ANSI31" / "SOLID" / None (exploded)
    layer: str
    source: str                          # "native" | "layer" | "geometric"
    handles: list = field(default_factory=list)

    @property
    def bbox(self):
        xs = [p[0] for p in self.boundary]; ys = [p[1] for p in self.boundary]
        return (min(xs), min(ys), max(xs), max(ys))

    def to_dict(self):
        return {"type": "hatch", "pattern": self.pattern, "layer": self.layer,
                "source": self.source, "boundary": [[round(x, 2), round(y, 2)] for x, y in self.boundary],
                "n_entities": len(self.handles)}


class HatchDetector:
    def __init__(self, dxf, *, hatch_layers=None, short_frac=0.03, cell_frac=0.006,
                 min_density=3, min_cells=4, min_lines_per_cell=2.0, dot_max_r_frac=0.004,
                 parallel_frac=0.0):
        self.doc = dxf if hasattr(dxf, "modelspace") else ezdxf.readfile(str(dxf))
        self.msp = self.doc.modelspace()
        self.hatch_layers = {l.lower() for l in (hatch_layers or [])}
        ext = _bbox.extents(self.msp)
        if ext is not None and ext.has_data:
            self.x0, self.y0 = ext.extmin.x, ext.extmin.y
            self.W = max(1e-6, ext.extmax.x - self.x0); self.H = max(1e-6, ext.extmax.y - self.y0)
        else:
            self.x0 = self.y0 = 0.0; self.W = self.H = 1.0
        self.diag = math.hypot(self.W, self.H)
        self.short_len = short_frac * self.diag
        self.cell = max(1e-6, cell_frac * self.diag)
        self.min_density = min_density; self.min_cells = min_cells
        self.min_lines_per_cell = min_lines_per_cell; self.parallel_frac = parallel_frac
        self.dot_max_r = dot_max_r_frac * self.diag
        self.regions: List[HatchRegion] = []
        self.hatch_handles = set()
        self._detect()

    # ----------------------------- public API ----------------------------- #
    def clean_entities(self):
        """Yield modelspace entities with all hatch removed (transformer input)."""
        for e in self.msp:
            if e.dxf.handle not in self.hatch_handles:
                yield e

    def summary(self):
        by = Counter(r.source for r in self.regions)
        return {"regions": len(self.regions), "by_source": dict(by),
                "hatch_entities": len(self.hatch_handles)}

    def rasterize(self, width, height, bbox=None):
        """Labeled raster mask: 0 = background, i = region i (1-based)."""
        import cv2
        x0, y0, x1, y1 = bbox or (self.x0, self.y0, self.x0 + self.W, self.y0 + self.H)
        sx = width / max(1e-6, x1 - x0); sy = height / max(1e-6, y1 - y0)
        mask = np.zeros((height, width), np.uint8)
        for i, r in enumerate(self.regions, 1):
            pts = np.array([[(px - x0) * sx, height - (py - y0) * sy] for px, py in r.boundary], np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], i)
        return mask

    # ----------------------------- detection ------------------------------ #
    def _detect(self):
        self._native()                       # 1. CAD-declared hatch (HATCH)
        self._solids()                       # 2. filled areas (SOLID/TRACE)
        layer_segs, geom_segs = [], []        # 3+4. remaining line work, split by layer hint
        for (p1, p2, hh, layer) in self._candidate_segments():
            if hh in self.hatch_handles:
                continue
            (layer_segs if layer.lower() in self.hatch_layers else geom_segs).append((p1, p2, hh))
        self._cluster(layer_segs, "layer", require_density=False)
        geom_segs = [s for s in geom_segs if s[2] not in self.hatch_handles]   # geometry on what's left
        self._cluster(geom_segs, "geometric", require_density=True)
        self._dots()                         # 5. stipple/dot hatch (unclaimed)

    def _solids(self):
        for e in self.msp:
            if e.dxf.handle in self.hatch_handles or e.dxftype() not in ("SOLID", "TRACE"):
                continue
            try:
                d = e.dxf
                quad = [(d.vtx0[0], d.vtx0[1]), (d.vtx1[0], d.vtx1[1]),
                        (d.vtx3[0], d.vtx3[1]), (d.vtx2[0], d.vtx2[1])]    # bowtie -> quad
            except Exception:
                continue
            poly = [p for j, p in enumerate(quad) if p not in quad[:j]]    # dedup triangles
            if len(poly) >= 3:
                self.regions.append(HatchRegion(poly, "SOLID", e.dxf.layer, "solid", [e.dxf.handle]))
                self.hatch_handles.add(e.dxf.handle)

    def _dots(self):
        items = []
        for e in self.msp:
            if e.dxf.handle in self.hatch_handles:
                continue
            t = e.dxftype()
            if t == "POINT":
                loc = e.dxf.location; items.append(((loc[0], loc[1]), e.dxf.handle))
            elif t == "CIRCLE" and e.dxf.radius <= self.dot_max_r:
                c = e.dxf.center; items.append(((c.x, c.y), e.dxf.handle))
        if len(items) < self.min_density:
            return
        nx = max(1, int(self.W / self.cell)); ny = max(1, int(self.H / self.cell))
        count = np.zeros((ny, nx), np.int32); cell_pts = {}
        for (x, y), hh in items:
            cx = min(nx - 1, max(0, int((x - self.x0) / self.cell)))
            cy = min(ny - 1, max(0, int((y - self.y0) / self.cell)))
            count[cy, cx] += 1; cell_pts.setdefault((cy, cx), []).append(hh)
        # dots may be spaced wider than a cell -> cluster occupied cells, not per-cell density
        occ = ndi.binary_closing(count >= 1, np.ones((3, 3)))
        lbl, n = ndi.label(occ)
        for i in range(1, n + 1):
            comp = lbl == i
            if int(comp.sum()) < self.min_cells:
                continue
            handles = []
            for cy, cx in zip(*np.where(comp)):
                handles += cell_pts.get((int(cy), int(cx)), [])
            handles = list(dict.fromkeys(handles))
            if len(handles) < max(self.min_density, 6):     # need a real dot cluster, not stray markers
                continue
            poly = self._contour_poly(comp)
            if len(poly) >= 3:
                self.regions.append(HatchRegion(poly, "DOTS", self._mode_layer(handles), "dots", handles))
                self.hatch_handles.update(handles)

    def _native(self):
        for h in self.msp.query("HATCH"):
            poly = self._hatch_boundary(h)
            if len(poly) >= 3:
                pat = getattr(h.dxf, "pattern_name", None) or ("SOLID" if getattr(h.dxf, "solid_fill", 0) else None)
                self.regions.append(HatchRegion(poly, pat, h.dxf.layer, "native", [h.dxf.handle]))
                self.hatch_handles.add(h.dxf.handle)

    @staticmethod
    def _hatch_boundary(hatch):
        pts = []
        try:
            for path in hatch.paths:
                if hasattr(path, "vertices") and path.vertices:
                    pts += [(v[0], v[1]) for v in path.vertices]
                elif hasattr(path, "edges"):
                    for edge in path.edges:
                        s, e = getattr(edge, "start", None), getattr(edge, "end", None)
                        if s is not None: pts.append((s[0], s[1]))
                        if e is not None: pts.append((e[0], e[1]))
                if pts:
                    break
        except Exception:
            pass
        return pts

    def _candidate_segments(self):
        """All LINE / polyline segments as (mid, angle_mod180, length, handle, layer)."""
        for e in self.msp:
            t = e.dxftype()
            if t == "LINE":
                a, b = e.dxf.start, e.dxf.end
                segs = [((a.x, a.y), (b.x, b.y))]
            elif t in ("LWPOLYLINE", "POLYLINE"):
                try:
                    pts = ([(p[0], p[1]) for p in e.get_points()] if t == "LWPOLYLINE"
                           else [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices])
                except Exception:
                    continue
                segs = list(zip(pts, pts[1:]))
            else:
                continue
            for (x1, y1), (x2, y2) in segs:
                if math.hypot(x2 - x1, y2 - y1) > 0:
                    yield (x1, y1), (x2, y2), e.dxf.handle, e.dxf.layer

    def _cluster(self, segs, source, require_density):
        if len(segs) < self.min_density:
            return
        nx = max(1, int(self.W / self.cell)); ny = max(1, int(self.H / self.cell))
        count = np.zeros((ny, nx), np.int32); cell_lines = {}
        for p1, p2, hh in segs:                            # rasterize the whole line (any length)
            L = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            for cy, cx in self._line_cells(p1, p2, nx, ny):
                count[cy, cx] += 1; cell_lines.setdefault((cy, cx), []).append((hh, L))
        occ = (count >= self.min_density) if require_density else (count >= 1)
        occ = ndi.binary_closing(occ, np.ones((3, 3)))
        lbl, n = ndi.label(occ)
        for i in range(1, n + 1):
            comp = lbl == i
            sz = int(comp.sum())
            if sz < self.min_cells:
                continue
            hl = {}
            for cy, cx in zip(*np.where(comp)):
                for hh, L in cell_lines.get((int(cy), int(cx)), []):
                    hl[hh] = L
            # geometric: only the SHORT strokes are hatch — keep long lines (walls)
            # that merely cross the region. layer mode trusts the layer (take all).
            handles = ([hh for hh, L in hl.items() if L <= self.short_len]
                       if require_density else list(hl.keys()))
            if len(handles) < self.min_density:
                continue
            if require_density and len(handles) / sz < self.min_lines_per_cell:
                continue
            poly = self._contour_poly(comp)
            if len(poly) >= 3:
                self.regions.append(HatchRegion(poly, None, self._mode_layer(handles), source, handles))
                self.hatch_handles.update(handles)

    def _line_cells(self, p1, p2, nx, ny):
        (x1, y1), (x2, y2) = p1, p2
        steps = max(2, int(math.hypot(x2 - x1, y2 - y1) / self.cell) + 1)
        out = set()
        for t in np.linspace(0.0, 1.0, steps):
            x = x1 + (x2 - x1) * t; y = y1 + (y2 - y1) * t
            cx = min(nx - 1, max(0, int((x - self.x0) / self.cell)))
            cy = min(ny - 1, max(0, int((y - self.y0) / self.cell)))
            out.add((cy, cx))
        return out

    def _contour_poly(self, comp):
        import cv2
        m = (comp * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return []
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
        return [(self.x0 + (col + 0.5) * self.cell, self.y0 + (row + 0.5) * self.cell) for col, row in c]

    def _mode_layer(self, handles):
        lays = [self.doc.entitydb.get(h).dxf.layer for h in handles if self.doc.entitydb.get(h) is not None]
        return Counter(lays).most_common(1)[0][0] if lays else "0"

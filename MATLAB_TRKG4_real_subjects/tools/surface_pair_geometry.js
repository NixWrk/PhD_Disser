/* Surface-only mirror helper. All coordinates are in millimetres.
 * Projection uses triangle interiors, not just the nearest mesh vertex.
 * Exported pairs are frozen explicit centres; the FEM loader projects them again.
 */
(function (root) {
  "use strict";
  const sub = (a, b) => a.map((x, i) => x - b[i]);
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const dist2 = (a, b) => dot(sub(a, b), sub(a, b));
  function closestPoint(p, a, b, c) {
    const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const aa = dot(ab, ab), bb = dot(ac, ac), cross = dot(ab, ac);
    const denominator = aa * bb - cross * cross;
    if (denominator > 1e-20) {
      const u = (dot(ap, ab) * bb - dot(ap, ac) * cross) / denominator;
      const v = (dot(ap, ac) * aa - dot(ap, ab) * cross) / denominator;
      if (u >= 0 && v >= 0 && u + v <= 1) {
        return a.map((x, i) => x + u * ab[i] + v * ac[i]);
      }
    }
    let best = a, bestDistance = dist2(p, a);
    for (const [start, end] of [[a, b], [b, c], [c, a]]) {
      const edge = sub(end, start), length2 = dot(edge, edge);
      const t = length2 ? Math.max(0, Math.min(1, dot(sub(p, start), edge) / length2)) : 0;
      const q = start.map((x, i) => x + t * edge[i]);
      const d = dist2(p, q);
      if (d < bestDistance) { best = q; bestDistance = d; }
    }
    return best;
  }
  function createSurface(vertices, faces) {
    if (!vertices.length || vertices.some(p => p.length !== 3 || !p.every(Number.isFinite))) {
      throw Error("Некорректная поверхность для парного выбора.");
    }
    const triangles = faces.map(face => {
      const points = face.map(i => vertices[i]);
      if (points.length !== 3 || points.some(p => !p)) throw Error("Некорректная грань поверхности.");
      return { points, low: [0, 1, 2].map(i => Math.min(...points.map(p => p[i]))),
        high: [0, 1, 2].map(i => Math.max(...points.map(p => p[i]))) };
    });
    const bounds = [0, 1, 2].map(axis => vertices.reduce(
      (r, p) => [Math.min(r[0], p[axis]), Math.max(r[1], p[axis])], [Infinity, -Infinity]));
    function nearest(p, axis, offset, wantedSide) {
      let best = null, bestDistance = Infinity;
      for (const tri of triangles) {
        if (wantedSide && (wantedSide > 0 ? tri.high[axis] <= offset : tri.low[axis] >= offset)) continue;
        let lowerBound = 0;
        for (let i = 0; i < 3; i++) {
          const delta = Math.max(tri.low[i] - p[i], 0, p[i] - tri.high[i]);
          lowerBound += delta * delta;
        }
        if (lowerBound >= bestDistance) continue;
        const q = closestPoint(p, ...tri.points);
        if (wantedSide && wantedSide * (q[axis] - offset) <= 1e-6) continue;
        const d = dist2(p, q);
        if (d < bestDistance) { best = q; bestDistance = d; }
      }
      return best && { point: best.slice(), distance: Math.sqrt(bestDistance) };
    }
    function pair(p, axis, offset) {
      if (![0, 1, 2].includes(axis) || !Number.isFinite(offset) || !p.every(Number.isFinite)) return null;
      if (offset <= bounds[axis][0] || offset >= bounds[axis][1]) return null;
      const side = Math.sign(p[axis] - offset);
      if (!side || Math.abs(p[axis] - offset) < 1e-6) return null;
      const selected = nearest(p, axis, offset, side);
      if (!selected) return null;
      const reflected = selected.point.slice();
      reflected[axis] = 2 * offset - reflected[axis];
      const opposite = nearest(reflected, axis, offset, -side);
      if (!opposite || dist2(selected.point, opposite.point) < 1e-12) return null;
      return { selected: selected.point, opposite: opposite.point, reflected,
        projectionMm: opposite.distance, side };
    }
    return { bounds, pair };
  }
  const api = { createSurface, closestPoint };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SurfacePairGeometry = api;
})(typeof globalThis !== "undefined" ? globalThis : this);


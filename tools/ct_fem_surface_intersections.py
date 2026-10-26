"""Conservative geometric adjudication of MeshLab's selected face pairs.

A 3D intersection must project into the intersection of the two 2D triangles.
Use an injective projection of the first triangle (drop its dominant normal
axis). Empty intersection proves separation. Intersection contained in a
shared topological vertex/edge proves that only allowed adjacency is possible.
Anything else remains a refusal, even if the pair might be separated in 3D.
No coordinate rounding, displacement, overlap-area tolerance or mesh repair.
"""
import numpy as np
from shapely.geometry import LineString, Point, Polygon


def projected_pair_is_separate_or_adjacent(first, second, common):
    normal = np.cross(first[1]-first[0], first[2]-first[0])
    if not np.isfinite(normal).all() or not np.linalg.norm(normal):
        return False
    axes = [i for i in range(3) if i != np.argmax(abs(normal))]
    a, b = Polygon(first[:, axes]), Polygon(second[:, axes])
    if not a.is_valid or not b.is_valid:
        return False
    intersection = a.intersection(b)
    if intersection.is_empty:
        return True
    if len(common) == 1:
        return Point(common[0, axes]).covers(intersection)
    if len(common) == 2:
        return LineString(common[:, axes]).covers(intersection)
    return False


def adjudicate(surface, selected_faces, maximum_pairs=100000):
    ids = np.asarray(selected_faces, dtype=int)
    report = dict(detector_selected_faces=int(len(ids)), candidate_pairs=0,
        dismissed_by_projection=0, unresolved_pairs=0, coordinates_modified=False,
        overlap_area_tolerance_mm2=0., projection_shared_topology_only=True)
    # Both faces of every reported intersection belong to the detector's
    # selected set. Only disprove its candidates; do not replace its search.
    if len(ids) > 5000:
        return dict(report, passed=False, refusal="Too many detector candidates for bounded adjudication")
    triangles = surface.triangles[ids]; faces = surface.faces[ids]
    lower, upper = triangles.min(axis=1), triangles.max(axis=1)
    for i in range(len(ids)):
        possible = np.flatnonzero(np.all(lower <= upper[i], axis=1) &
            np.all(upper >= lower[i], axis=1) & (np.arange(len(ids)) > i))
        for j in possible:
            report["candidate_pairs"] += 1
            if report["candidate_pairs"] > maximum_pairs:
                return dict(report, passed=False, refusal="Candidate-pair resource limit")
            shared = np.intersect1d(faces[i], faces[j])
            if projected_pair_is_separate_or_adjacent(triangles[i], triangles[j], surface.vertices[shared]):
                report["dismissed_by_projection"] += 1
            else:
                report["unresolved_pairs"] += 1
    return dict(report, passed=report["unresolved_pairs"] == 0)

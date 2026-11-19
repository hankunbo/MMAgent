"""Disc-rooted branch geometry, independent of the preserved official PVBM baseline.

CPU research measurement. No model, patient label, medical threshold or database access.
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

from Algorithm_Tool.vessel_branch_geometry.policy import POLICY, VERSION


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def seal(value):
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def binary(value):
    a = np.asarray(value)
    if a.ndim != 2 or not a.size or a.size > POLICY["maximum_pixels"]:
        raise ValueError("mask_dimensions")
    if a.dtype.kind not in "bui" or not set(np.unique(a)).issubset({0, 1, 255}):
        raise ValueError("mask_not_binary")
    return a > 0


def read_mask(path):
    with path.open("rb") as source:
        data = source.read(16_777_217)
    if len(data) > 16_777_216:
        raise ValueError("encoded_size_limit")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.mode != "L":
            raise ValueError("binary_grayscale_png_required")
        if image.width * image.height > POLICY["maximum_pixels"]:
            raise ValueError("expanded_size_limit")
        a = binary(np.array(image))
    return a, {"sha256": digest(data), "bytes": len(data), "dimensions_hw": list(a.shape)}


def normalized(mask, center, radius):
    """Nearest pixel-centre sampling and a lossless canonical rotation.

    This removes scan-order rotation bias; it does not correct segmentation errors.
    Radius-relative resampling makes a nearest-neighbour enlarged input equivalent.
    """
    h, w = mask.shape
    factor = POLICY["reference_disc_diameter"] / (2 * radius)
    nh, nw = round(h * factor), round(w * factor)
    if min(nh, nw) < 8 or nh * nw > POLICY["maximum_pixels"]:
        raise ValueError("normalization_size_limit")
    candidates = []
    for k in range(4):
        original = np.rot90(mask, k)
        hh, ww = (nh, nw) if k % 2 == 0 else (nw, nh)
        oh, ow = original.shape
        yi = np.minimum(((np.arange(hh) + 0.5) * oh / hh).astype(int), oh - 1)
        xi = np.minimum(((np.arange(ww) + 0.5) * ow / ww).astype(int), ow - 1)
        a = original[np.ix_(yi, xi)]
        payload = str(a.shape).encode() + np.packbits(a).tobytes()
        cx, cy = center
        rh, rw = h, w
        for _ in range(k):
            cx, cy, rh, rw = cy, rw - 1 - cx, rw, rh
        cx, cy = (np.array([cx, cy]) + 0.5) * [ww / rw, hh / rh] - 0.5
        # Orient by disc position first. Hash is only a deterministic tie break
        # for a geometrically symmetric frame, not a global trigger that changes
        # skeletonization direction when one unrelated vessel pixel changes.
        candidates.append(
            ((cx + 0.5) / ww, (cy + 0.5) / hh, hashlib.sha256(payload).digest(), cx, cy, k, a)
        )
    _, _, _, cx, cy, turns, resized = min(candidates, key=lambda item: item[:6])

    def restore(xy):
        x, y = xy
        hh, ww = (nh, nw) if turns % 2 == 0 else (nw, nh)
        for _ in range(turns):
            x, y, hh, ww = hh - 1 - y, x, ww, hh
        return [float((x + 0.5) * w / nw - 0.5), float((y + 0.5) * h / nh - 0.5)]

    return resized.copy(), (cx, cy), restore, (nh, nw), turns


def pixel_graph(skeleton):
    pixels = {tuple(p) for p in np.argwhere(skeleton)}
    if len(pixels) > 100000:
        raise ValueError("skeleton_complexity_limit")
    g = nx.Graph()
    g.add_nodes_from(sorted(pixels))
    for y, x in sorted(pixels):
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            q = y + dy, x + dx
            if q not in pixels:
                continue
            if dy and dx and ((y + dy, x) in pixels or (y, x + dx) in pixels):
                continue  # no triangular diagonal shortcuts
            g.add_edge((y, x), q, weight=math.hypot(dy, dx))
    return g


def compressed(g):
    junctions = {p for p in g if g.degree(p) >= 3}
    if len(junctions) > 2048:
        raise ValueError("junction_complexity_limit")
    groups = nx.utils.UnionFind(junctions)
    merge = POLICY["junction_merge_disc_diameters"] * POLICY["reference_disc_diameter"]
    for p in sorted(junctions):
        # Merge only along short skeleton paths, never across adjacent vessels.
        lengths = nx.single_source_dijkstra_path_length(g, p, cutoff=merge, weight="weight")
        for q in junctions.intersection(lengths):
            groups.union(p, q)
    clusters = {}
    for p in sorted(junctions):
        clusters.setdefault(groups[p], set()).add(p)
    # Include degree-2 pixels lying along the short junction bridges.
    for cluster in clusters.values():
        origins = sorted(cluster)
        for p in origins:
            paths = nx.single_source_dijkstra_path(g, p, cutoff=merge, weight="weight")
            for q in origins:
                if q in paths:
                    cluster.update(paths[q])
    nodes = sorted(clusters.values(), key=lambda s: min(s))
    covered = set().union(*nodes) if nodes else set()
    nodes += [{p} for p in sorted(g) if g.degree(p) != 2 and p not in covered]
    lookup = {p: n for n, group in enumerate(nodes) for p in group}
    graph = nx.MultiGraph()
    for n, group in enumerate(nodes):
        graph.add_node(n, xy=np.mean(list(group), axis=0)[::-1], pixels=group)
    used = set()
    for n, group in enumerate(nodes):
        for p in sorted(group):
            for q in sorted(g[p]):
                if q in group or frozenset((p, q)) in used:
                    continue
                path = [p, q]
                used.add(frozenset((p, q)))
                while path[-1] not in lookup:
                    nexts = [z for z in g[path[-1]] if z != path[-2]]
                    if len(nexts) != 1:
                        raise ValueError("skeleton_compression_invariant")
                    z = nexts[0]
                    used.add(frozenset((path[-1], z)))
                    path.append(z)
                end = lookup[path[-1]]
                if end == n:
                    graph.nodes[n]["internal_loop"] = True
                    continue
                graph.add_edge(n, end, path=np.array(path)[:, ::-1].astype(float), start=n)
    return graph


def prune_spurs(g):
    """Remove short terminal raster spurs; never cut a link between junctions."""
    limit = POLICY["spur_max_disc_diameters"] * POLICY["reference_disc_diameter"]
    removed = 0
    for _ in range(3):
        discard = set()
        for p in sorted(g):
            if g.degree(p) != 1:
                continue
            path, length = [p], 0.0
            while True:
                choices = [q for q in g[path[-1]] if len(path) == 1 or q != path[-2]]
                if len(choices) != 1:
                    break
                q = choices[0]
                length += g[path[-1]][q]["weight"]
                if length > limit:
                    break
                if g.degree(q) >= 3:
                    discard.update(path)
                    break
                path.append(q)
        if not discard:
            break
        removed += len(discard)
        g.remove_nodes_from(discard)
    return removed


def arm(graph, node, other, key):
    edge = graph[node][other][key]
    path = edge["path"] if edge["start"] == node else edge["path"][::-1]
    xy = graph.nodes[node]["xy"]
    points = np.vstack([xy, path])
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    start = POLICY["sample_start_disc_diameters"] * POLICY["reference_disc_diameter"]
    end = POLICY["sample_end_disc_diameters"] * POLICY["reference_disc_diameter"]
    if lengths[-1] < end:
        return None
    a = np.array([np.interp(start, lengths, points[:, d]) for d in range(2)])
    b = np.array([np.interp(end, lengths, points[:, d]) for d in range(2)])
    vector = b - a
    norm = np.linalg.norm(vector)
    if norm < (end - start) * 0.5:
        return None  # folded/looping branch does not have a stable local direction
    samples = np.stack(
        [np.interp(np.linspace(start, end, 13), lengths, points[:, d]) for d in range(2)], axis=1
    )
    mean = samples.mean(axis=0)
    _, axes = np.linalg.eigh((samples - mean).T @ (samples - mean))
    direction = axes[:, -1]
    if np.dot(direction, vector) < 0:
        direction = -direction
    fitted = [mean + np.dot(p - mean, direction) * direction for p in [a, b]]
    return {"vector": direction, "sample_xy": fitted, "length": float(lengths[-1])}


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def validate_disc(mask, center, radius):
    if (
        len(center) != 2
        or not np.isfinite([*center, radius]).all()
        or radius <= 2
        or not 0 <= center[0] < mask.shape[1]
        or not 0 <= center[1] < mask.shape[0]
    ):
        raise ValueError("invalid_disc_geometry")


def measure_once(mask, center, radius):
    mask = binary(mask)
    validate_disc(mask, center, radius)
    a, c, restore, normalized_shape, turns = normalized(mask, center, radius)
    yy, xx = np.indices(a.shape)
    r = POLICY["reference_disc_diameter"] / 2
    a &= np.hypot(xx - c[0], yy - c[1]) > r
    pixels = pixel_graph(skeletonize(a))
    pruned = prune_spurs(pixels)
    graph = compressed(pixels)
    rows = []
    for component in nx.connected_components(graph):
        sub = graph.subgraph(component)
        terminals = sorted(
            (float(np.linalg.norm(graph.nodes[n]["xy"] - c)), n)
            for n in component
            if sub.degree(n) == 1
        )
        root_reason = None
        if not terminals or terminals[0][0] > POLICY["root_max_disc_radii"] * r:
            root_reason = "no_disc_connected_root"
        elif len(terminals) > 1 and terminals[1][0] - terminals[0][0] < (
            POLICY["root_tie_disc_diameters"] * 2 * r
        ):
            root_reason = "ambiguous_disc_root"
        root = None if root_reason else terminals[0][1]
        simple = nx.Graph(sub)
        cycle_nodes = set().union(*[set(v) for v in nx.cycle_basis(simple)])
        cycle_nodes.update(n for n in component if graph.nodes[n].get("internal_loop"))
        for u, v in simple.edges:
            if sub.number_of_edges(u, v) > 1:
                cycle_nodes.update((u, v))
        bridges = {frozenset(e) for e in nx.bridges(simple)}
        paths = nx.single_source_shortest_path(simple, root) if root is not None else {}
        for n in sorted(component):
            degree = sub.degree(n)
            if degree < 3:
                continue
            xy = graph.nodes[n]["xy"]
            row = {
                "vertex_xy": restore(xy),
                "arms": degree,
                "status": "excluded",
                "reason": None,
                "angle_degrees": None,
            }
            rows.append(row)
            edges = list(sub.edges(n, keys=True))
            sampled = [arm(graph, n, v, k) for _, v, k in edges]
            if degree == 4:
                crossing = False
                if all(v is not None for v in sampled):
                    vectors = [v["vector"] for v in sampled]
                    for pairs in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
                        if all(angle(vectors[i], vectors[j]) >= 155 for i, j in pairs):
                            crossing = True
                row["reason"] = "crossing_candidate" if crossing else "complex_four_arm_junction"
            elif degree != 3:
                row["reason"] = "multiple_branches"
            elif n in cycle_nodes:
                row["reason"] = "loop_ambiguous"
            elif root_reason:
                row["reason"] = root_reason
            elif not all(v is not None for v in sampled):
                row["reason"] = "short_or_folded_arm"
            else:
                parent = paths[n][-2] if len(paths[n]) > 1 else None
                if parent is None or any(frozenset((u, v)) not in bridges for u, v, _ in edges):
                    row["reason"] = "parent_not_unique"
                    continue
                children = [s for (_, v, _), s in zip(edges, sampled, strict=True) if v != parent]
                parents = [s for (_, v, _), s in zip(edges, sampled, strict=True) if v == parent]
                if len(children) != 2 or len(parents) != 1:
                    raise ValueError("parent_count_invariant")
                native_children = [[restore(p) for p in s["sample_xy"]] for s in children]
                directions = [np.subtract(points[1], points[0]) for points in native_children]
                directions = [d / np.linalg.norm(d) for d in directions]
                row.update(
                    status="measured",
                    reason=None,
                    angle_degrees=angle(*directions),
                    parent_samples_xy=[restore(p) for p in parents[0]["sample_xy"]],
                    daughter_samples_xy=native_children,
                    root_xy=restore(graph.nodes[root]["xy"]),
                )
    rows.sort(key=lambda row: (row["vertex_xy"][1], row["vertex_xy"][0]))
    for i, row in enumerate(rows):
        row["junction_id"] = f"J{i + 1:03d}"
    values = [row["angle_degrees"] for row in rows if row["status"] == "measured"]
    result = {
        "schema_version": VERSION,
        "policy": POLICY,
        "policy_checksum": seal(POLICY),
        "status": "measured" if values else "no_valid_bifurcation",
        "unit": "degree",
        "count": len(values),
        "median_degrees": float(np.median(values)) if values else None,
        "mean_degrees": float(np.mean(values)) if values else None,
        "junctions": rows,
        "disc_center_xy": list(center),
        "disc_radius_px": radius,
        "normalization": {
            "shape_hw": normalized_shape,
            "quarter_turns": turns,
            "pruned_terminal_pixels": pruned,
        },
        "research_only": True,
        "clinical_threshold": None,
        "baseline": "PVBM unmodified; independent topology-aware algorithm, not SIVA",
    }
    return json.loads(json.dumps(result, allow_nan=False))


def measure(mask, center, radius):
    from Algorithm_Tool.vessel_branch_geometry.stability import checked_measure

    mask = binary(mask)
    validate_disc(mask, center, radius)
    return checked_measure(measure_once, normalized, mask, center, radius, POLICY)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vein-mask", type=Path, required=True)
    p.add_argument("--disc-mask", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    vein, vein_binding = read_mask(args.vein_mask)
    disc, disc_binding = read_mask(args.disc_mask)
    if vein.shape != disc.shape or not disc.any():
        raise ValueError("aligned_nonempty_disc_required")
    from scipy.ndimage import label

    if (
        label(disc)[1] != 1
        or disc[0].any()
        or disc[-1].any()
        or disc[:, 0].any()
        or disc[:, -1].any()
    ):
        raise ValueError("disc_component_or_clipping")
    y, x = np.nonzero(disc)
    result = measure(vein, [float(x.mean()), float(y.mean())], math.sqrt(disc.sum() / math.pi))
    result["input_bindings"] = {"vein_mask": vein_binding, "disc_mask": disc_binding}
    result["checksum"] = seal(result)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("status", "count", "median_degrees", "checksum")}))


if __name__ == "__main__":
    main()

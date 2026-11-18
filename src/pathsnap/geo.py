import numpy as np
from pyproj import CRS, Transformer


def metric_transformer(lon: float, lat: float):
    local = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")
    return Transformer.from_crs("EPSG:4326", local, always_xy=True)


def edge_midpoints(nodes: np.ndarray, arcs: np.ndarray) -> np.ndarray:
    return (nodes[arcs[:, 0]] + nodes[arcs[:, 1]]) / 2


def candidates(track: np.ndarray, nodes: np.ndarray, arcs: np.ndarray, radius_m=100, limit=8):
    """Return candidate arc IDs and distances per observation; local metric projection."""
    mid = edge_midpoints(nodes, arcs)
    center = np.nanmean(track[:, :2], axis=0)
    tf = metric_transformer(*center)
    mx, my = tf.transform(mid[:, 0], mid[:, 1])
    gx, gy = tf.transform(track[:, 0], track[:, 1])
    edge_xy = np.c_[mx, my]
    out = []
    for x, y in zip(gx, gy):
        d = np.hypot(edge_xy[:, 0] - x, edge_xy[:, 1] - y)
        ids = np.flatnonzero(d <= radius_m)
        if not len(ids):
            ids = np.argsort(d)[:1]
        ids = ids[np.argsort(d[ids])[:limit]]
        out.append([(int(i), float(d[i])) for i in ids])
    return out

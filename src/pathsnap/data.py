from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray  # longitude, latitude, timestamp seconds
    nodes: np.ndarray  # longitude, latitude
    arcs: np.ndarray   # from_node, to_node; row index is arc id
    route: np.ndarray | None


def _read(path: Path, columns: int, dtype=float) -> np.ndarray:
    if not path.exists():
        return np.empty((0, columns), dtype=dtype)
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if arr.shape[1] != columns:
        raise ValueError(f"{path}: expected {columns} columns, found {arr.shape[1]}")
    return arr


def load_trip(folder: str | Path) -> Trip:
    folder = Path(folder)
    files = {p.suffix: p for p in folder.iterdir() if p.is_file()}
    trip_id = folder.name
    route = _read(files[".route"], 1, int).ravel() if ".route" in files else None
    track = _read(files[".track"], 3)
    nodes = _read(files[".nodes"], 2)
    arcs = _read(files[".arcs"], 2, int)
    if not len(track) or not len(nodes) or not len(arcs):
        raise ValueError(f"Trip {trip_id} must include nonempty .track, .nodes, and .arcs files")
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError(f"Trip {trip_id} has non-finite GPS/network coordinates")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"Trip {trip_id} has arc endpoints outside its node table")
    return Trip(trip_id, track, nodes, arcs, route)


def find_trip(data_root: str | Path, trip_id: str) -> tuple[str, Trip]:
    root = Path(data_root).resolve()
    # IDs are searched only in explicitly allowed partitions by the caller.
    matches = list(root.glob(f"*/{trip_id}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Trip {trip_id!r} was not found uniquely under the configured data root")
    return matches[0].parent.name, load_trip(matches[0])

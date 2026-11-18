import argparse
from pathlib import Path
import csv
import numpy as np
import folium
from pathsnap.data import load_trip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default="reports/exploration")
    p.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    args = p.parse_args()
    root, out = Path(args.data_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for partition in args.partitions:
        base = root / partition
        if not base.exists():
            continue
        for folder in sorted(x for x in base.iterdir() if x.is_dir()):
            try:
                trip = load_trip(folder)
                gaps = np.diff(trip.track[:, 2])
                row = {"partition": partition, "trip_id": trip.trip_id, "gps_points": len(trip.track),
                       "nodes": len(trip.nodes), "arcs": len(trip.arcs), "route_arcs": len(trip.route) if trip.route is not None else "",
                       "missing_gps": int((~np.isfinite(trip.track)).any(axis=1).sum()),
                       "non_increasing_timestamps": int((gaps <= 0).sum()), "gaps_over_5s": int((gaps > 5).sum()),
                       "track_bytes": (folder / f"{trip.trip_id}.track").stat().st_size}
                rows.append(row)
                if len(rows) <= 5:
                    center = trip.track[:, 1].mean(), trip.track[:, 0].mean()
                    m = folium.Map(location=center, zoom_start=14)
                    folium.PolyLine(trip.track[:, [1, 0]].tolist(), color="blue", tooltip="Observed GPS").add_to(m)
                    folium.PolyLine(trip.nodes[trip.arcs[:, 0]][:, [1, 0]].tolist(), color="gray", opacity=.45, tooltip="Road arcs (start nodes)").add_to(m)
                    folium.LayerControl().add_to(m)
                    m.save(out / f"{partition}_{trip.trip_id}.html")
            except Exception as exc:
                rows.append({"partition": partition, "trip_id": folder.name, "error": str(exc)})
    if rows:
        fields = sorted({k for r in rows for k in r})
        with (out / "inventory.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} trip inventory rows to {out}")


if __name__ == "__main__":
    main()

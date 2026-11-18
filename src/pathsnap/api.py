import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from .data import find_trip
from .geo import candidates
from .decoder import viterbi

DATA_ROOT = Path(os.getenv("PATHSNAP_DATA", "data")).resolve()
ALLOWED = {x.strip() for x in os.getenv("PATHSNAP_ALLOWED_TRIPS", "").split(",") if x.strip()}
app = FastAPI(title="PathSnap API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("PATHSNAP_CORS_ORIGINS", "*").split(","), allow_methods=["GET"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "data_configured": DATA_ROOT.exists(), "allowed_trip_count": len(ALLOWED)}


@app.get("/trips")
def trips():
    if not ALLOWED:
        raise HTTPException(503, "Set PATHSNAP_ALLOWED_TRIPS to the trip IDs enabled for this evaluation stage")
    return {"trips": sorted(ALLOWED)}


@app.get("/predict/{trip_id}")
def predict(trip_id: str):
    if trip_id not in ALLOWED:
        raise HTTPException(404, "Trip is not enabled for this evaluation stage")
    try:
        partition, trip = find_trip(DATA_ROOT, trip_id)
        cand = candidates(trip.track, trip.nodes, trip.arcs)
        # Transparent geometry baseline, replaced by the independently trained model artifact.
        ids = [[a for a, _ in row] for row in cand]
        weights = [[float(np.exp(-d / 25.0)) for _, d in row] for row in cand]
        pred = viterbi(ids, weights, trip.arcs, timestamps=trip.track[:, 2], gap_break_s=30)
    except (OSError, ValueError, IndexError) as exc:
        raise HTTPException(500, f"Unable to process trip: {exc}") from exc
    # Never return reference labels or evaluator scores to the browser.
    return {"trip_id": trip_id, "observations": trip.track[:, :2].tolist(), "predicted_arc_ids": pred,
            "nodes": trip.nodes.tolist(), "arcs": trip.arcs.tolist(), "partition": partition,
            "mode": "geometry_baseline_replace_with_trained_model"}

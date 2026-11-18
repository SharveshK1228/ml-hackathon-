# PathSnap

PathSnap reconstructs road routes from noisy GPS observations using a road network, a trained candidate classifier, and sequential route decoding.

## Project layout

- `src/pathsnap/data.py` reads and validates the supplied trip files.
- `src/pathsnap/geo.py` provides local metric coordinates and candidate generation.
- `src/pathsnap/decoder.py` performs graph-aware Viterbi decoding.
- `src/pathsnap/api.py` exposes trips and predictions through FastAPI.
- `ui/` contains the evaluator-facing browser UI.
- `scripts/explore.py` creates a dataset inventory and Folium trip maps.
- `reports/` is for actual experiment outputs and the final report.

Model training is intentionally separate; place the other laptop's trained model and preprocessing artifacts in `artifacts/` and configure `PATHSNAP_MODEL`. Do not commit evaluator labels or fabricated metrics. The current demo can show generated candidates as a clearly labeled baseline until the trained model artifact is supplied.

## Data

Extract the provided `Student folder` so it contains `train/` and `student_test/` directories. Each trip folder is expected to contain matching `.track`, `.nodes`, `.arcs`, and, where available, `.route` files. The evaluator partition is not included in this repository; provide it locally as `PATHSNAP_EVALUATOR_DATA` only during the final evaluation setup. Keep its labels server-side and never expose them to the UI.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Explore data and generate maps

```powershell
python scripts/explore.py --data-root "C:\path\to\Student folder" --output reports/exploration
```

This writes an inventory CSV and Folium maps for selected trips. Exploration and maps should use training or student-testing trips only.

## Run API and UI

On Laptop 1:

```powershell
$env:PATHSNAP_DATA="C:\path\to\Student folder"
uvicorn pathsnap.api:app --app-dir src --host 0.0.0.0 --port 8000
```

On Laptop 2, open `ui/index.html`, set the API URL to Laptop 1's address, select a trip, and request its route. Configure `PATHSNAP_ALLOWED_TRIPS` with comma-separated `student_test` trip IDs for the student-stage demo. At final evaluation, use the ten reserved IDs supplied by organisers. The browser sends only a trip ID; labels remain on the server.

The `/health`, `/trips`, and `/predict/{trip_id}` endpoints are documented at `http://localhost:8000/docs`.

## Results and report

Record real runs, settings, seeds, timings, and metrics under `reports/`. The final submission needs evidence for five model comparisons, Bayesian tuning, classifier-only versus HMM decoding, actual Folium visualisations, and screenshots of the working UI/API. This repository does not assert any experimental results.

## Academic integrity

Implement and explain your team's independent work. Do not exchange models, code, configurations, or results with other teams. Disclose AI assistance according to event rules.

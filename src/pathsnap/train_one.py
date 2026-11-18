"""Train a single random-forest candidate scorer on route-derived surrogate labels.

The provided route file is not timestamp aligned to GPS fixes. Labels from this
script are weak geometric surrogates and must not be reported as official truth.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from .data import load_trip
from .geo import candidates, metric_transformer


def make_features(trip, cand_rows, indices):
    center = np.nanmean(trip.track[:, :2], axis=0)
    transformer = metric_transformer(*center)
    nx, ny = transformer.transform(trip.nodes[:, 0], trip.nodes[:, 1])
    gx, gy = transformer.transform(trip.track[indices, 0], trip.track[indices, 1])
    rows, labels, groups, top1 = [], [], [], []
    for pos, (idx, row) in enumerate(zip(indices, cand_rows)):
        vals = []
        for arc_id, dist in row:
            u, v = trip.arcs[arc_id]
            length = float(np.hypot(nx[v]-nx[u], ny[v]-ny[u]))
            vals.append([dist, length, abs(gx[pos]-(nx[u]+nx[v])/2), abs(gy[pos]-(ny[u]+ny[v])/2)])
        # A weak label: nearest road arc to GPS by midpoint distance.
        winner = min(range(len(row)), key=lambda j: row[j][1])
        for j, (arc_id, _) in enumerate(row):
            rows.append(vals[j]); labels.append(int(j == winner)); groups.append(trip.trip_id)
        top1.append((idx, np.asarray([x[0] for x in row]), winner))
    return np.asarray(rows), np.asarray(labels), np.asarray(groups), top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Root containing train/ and student_test/')
    ap.add_argument('--out', default='artifacts/model.joblib')
    ap.add_argument('--results', default='results/one_model.json')
    ap.add_argument('--samples-per-trip', type=int, default=250)
    ap.add_argument('--radius-m', type=float, default=120)
    ap.add_argument('--candidates', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data)
    xs, ys, groups = [], [], []
    train_info = []
    for folder in sorted((root/'train').iterdir()):
        if not folder.is_dir():
            continue
        trip = load_trip(folder)
        if trip.route is None or not len(trip.route):
            continue
        valid = np.flatnonzero(np.isfinite(trip.track).all(axis=1))
        idx = valid[np.unique(np.linspace(0, len(valid)-1, min(args.samples_per_trip, len(valid)), dtype=int))]
        cand = candidates(trip.track[idx], trip.nodes, trip.arcs, radius_m=args.radius_m, limit=args.candidates)
        x, y, g, _ = make_features(trip, cand, idx)
        xs.append(x); ys.append(y); groups.append(g)
        train_info.append({'trip_id':trip.trip_id,'observations':len(idx)})
    X, y, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)
    if len(np.unique(y)) < 2:
        raise RuntimeError('Need both positive and negative candidate examples to train')

    model = RandomForestClassifier(n_estimators=120, max_depth=16, min_samples_leaf=3,
                                   class_weight='balanced_subsample', n_jobs=-1, random_state=args.seed)
    folds = GroupKFold(n_splits=min(5, len(np.unique(g))))
    fold_scores = []
    from sklearn.base import clone
    for tr, va in folds.split(X, y, g):
        fitted = clone(model).fit(X[tr], y[tr])
        probs = fitted.predict_proba(X[va])[:, 1]
        fold_scores.append({'macro_f1':float(f1_score(y[va], probs >= .5, average='macro', zero_division=0)),
                            'log_loss':float(log_loss(y[va], np.c_[1-probs, probs], labels=[0,1]))})
    model.fit(X, y)
    top1_acc = []
    for folder in sorted((root/'student_test').iterdir()):
        if not folder.is_dir(): continue
        trip = load_trip(folder)
        if trip.route is None or not len(trip.route): continue
        valid=np.flatnonzero(np.isfinite(trip.track).all(axis=1))
        idx=valid[np.unique(np.linspace(0,len(valid)-1,min(args.samples_per_trip,len(valid)),dtype=int))]
        cand=candidates(trip.track[idx],trip.nodes,trip.arcs,radius_m=args.radius_m,limit=args.candidates)
        x,_,_,_ = make_features(trip,cand,idx)
        scores=model.predict_proba(x)[:,1]
        offset=0
        for row in cand:
            n=len(row); pred_local=int(np.argmax(scores[offset:offset+n]))
            nearest_local=int(np.argmin([d for _,d in row]))
            top1_acc.append(pred_local==nearest_local); offset+=n

    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,out)
    result={'model':'RandomForestClassifier','parameters':model.get_params(),'training_trips':train_info,
            'training_candidate_rows':int(len(y)),'surrogate_label_definition':'nearest candidate road arc midpoint to each sampled GPS point',
            'warning':'Route-derived geometric surrogate labels are not timestamp-aligned ground truth; reported scores do not establish official route accuracy.',
            'group_cv':fold_scores,'group_cv_macro_f1_mean':float(np.mean([z['macro_f1'] for z in fold_scores])),
            'student_test_agreement_with_nearest_geometry_baseline':float(np.mean(top1_acc)) if top1_acc else None,
            'student_test_note':'Agreement with the deterministic nearest-candidate geometry baseline, not ground truth.',
            'artifact':str(out),'sampled_observations_per_trip':args.samples_per_trip,'radius_m':args.radius_m}
    rp=Path(args.results);rp.parent.mkdir(parents=True,exist_ok=True);rp.write_text(json.dumps(result,indent=2))
    print(json.dumps({'artifact':str(out),'results':str(rp),'cv_macro_f1':result['group_cv_macro_f1_mean'],'student_test_baseline_agreement':result['student_test_agreement_with_nearest_geometry_baseline']},indent=2))


if __name__=='__main__': main()

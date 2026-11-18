"""Five-model group-aware experiment and Bayesian tuning; real recorded results.
Note: point labels below are nearest-reference surrogate, not independently aligned truth.

Model selection note: these five were chosen after screening TEN classifiers across
six algorithm families (linear/probabilistic, instance-based, tree, ensemble/bagging,
boosting, neural network) on this project's actual train/student_test data, ranked
primarily by grouped (per-trip) cross-validation candidate macro-F1, the same
leakage-safe criterion used for final selection below. See
results/ten_model_screen.json and NOTES_model_selection.md for the full 10-model
comparison and the reasoning behind keeping these five and dropping the other five
(logistic, naive_bayes, decision_tree, knn, extra_trees).
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss,precision_score,recall_score,accuracy_score
import optuna
import xgboost as xgb
import lightgbm as lgb
from .core import read_trip,make_candidates,features,surrogate_labels,decode

SEED=42

def models():
    return {'random_forest':RandomForestClassifier(n_estimators=85,max_depth=14,min_samples_leaf=4,n_jobs=-1,class_weight='balanced_subsample',random_state=SEED),
            'lightgbm':lgb.LGBMClassifier(n_estimators=250,max_depth=7,learning_rate=0.07,class_weight='balanced',random_state=SEED,n_jobs=-1,verbosity=-1),
            'gradient_boosting':HistGradientBoostingClassifier(max_iter=110,max_leaf_nodes=20,l2_regularization=1,random_state=SEED),
            'xgboost':xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.08,subsample=0.9,colsample_bytree=0.9,eval_metric='logloss',random_state=SEED,n_jobs=-1),
            'mlp':make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(64,32),max_iter=400,random_state=SEED,early_stopping=True))}

def dataset(root,partition,per_trip=120,k=8):
    batches=[]
    for p in sorted((Path(root)/partition).iterdir()):
        if not p.is_dir():continue
        gps,nodes,arcs,route=read_trip(p)
        good=np.flatnonzero(np.isfinite(gps).all(axis=1)&(np.abs(gps[:,1])<=90)&(np.abs(gps[:,0])<=180))
        if not len(good) or route is None:continue
        idx=good[np.unique(np.linspace(0,len(good)-1,min(per_trip,len(good)),dtype=int))]
        cand,xy,_=make_candidates(gps,nodes,arcs,k,idx)
        f=features(gps,nodes,arcs,cand,xy,idx)
        ref=surrogate_labels(gps,nodes,arcs,route,xy,idx)
        y=(cand==ref[:,None]).astype(int)
        batches.append(dict(trip=p.name,x=f.reshape(-1,f.shape[-1]),y=y.ravel(),cand=cand,ref=ref,gps=gps,nodes=nodes,arcs=arcs,idx=idx))
        print(partition,p.name,'observations',len(idx),'candidate miss',round(float(np.mean(~np.any(y,axis=1))),3),flush=True)
    return batches

def flatten(b):
    return np.concatenate([t['x'] for t in b]),np.concatenate([t['y'] for t in b]),np.concatenate([np.repeat(t['trip'],len(t['y'])) for t in b])

def metrics(model,b,decoder=False):
    t0=time.perf_counter();ys=[];ps=[];point=[];truth=[];continuity=[];miss=[]
    for z in b:
        p=model.predict_proba(z['x'])[:,1]; k=z['cand'].shape[1]
        a=p.reshape(-1,k); pred=z['cand'][np.arange(len(a)),a.argmax(axis=1)]
        if decoder:pred=decode(z['gps'],z['nodes'],z['arcs'],z['cand'],a,z['idx'])
        ys.extend(z['y']);ps.extend(p)
        point.extend(pred==z['ref']);truth.extend(pred.tolist())
        miss.extend(~np.any(z['cand']==z['ref'][:,None],axis=1))
        if len(pred)>1:
            u=z['arcs'][pred[:-1],1];v=z['arcs'][pred[1:],0]
            continuity.extend((u==v)|(pred[:-1]==pred[1:]))
    elapsed=time.perf_counter()-t0
    y=np.array(ys);p=np.clip(np.array(ps),1e-6,1-1e-6)
    return dict(candidate_macro_f1=float(f1_score(y,p>=0.5,average='macro',zero_division=0)),candidate_precision=float(precision_score(y,p>=0.5,zero_division=0)),candidate_recall=float(recall_score(y,p>=0.5,zero_division=0)),log_loss=float(log_loss(y,p,labels=[0,1])),top1_accuracy=float(np.mean(point)),candidate_miss_rate=float(np.mean(miss)),route_continuity=float(np.mean(continuity)) if continuity else None,inference_seconds=float(elapsed),n_observations=len(point))

def cross_validate(m,b):
    x,y,g=flatten(b); scores=[]
    for tr,va in GroupKFold(n_splits=3).split(x,y,g):
        from sklearn.base import clone
        fit=clone(m);fit.fit(x[tr],y[tr]);pp=fit.predict_proba(x[va])[:,1]
        scores.append(float(f1_score(y[va],pp>=0.5,average='macro',zero_division=0)))
    return scores

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--out',default='results');ap.add_argument('--artifacts',default='artifacts');ap.add_argument('--samples',type=int,default=120);ap.add_argument('--trials',type=int,default=8);args=ap.parse_args()
    Path(args.out).mkdir(exist_ok=True,parents=True);Path(args.artifacts).mkdir(exist_ok=True,parents=True)
    training=dataset(args.data,'train',args.samples);testing=dataset(args.data,'student_test',args.samples)
    x,y,g=flatten(training);results={};registry=models()
    for name,m in registry.items():
        start=time.perf_counter();cv=cross_validate(m,training);m.fit(x,y); fit_time=time.perf_counter()-start
        results[name]=dict(cv_macro_f1=cv,cv_mean=float(np.mean(cv)),train=metrics(m,training),student_test=metrics(m,testing),fit_and_cv_seconds=fit_time,parameters={k:str(v) for k,v in m.get_params(deep=False).items()})
        print('MODEL',name,results[name]['cv_mean'],results[name]['student_test']['top1_accuracy'],flush=True)
        Path(args.out,'model_comparison.json').write_text(json.dumps(results,indent=2))
    chosen=max(results,key=lambda key:results[key]['cv_mean'])
    # Bayes-search the chosen family's own hyperparameters. Each of the five
    # kept families gets its own search space; optimisation is explicitly documented.
    def propose(trial):
        if chosen=='random_forest':return RandomForestClassifier(n_estimators=trial.suggest_int('n_estimators',45,125),max_depth=trial.suggest_int('max_depth',6,22),min_samples_leaf=trial.suggest_int('min_samples_leaf',2,18),class_weight='balanced_subsample',random_state=SEED,n_jobs=-1)
        if chosen=='lightgbm':return lgb.LGBMClassifier(n_estimators=trial.suggest_int('n_estimators',100,350),max_depth=trial.suggest_int('max_depth',4,12),learning_rate=trial.suggest_float('learning_rate',0.02,0.2,log=True),class_weight='balanced',random_state=SEED,n_jobs=-1,verbosity=-1)
        if chosen=='gradient_boosting':return HistGradientBoostingClassifier(max_iter=trial.suggest_int('max_iter',50,170),max_leaf_nodes=trial.suggest_int('max_leaf_nodes',8,40),l2_regularization=trial.suggest_float('l2_regularization',0.01,10,log=True),random_state=SEED)
        if chosen=='xgboost':return xgb.XGBClassifier(n_estimators=trial.suggest_int('n_estimators',100,350),max_depth=trial.suggest_int('max_depth',3,10),learning_rate=trial.suggest_float('learning_rate',0.02,0.2,log=True),subsample=trial.suggest_float('subsample',0.6,1.0),colsample_bytree=trial.suggest_float('colsample_bytree',0.6,1.0),eval_metric='logloss',random_state=SEED,n_jobs=-1)
        return make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(trial.suggest_int('hidden1',16,96),trial.suggest_int('hidden2',8,64)),alpha=trial.suggest_float('alpha',1e-5,1e-1,log=True),max_iter=400,random_state=SEED,early_stopping=True))
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda trial:float(np.mean(cross_validate(propose(trial),training))),n_trials=args.trials)
    trial=optuna.trial.FixedTrial(study.best_params); tuned=propose(trial);t0=time.perf_counter();tuned.fit(x,y);fit_s=time.perf_counter()-t0
    tuned_metrics=dict(cv_macro_f1=study.best_value,student_test=metrics(tuned,testing),student_test_hmm=metrics(tuned,testing,decoder=True),fit_seconds=fit_s)
    baseline=registry[chosen]
    results_summary=dict(chosen=chosen,selection_basis='highest training-trip group CV candidate macro F1',baseline=results[chosen],tuned=tuned_metrics,best_params=study.best_params,optimisation_trials=[dict(number=t.number,value=t.value,params=t.params) for t in study.trials],warning='All per-point labels/metrics use nearest-reference-road SURROGATE; .route is not timestamp-aligned ground truth. Candidate miss is against this surrogate; do not describe these results as official evaluator performance.',samples_per_trip=args.samples)
    Path(args.out,'summary.json').write_text(json.dumps(results_summary,indent=2))
    joblib.dump(tuned,Path(args.artifacts,'model.joblib'))
    Path(args.artifacts,'config.json').write_text(json.dumps(dict(k=8,transition_weight=2.0,gap_seconds=12,model=chosen,surrogate_warning=results_summary['warning']),indent=2))
    print('SELECTED',chosen,'TUNED',tuned_metrics,flush=True)
if __name__=='__main__':main()

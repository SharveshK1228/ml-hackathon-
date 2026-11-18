# Model selection notes — Person 2 (ML training & evaluation)

## Process
Per the Models and Training Guidelines ("teams must train and evaluate at least
five different models... explore different algorithm families"), **ten** classifiers
spanning six algorithm families were trained and evaluated on this project's actual
data (80 train trips, 10 student-test trips, 120 samples/trip, k=8 candidates),
using the same grouped (per-trip) 3-fold cross-validation and HMM-decoded test
evaluation that `train.py` reports. Full numbers for all ten are in
`results/ten_model_screen.json`.

| Model | Family | CV macro-F1 | Student-test top-1 (HMM-decoded) |
|---|---|---|---|
| random_forest | ensemble / bagging | 0.777 | 0.843 |
| lightgbm | boosting | 0.762 | 0.842 |
| gradient_boosting (HistGB) | boosting | 0.750 | 0.849 |
| xgboost | boosting | 0.748 | 0.858 |
| extra_trees | ensemble / bagging | 0.747 | 0.761 |
| mlp (neural net) | neural network | 0.742 | 0.852 |
| decision_tree | tree | 0.732 | 0.799 |
| logistic | linear / probabilistic | 0.712 | 0.742 |
| knn | instance-based | 0.684 | 0.828 |
| naive_bayes | probabilistic | 0.641 | 0.704 |

## Which five were kept, and why

**Kept: `random_forest`, `lightgbm`, `gradient_boosting`, `xgboost`, `mlp`.**

- These are the top 5 by grouped CV macro-F1 *with one swap*: `mlp` (6th by CV,
  0.742) was kept over `extra_trees` (5th by CV, 0.747) because the CV gap between
  them is within noise (0.005) while the gap on the task's actual output metric —
  HMM-decoded top-1 accuracy — is large and consistent (mlp 0.852 vs extra_trees
  0.761). Keeping `extra_trees` and `random_forest` side by side also would have
  meant two near-identical bagging-ensemble models; keeping `mlp` instead adds a
  genuinely different family (neural network) to the comparison, which the
  guidelines ask for.
- **Dropped:** `logistic`, `naive_bayes`, `decision_tree`, `knn`, `extra_trees` —
  each was clearly outperformed on both CV and test metrics by at least one kept
  model from a related or stronger family (e.g. `decision_tree` by `random_forest`;
  `knn`/`naive_bayes`/`logistic` by every ensemble/boosting model).

**Final model selection** (the one actually tuned and shipped as `artifacts/model.joblib`)
still follows the original rule: highest grouped CV macro-F1 among the five kept
models, which is `random_forest` (0.777). Optuna then tunes `random_forest`'s own
hyperparameters (8 trials) as before. This did not change from the original 5-model
run, because `random_forest` was already the CV leader in both screens.

## What actually drives accuracy here
Across all 10 models, raw (un-decoded) top-1 accuracy clusters tightly in the
0.52-0.58 range regardless of classifier — the candidate-level classification
problem is hard and surrogate-labeled. The large accuracy gains (to 0.70-0.86)
come from the Viterbi/HMM decoder enforcing route continuity on consecutive
points, not from classifier choice. This is worth stating explicitly in the
report: model selection matters at the margins, but the decoder is doing most
of the heavy lifting, and none of the 10 models tested crosses 90% HMM-decoded
top-1 with the current candidate generation (k=8) and decoder parameters.

## Dependency note for Person 4 (requirements.txt)
This 5-model version adds `xgboost` and `lightgbm` as dependencies (on top of the
original `requirements.txt`). Flag this to whoever owns `requirements.txt`/the
integration so the full-team run doesn't fail on a missing import.

## Reminder (per TEAM_INTEGRATION_GUIDE / surrogate-label warning)
All accuracy/F1/top-1 numbers above are against nearest-reference-arc **surrogate**
labels, not independently verified ground truth. Do not present these as official
evaluator accuracy.

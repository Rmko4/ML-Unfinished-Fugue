from typing import Callable, Dict

import numpy as np
from sklearn.model_selection import ParameterSampler, ShuffleSplit
from sklearn.utils.fixes import loguniform

from data_io.model_data import (convert_raw_to_training_data,
                                create_sliding_window_view_data, load_data_raw)
from estimator import Estimator, EstimatorLinearRegressionMC, EstimatorMultiLayerPerceptronMC, EstimatorRidgeRegressionMC
from metrics import accuracy_score


if __name__ == "__main__":
    FILENAME_F = "F.txt"  # Requires tab delimited csv

# batch_size is power of two
PARAM_DIST_LINEAR = {"window_length": np.linspace(1, 400, 400, dtype=int)}

PARAM_DIST_RIDGE = {"alpha": loguniform(1e-5, 1e3),
                    "window_length": np.linspace(1, 400, 400, dtype=int)}

PARAM_DIST_MLP = {"batch_size": [2**x for x in range(4, 9)],
                  "hidden_units": np.linspace(4, 1024, 1021, dtype=int),
                  "l2": loguniform(1e-8, 1e-2),
                  "window_length": np.linspace(1, 400, 400, dtype=int)}

N_ITER = 100
N_SPLIT = 5
VALIDATION_SPLIT = 0.2

MAX_EPOCHS_MLP = 100
EARLY_STOPPING_PATIENCE_MLP = 3


def cross_validate(u: np.ndarray, y: np.ndarray,
                   model: Estimator,
                   param_dist: Dict,
                   scoring_func: Callable[[np.ndarray, np.ndarray], float],
                   validation_split=0.1,
                   n_splits=5, n_iter=10) -> Dict:

    params = ParameterSampler(param_dist, n_iter)
    scores = []

    for i, param in enumerate(params):
        print(f'{i}/{n_iter} - {param}', end='')

        window_length = param.pop("window_length")
        u_sw, y_sw = create_sliding_window_view_data(u, y, window_length)

        ss = ShuffleSplit(n_splits, test_size=validation_split)
        score = 0
        for train_i, val_i in ss.split(u_sw):
            train_u = u_sw[train_i]
            val_u = u_sw[val_i]

            if not isinstance(y, list):
                train_y = y_sw[train_i]
                val_y = y_sw[val_i]
            else:
                train_y = [x[train_i] for x in y_sw]
                val_y = [x[val_i] for x in y_sw]

            mdl = model(**param)
            mdl.fit(train_u, train_y,
                    validation_data=(val_u, val_y))

            pred_y = mdl.predict(val_u)
            score += scoring_func(val_y, pred_y)

        score /= n_splits
        print(f' - {score}')
        scores.append(score)

    order = np.argsort(scores)[::-1]
    ordered_params = np.array(list(params))[order]
    ordered_scores = np.array(scores)[order]

    scored_params = list(zip(ordered_scores, ordered_params))
    if n_iter < 5:
        print(scored_params)
    else:
        print(scored_params[:5])

    return ordered_params[0]


def cv_mlp():
    midi_raw = load_data_raw(FILENAME_F)[:-16, :]
    # Deferring windowing
    data, ove, ive = convert_raw_to_training_data(
        midi_raw, flatten_output=False, defer_windowing=True)

    fn_acc_score = accuracy_score(ove)
    mlp = EstimatorMultiLayerPerceptronMC(
        ive, ove, epochs=MAX_EPOCHS_MLP,
        early_stopping_patience=EARLY_STOPPING_PATIENCE_MLP)

    cross_validate(*data, mlp, param_dist=PARAM_DIST_MLP, scoring_func=fn_acc_score,
                   validation_split=VALIDATION_SPLIT, n_splits=N_SPLIT, n_iter=N_ITER)


def cv_lr():
    midi_raw = load_data_raw(FILENAME_F)[:-16, :]
    data, ove, ive = convert_raw_to_training_data(
        midi_raw, flatten_output=True, defer_windowing=True)

    fn_acc_score = accuracy_score(ove)
    linear_regressor = EstimatorLinearRegressionMC(ive, ove)

    cross_validate(*data, linear_regressor, param_dist=PARAM_DIST_LINEAR,
                   scoring_func=fn_acc_score, validation_split=VALIDATION_SPLIT,
                   n_splits=N_SPLIT, n_iter=N_ITER)


def cv_ridge():
    midi_raw = load_data_raw(FILENAME_F)[:-16, :]
    data, ove, ive = convert_raw_to_training_data(
        midi_raw, flatten_output=True, defer_windowing=True)

    fn_acc_score = accuracy_score(ove)
    ridge_regressor = EstimatorRidgeRegressionMC(ive, ove)

    cross_validate(*data, ridge_regressor, param_dist=PARAM_DIST_RIDGE,
                   scoring_func=fn_acc_score, validation_split=VALIDATION_SPLIT,
                   n_splits=N_SPLIT, n_iter=N_ITER)
                   
def cv_all_models():
    # cv_lr()
    cv_ridge()
    # cv_mlp()


if __name__ == "__main__":
    cv_all_models()

from typing import Dict
from sklearn.model_selection import ParameterSampler, ShuffleSplit
from sklearn.utils.fixes import loguniform
import numpy as np

from data_io.model_data import convert_raw_to_training_data, create_sliding_window_view_data, load_data_raw
from data_io.vector_encoders import InputVectorEncoderMC, OutputVectorEncoderMC
from mlp import MultiLayerPerceptronMC

if __name__ == "__main__":
    FILENAME_F = "F.txt"  # Requires tab delimited csv

# batch_size is power of two
PARAM_DIST_LINEAR = {"window_length": np.linspace(1, 200, 200, dtype=int)}

PARAM_DIST_RIDGE = {"alpha": loguniform(1e-1, 1e3),
                    "window_length": np.linspace(1, 200, 200, dtype=int)}

PARAM_DIST_MLP = {"batch_size": [2**x for x in range(4, 9)],
                  "hidden_units": np.linspace(4, 256, 253, dtype=int),
                  "window_length": np.linspace(1, 200, 200, dtype=int)}

N_ITER = 100
N_SPLIT = 5
VALIDATION_SPLIT = 0.1

def accuracy_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y == y_pred)

def cross_validate_mlp(u, y, param_dist: Dict, ive: InputVectorEncoderMC,
                       ove: OutputVectorEncoderMC, validation_split=0.1,
                       n_splits=5, n_iter=10):
    params = ParameterSampler(param_dist, n_iter)
    scores = []

    for param in params:
        print("\n" + str(param))

        window_length = param.pop("window_length")
        u_sw, y_sw = create_sliding_window_view_data(u, y, window_length)

        ss = ShuffleSplit(n_splits, test_size=validation_split)
        score = 0
        for train_i, val_i in ss.split(u_sw):
            train_u = u_sw[train_i]
            train_y = [x[train_i] for x in y_sw]

            val_u = u_sw[val_i]
            y_val = [x[val_i] for x in y_sw]

            mlp = MultiLayerPerceptronMC(**param, ive=ive, ove=ove)
            mlp.compile()
            import tensorflow.keras as keras
            callbacks = [keras.callbacks.EarlyStopping(patience=3)]
            mlp.fit(train_u, train_y,
                    validation_data=(val_u, y_val), callbacks=callbacks, epochs=10)

            y_pred = mlp.predict(val_u)
            pred_midi = ove.inv_transform_maximum_likelihood(y_pred)
            real_midi = ove.inv_transform_maximum_likelihood(y_val)
            score += accuracy_score(real_midi, pred_midi)

        score /= n_splits 
        scores.append(score)

    order = np.argsort(scores)[::-1]
    best_params = np.array(list(params))[order]
    scored_params = list(zip(scores, best_params))
    if n_iter < 5:
        print(scored_params)
    else:
        print(scored_params[:5])

    return best_params[0]



def main():
    midi_raw = load_data_raw(FILENAME_F)[:-16, :]
    # Deferring windowing
    data, ove, ive = convert_raw_to_training_data(
        midi_raw, flatten_output=False, defer_windowing=True)

    cross_validate_mlp(*data, PARAM_DIST_MLP, ive, ove,
                       VALIDATION_SPLIT, N_SPLIT, N_ITER)


if __name__ == "__main__":
    main()

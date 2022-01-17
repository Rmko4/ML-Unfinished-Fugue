import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data_io.midi_duration import (MIDI_COMPACT_SC,
                                   midi_tones_file_to_midi_compact)
from data_io.midi_file import MODULATION, TEMPO, midi_compact_to_midi_file, midi_tones_to_midi_file
from data_io.model_data import convert_midi_compact_sc_to_training_data, convert_raw_to_training_data, load_data_raw
from data_io.vector_encoders import (InputVectorEncoderMC,
                                     InputVectorEncoderSC,
                                     OutputVectorEncoderMC,
                                     OutputVectorEncoderSC)

if __name__ == "__main__":
    FILENAME_F = "F.txt"  # Requires tab delimited csv
    OMIT_REST = True
    CHANNEL = 0
    WINDOW_LENGTH_SC = 10
    WINDOW_LENGTH_MC = 48
    N_NEW_SYMBOLS = 160  # Roughly 20 seconds considering bpm 120 and 4 symbols per beat


class LinearRegressionSC(LinearRegression):
    def __init__(self, ove: OutputVectorEncoderSC, ive: InputVectorEncoderSC,
                 fit_intercept=True, normalize=False,
                 copy_X=True, n_jobs=None, positive=False) -> None:
        super().__init__(fit_intercept=fit_intercept, normalize=normalize,
                         copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        self.ove = ove
        self.ive = ive

    def predict_sequence(self, X: np.ndarray, duration=64) -> MIDI_COMPACT_SC:
        mc_pred_seq = []
        u = self.ive.transform(X)
        time = 0
        while time < duration:
            # Create single window view
            u_sw = sliding_window_view(u, len(X), axis=0)
            u_sw = u_sw.reshape(u_sw.shape[0], -1)

            # Raw output
            y_pred = self.predict(u_sw)
            # Select note and duration on maximum likelihood
            out = self.ove.inv_transform_maximum_likelihood(y_pred)
            mc_pred = (int(out[0][0]), int(out[0][1]))

            # Add the prediction to the sequence
            mc_pred_seq.append(mc_pred)

            # Update window with new prediction
            new_u = self.ive.transform(out)
            u = np.concatenate((u[1:], new_u))

            # Update total duration with new note
            time += mc_pred[1]

        return mc_pred_seq


class LinearRegressionMC(LinearRegression):
    def __init__(self, ove: OutputVectorEncoderMC, ive: InputVectorEncoderMC,
                 fit_intercept=True, normalize=False,
                 copy_X=True, n_jobs=None, positive=False) -> None:
        super().__init__(fit_intercept=fit_intercept, normalize=normalize,
                         copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        self.ove = ove
        self.ive = ive

    def predict_sequence(self, X: np.ndarray, duration=64) -> np.ndarray:
        mc_pred_seq = []
        u = self.ive.transform(X)

        for _ in range(duration):
            # Create single window view
            u_sw = sliding_window_view(u, len(X), axis=0)
            u_sw = u_sw.reshape(u_sw.shape[0], -1)

            # Raw output
            y_pred = self.predict(u_sw)
            # Select note and duration on maximum likelihood
            out = self.ove.inv_transform_maximum_likelihood(y_pred)

            # Add the prediction to the sequence
            mc_pred_seq.append(*out)

            # Update window with new prediction
            new_u = self.ive.transform(out)
            u = np.concatenate((u[1:], new_u))

        return np.array(mc_pred_seq)


def apply_linear_regression_SC():
    midi_compact = midi_tones_file_to_midi_compact(
        FILENAME_F, omit_rest=OMIT_REST)
    # Only use channel CHANNEL; Last measure is omitted as this one is incomplete
    in_d = midi_compact[CHANNEL][:-16]
    data, _, ove, ive = convert_midi_compact_sc_to_training_data(
        in_d, validation_split=0.1, window_length=WINDOW_LENGTH_SC)

    u, y = data

    train_u, val_u, train_y, val_y = train_test_split(u, y, test_size=0.2)

    lreg = LinearRegressionSC(ove, ive)

    lreg.fit(train_u, train_y)
    print(lreg.score(val_u, val_y))

    # pred_val_y = lreg.predict(val_u)
    #
    # pd.DataFrame(pred_val_y).to_csv(
    #     "postprocessing/probabilities.csv", header=None, index=None)
    # read = pd.read_csv("postprocessing/probabilities.csv",
    #                    header=None).to_numpy()

    predicted_sequence = lreg.predict_sequence(
        in_d[-WINDOW_LENGTH_SC:], duration=N_NEW_SYMBOLS)

    full_sequence = in_d.copy()
    full_sequence.extend(predicted_sequence)
    midi_compact_fs = [full_sequence]

    midi_compact_to_midi_file(
        midi_compact_fs, "pred.mid", tempo=TEMPO, modulation=MODULATION)

    # Might one to apply something like softmax.
    # However, it is not trained on this softmax so it is not really going to represent the probability.


def apply_linear_regression_MC():
    midi_raw = load_data_raw(FILENAME_F)[:-16, :]
    data, ove, ive = convert_raw_to_training_data(
        midi_raw, window_length=WINDOW_LENGTH_MC, flatten_output=True)
    # Training data must be flattened for linear regressor

    u, y = data

    train_u, val_u, train_y, val_y = train_test_split(u, y, test_size=0.2)

    lreg = LinearRegressionMC(ove, ive)

    lreg.fit(train_u, train_y)
    print(lreg.score(val_u, val_y))

    # pred_val_y = lreg.predict(val_u)
    #
    # pd.DataFrame(pred_val_y).to_csv(
    #     "postprocessing/probabilities.csv", header=None, index=None)
    # read = pd.read_csv("postprocessing/probabilities.csv",
    #                    header=None).to_numpy()

    predicted_sequence = lreg.predict_sequence(
        midi_raw[-WINDOW_LENGTH_MC:], duration=N_NEW_SYMBOLS)

    full_sequence = np.concatenate((midi_raw, predicted_sequence), axis=0)
    midi_tones_to_midi_file(predicted_sequence, "pred_2.mid",
                            tempo=TEMPO, modulation=MODULATION)

    # Might one to apply something like softmax.
    # However, it is not trained on this softmax so it is not really going to represent the probability.


if __name__ == "__main__":
    apply_linear_regression_SC()
    apply_linear_regression_MC()

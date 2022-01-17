from sklearn.linear_model import LinearRegression
from data_io.midi_duration import MIDI_COMPACT_SC, midi_tones_file_to_midi_compact
from data_io.midi_file import MODULATION, TEMPO, midi_compact_to_midi_file
from data_io.model_data import convert_midi_compact_sc_to_training_data
from scipy.special import softmax
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from data_io.vector_encoders import InputVectorEncoderSC, TeacherVectorEncoderSC


if __name__ == "__main__":
    FILENAME_F = "F.txt"  # Requires tab delimited csv
    OMIT_REST = True
    CHANNEL = 0
    WINDOW_LENGTH = 10
    N_NEW_SYMBOLS = 160  # Roughly 20 seconds considering bpm 120 and 4 symbols per beat


class LinearRegressionSC(LinearRegression):
    def __init__(self, tve: TeacherVectorEncoderSC, ive: InputVectorEncoderSC,
                 fit_intercept=True, normalize=False,
                 copy_X=True, n_jobs=None, positive=False) -> None:
        super().__init__(fit_intercept=fit_intercept, normalize=normalize,
                         copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        self.tve = tve
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
            out = self.tve.inv_transform_maximum_likelihood(y_pred)
            mc_pred = (int(out[0][0]), int(out[0][1]))

            # Add the prediction to the sequence
            mc_pred_seq.append(mc_pred)

            # Update window with new prediction
            new_u = self.ive.transform(out)
            u = np.concatenate((u[1:], new_u))

            # Update total duration with new note
            time += mc_pred[1]

        return mc_pred_seq


def apply_linear_regression():
    midi_compact = midi_tones_file_to_midi_compact(
        FILENAME_F, omit_rest=OMIT_REST)
    # Only use channel CHANNEL; Last measure is omitted as this one is incomplete
    in_d = midi_compact[CHANNEL][:-16]
    data, _, tve, ive = convert_midi_compact_sc_to_training_data(
        in_d, validation_split=0.1, window_length=WINDOW_LENGTH)

    u, y = data

    train_u, val_u, train_y, val_y = train_test_split(u, y, test_size=0.2)

    lreg = LinearRegressionSC(tve, ive)

    lreg.fit(train_u, train_y)
    print(lreg.score(val_u, val_y))

    pred_val_y = lreg.predict(val_u)

    pd.DataFrame(pred_val_y).to_csv("postprocessing/probabilities.csv", header=None, index=None)
    read = pd.read_csv("postprocessing/probabilities.csv", header=None).to_numpy()

    predicted_sequence = lreg.predict_sequence(in_d[-10:], duration=N_NEW_SYMBOLS)

    full_sequence = in_d.copy()
    full_sequence.extend(predicted_sequence)
    midi_compact_fs = [full_sequence]

    midi_compact_to_midi_file(midi_compact_fs, "pred.mid", tempo=TEMPO, modulation=MODULATION)

    # Might one to apply something like softmax.
    # However, it is not trained on this softmax so it is not really going to represent the probability.



if __name__ == "__main__":
    apply_linear_regression()

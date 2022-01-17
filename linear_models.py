from sklearn.linear_model import LinearRegression
from data_io.midi_duration import MIDI_COMPACT_SC, midi_tones_file_to_midi_compact
from data_io.midi_file import MODULATION, TEMPO, midi_compact_to_midi_file
from data_io.model_data import convert_to_training_data
from scipy.special import softmax
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np

from data_io.vector_encoders import InputVectorEncoder, TeacherVectorEncoder


if __name__ == "__main__":
    FILENAME_F = "F.txt"  # Requires tab delimited csv
    OMIT_REST = True
    CHANNEL = 0
    WINDOW_LENGTH = 10
    N_NEW_SYMBOLS = 160  # Roughly 20 seconds considering bpm 120 and 4 symbols per beat


class LinearRegressionSC(LinearRegression):
    def __init__(self, tve: TeacherVectorEncoder, ive: InputVectorEncoder,
                 fit_intercept=True, normalize=False,
                 copy_X=True, n_jobs=None, positive=False) -> None:
        super().__init__(fit_intercept=fit_intercept, normalize=normalize,
                         copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        self.tve = tve
        self.ive = ive

    def predict_sequence(self, X: np.ndarray, duration=64) -> MIDI_COMPACT_SC:
        mc_pred_seq = []
        u = self.ive.transform(X)
        duration = 0
        while duration < N_NEW_SYMBOLS:
            # Create single window view
            u_sw = sliding_window_view(u, WINDOW_LENGTH, axis=0)
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
            duration += mc_pred[1]

        return mc_pred_seq


def apply_linear_regression():
    midi_compact = midi_tones_file_to_midi_compact(
        FILENAME_F, omit_rest=OMIT_REST)
    # Only use channel CHANNEL; Last measure is omitted as this one is incomplete
    in_d = midi_compact[CHANNEL][:-16]
    train, val, tve, ive = convert_to_training_data(
        in_d, validation_split=0, window_length=WINDOW_LENGTH)

    u, y = train

    lreg = LinearRegressionSC(tve, ive)

    lreg.fit(u, y)
    print(lreg.score(u, y))

    predicted_sequence = lreg.predict_sequence(in_d[-10:])

    full_sequence = in_d.copy()
    full_sequence.extend(predicted_sequence)
    midi_compact_fs = [full_sequence]

    midi_compact_to_midi_file(
        midi_compact_fs, "pred.mid", tempo=TEMPO, modulation=MODULATION)

    # Might one to apply something like softmax.
    # However, it is not trained on this softmax so it is not really going to represent the probability.


if __name__ == "__main__":
    apply_linear_regression()

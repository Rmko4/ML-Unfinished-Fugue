from sklearn.linear_model import Ridge
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np

from data_io.vector_encoders import InputVectorEncoder, TeacherVectorEncoder


class RidgeRegressionSC(Ridge):

    def __init__(self, tve: TeacherVectorEncoder, ive: InputVectorEncoder, alpha=1.0,
                 fit_intercept=True, copy_X=True, n_jobs=None,
                 positive=False) -> None:
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, copy_X=copy_X,
                         n_jobs=n_jobs, positive=positive)
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

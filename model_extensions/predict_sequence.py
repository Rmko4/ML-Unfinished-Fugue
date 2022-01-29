from typing import Callable, List, Union
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from data_io.vector_encoders import OVE_OUT, InputVectorEncoderMC, OutputVectorEncoderMC


class SequencePredictorMixin():
    ''' 
    Mixin classbu
    Requires (predictor) class that implements predict function.
    '''

    def __init__(self, ive: InputVectorEncoderMC,
                 ove: OutputVectorEncoderMC, **kwargs):
        super().__init__(**kwargs)
        self.ive = ive
        self.ove = ove

    def predict_sequence(self, X: np.ndarray, steps=64, inv_transform_fn: Callable[[OVE_OUT, np.ndarray], np.ndarray] = None) -> np.ndarray:
        mc_pred_seq = []
        u = self.ive.transform(X)

        for _ in range(steps):
            # Create single window view
            u_sw = sliding_window_view(u, len(X), axis=0)
            u_sw = u_sw.reshape(u_sw.shape[0], -1)

            # Raw output
            y_pred = self.predict(u_sw)

            # Select note and duration on maximum probability
            if inv_transform_fn:
                out = inv_transform_fn(y_pred, self.ove)
            else:
                out = self.ove.inv_transform_max_probability(y_pred)[0]

            # Add the prediction to the sequence
            mc_pred_seq.append(out)

            # Update window with new prediction
            new_u = self.ive.transform([out])
            u = np.concatenate((u[1:], new_u))

        return np.array(mc_pred_seq)

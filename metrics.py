from typing import Callable

import numpy as np

from data_io.vector_encoders import OutputVectorEncoderMC


def accuracy_score(ove: OutputVectorEncoderMC,) -> Callable[[np.ndarray, np.ndarray], float]:
    def _acc_score(y, y_pred):
        real_midi = ove.inv_transform_max_probability(y)
        pred_midi = ove.inv_transform_max_probability(y_pred)
        return np.mean(real_midi == pred_midi)
    return _acc_score

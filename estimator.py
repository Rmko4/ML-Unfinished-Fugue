from typing import Any, Union

from tensorflow.keras.callbacks import EarlyStopping

from data_io.vector_encoders import InputVectorEncoderMC, OutputVectorEncoderMC
from linear_models import LinearRegressionMC, RidgeRegressionMC
from mlp import MultiLayerPerceptronMC


class Estimator:
    '''
        Can be implemented and used for typing.
    '''

    def __call__(self, *args: Any, **kwds: Any) -> 'Estimator':
        raise NotImplementedError

    def fit(self, X, y, validation_data):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class EstimatorMultiLayerPerceptronMC():
    def __init__(self, ive: InputVectorEncoderMC, ove: OutputVectorEncoderMC,
                 epochs=100, early_stopping_patience: Union[int, None] = 1):
        self.ive = ive
        self.ove = ove
        self.epochs = epochs
        if early_stopping_patience:
            self.callbacks = [EarlyStopping(patience=early_stopping_patience)]

    def __call__(self, **kwargs):
        self.mlp = MultiLayerPerceptronMC(self.ive, self.ove, **kwargs)
        self.mlp.compile()
        return self

    def fit(self, X, y, validation_data, **kwargs):
        self.mlp.fit(X, y, validation_data=validation_data,
                     callbacks=self.callbacks, epochs=self.epochs, verbose=0
                     )

    def predict(self, X):
        return self.mlp.predict(X)


class EstimatorRidgeRegressionMC(RidgeRegressionMC):
    def __init__(self, ive: InputVectorEncoderMC, ove: OutputVectorEncoderMC):
        self.ive = ive
        self.ove = ove

    def __call__(self, **kwargs):
        super().__init__(self.ive, self.ove, **kwargs)
        return self

    def fit(self, X, y, **kwargs):
        super().fit(X, y)


class EstimatorLinearRegressionMC(LinearRegressionMC):
    def __init__(self, ive: InputVectorEncoderMC, ove: OutputVectorEncoderMC):
        self.ive = ive
        self.ove = ove

    def __call__(self, **kwargs):
        super().__init__(self.ive, self.ove, **kwargs)
        return self

    def fit(self, X, y, **kwargs):
        super().fit(X, y)

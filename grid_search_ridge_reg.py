from curses import window
from data_io.model_data import convert_to_training_data
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from data_io.midi_duration import midi_tones_file_to_midi_compact
from data_io.model_data import convert_to_training_data
import pandas as pd
import numpy as np

K_FOLDS = 5
FILENAME_F = "F.txt"  # Requires tab delimited csv
OMIT_REST = True
CHANNEL = 0


def grid_search(data, k_folds, window_lens, alphas):
    zeros = np.zeros((len(alphas), len(window_lens)))
    scores_df = pd.DataFrame(zeros, columns=window_lens, index=alphas)
    test = scores_df.loc['0.1', '5']

    # Setup cross validation splitter with seed
    cv_splitter = KFold(k_folds, shuffle=True, random_state=3)
    for window_len in window_lens:
        # Get x and y data with appropriate window length
        train, _, _, _ = convert_to_training_data(data, window_length=window_len)
        u, y = train
        for alpha in alphas:
            model = Ridge(alpha, fit_intercept=True, copy_X=True)
            scores = cross_validate(model, u, y, cv=cv_splitter)
            print(scores)


if __name__ == '__main__':
    window_lens = [5]
    alphas = [0.1, 0.5]

    midi_compact = midi_tones_file_to_midi_compact(FILENAME_F, omit_rest=OMIT_REST)
    # Only use channel CHANNEL; Last measure is omitted as this one is incomplete
    data = midi_compact[CHANNEL][:-16]
    grid_search(data, K_FOLDS, window_lens, alphas)

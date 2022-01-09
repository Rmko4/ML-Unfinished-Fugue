from midi_to_5d_vec import midi_to_5d_vec
import pandas as pd
import numpy as np
from pandas import DataFrame

IGNORE_PAUSES = True
IMPORT_FILE = 'F.txt'


def get_6d_vec(midi_value, duration) -> list:
    """
    Returns the midi value as a 5d vec representation plus
    an extra entry for the duration
    Duration is encoded as log2(duration)
    [pitch, chroma_x, chroma_y, circle5_x, circle5_y, log2(duration)]
    """
    vec = midi_to_5d_vec(midi_value)
    vec.append(np.log2(duration))
    return vec


def gen_6d_train_vecs(midi_values) -> list:
    """
    Converts a sequence of midi values to a 6d representation
    """
    # NOTE: loop is based on the assumption last midi entry is not different
    # than second to last entry
    train_vectors = []
    duration = 1
    for i in range(midi_values.size - 1):
        if midi_values[i] == midi_values[i + 1]:
            duration += 1
        else:
            # If one wants to ignore pauses in the voices
            # @TODO how do we represent pauses?
            if midi_values[i] == 0 and IGNORE_PAUSES:
                pass
            else:
                vec = get_6d_vec(midi_values[i], duration)
                train_vectors.append(vec)
            # Reset duration to 1
            duration = 1
    return train_vectors


def main():
    voices_df: DataFrame = pd.read_csv(IMPORT_FILE, sep="\t", header=None)
    voices = voices_df.to_numpy()
    voices = voices.T

    # Only select first voice (for now) and discard last 16 midi notes
    midi_values = voices[0, :-16]

    x = gen_6d_train_vecs(midi_values)
    x = np.array(x)

    # Save as np array and as txt (for vizability)
    np.save('data/x', x)
    np.savetxt('data/x.txt', x, fmt='%s')


if __name__ == '__main__':
    print('Generating...')
    main()
    print('Done!')

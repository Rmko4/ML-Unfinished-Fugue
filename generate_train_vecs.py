from midi_to_5d_vec import midi_to_5d_vec
import pandas as pd
import numpy as np

ENCODE_PAUSES = True
BIAS = True

IMPORT_FILE = 'F.txt'
OUTPUT_FILE = 'x'

# Vector representation:
# [music playing (0,1), pitch, chroma_x, chroma_y, circle5_x, circle5_y, log2(duration)] + (bias)


def get_pause_vector(duration) -> list:
    """
    Function that returns a vector of with 6 zero entries plus the log2(duration) and a possible bias
    """
    vec = [0] * 6
    vec.append(np.log2(duration))
    if BIAS:
        vec.append(1)
    return vec


def get_vector(midi_value, duration) -> list:
    """
    Returns the midi value as a 5d vec representation plus
    an extra entry for the duration
    Duration is encoded as log2(duration)
    [music playing, pitch, chroma_x, chroma_y, circle5_x, circle5_y, log2(duration)] + bias
    """
    vec = []
    vec.append(1)  # Music playing
    vec.extend(midi_to_5d_vec(midi_value))
    vec.append(np.log2(duration))
    if BIAS:
        vec.append(1)
    return vec


def generate_training_vectors(midi_values) -> list:
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
            if midi_values[i] == 0:
                if ENCODE_PAUSES:
                    vec = get_pause_vector(7)
                    train_vectors.append(vec)
            else:
                vec = get_vector(midi_values[i], duration)
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

    x = generate_training_vectors(midi_values)
    x = np.array(x)

    # Save as np array and as txt (for vizability)
    np.save(f'data/{OUTPUT_FILE}', x)
    np.savetxt(f'data/{OUTPUT_FILE}.txt', x, fmt='%s')


if __name__ == '__main__':
    print('Generating...')
    main()
    print('Done!')

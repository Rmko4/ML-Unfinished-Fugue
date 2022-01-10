import numpy as np
import pandas as pd

from midi_to_train_vecs import get_input_vec, get_pause_vec, get_teacher_vec

ENCODE_PAUSES = False
BIAS = False

IMPORT_FILE = 'F.txt'
INPUT_VECS_FILE = 'x'
TEACHER_VECS_FILE = 'y'


def get_train_vecs(midi_value, duration) -> list:
    """
    Function that calls the convertion functions to obtain the input and teacher vectors
    """
    x, y = None, None
    if midi_value == 0:
        if ENCODE_PAUSES:
            x = get_pause_vec(duration, bias=BIAS)
            y = get_teacher_vec(midi_value, duration)
        else:
            return x, y

    x = get_input_vec(midi_value, duration, bias=BIAS)
    y = get_teacher_vec(midi_value, duration)
    return x, y


def generate_train_vecs(midi_values) -> list:
    """
    Function to loop over a sequence of midi values, and converts 
    them to training vectors (input and teacher vectors)
    """
    # NOTE: loop is based on the assumption last midi entry is not different
    # than second to last entry
    input_vecs = []
    teacher_vecs = []
    duration = 1
    for i in range(midi_values.size - 1):
        if midi_values[i] == midi_values[i + 1]:
            duration += 1
        else:
            x, y = get_train_vecs(midi_values[i], duration)
            if x and y is not None:
                input_vecs.append(x)
                teacher_vecs.append(y)
            duration = 1
    return input_vecs, teacher_vecs


def main():
    voices_df: DataFrame = pd.read_csv(IMPORT_FILE, sep="\t", header=None)
    voices = voices_df.to_numpy()
    voices = voices.T

    # Only select first voice (for now) and discard last 16 midi notes
    midi_values = voices[0, :-16]

    x, y = generate_train_vecs(midi_values)
    x = np.array(x)
    y = np.array(y)

    # Save as np array and as txt (for vizability)
    np.save(f'data/{INPUT_VECS_FILE}', x)
    np.savetxt(f'data/{INPUT_VECS_FILE}.txt', x, fmt='%s')

    np.save(f'data/{TEACHER_VECS_FILE}', y)
    np.savetxt(f'data/{TEACHER_VECS_FILE}.txt', y, fmt='%s')


if __name__ == '__main__':
    print('Generating...')
    main()
    print('Done!')

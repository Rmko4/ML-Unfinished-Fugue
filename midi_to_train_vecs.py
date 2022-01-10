from os import name
import numpy as np
from midi_to_5d_vec import midi_to_5d_vec, MIN_NOTE, MAX_NOTE

NOTE_RANGE = MAX_NOTE - MIN_NOTE + 1  # As it is up to AND INCLUDING
DURATION = 24

# So the length of the teacher vector will be:
# 1 + NOTE_RANGE + DURATION
# Where 1 corresponds to 0/1 for music playing
# NOTE_RANGE is a one hot encoding for the corresponding note
# DURATION is a one hot encoding for the corresponding duration of the note

# input vector representation:
# [music playing (0,1), pitch, chroma_x, chroma_y, circle5_x, circle5_y, log2(duration)] + (bias)


def get_zero_vector(n):
    return [0] * n


def get_pause_vec(duration, bias=True) -> list:
    """
    Function that returns a vector with 6 zero entries plus the log2(duration) and a possible bias
    """
    vec = get_zero_vector(6)
    vec.append(np.log2(duration))
    if bias:
        vec.append(1)
    return vec


def get_input_vec(midi_value, duration, bias=True) -> list:
    """
    Returns the input vector in the form:
    [music playing, pitch, chroma_x, chroma_y, circle5_x, circle5_y, log2(duration)] + bias
    """
    vec = []
    vec.append(1)  # Music playing
    vec.extend(midi_to_5d_vec(midi_value))
    vec.append(np.log2(duration))
    if bias:
        vec.append(1)
    return vec


def get_teacher_vec(midi_value, duration) -> list:
    """
    Returns the midi value and duration as a teacher vector of size: 1 + NOTE_RANGE + DURATION
    Where 1 corresponds to 0/1 for music playing
    NOTE_RANGE is a one hot encoding for the corresponding note
    DURATION is a one hot encoding for the corresponding duration of the note
    """
    y = get_zero_vector(1 + NOTE_RANGE + DURATION)
    # If the midi value is not a pause, fill up the vector with the note
    if midi_value != 0:
        # Set playing music to 1
        y[0] = 1
        # Get the index of the midi value relative to the teacher vector
        note_idx = 1 + midi_value - MIN_NOTE
        y[note_idx] = 1
    # Get the index of the duration relative to the teacher vector
    # NOTE no plus
    duration_idx = NOTE_RANGE + duration
    y[duration_idx] = 1
    return y


if __name__ == '__main__':
    y = get_teacher_vec(0, 23)
    print(y)

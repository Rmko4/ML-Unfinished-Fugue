
from typing import List, Union
import numpy as np
from music_utils.music_circle import get_chroma_coords, get_circle5_coords
from midi_duration import MIDI_COMPACT_SC


class TeacherVectorEncoder:
    def __init__(self, midi_raw: np.ndarray):
        self.n_channels = midi_raw.shape[1]

        midi_notes_masked = np.ma.masked_equal(midi_raw, 0)
        self.note_min = midi_notes_masked.min(axis=0)
        self.note_max = midi_notes_masked.max(axis=0)

        # Notes in [self.note_min, self.note_max]
        self.note_range = self.note_max - self.note_min + 1
        self.encoder_len = self.note_range + 1

        self._playing_idx = 0
        self._notes_idx = 1

    def get_teacher_vector(self, midi_notes: np.ndarray,
                           flatten=True) -> Union[List[np.ndarray], np.ndarray]:
        teacher_vector = []

        for channel in range(self.n_channels):
            y = np.zeros((self.encoder_len[channel]))
            midi_note = midi_notes[channel]

            if midi_note == 0:
                y[self._playing_idx] = 1
            else:
                note_idx = self._notes_idx + midi_note - self.note_min[channel]
                y[note_idx] = 1
            teacher_vector.append(y)

        if flatten:
            return np.concatenate(teacher_vector)
        else:
            return teacher_vector

    def transform(self, freq_raw: np.ndarray, flatten=True) -> np.ndarray:
        y = []
        for x in freq_raw:
            teacher_vec = self.get_teacher_vector(x, flatten)
            y.append(teacher_vec)

        if flatten:
            return np.array(y)
        else:
            y_conv = []
            for channel in range(self.n_channels):
                sc_encodings = [x[channel] for x in y]
                y_conv.append(np.array(sc_encodings))

            return y_conv

    # def get_playing_prob(self, X: np.ndarray) -> np.ndarray:
    #     return X[:, self._playing_idx]

    # def get_notes_prob(self, X: np.ndarray) -> np.ndarray:
    #     return X[:, self._notes_idx:self._durations_idx]

    # def get_durations_prob(self, X: np.ndarray) -> np.ndarray:
    #     return X[:, self._durations_idx:]

    # def inv_transform_maximum_likelihood(self, X: np.ndarray) -> np.ndarray:
    #     # Ignores playing probability (for now)
    #     notes_prob = self.get_notes_prob(X)
    #     durations_prob = self.get_durations_prob(X)
    #     notes = notes_prob.argmax(axis=1) + self.note_min
    #     durations = durations_prob.argmax(axis=1) + self.duration_min
    #     mc_ndarr = np.column_stack((notes, durations))
    #     # For MIDI_COMPACT_SC: list(map(tuple, mc_ndarr))
    #     return mc_ndarr


class InputVectorEncoder():
    def __init__(self, midi_raw: np.ndarray):   
        self.n_channels = midi_raw.shape[1]

        midi_notes_masked = np.ma.masked_equal(midi_raw, 0)
        self.note_min = midi_notes_masked.min(axis=0)
        self.note_max = midi_notes_masked.max(axis=0)

        self.encoder_len = self.n_channels * 5

        min_p = 2 * np.log2(2**((self.note_min - 69)/12) * 440)
        max_p = 2 * np.log2(2**((self.note_max - 69)/12) * 440)

        self._pitch_offset = (max_p - min_p)/2 - max_p

    def get_input_vector(self, midi_notes: int, flatten=True) -> np.ndarray:
        """
        Takes in a MIDI value and duration and returns the
        [pitch, chroma_x, chroma_y, circle5_x, circle5_y, log_duration] as a ndarray
        """
        input_vector = []
        # TODO: What if zero midi
        for channel in range(self.n_channels):
            # Convert midi_note to an int [0,11]
            # 55 is MIDI value of note G3
            midi_note = midi_notes[channel]

            note = (midi_note - 55) % 12

            chroma_x, chroma_y = get_chroma_coords(note)
            circle5_x, circle5_y = get_circle5_coords(note)

            # d is the distance (in semitones) of midi_note from A4 (69 in MIDI),
            # whose frequency is 440 Hz. fx is the frequency of the note
            d = midi_note - 69
            fx = 2**(d/12) * 440

            # the representation of pitch is scaled in such a way that a pitch
            # distance of 1 octave in the first dimension, is equal to the distance of
            # notes on the opposite sides on the chroma circle or the circle of fifths
            # In addition the pitch is offset, such that zero alligns with the pitch inbetween min_p and max_p
            pitch = 2 * np.log2(fx) + self._pitch_offset[channel]

            input_vector.append(np.array([pitch, chroma_x, chroma_y, circle5_x, circle5_y]))

        if flatten:
            return np.concatenate(input_vector)
        else:
            return input_vector

    def transform(self, freq_raw: np.ndarray, flatten=True) -> np.ndarray:
        y = []
        for x in freq_raw:
            teacher_vec = self.get_input_vector(x, flatten)
            y.append(teacher_vec)

        if flatten:
            return np.array(y)
        else:
            y_conv = []
            for channel in range(self.n_channels):
                sc_encodings = [x[channel] for x in y]
                y_conv.append(np.array(sc_encodings))

            return y_conv


class TeacherVectorEncoderSC:
    def __init__(self, midi_compact: MIDI_COMPACT_SC) -> None:
        mc_ndarr = np.array(midi_compact)

        self.duration_min = int(mc_ndarr[:, 1].min())
        self.duration_max = int(mc_ndarr[:, 1].max())

        # Durations in [self.duration_min, self.duration_max]
        self.duration_range = self.duration_max - self.duration_min + 1

        midi_notes_masked = np.ma.masked_equal(mc_ndarr[:, 0], 0)
        self.note_min = int(midi_notes_masked.min())
        self.note_max = int(midi_notes_masked.max())

        # Notes in [self.note_min, self.note_max]
        self.note_range = self.note_max - self.note_min + 1

        self._playing_idx = 0
        self._notes_idx = 1
        self._durations_idx = 1 + self.note_range

    def get_teacher_vector(self, midi_note: int, duration: int) -> np.ndarray:
        """
        Returns the midi value and duration as a teacher vector of size: 1 + NOTE_RANGE + DURATION
        Where 1 corresponds to 0/1 for music playing
        NOTE_RANGE is a one hot encoding for the corresponding note
        DURATION is a one hot encoding for the corresponding duration of the note
        """
        y = np.zeros((1 + self.note_range + self.duration_range))
        # If the midi value is not a pause, fill up the vector with the note
        if midi_note != 0:
            # Set playing music to 1
            y[0] = 1
            # Get the index of the midi value relative to the teacher vector
            note_idx = 1 + midi_note - self.note_min
            y[note_idx] = 1
        # Get the index of the duration relative to the teacher vector
        duration_idx = 1 + self.note_range + duration - self.duration_min
        y[duration_idx] = 1

        return y

    def transform(self, midi_compact: Union[MIDI_COMPACT_SC, np.ndarray]) -> np.ndarray:
        y = []
        for x in midi_compact:
            teacher_vec = self.get_teacher_vector(*x)
            y.append(teacher_vec)

        return np.array(y)

    def get_playing_prob(self, X: np.ndarray) -> np.ndarray:
        return X[:, self._playing_idx]

    def get_notes_prob(self, X: np.ndarray) -> np.ndarray:
        return X[:, self._notes_idx:self._durations_idx]

    def get_durations_prob(self, X: np.ndarray) -> np.ndarray:
        return X[:, self._durations_idx:]

    def inv_transform_maximum_likelihood(self, X: np.ndarray) -> np.ndarray:
        # Ignores playing probability (for now)
        notes_prob = self.get_notes_prob(X)
        durations_prob = self.get_durations_prob(X)
        notes = notes_prob.argmax(axis=1) + self.note_min
        durations = durations_prob.argmax(axis=1) + self.duration_min
        mc_ndarr = np.column_stack((notes, durations))
        # For MIDI_COMPACT_SC: list(map(tuple, mc_ndarr))
        return mc_ndarr


class InputVectorEncoderSC():
    def __init__(self, midi_compact: MIDI_COMPACT_SC) -> None:
        mc_ndarr = np.array(midi_compact)

        midi_notes_masked = np.ma.masked_equal(mc_ndarr[:, 0], 0)
        self.note_min = int(midi_notes_masked.min())
        self.note_max = int(midi_notes_masked.max())

        min_p = 2 * np.log2(2**((self.note_min - 69)/12) * 440)
        max_p = 2 * np.log2(2**((self.note_max - 69)/12) * 440)

        self._pitch_offset = (max_p - min_p)/2 - max_p

    def get_input_vector(self, midi_note: int, duration: int) -> np.ndarray:
        """
        Takes in a MIDI value and duration and returns the
        [pitch, chroma_x, chroma_y, circle5_x, circle5_y, log_duration] as a ndarray
        """

        # Convert midi_note to an int [0,11]
        # 55 is MIDI value of note G3
        note = (midi_note - 55) % 12

        chroma_x, chroma_y = get_chroma_coords(note)
        circle5_x, circle5_y = get_circle5_coords(note)

        # d is the distance (in semitones) of midi_note from A4 (69 in MIDI),
        # whose frequency is 440 Hz. fx is the frequency of the note
        d = midi_note - 69
        fx = 2**(d/12) * 440

        # the representation of pitch is scaled in such a way that a pitch
        # distance of 1 octave in the first dimension, is equal to the distance of
        # notes on the opposite sides on the chroma circle or the circle of fifths
        # In addition the pitch is offset, such that zero alligns with the pitch inbetween min_p and max_p
        pitch = 2 * np.log2(fx) + self._pitch_offset

        log_duration = np.log2(duration)

        return np.array([pitch, chroma_x, chroma_y, circle5_x, circle5_y, log_duration])

    def transform(self, midi_compact: Union[MIDI_COMPACT_SC, np.ndarray]) -> np.ndarray:
        y = []
        for x in midi_compact:
            teacher_vec = self.get_input_vector(*x)
            y.append(teacher_vec)

        return np.array(y)

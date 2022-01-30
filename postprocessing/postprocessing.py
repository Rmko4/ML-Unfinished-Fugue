# For each voice:
#   -Notes of Dminor are more likely
#   -Dminor notes are more likly to start a mesure
#   -Notes are unlikly to start before a Bar and end after the bar ( a mesure usually starts with a note)

#   -Notes of Dminor are more likly to be long
#
# For combination of voices:
#   -Some differences between notes in different voices at a time are less or more likly.

from data_io.vector_encoders import OVE_OUT, OutputVectorEncoderMC
import os
import random
import sys
from typing import List

import numpy as np

# Quick fix to be able to import data_io module from the parent folder
sys.path.append(os.path.join(sys.path[0], '..'))

# raise probabilities to this power to increse high probabilities and reduce smaller ones
POWER_PROBABILITIES = 2
# raise the probability for D_minor notes in the beginning of a measure
INCREASE_DMINOR_PROB_START_MEASURE = 10
# Decrease the liklyhood for notes to start before and end after a bar
DECRESE__ACROSS_BAR_PROB = 0.3
# Increases liklyhood for long notes if choosen note is in Dminor. Should be above 1
D_MINOR_LEN_ADAPT = 5
# Decreases liklyhood for long notes if choosen note is not in Dminor. Should be below 1
NON_DMINOR_LEN_ADAPT = 0.33

# probability of pitch difference between voices( index 0 coresponds to the same pitch, index 1 corresponds to one half note difference etc.)
PROB_VOICE_DIFFERENCES = [8.333, 2.652, 16.919, 59.848, 53.22, 53.409, 31.881, 84.722, 55.366, 76.01, 34.848, 8.586, 77.399, 5.997, 22.096, 72.222, 49.306, 41.793, 26.705, 43.182, 22.096,
                          34.028, 18.939, 5.303, 29.545, 1.894, 8.775, 21.023, 13.258, 8.144, 2.336, 3.157, 2.083, 2.02, 0.758, 0.126, 0.884, 0.253, 0.0, 0.758, 0.0, 0.126, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# #adapt probabilites for each note
ADAPT_PITCH_C = 6
ADAPT_PITCH_D_B = 1
ADAPT_PITCH_D = 6
ADAPT_PITCH_E_B = 1
ADAPT_PITCH_E = 6
ADAPT_PITCH_F = 6
ADAPT_PITCH_G_B = 1
ADAPT_PITCH_G = 6
ADAPT_PITCH_A_B = 1
ADAPT_PITCH_A = 6
ADAPT_PITCH_B_B = 6
ADAPT_PITCH_B = 1
ADAPT_PITCH_NOTES = [ADAPT_PITCH_A, ADAPT_PITCH_B_B, ADAPT_PITCH_B, ADAPT_PITCH_C, ADAPT_PITCH_D_B,
                     ADAPT_PITCH_D, ADAPT_PITCH_E_B, ADAPT_PITCH_E, ADAPT_PITCH_F, ADAPT_PITCH_G_B, ADAPT_PITCH_G, ADAPT_PITCH_A_B]

# adapt length for each note
ADAPT_LENGTH_C = 3
ADAPT_LENGTH_D_B = 1
ADAPT_LENGTH_D = 3
ADAPT_LENGTH_E_B = 1
ADAPT_LENGTH_E = 3
ADAPT_LENGTH_F = 3
ADAPT_LENGTH_G_B = 1
ADAPT_LENGTH_G = 3
ADAPT_LENGTH_A_B = 1
ADAPT_LENGTH_A = 3
ADAPT_LENGTH_B_B = 3
ADAPT_LENGTH_B = 1
ADAPT_LENGTH_NOTES = [ADAPT_LENGTH_A, ADAPT_LENGTH_B_B, ADAPT_LENGTH_B, ADAPT_LENGTH_C, ADAPT_LENGTH_D_B,
                      ADAPT_LENGTH_D, ADAPT_LENGTH_E_B, ADAPT_LENGTH_E, ADAPT_LENGTH_F, ADAPT_LENGTH_G_B, ADAPT_LENGTH_G, ADAPT_LENGTH_A_B, ]


# normalize vector such that it's sum is 1
def normalize_vector(vector):
    sum = 0
    for x in vector:
        sum += x
    return [x / sum for x in vector]


# create to probability vector as predicted by the network without handcrafted postprocessing
# necessary because model output may contain sub-zero numbers
# Takes each value to a power to increase differences between high and low probabilities
def output_to_probability_vector(vector):
    # remove sub zero numbers
    vector = [x if x > 0 else 0 for x in vector]
    vector = normalize_vector(vector)  # Note: Maybe not necessary?
    # take values to power to decrese entropy
    vector = [x ** POWER_PROBABILITIES for x in vector]

    return vector

# adapt probabilty of each note with handcrafted value
# assumes first note is an A


def adapt_prob_for_each_note(prob_notes):
    for i in range(len(prob_notes)):
        prob_notes[i] = prob_notes[i] * ADAPT_LENGTH_NOTES[i % 12]
    return prob_notes


def is_diminor(note):
    # assumes first probability corresponses to A_3 (220	55 Hz)
    d_minor = [0, 1, 3, 5, 7, 8, 10]
    for x in d_minor:
        if note % 12 == x:
            return True
    return False


def increase_dminor_prob(prob_notes):
    for i in range(len(prob_notes)):
        if is_diminor(prob_notes[i]):
            prob_notes[i] *= INCREASE_DMINOR_PROB_START_MEASURE
    return prob_notes


def make_across_bar_probability_unlikley(prob_len, measure_position):
    for i in range(len(prob_len)):
        if measure_position + i/16 > 0:
            prob_len[i] *= DECRESE__ACROSS_BAR_PROB
    return prob_len


def update_based_on_difference_to_other_voices(prob_notes, other_voice):
    for i in range(len(prob_notes)):
        # -37 because assumption of starting A_3 220 Hz
        difference = abs(i - (other_voice - 37))
        if difference < len(PROB_VOICE_DIFFERENCES):
            prob_notes[i] *= 5 * \
                PROB_VOICE_DIFFERENCES[difference]**2 + 1  # TODO Adapt!
    return prob_notes


def choose_from_probs(probs):
    probs = normalize_vector(probs)

    decision = random.choices([x for x in range(len(probs))], probs, k=1)[0]
    return decision


def len_based_on_dminor(prob_len, dminor):
    if dminor:
        weights = np.linspace(1, D_MINOR_LEN_ADAPT, len(prob_len))
    else:
        weights = np.linspace(1, NON_DMINOR_LEN_ADAPT, len(prob_len))

    prob_len = [x * weight for x, weight in zip(prob_len, weights)]
    return prob_len


# inout the raw output of the model  as one probability vector for note and one for length.
# measure_position:  fraction of measure since last bar (0 if start of measure, 15/16 if last 1/16 before new measure starts)
# prob_len: probability for len of measure. index 1 correspons to 1/16, index 16 corresponses to one full measure
def mode_outpout_to_note(prob_notes, prob_len, measure_position=0, notes_other_voices=[None, None, None]):

    prob_notes = output_to_probability_vector(prob_notes)
    prob_len = output_to_probability_vector(prob_len)

    prob_notes = adapt_prob_for_each_note(prob_notes)

    if measure_position == 0:
        prob_notes = increase_dminor_prob(prob_notes)

    prob_len = make_across_bar_probability_unlikley(prob_len, measure_position)

    for other_voice in notes_other_voices:
        if other_voice != None:
            prob_notes = update_based_on_difference_to_other_voices(
                prob_notes, other_voice)

    index_decision = choose_from_probs(prob_notes)
    note = index_decision + 37  # Assumes first note corresponses to 37 A_3 220 Hz

    prob_len = len_based_on_dminor(prob_len, is_diminor(index_decision))

    length = choose_from_probs(prob_len) / 16

    return note, length


class PostProcessorMC:
    def __init__(self, ove: OutputVectorEncoderMC, Y_prior: np.ndarray,
                 measure_length: int):
        '''
        ove: The output vector encoder of multi channel music
        Y_prior: The midi values that are used to identify what midi values
        have been generated upto this point.
        '''

        self.ove = ove

        # The length in symbols of a single measure.
        self.measure_length = measure_length
        # The current position within a measure in symbols (starts at 0).
        self.measure_idx = (Y_prior.shape[0] - 1) % self.measure_length

        # The duration for which the current note is playing (minimum 1).
        self.duration = np.empty((ove.n_channels))
        # The current note that is playing.
        self.current_note = np.empty((ove.n_channels))
        # The note prior to the current note.
        self.prev_note = np.empty((ove.n_channels))

        # A True or False mask indicating where the note changes
        # in the next time step for a each channel.
        change_indicator = np.diff(Y_prior, axis=0) != 0

        for channel in range(ove.n_channels):
            mask = change_indicator[:, channel]
            # The last time step at which the note changed
            last_change_idx = np.where(mask)[0][-1]

            # The previous note is still played at index last_change_idx
            self.prev_note[channel] = Y_prior[last_change_idx, channel]
            # The last note that is played for that channel
            self.current_note[channel] = Y_prior[-1, channel]
            # The new note starts at the following index, therefore the
            # duration is minus one.
            self.duration[channel] = Y_prior.shape[0] - last_change_idx - 1

        # What I need as an additional input for the function "post_process_output" is:
        # -How long is the current note already playing?
        # -What was the pitch of the note before that?
        # -What is the current position in the measure?

    def __call__(self, y: OVE_OUT, *args) -> np.ndarray:
        '''
        This function is called when the object is called.
        Here should be the main body of the post processing functionality.
        Any additional keyword arguments that are passed on can be ignored.
        '''
        # Initially go to the next position in the measure or into a new measure
        self.forward_measure()

        midi_notes = []
        output_vectors = []

        for channel in range(self.ove.n_channels):
            if self.ove.flatten:
                output_vectors.append(
                    self.ove.output_vector_channel(y, channel)[0])
            else:
                output_vectors.append(y[channel][0])

        for i, output_vector in enumerate(output_vectors):
            midi_notes.append(random.choices(
                [0] + list(range(self.ove.note_min[i], self.ove.note_max[i] + 1)), output_vector, k=1)[0])

        # output_vectors is now a list of numpy arrays constructed from the
        # raw output (y: OVE_OUT).
        # Indexing the list will yield the probabilities (np.ndarray) of the
        # notes for channel 0 to ove.n_channels.
        # The probability at index 0 (ove.playing_idx) is the probability to
        # play no note.
        # ove.note_min[channel] will give the minimum note that can be played
        # by that channel. This is represented by index 1 (ove.notes_idx) for
        # that channel.
        # E.g. the probability that channel (or voice) 2 plays the lowest note
        # is output_vectors[2][1]. The midi value belonging to this probability
        # is ove.note_min[2].
        # The last value in the probability array will correspond to ove.note_max[channel]
        # The vocal range of notes is ove.note_range[channel].

        # Also note that in case of the MLP that the output_vectors for each channel
        # are an actual probability vector. They sum to one and are all positive
        # values, including whether to play or not. In case of linear and ridge
        # regression this is not the case.

        # The output should be a numpy array of length ove.n_channels,
        # containing the integers of the midi value.
        return np.array(midi_notes)

        # I hope that helps - Remco

    def forward_measure(self):
        '''
        Go one position forward in the measure.
        0 is the first time step in a new measure.
        If self.measure_length is 16, then 15 is the last time step in that measure
        '''
        self.measure_idx = (self.measure_idx + 1) % self.measure_length

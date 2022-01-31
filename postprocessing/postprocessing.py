# For each voice:
#   -Notes of Dminor are more likely
#   -Dminor notes are more likly to start a mesure
#   -Notes are unlikly to start before a Bar and end after the bar ( a mesure usually starts with a note)

#   -Notes of Dminor are more likly to be long
#
# For combination of voices:
#   -Some differences between notes in different voices at a time are less or more likly.

from unittest.util import _MAX_LENGTH
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
#Maximum length possible
MAX_LENGTH = 20
#probability adaptation for possible length, missing key means probability of 0
LENGTH_PROBABILITIES = {1:50,  2:2, 4: 3, 6:12, 8:3, 10:5, 12: 3, 14:8, 16:1, 18:1, 20:1}


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

        for i  in range(len(output_vectors)):
            output_vectors[i] = self.output_to_probability_vector(output_vectors[i])

            if self.prev_note[i] == 0:
                index_last_note = 0 
            else:
                index_last_note = int(1 + self.prev_note[i] - self.ove.note_min[i])
            #adapt length probabilities
            try:
                output_vectors[i][index_last_note] *= LENGTH_PROBABILITIES[int(self.duration[i])]
            except KeyError:
                #last note MUST be repeated
                output_vectors[i] = [ 1 if x == index_last_note else 0 for x in range(len(output_vectors[i]))]

            
            if self.duration[i] == MAX_LENGTH:
                #A different note must be played
                output_vectors[i][index_last_note] = 0 
                if sum(output_vectors[i]) == 0:
                    #edge case in which no note is good - a random one is taken
                    output_vectors[i] = [1 for x in output_vectors[i] ]




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

        for i, output_vector in enumerate(output_vectors):
            midi_notes.append(random.choices(
                [0] + list(range(self.ove.note_min[i], self.ove.note_max[i] + 1)), output_vector, k=1)[0])

        self.remember_state(midi_notes)
        return np.array(midi_notes)

        # I hope that helps - Remco

    def remember_state(self,midi_notes):
        for i in range(len(midi_notes)):
            if midi_notes[i] == self.prev_note[i]:
                self.duration[i] +=1
            else:
                self.duration[i] = 1
            self.prev_note[i] = midi_notes[i]
        
    def forward_measure(self):
        '''
        Go one position forward in the measure.
        0 is the first time step in a new measure.
        If self.measure_length is 16, then 15 is the last time step in that measure
        '''
        self.measure_idx = (self.measure_idx + 1) % self.measure_length

    # normalize vector such that it's sum is 1
    def normalize_vector(self,vector):
        sum = 0
        for x in vector:
            sum += x
        return [x / sum for x in vector]


    # create to probability vector
    # model output may contain sub-zero numbers
    # Takes each value to a power to increase differences between high and low probabilities
    def output_to_probability_vector(self,vector):
        # remove sub zero numbers
        vector = [x if x > 0 else 0 for x in vector]
        vector = self.normalize_vector(vector)  # Note: Maybe not necessary?
        # take values to power to decrese entropy
        vector = [x ** POWER_PROBABILITIES for x in vector]

        return vector
